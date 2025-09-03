!------------------------------------------------------------------------------
! Provides methods for mapping from regular longitude / latitude grid
! to physics grid to via ESMF regridding capabilities
!------------------------------------------------------------------------------
module esmf_lonlat2phys_mod
  use shr_kind_mod, only: r8 => shr_kind_r8
  use cam_logfile,  only: iulog
  use cam_abortutils, only: endrun
  use spmd_utils, only: masterproc
  use ppgrid, only: pver

  use ESMF, only: ESMF_RouteHandle, ESMF_Field, ESMF_ArraySpec, ESMF_ArraySpecSet
  use ESMF, only: ESMF_FieldCreate, ESMF_FieldRegridStore
  use ESMF, only: ESMF_FieldGet, ESMF_FieldRegrid
  use ESMF, only: ESMF_KIND_I4, ESMF_KIND_R8, ESMF_TYPEKIND_R8
  use ESMF, only: ESMF_REGRIDMETHOD_BILINEAR, ESMF_POLEMETHOD_NONE, ESMF_EXTRAPMETHOD_NEAREST_IDAVG
  use ESMF, only: ESMF_TERMORDER_SRCSEQ, ESMF_MESHLOC_ELEMENT, ESMF_STAGGERLOC_CENTER
  use ESMF, only: ESMF_FieldDestroy, ESMF_RouteHandleDestroy
  use esmf_check_error_mod, only: check_esmf_error

  implicit none

  private

  public :: esmf_lonlat2phys_init
  public :: esmf_lonlat2phys_regrid
  public :: esmf_lonlat2phys_destroy
  public :: fields_bundle_t
  public :: n_flx_flds

  type(ESMF_RouteHandle) :: rh_lonlat2phys_3d

  type(ESMF_Field) :: physfld_3d
  type(ESMF_Field) :: lonlatfld_3d

  interface esmf_lonlat2phys_regrid
     module procedure esmf_lonlat2phys_regrid_3d
  end interface esmf_lonlat2phys_regrid

  type :: fields_bundle_t
     real(r8), pointer :: fld(:,:,:) => null()
  end type fields_bundle_t

  integer, parameter :: n_flx_flds = 2 ! 3D uflux and vflux

contains

  !------------------------------------------------------------------------------
  !------------------------------------------------------------------------------
  subroutine esmf_lonlat2phys_init()
    use esmf_phys_mesh_mod, only: physics_grid_mesh
    use esmf_lonlat_grid_mod, only: lonlat_grid

    type(ESMF_ArraySpec) :: arrayspec
    integer(ESMF_KIND_I4), pointer :: factorIndexList(:,:)
    real(ESMF_KIND_R8),    pointer :: factorList(:)
    integer                        :: smm_srctermproc,  smm_pipelinedep, rc

    character(len=*), parameter :: subname  = 'esmf_lonlat2phys_init: '

    smm_srctermproc = 0
    smm_pipelinedep = 16

    ! create ESMF fields

    ! 3D phys fld
    call ESMF_ArraySpecSet(arrayspec, 3, ESMF_TYPEKIND_R8, rc=rc)
    call check_esmf_error(rc, subname//'ESMF_ArraySpecSet 3D phys fld ERROR')

    physfld_3d = ESMF_FieldCreate(physics_grid_mesh, arrayspec, &
                                  gridToFieldMap=(/3/), meshloc=ESMF_MESHLOC_ELEMENT, &
                                  ungriddedLBound=(/1,1/), ungriddedUBound=(/pver,n_flx_flds/), rc=rc)
    call check_esmf_error(rc, subname//'ESMF_FieldCreate 3D phys fld ERROR')

    ! 3D lon lat grid
    call ESMF_ArraySpecSet(arrayspec, 4, ESMF_TYPEKIND_R8, rc=rc)
    call check_esmf_error(rc, subname//'ESMF_ArraySpecSet 3D lonlat fld ERROR')

    lonlatfld_3d = ESMF_FieldCreate( lonlat_grid, arrayspec, staggerloc=ESMF_STAGGERLOC_CENTER, &
                                     ungriddedLBound=(/1,1/), ungriddedUBound=(/pver,n_flx_flds/), rc=rc)
    call check_esmf_error(rc, subname//'ESMF_FieldCreate 3D lonlat fld ERROR')

    call ESMF_FieldRegridStore(srcField=lonlatfld_3d, dstField=physfld_3d, &
         regridMethod=ESMF_REGRIDMETHOD_BILINEAR,                          &
         polemethod=ESMF_POLEMETHOD_NONE,                                &
         extrapMethod=ESMF_EXTRAPMETHOD_NEAREST_IDAVG,                     &
         routeHandle=rh_lonlat2phys_3d, factorIndexList=factorIndexList,   &
         factorList=factorList, srcTermProcessing=smm_srctermproc,         &
         pipelineDepth=smm_pipelinedep, rc=rc)
    call check_esmf_error(rc, subname//'ESMF_FieldRegridStore 3D routehandle ERROR')

  end subroutine esmf_lonlat2phys_init

  !------------------------------------------------------------------------------
  !------------------------------------------------------------------------------
  subroutine esmf_lonlat2phys_regrid_3d(lonlatflds, physflds)
    use esmf_lonlat_grid_mod, only: lon_beg,lon_end,lat_beg,lat_end
    use ppgrid, only: pcols, pver, begchunk, endchunk
    use phys_grid, only: get_ncols_p

    type(fields_bundle_t), intent(in) :: lonlatflds(n_flx_flds)
    type(fields_bundle_t), intent(inout) :: physflds(n_flx_flds)

    integer :: i, ichnk, ncol, ifld, ilev, icol, rc
    real(ESMF_KIND_R8), pointer :: physptr(:,:,:)
    real(ESMF_KIND_R8), pointer :: lonlatptr(:,:,:,:)

    character(len=*), parameter :: subname = 'esmf_lonlat2phys_regrid_3d: '

    ! set values of lonlatfld_3d ESMF field
    call ESMF_FieldGet(lonlatfld_3d, localDe=0, farrayPtr=lonlatptr, rc=rc)
    call check_esmf_error(rc, subname//'ESMF_FieldGet lonlatptr')

    do ifld = 1,n_flx_flds
       lonlatptr(lon_beg:lon_end,lat_beg:lat_end,1:pver,ifld) = lonlatflds(ifld)%fld(lon_beg:lon_end,lat_beg:lat_end,1:pver)
    end do

    ! regrid
    call ESMF_FieldRegrid(lonlatfld_3d, physfld_3d, rh_lonlat2phys_3d, &
              termorderflag=ESMF_TERMORDER_SRCSEQ, rc=rc)
    call check_esmf_error(rc, subname//'ESMF_FieldRegrid lonlatfld_3d->physfld_3d')

    ! get values from physfld_3d ESMF field
    call ESMF_FieldGet(physfld_3d, localDe=0, farrayPtr=physptr, rc=rc)
    call check_esmf_error(rc, subname//'ESMF_FieldGet physptr')

    i = 0
    do ichnk = begchunk, endchunk
       ncol = get_ncols_p(ichnk)
       do icol = 1,ncol
          i = i+1
          do ifld = 1,n_flx_flds
             do ilev = 1,pver
                physflds(ifld)%fld(ilev,icol,ichnk) = physptr(ilev,ifld,i)
             end do
          end do
       end do
    end do

  end subroutine esmf_lonlat2phys_regrid_3d

  !------------------------------------------------------------------------------
  !------------------------------------------------------------------------------
  subroutine esmf_lonlat2phys_destroy()

    integer :: rc
    character(len=*), parameter :: subname = 'esmf_lonlat2phys_destroy: '

    call ESMF_RouteHandleDestroy(rh_lonlat2phys_3d, rc=rc)
    call check_esmf_error(rc, subname//'ESMF_FieldDestroy rh_lonlat2phys_3d')

    call ESMF_FieldDestroy(lonlatfld_3d, rc=rc)
    call check_esmf_error(rc, subname//'ESMF_FieldDestroy lonlatfld_3d')

    call ESMF_FieldDestroy(physfld_3d, rc=rc)
    call check_esmf_error(rc, subname//'ESMF_FieldDestroy physfld_3d')

  end subroutine esmf_lonlat2phys_destroy

end module esmf_lonlat2phys_mod
