!-----------------------------------------------------------------------------
! utilities to gather and distribute columns and remap them to/from
! cubed-sphere grid to a lat-lon grid.
!-----------------------------------------------------------------------------
module nlgw_remap_mod
  use shr_kind_mod, only: r8 => shr_kind_r8, cx => SHR_KIND_CX
  use ppgrid, only: begchunk, endchunk, pcols, pver, pverp
  use physics_types, only: physics_state
  use phys_grid, only: get_ncols_p
  use spmd_utils, only: masterproc
  use ref_pres, only: pref_mid
  use esmf_lonlat_grid_mod, only: beglon=>lon_beg, endlon=>lon_end, beglat=>lat_beg, endlat=>lat_end
  use cam_history,  only: addfld, outfld, horiz_only
  use cam_history_support, only : fillvalue
  use perf_mod, only: t_startf, t_stopf
  use cam_logfile, only: iulog
  use cam_abortutils, only: endrun

  implicit none

  private

  public :: nlgw_regrid_init
  public :: nlgw_regrid
  public :: nlgw_regrid_final

contains

  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  subroutine nlgw_regrid_init()
    use cam_grid_support, only: horiz_coord_t, horiz_coord_create, iMap, cam_grid_register
    use esmf_lonlat_grid_mod, only: glats, nlat, glons, nlon
    use esmf_lonlat_grid_mod, only: esmf_lonlat_grid_init
    use esmf_phys_mesh_mod, only: esmf_phys_mesh_init
    use esmf_phys2lonlat_mod, only: esmf_phys2lonlat_init

    integer, parameter :: reg_decomp = 332

    integer(iMap),       pointer :: grid_map(:,:)

    integer(iMap),       pointer :: coord_map(:) => null()
    type(horiz_coord_t), pointer :: lon_coord
    type(horiz_coord_t), pointer :: lat_coord
    integer :: i, j, ind, astat

    character(len=*), parameter :: subname = 'ctem_diags_reg: '

    ! initialize grids and mapping
    call esmf_lonlat_grid_init(64)
    call esmf_phys_mesh_init()
    call esmf_phys2lonlat_init()

    ! for the lon-lat grid
    allocate(grid_map(4, ((endlon - beglon + 1) * (endlat - beglat + 1))), stat=astat)
    if (astat/=0) then
       call endrun(subname//'not able to allocate grid_map array')
    end if

    ind = 0
    do i = beglat, endlat
       do j = beglon, endlon
          ind = ind + 1
          grid_map(1, ind) = j
          grid_map(2, ind) = i
          grid_map(3, ind) = j
          grid_map(4, ind) = i
       end do
    end do

    allocate(coord_map(endlat - beglat + 1), stat=astat)
    if (astat/=0) then
       call endrun(subname//'not able to allocate coord_map array')
    end if

    if (beglon==1) then
       coord_map = (/ (i, i = beglat, endlat) /)
    else
       coord_map = 0
    end if
    lat_coord => horiz_coord_create('reglat', '', nlat, 'latitude',  'degrees_north', beglat, endlat, &
                                    glats(beglat:endlat),  map=coord_map)

    nullify(coord_map)

    allocate(coord_map(endlon - beglon + 1), stat=astat)
    if (astat/=0) then
       call endrun(subname//'not able to allocate coord_map array')
    end if

    if (beglat==1) then
       coord_map = (/ (i, i = beglon, endlon) /)
    else
       coord_map = 0
    end if

    lon_coord => horiz_coord_create('reglon', '', nlon, 'longitude',  'degrees_east', beglon, endlon, &
                                    glons(beglon:endlon),  map=coord_map)

    nullify(coord_map)

    call cam_grid_register('ctem_lonlat', reg_decomp, lat_coord, lon_coord, grid_map, unstruct=.false.)

    nullify(grid_map)

  end subroutine nlgw_regrid_init

  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  subroutine nlgw_regrid(phys_state)
    use air_composition, only: mbarv ! g/mole
    use shr_const_mod, only: rgas => shr_const_rgas ! J/K/kmole
    use shr_const_mod, only: grav => shr_const_g ! m/s2
    use esmf_phys2lonlat_mod, only: esmf_phys2lonlat_regrid
    use esmf_zonal_mean_mod, only: esmf_zonal_mean_calc, esmf_zonal_mean_wsums, esmf_zonal_mean_masked
    use interpolate_data, only: lininterp
    use esmf_phys2lonlat_mod, only: fields_bundle_t, nflds

    type(physics_state), intent(in) :: phys_state(begchunk:endchunk)

    ! arrays on physics grid
    real(r8), target :: u_phys(pver,pcols,begchunk:endchunk)
    real(r8), target :: v_phys(pver,pcols,begchunk:endchunk)
    real(r8), target :: w_phys(pver,pcols,begchunk:endchunk)
    real(r8), target :: t_phys(pver,pcols,begchunk:endchunk)
    real(r8), target :: pmid_phys(pver,pcols,begchunk:endchunk)
    real(r8) :: ps_phys(pcols,begchunk:endchunk)
    real(r8) :: lat_phys(pcols,begchunk:endchunk)
    real(r8) :: lon_phys(pcols,begchunk:endchunk)

    ! arrays on latlon grid
    real(r8), target :: u_lonlat(beglon:endlon,beglat:endlat,pver)
    real(r8), target :: v_lonlat(beglon:endlon,beglat:endlat,pver)
    real(r8), target :: w_lonlat(beglon:endlon,beglat:endlat,pver)
    real(r8), target :: t_lonlat(beglon:endlon,beglat:endlat,pver)
    real(r8), target :: pmid_lonlat(beglon:endlon,beglat:endlat,pver)
    real(r8) :: ps_lonlat(beglon:endlon,beglat:endlat)
    real(r8) :: lat_lonlat(beglon:endlon,beglat:endlat)
    real(r8) :: lon_lonlat(beglon:endlon,beglat:endlat)

    integer  :: lchnk, ncol, i

    type(fields_bundle_t) :: physflds(nflds)
    type(fields_bundle_t) :: lonlatflds(nflds)

    call t_startf('nlgw_gather')

    call t_startf('nlgw_unchunk')

    do lchnk = begchunk,endchunk
       ncol = phys_state(lchnk)%ncol
       do i = 1,ncol
          ! wind components
          u_phys(:,i,lchnk)    = phys_state(lchnk)%u(i,:)
          v_phys(:,i,lchnk)    = phys_state(lchnk)%v(i,:)
          w_phys(:,i,lchnk)    = phys_state(lchnk)%omega(i,:)
          t_phys(:,i,lchnk)    = phys_state(lchnk)%t(i,:)
          pmid_phys(:,i,lchnk) = phys_state(lchnk)%pmid(i,:)

          ps_phys(i,lchnk) = phys_state(lchnk)%ps(i)
          lat_phys(i,lchnk) = phys_state(lchnk)%lat(i)
          lon_phys(i,lchnk) = phys_state(lchnk)%lon(i)

       end do
    end do

    call t_stopf('nlgw_unchunk')

    call t_startf('nlgw_regrid')

    ! regrid to lon/lat grid

    physflds(1)%fld => u_phys
    physflds(2)%fld => v_phys
    physflds(3)%fld => w_phys
    physflds(4)%fld => t_phys
    physflds(5)%fld => pmid_phys

    lonlatflds(1)%fld => u_lonlat
    lonlatflds(2)%fld => v_lonlat
    lonlatflds(3)%fld => w_lonlat
    lonlatflds(4)%fld => t_lonlat
    lonlatflds(5)%fld => pmid_lonlat

    call esmf_phys2lonlat_regrid(physflds, lonlatflds)

    call esmf_phys2lonlat_regrid(ps_phys, ps_lonlat)
    call esmf_phys2lonlat_regrid(lat_phys, lat_lonlat)
    call esmf_phys2lonlat_regrid(lon_phys, lon_lonlat)

    call t_stopf('nlgw_regrid')

    call t_stopf('nlgw_gather')

  end subroutine nlgw_regrid

  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  subroutine nlgw_regrid_final()
    use esmf_phys2lonlat_mod, only: esmf_phys2lonlat_destroy
    use esmf_lonlat_grid_mod, only: esmf_lonlat_grid_destroy
    use esmf_phys_mesh_mod, only: esmf_phys_mesh_destroy

    if (.not.ctem_diags_active) return

    call esmf_phys2lonlat_destroy()
    call esmf_lonlat_grid_destroy()
    call esmf_phys_mesh_destroy()

  end subroutine nlgw_regrid_final

end module nlgw_remap_mod
