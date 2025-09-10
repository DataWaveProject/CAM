!-----------------------------------------------------------------------------
! utilities to gather and distribute columns and remap them to/from
! cubed-sphere grid to a lat-lon grid.
!-----------------------------------------------------------------------------
module nlgw_remap_mod
  use shr_kind_mod, only: r8 => shr_kind_r8, cx => SHR_KIND_CX
  use ppgrid, only: begchunk, endchunk, pcols, pver, pverp
  use physics_types, only: physics_state
  use phys_grid, only: get_ncols_p
  use spmd_utils, only: masterproc, npes
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

  ! these arrays contain the regridded variables of interest for the NN
  real(r8), dimension(:, :), allocatable, public :: phis_grid
  real(r8), dimension(:,:,:), allocatable, public :: u_grid, v_grid, w_grid, t_grid
  real(r8), dimension(:,:,:), allocatable, public :: utgw_grid, vtgw_grid

contains

  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  subroutine nlgw_regrid_init()
    use cam_grid_support, only: horiz_coord_t, horiz_coord_create, iMap, cam_grid_register
    use esmf_lonlat_grid_mod, only: glats, nlat, glons, nlon
    use esmf_lonlat_grid_mod, only: esmf_lonlat_grid_init
    use esmf_phys_mesh_mod, only: esmf_phys_mesh_init
    use esmf_phys2lonlat_mod, only: esmf_phys2lonlat_init
    use esmf_lonlat2phys_mod, only: esmf_lonlat2phys_init

    integer, parameter :: reg_decomp = 332

    integer(iMap),       pointer :: grid_map(:,:)

    integer(iMap),       pointer :: coord_map(:) => null()
    type(horiz_coord_t), pointer :: lon_coord
    type(horiz_coord_t), pointer :: lat_coord
    integer :: i, j, ind, astat

    character(len=*), parameter :: subname = 'ctem_diags_reg: '

    ! initialize grids and mapping
    call esmf_lonlat_grid_init(64, 128)
    call esmf_phys_mesh_init()
    call esmf_phys2lonlat_init()
    call esmf_lonlat2phys_init()

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
    use esmf_lonlat_grid_mod, only: nlat, nlon
    use esmf_phys2lonlat_mod, only: esmf_phys2lonlat_regrid
    use esmf_zonal_mean_mod, only: esmf_zonal_mean_calc, esmf_zonal_mean_wsums, esmf_zonal_mean_masked
    use interpolate_data, only: lininterp
    use esmf_phys2lonlat_mod, only: fields_bundle_t, nflds
    use esmf_lonlat2phys_mod, only: esmf_lonlat2phys_regrid, n_flx_flds
    use mpishorthand

    type(physics_state), intent(in) :: phys_state(begchunk:endchunk)

    ! arrays on physics grid
    real(r8), target :: u_phys(pver,pcols,begchunk:endchunk)
    real(r8), target :: v_phys(pver,pcols,begchunk:endchunk)
    real(r8), target :: w_phys(pver,pcols,begchunk:endchunk)
    real(r8), target :: t_phys(pver,pcols,begchunk:endchunk)
    real(r8) :: phis_phys(pcols,begchunk:endchunk)
    ! for debugging only
    ! real(r8) :: lat_phys(pcols,begchunk:endchunk)
    ! real(r8) :: lon_phys(pcols,begchunk:endchunk)

    ! arrays on latlon grid
    real(r8), target :: u_lonlat(beglon:endlon,beglat:endlat,pver)
    real(r8), target :: v_lonlat(beglon:endlon,beglat:endlat,pver)
    real(r8), target :: w_lonlat(beglon:endlon,beglat:endlat,pver)
    real(r8), target :: t_lonlat(beglon:endlon,beglat:endlat,pver)
    real(r8) :: phis_lonlat(beglon:endlon,beglat:endlat)

    real(r8), allocatable :: flat_array(:)

    integer  :: lchnk, ncol, i, sendcnt, disp_sum
    integer  :: lonsize, latsize

    integer, allocatable :: recvcnts(:), displs(:)
    integer, allocatable :: beglats(:), beglons(:)
    integer, allocatable :: endlats(:), endlons(:)

    type(fields_bundle_t) :: physflds(nflds)
    type(fields_bundle_t) :: lonlatflds(nflds)

    type(fields_bundle_t) :: phys_flx_flds(n_flx_flds)
    type(fields_bundle_t) :: lonlat_flx_flds(n_flx_flds)

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

          phis_phys(i,lchnk) = phys_state(lchnk)%ps(i)
          ! for debugging only
          ! lat_phys(i,lchnk) = phys_state(lchnk)%lat(i)
          ! lon_phys(i,lchnk) = phys_state(lchnk)%lon(i)

       end do
    end do

    call t_stopf('nlgw_unchunk')

    call t_startf('nlgw_regrid')
    ! this subsection does regridding

    physflds(1)%fld => u_phys
    physflds(2)%fld => v_phys
    physflds(3)%fld => w_phys
    physflds(4)%fld => t_phys

    lonlatflds(1)%fld => u_lonlat
    lonlatflds(2)%fld => v_lonlat
    lonlatflds(3)%fld => w_lonlat
    lonlatflds(4)%fld => t_lonlat

    ! actual call to regrid to lon/lat grid
    call esmf_phys2lonlat_regrid(physflds, lonlatflds)
    call esmf_phys2lonlat_regrid(phis_phys, phis_lonlat)

    ! TODO
    ! convert t to theta before gathering
    ! we dont need ps we need phis

    call t_stopf('nlgw_regrid')

    call t_startf('nlgw_mpigather')
    ! this subsection gathers all variables onto a single process

    allocate(recvcnts(npes))
    allocate(displs(npes))
    allocate(beglats(npes))
    allocate(beglons(npes))
    allocate(endlats(npes))
    allocate(endlons(npes))

    sendcnt = (endlon - beglon + 1) * (endlat - beglat + 1)

    ! mpi gather book-keeping
    call mpigather(sendcnt, 1, mpiint, recvcnts, 1, mpiint, 0, mpicom)
    call mpigather(beglat, 1, mpiint, beglats, 1, mpiint, 0, mpicom)
    call mpigather(beglon, 1, mpiint, beglons, 1, mpiint, 0, mpicom)
    call mpigather(endlat, 1, mpiint, endlats, 1, mpiint, 0, mpicom)
    call mpigather(endlon, 1, mpiint, endlons, 1, mpiint, 0, mpicom)

    if (masterproc) then
      disp_sum = 0
      do i = 1, npes
        displs(i) = disp_sum
        disp_sum = disp_sum + recvcnts(i)
      end do
    end if

    allocate(flat_array(nlon * nlat))

    call gather_2d(phis_lonlat(beglon:endlon, beglat:endlat), sendcnt, flat_array, recvcnts, displs, &
                  phis_grid, beglons, endlons, beglats, endlats)


    sendcnt = sendcnt * pver
    if (masterproc) then
      do i = 1, npes
        displs(i) = displs(i) * pver
        recvcnts(i) = recvcnts(i) * pver
      end do
    end if
    deallocate(flat_array)
    allocate(flat_array(nlon * nlat * pver))

    call gather_3d(u_lonlat(beglon:endlon, beglat:endlat, 1:pver), sendcnt, flat_array, recvcnts, displs, &
                u_grid, beglons, endlons, beglats, endlats)
    call gather_3d(v_lonlat(beglon:endlon, beglat:endlat, 1:pver), sendcnt, flat_array, recvcnts, displs, &
                v_grid, beglons, endlons, beglats, endlats)
    call gather_3d(w_lonlat(beglon:endlon, beglat:endlat, 1:pver), sendcnt, flat_array, recvcnts, displs, &
                w_grid, beglons, endlons, beglats, endlats)
    call gather_3d(t_lonlat(beglon:endlon, beglat:endlat, 1:pver), sendcnt, flat_array, recvcnts, displs, &
                t_grid, beglons, endlons, beglats, endlats)

    call t_stopf('nlgw_mpigather')

    ! TODO
    ! convert fluxes to tendencies after regridding back to cubed sphere
    ! that way we dont need pmid

    call t_stopf('nlgw_gather')

  end subroutine nlgw_regrid

  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  subroutine gather_2d(local_array, sendcnt, flat_array, recvcnts, displs, grid_out, beglons, endlons, beglats, endlats)
    use mpishorthand
    use esmf_lonlat_grid_mod, only: nlat, nlon
    real(r8), intent(in) :: local_array(:,:)  ! Local 2D array section
    integer, intent(in) :: sendcnt
    real(r8), intent(inout) :: flat_array(:)     ! Flattened array for gathering
    integer, intent(in) :: recvcnts(:), displs(:)
    real(r8), allocatable, intent(out) :: grid_out(:,:)    ! Full gathered grid
    integer, intent(in) :: beglons(:), endlons(:)
    integer, intent(in) :: beglats(:), endlats(:)

    integer :: i, lonsize, latsize

    ! gather variables onto master proc into a flat array (can't do 2D/3D mpigather)
    call mpigatherv(local_array, sendcnt, mpir8, flat_array, recvcnts, displs, mpir8, 0, mpicom)

    if (masterproc) then
        allocate(grid_out(nlon, nlat))
        do i = 1, npes
            lonsize = endlons(i) - beglons(i) + 1
            latsize = endlats(i) - beglats(i) + 1
            ! reshape each ranks flattended data and populate each block into a single lonlat grid
            grid_out(beglons(i):endlons(i), beglats(i):endlats(i)) = &
                reshape(flat_array(displs(i)+1:displs(i)+sendcnt), (/ lonsize, latsize /))
        end do
    end if
  end subroutine gather_2d

  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  subroutine gather_3d(local_array, sendcnt, flat_array, recvcnts, displs, grid_out, beglons, endlons, beglats, endlats)
    use mpishorthand
    use esmf_lonlat_grid_mod, only: nlat, nlon
    real(r8), intent(in) :: local_array(:,:,:)  ! Local 2D array section
    integer, intent(in) :: sendcnt
    real(r8), intent(inout) :: flat_array(:)     ! Flattened array for gathering
    integer, intent(in) :: recvcnts(:), displs(:)
    real(r8), allocatable, intent(out) :: grid_out(:,:,:)    ! Full gathered grid
    integer, intent(in) :: beglons(:), endlons(:)
    integer, intent(in) :: beglats(:), endlats(:)

    integer :: i, lonsize, latsize

    ! gather variables onto master proc into a flat array (can't do 2D/3D mpigather)
    call mpigatherv(local_array, sendcnt, mpir8, flat_array, recvcnts, displs, mpir8, 0, mpicom)

    if (masterproc) then
        allocate(grid_out(nlon, nlat, pver))
        do i = 1, npes
            lonsize = endlons(i) - beglons(i) + 1
            latsize = endlats(i) - beglats(i) + 1
            ! reshape each ranks flattended data and populate each block into a single lonlat grid
            grid_out(beglons(i):endlons(i), beglats(i):endlats(i), 1:pver) = &
                reshape(flat_array(displs(i)+1:displs(i)+sendcnt), (/ lonsize, latsize, pver /))
        end do
    end if
  end subroutine gather_3d

  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  subroutine nlgw_regrid_final()
    use esmf_phys2lonlat_mod, only: esmf_phys2lonlat_destroy
    use esmf_lonlat2phys_mod, only: esmf_lonlat2phys_destroy
    use esmf_lonlat_grid_mod, only: esmf_lonlat_grid_destroy
    use esmf_phys_mesh_mod, only: esmf_phys_mesh_destroy

    call esmf_phys2lonlat_destroy()
    call esmf_lonlat2phys_destroy()
    call esmf_lonlat_grid_destroy()
    call esmf_phys_mesh_destroy()

    if (masterproc) then
      ! TODO ALL deallocates here
      deallocate(phis_grid)
      deallocate(u_grid)
      deallocate(v_grid)
      deallocate(w_grid)
      deallocate(t_grid)
      deallocate(utgw_grid)
      deallocate(vtgw_grid)
    end if


  end subroutine nlgw_regrid_final

end module nlgw_remap_mod
