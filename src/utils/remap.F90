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
  use gw_nlgw_utils, only: phys_vars, lonlat_vars

  implicit none

  private

  public :: nlgw_regrid_init
  public :: nlgw_latlon_gather
  public :: nlgw_latlon_scatter
  public :: nlgw_regrid_final

  ! private (for book-keeping in MPI calls)
  integer, allocatable :: recvcnts(:), displs(:)
  integer, allocatable :: beglats(:), beglons(:)
  integer, allocatable :: endlats(:), endlons(:)

  logical, parameter, public :: debug = .true.


contains

  !-----------------------------------------------------------------------------
  ! Initialize arrays and grids for regridding/MPI calls
  !-----------------------------------------------------------------------------
  subroutine nlgw_regrid_init(phys, lonlat, gathered_lonlat)
    use cam_grid_support,     only: horiz_coord_t, horiz_coord_create, iMap, cam_grid_register
    use esmf_lonlat_grid_mod, only: glats, glons
    use esmf_lonlat_grid_mod, only: esmf_lonlat_grid_init
    use esmf_phys_mesh_mod,   only: esmf_phys_mesh_init
    use esmf_phys2lonlat_mod, only: esmf_phys2lonlat_init
    use esmf_lonlat2phys_mod, only: esmf_lonlat2phys_init
    use gw_nlgw_utils,        only: nlon, nlat

    integer, parameter :: reg_decomp = 332

    integer(iMap),       pointer :: grid_map(:,:)

    integer(iMap),       pointer :: coord_map(:) => null()
    type(horiz_coord_t), pointer :: lon_coord
    type(horiz_coord_t), pointer :: lat_coord
    integer :: i, j, ind, astat

    type(phys_vars), intent(inout), target :: phys
    type(lonlat_vars), intent(inout), target :: lonlat
    type(lonlat_vars), intent(inout), target :: gathered_lonlat

    character(len=*), parameter :: subname = 'ctem_diags_reg: '

    ! initialize grids and mapping
    call esmf_lonlat_grid_init(nlat, nlon)
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

    allocate(recvcnts(npes))
    allocate(displs(npes))
    allocate(beglats(npes))
    allocate(beglons(npes))
    allocate(endlats(npes))
    allocate(endlons(npes))

    allocate(phys%u(pver,pcols,begchunk:endchunk))
    allocate(phys%v(pver,pcols,begchunk:endchunk))
    allocate(phys%theta(pver,pcols,begchunk:endchunk))
    allocate(phys%w(pver,pcols,begchunk:endchunk))
    allocate(phys%uflux(pver,pcols,begchunk:endchunk))
    allocate(phys%vflux(pver,pcols,begchunk:endchunk))
    allocate(phys%utgw(pver,pcols,begchunk:endchunk))
    allocate(phys%vtgw(pver,pcols,begchunk:endchunk))


    allocate(lonlat%u(beglon:endlon,beglat:endlat,pver))
    allocate(lonlat%v(beglon:endlon,beglat:endlat,pver))
    allocate(lonlat%w(beglon:endlon,beglat:endlat,pver))
    allocate(lonlat%theta(beglon:endlon,beglat:endlat,pver))
    allocate(lonlat%uflux(beglon:endlon,beglat:endlat,pver))
    allocate(lonlat%vflux(beglon:endlon,beglat:endlat,pver))

    if (debug) then
      allocate(phys%pmid(pver,pcols,begchunk:endchunk))
      allocate(phys%lon(pcols,begchunk:endchunk))
      allocate(phys%lat(pcols,begchunk:endchunk))
    end if

    ! gathered grids only exist on masterproc
    if (masterproc) then
      allocate(gathered_lonlat%u(nlon, nlat, pver))
      allocate(gathered_lonlat%v(nlon, nlat, pver))
      allocate(gathered_lonlat%w(nlon, nlat, pver))
      allocate(gathered_lonlat%theta(nlon, nlat, pver))
      allocate(gathered_lonlat%uflux(nlon, nlat, pver))
      allocate(gathered_lonlat%vflux(nlon, nlat, pver))
    end if
  end subroutine nlgw_regrid_init


  !-----------------------------------------------------------------------------
  ! This routine takes variables on the regular lat/lon grid:
  !   * uses MPI_Scatter to broadcast them from masterproc back to all ranks
  !   * interpolates them back onto the cubed-sphere grid
  !   * finally re-chunks the data so it can be used elsewhere
  !-----------------------------------------------------------------------------
  subroutine nlgw_latlon_scatter(phys, lonlat, gathered_lonlat)
    use gw_nlgw_utils,        only: nlon, nlat
    use esmf_lonlat2phys_mod, only: fields_bundle_t, n_flx_flds, esmf_lonlat2phys_regrid
    use mpishorthand

    type(phys_vars), intent(inout), target :: phys
    type(lonlat_vars), intent(inout), target :: lonlat
    type(lonlat_vars), intent(inout), target :: gathered_lonlat

    real(r8), allocatable :: flat_array(:)

    integer  :: lchnk, ncol, i, sendcnt, disp_sum

    type(fields_bundle_t) :: phys_flx_flds(n_flx_flds)
    type(fields_bundle_t) :: lonlat_flx_flds(n_flx_flds)

    call t_startf('nlgw_scatter')

    call t_startf('nlgw_mpiscatter')
    ! this subsection gathers all variables onto a single process

    sendcnt = (endlon - beglon + 1) * (endlat - beglat + 1) * pver

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
    allocate(flat_array(nlon * nlat * pver))

    ! unlike in gather case all ranks needs the displs and recvcnts
    call mpibcast(displs, npes, mpiint, 0, mpicom)
    call mpibcast(recvcnts, npes, mpiint, 0, mpicom)

    call scatter_3d(gathered_lonlat%uflux, sendcnt, flat_array, lonlat%uflux(beglon:endlon, beglat:endlat, 1:pver))
    call scatter_3d(gathered_lonlat%vflux, sendcnt, flat_array, lonlat%vflux(beglon:endlon, beglat:endlat, 1:pver))

    deallocate(flat_array)

    call t_stopf('nlgw_mpiscatter')

    call t_startf('nlgw_latlon_gather')
    ! this subsection does regridding

    phys_flx_flds(1)%fld => phys%uflux
    phys_flx_flds(2)%fld => phys%vflux

    lonlat_flx_flds(1)%fld => lonlat%uflux
    lonlat_flx_flds(2)%fld => lonlat%vflux

    ! actual call to regrid to lon/lat grid
    call esmf_lonlat2phys_regrid(lonlat_flx_flds, phys_flx_flds)

    call t_stopf('nlgw_latlon_gather')

    call t_stopf('nlgw_scatter')

  end subroutine nlgw_latlon_scatter


  !-----------------------------------------------------------------------------
  ! This routine takes variables on the irregular cubed-sphere grid:
  !   * gathers all the chunks into a single data structure on each rank
  !   * interpolates cubed-sphere variable to a regular lonlat grid
  !   * uses MPI_Gather to collect regridded data from all ranks to the masterproc
  !-----------------------------------------------------------------------------
  subroutine nlgw_latlon_gather(phys_state, phys, lonlat, gathered_lonlat)
    use gw_nlgw_utils,        only: nlon, nlat
    use esmf_phys2lonlat_mod, only: fields_bundle_t, nflds, esmf_phys2lonlat_regrid
    use gw_nlgw_utils,        only: p0
    use physconst,            only: cappa
    use mpishorthand

    type(physics_state), intent(in) :: phys_state(begchunk:endchunk)

    type(phys_vars), intent(inout), target :: phys
    type(lonlat_vars), intent(inout), target :: lonlat
    type(lonlat_vars), intent(inout), target :: gathered_lonlat

    real(r8), allocatable :: flat_array(:)

    integer  :: lchnk, ncol, i, sendcnt, disp_sum

    type(fields_bundle_t) :: physflds(nflds)
    type(fields_bundle_t) :: lonlatflds(nflds)

    call t_startf('nlgw_gather')


    call t_startf('nlgw_unchunk')

    do lchnk = begchunk,endchunk
       ncol = phys_state(lchnk)%ncol
       do i = 1,ncol
          ! wind components
          phys%u(:,i,lchnk)     = phys_state(lchnk)%u(i,:)
          phys%v(:,i,lchnk)     = phys_state(lchnk)%v(i,:)
          phys%w(:,i,lchnk)     = phys_state(lchnk)%omega(i,:)
          phys%theta(:,i,lchnk) = phys_state(lchnk)%t(i,:) * (p0 / phys_state(lchnk)%pmid(i,:)) ** cappa

          ! for debugging only
          if (debug) then
            phys%pmid(:,i,lchnk)  = phys_state(lchnk)%pmid(i,:)
            phys%lat(i,lchnk) = phys_state(lchnk)%lat(i)
            phys%lon(i,lchnk) = phys_state(lchnk)%lon(i)
          end if

       end do
    end do

    call t_stopf('nlgw_unchunk')

    call t_startf('nlgw_latlon_gather')

    physflds(1)%fld => phys%u
    physflds(2)%fld => phys%v
    physflds(3)%fld => phys%w
    physflds(4)%fld => phys%theta

    lonlatflds(1)%fld => lonlat%u
    lonlatflds(2)%fld => lonlat%v
    lonlatflds(3)%fld => lonlat%w
    lonlatflds(4)%fld => lonlat%theta

    ! actual call to regrid to lon/lat grid
    call esmf_phys2lonlat_regrid(physflds, lonlatflds)

    call t_stopf('nlgw_latlon_gather')

    call t_startf('nlgw_mpigather')
    ! this subsection gathers all variables onto a single process

    sendcnt = (endlon - beglon + 1) * (endlat - beglat + 1) * pver

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
    allocate(flat_array(nlon * nlat * pver))

    call gather_3d(lonlat%u(beglon:endlon, beglat:endlat, 1:pver), sendcnt, flat_array, gathered_lonlat%u)
    call gather_3d(lonlat%v(beglon:endlon, beglat:endlat, 1:pver), sendcnt, flat_array, gathered_lonlat%v)
    call gather_3d(lonlat%w(beglon:endlon, beglat:endlat, 1:pver), sendcnt, flat_array, gathered_lonlat%w)
    call gather_3d(lonlat%theta(beglon:endlon, beglat:endlat, 1:pver), sendcnt, flat_array, gathered_lonlat%theta)

    deallocate(flat_array)

    call t_stopf('nlgw_mpigather')

    call t_stopf('nlgw_gather')

  end subroutine nlgw_latlon_gather

  !-----------------------------------------------------------------------------
  ! Utility function for gathering 2D data into a single array
  !-----------------------------------------------------------------------------
  subroutine gather_2d(local_array, sendcnt, flat_array, grid_out)
    use mpishorthand
    real(r8), intent(in) :: local_array(:,:)  ! Local 2D array section
    integer, intent(in) :: sendcnt
    real(r8), intent(inout) :: flat_array(:)     ! Flattened array for gathering
    real(r8), allocatable, intent(inout) :: grid_out(:,:)    ! Full gathered grid

    integer :: i, lonsize, latsize

    ! gather variables onto master proc into a flat array (can't do 2D/3D mpigather)
    call mpigatherv(local_array, sendcnt, mpir8, flat_array, recvcnts, displs, mpir8, 0, mpicom)

    if (masterproc) then
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
  ! Utility function for gathering 3D data into a single array
  !-----------------------------------------------------------------------------
  subroutine gather_3d(local_array, sendcnt, flat_array, grid_out)
    use mpishorthand
    real(r8), intent(in) :: local_array(:,:,:)  ! Local 3D array section
    integer, intent(in) :: sendcnt
    real(r8), intent(inout) :: flat_array(:)     ! Flattened array for gathering
    real(r8), allocatable, intent(inout) :: grid_out(:,:,:)    ! Full gathered grid

    integer :: i, lonsize, latsize

    ! gather variables onto master proc into a flat array (can't do 2D/3D mpigather)
    call mpigatherv(local_array, sendcnt, mpir8, flat_array, recvcnts, displs, mpir8, 0, mpicom)

    if (masterproc) then
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
  ! Utility function for scattering 3D data into lonlat arrays
  !-----------------------------------------------------------------------------
  subroutine scatter_3d(grid_in, sendcnt, flat_array, lonlat_out)
    use mpishorthand
    real(r8), allocatable, intent(in) :: grid_in(:,:,:)  ! Local 3D array section
    integer, intent(in) :: sendcnt
    real(r8), intent(inout) :: flat_array(:)     ! temporary storage in flat array
    real(r8), target, intent(inout) :: lonlat_out(:,:,:)    ! Full scattered grid

    integer :: i, lonsize, latsize

    if (masterproc) then
        do i = 1, npes
            lonsize = endlons(i) - beglons(i) + 1
            latsize = endlats(i) - beglats(i) + 1
            flat_array(displs(i)+1:displs(i)+sendcnt) = &
              reshape(grid_in(beglons(i):endlons(i), beglats(i):endlats(i), 1:pver), (/lonsize* latsize * pver/))
        end do
    end if

    ! scatter variables from flat_array back to all processes
    call mpiscatterv(flat_array, recvcnts, displs, mpir8, lonlat_out, sendcnt, mpir8, 0, mpicom)

  end subroutine scatter_3d

  !-----------------------------------------------------------------------------
  ! Tidy up (free allocated memory)
  !-----------------------------------------------------------------------------
  subroutine nlgw_regrid_final(phys, lonlat, gathered_lonlat)
    use esmf_phys2lonlat_mod, only: esmf_phys2lonlat_destroy
    use esmf_lonlat2phys_mod, only: esmf_lonlat2phys_destroy
    use esmf_lonlat_grid_mod, only: esmf_lonlat_grid_destroy
    use esmf_phys_mesh_mod, only: esmf_phys_mesh_destroy

    type(phys_vars), intent(inout), target :: phys
    type(lonlat_vars), intent(inout), target :: lonlat
    type(lonlat_vars), intent(inout), target :: gathered_lonlat

    call esmf_phys2lonlat_destroy()
    call esmf_lonlat2phys_destroy()
    call esmf_lonlat_grid_destroy()
    call esmf_phys_mesh_destroy()

    ! TODO double check ALL deallocates here
    if (masterproc) then
      deallocate(gathered_lonlat%u)
      deallocate(gathered_lonlat%v)
      deallocate(gathered_lonlat%w)
      deallocate(gathered_lonlat%theta)
      deallocate(gathered_lonlat%uflux)
      deallocate(gathered_lonlat%vflux)
    end if

    deallocate(phys%u)
    deallocate(phys%v)
    deallocate(phys%theta)
    deallocate(phys%w)
    deallocate(phys%uflux)
    deallocate(phys%vflux)
    deallocate(phys%utgw)
    deallocate(phys%vtgw)

    deallocate(lonlat%u)
    deallocate(lonlat%v)
    deallocate(lonlat%w)
    deallocate(lonlat%theta)
    deallocate(lonlat%uflux)
    deallocate(lonlat%vflux)

    if (debug) then
      deallocate(phys%pmid)
      deallocate(phys%lon)
      deallocate(phys%lat)
    end if

    deallocate(recvcnts)
    deallocate(displs)
    deallocate(beglats)
    deallocate(beglons)
    deallocate(endlats)
    deallocate(endlons)
  end subroutine nlgw_regrid_final

end module nlgw_remap_mod
