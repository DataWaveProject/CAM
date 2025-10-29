module gw_nlgw_utils

use gw_utils, only: r8, r4
use ppgrid, only: begchunk, endchunk, pcols, pver, pverp

implicit none

public :: cbrt, flux_to_forcing
public :: phys_vars, lonlat_vars
integer, parameter, public :: p0 = 100000 ! 1000 hPa (Pa)
integer, parameter, public :: nlon = 288  ! number of longitude points on lonlat grid
integer, parameter, public :: nlat = 192  ! number of latitude points on lonlat grid

private

! variables on cubed-sphere "phys" grid
type phys_vars
!dimension(pver,pcols,begchunk:endchunk)
real(r8), dimension(:,:,:), allocatable :: &
  u,       &! zonal wind (m/s)
  v,       &! meridional wind (m/s)
  theta,   &! temperature (K)
  w,       &! vertical pressure velocity (Pa/s)
  pmid      ! midpoint pressure (Pa)

real(r8), dimension(:,:,:), allocatable :: &
  uflux,   &! zonal fluxes
  vflux     ! meridional fluxes

real(r8), dimension(:,:,:), allocatable :: &
  utgw,    &! zonal tendencies
  vtgw      ! meridional tendencies

! for debugging only
! dimension(pcols,begchunk:endchunk)
real(r8), dimension(:,:), allocatable :: &
  lat,     &
  lon
end type

! variables on regular lonlat grid
type lonlat_vars
! dimension(lon,lat,pver)
real(r8), dimension(:,:,:), allocatable :: &
  u,       &! zonal wind (m/s)
  v,       &! meridional wind (m/s)
  theta,   &! temperature (K)
  w         ! vertical pressure velocity (Pa/s)

real(r8), dimension(:,:,:), allocatable :: &
  uflux,    &! zonal fluxes
  vflux      ! meridional fluxes
end type

contains

elemental function cbrt(a) result(root)
  real(r8), intent(in) :: a
  real(r8), parameter :: one_third = 1._r8/3._r8
  real(r8) :: root
  root = sign(abs(a)**one_third, a)
end function cbrt

subroutine flux_to_forcing(flux, forcing, pmid, ncol)

  real(r8), intent(in), dimension(:,:) :: flux ! flux (Pa m/s^2) !TODO check with Aman
  real(r8), intent(in), dimension(:,:) :: pmid ! midpoint pressure (Pa)
  real(r8), intent(out), dimension(:,:) :: forcing ! forcing = -d(u'\omega')/d(p), units = m/s^2
  integer, intent(in) :: ncol

  integer :: level, col

  ! convert fluxes to tendencies
  ! pressure profile must be in Pascals

  do col = 1, ncol
    forcing(col,1) = -1*(flux(col,2) - flux(col,1))/(pmid(col,2) - pmid(col,1))
    do level = 2, pver-1
      forcing(col,level) = (flux(col,level+1) - flux(col,level-1)) / (pmid(col,level)*(log(pmid(col,level+1)) - log(pmid(col,level-1))))
    end do
    forcing(col,pver) = -1*(flux(col,pver) - flux(col,pver-1)) / (pmid(col,pver) - pmid(col,pver-1))
  end do

end subroutine flux_to_forcing

end module gw_nlgw_utils
