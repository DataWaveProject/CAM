module gw_nlgw_utils

use gw_utils, only: r8, r4
use ppgrid,   only: pver !vertical levels

implicit none

public :: cbrt, flux_to_forcing

private

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
      forcing(col,level) = -1*(flux(col,level+1) - flux(col,level-1)) / (pmid(col,level)*(log(pmid(col,level+1)) - log(pmid(col,level-1))))
    end do
    forcing(col,pver) = -1*(flux(col,pver) - flux(col,pver-1)) / (pmid(col,pver) - pmid(col,pver-1))
  end do

end subroutine flux_to_forcing

end module gw_nlgw_utils
