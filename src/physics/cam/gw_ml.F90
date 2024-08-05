module gw_ml

!
! This module handles machine learnt gravity wave schemes developed by
! the DataWave Project
!

use gw_utils, only: r8
use ppgrid,   only: pver

use ftorch

implicit none
private
save

public :: gw_drag_convect_dp_ml

contains

!==========================================================================

subroutine gw_drag_convect_dp_ml(convect_net, &
                                 ncol, dt, &
                                 u, v, t, dse, nm, netdt, zm, rhoi, ps, lat, lon, &
                                 utgw, vtgw)

  ! Take data from CAM, normalise and concatenate before passing it to the Torch neural
  ! net to calculate u and v tendencies.

  ! Neural Net as read in by FTorch
  type(torch_model) :: convect_net

  ! Column dimension.
  integer, intent(in) :: ncol

  ! Time step.
  real(r8), intent(in) :: dt

  ! Midpoint zonal/meridional winds.
  real(r8), intent(in) :: u(ncol,pver), v(ncol,pver)
  ! Midpoint and interface temperatures.
  real(r8), intent(in) :: t(ncol,pver)
  ! Dry static energy.
  real(r8), intent(in) :: dse(ncol,pver)
  ! Midpoint and interface Brunt-Vaisalla frequencies.
  real(r8), intent(in) :: nm(ncol,pver)
  ! Heating rate due to convection.
  real(r8), intent(in) :: netdt(:,:)
  ! Midpoint geopotential altitudes.
  real(r8), intent(in) :: zm(ncol,pver)
  ! Interface densities.
  real(r8), intent(in) :: rhoi(ncol,pver+1)
  ! Surface Pressure
  real(r8), intent(in) :: ps(ncol)
  ! Latitude in radians.
  real(r8), intent(in) :: lat(ncol)
  ! Longitude in radians.
  real(r8), intent(in) :: lon(ncol)

  ! Zonal/meridional wind tendencies.
  real(r8), intent(out) :: utgw(ncol,pver), vtgw(ncol,pver)


  !---------------------------Local storage-------------------------------

  ! Level, wavenumber, constituent and column loop indices.
  integer :: k, l, m, i

  real(r8), dimension(:,:), target :: net_inputs(8*pver+4, ncol)
  real(r8), dimension(:,:), target :: net_outputs(2*pver, ncol)
  type(torch_tensor) :: net_input_tensors(1), net_output_tensors(1)
  integer :: ninputs = 1
  integer :: noutputs = 1
  integer, dimension(1) :: layout = [1]

  ! Concatenate data into input array and map to a torch_tensor
  net_inputs(:pver, :)             = transpose(u(:, :))
  net_inputs(pver+1:2*pver, :)     = transpose(v(:, :))
  net_inputs(2*pver+1:3*pver, :)   = transpose(t(:, :))
  net_inputs(3*pver+1:4*pver, :)   = transpose(dse(:, :))
  net_inputs(4*pver+1:5*pver, :)   = transpose(nm(:, :))
  net_inputs(5*pver+1:6*pver, :)   = transpose(netdt(:, :))
  net_inputs(6*pver+1:7*pver, :)   = transpose(zm(:, :))
  net_inputs(7*pver+1:8*pver+1, :) = transpose(rhoi(:, :))
  net_inputs(8*pver+2, :)          = ps(:)
  net_inputs(8*pver+3, :)          = lat(:)
  net_inputs(8*pver+4, :)          = lon(:)

  ! Loop over columns, create input and infer
  do i = 1, ncol

      call torch_tensor_from_array(net_input_tensors(1), net_inputs(:, i), layout, torch_kCPU)
      call torch_tensor_from_array(net_output_tensors(1), net_outputs(:, i), layout, torch_kCPU)

      ! Run net forward on data
      call torch_model_forward(convect_net, net_input_tensors, net_output_tensors)

  end do

  ! Clean up the tensors
  call torch_delete(net_input_tensors)
  call torch_delete(net_output_tensors)

  ! Extract data and return
  do i = 1, ncol
      utgw(i, :) = net_outputs(1:pver, i)
      vtgw(i, :) = net_outputs(pver+1:2*pver, i)
  end do

end subroutine gw_drag_convect_dp_ml

end module gw_ml
