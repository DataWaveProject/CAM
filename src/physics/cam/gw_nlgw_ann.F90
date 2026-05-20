module gw_nlgw_ann

!
! This module predicts gravity wave forcings via PyTorch NNs trained to include non-local gravity wave effects
!

use gw_utils, only: r8, r4
use ppgrid,   only: pver !vertical levels
use physics_types,  only: physics_state, physics_ptend
use spmd_utils,     only: mpicom, mstrid=>masterprocid, masterproc, mpi_real8, iam
use cam_abortutils, only: endrun
use cam_logfile,    only: iulog
use physconst,      only: cappa, pi
use gw_nlgw_utils,  only: p0
use interpolate_data, only: lininterp
use cam_history,    only: outfld, addfld

use ftorch

implicit none

public :: gw_nlgw_ann_infer, gw_nlgw_ann_init, gw_nlgw_ann_finalize

private

type(torch_model) :: nlgw_model ! pytorch model

integer :: ncol ! number of vertical columns

real(r8), dimension(:), allocatable :: &
  lat,     &! latitude (radians)
  lon,     &! longitude (radians)
  ps,      &! surface pressure
  phis      ! surface geopotential
real(r8), dimension(:,:), allocatable :: &
  u,       &! zonal wind (m/s)
  v,       &! meridional wind (m/s)
  omega,   &! vertical pressure velocity (Pa/s)
  t,       &! temperature (K)
  theta,   &! potential temperature (K)
  pmid      ! midpoint pressure (Pa)

real(r8), dimension(:,:), allocatable :: &
  uflux,   &! zonal wind flux (Pa)
  vflux,   &! meridional wind flux (Pa)
  utgw,    &! zonal wind tendency (m/s^2)
  vtgw      ! meridional wind tendency (m/s^2)

real(r4), dimension(:,:), allocatable, target :: net_inputs
real(r4), dimension(:,:), allocatable, target :: net_outputs

! normalisation means and std devs
real(r8) :: u_mean, v_mean, omega_mean, theta_mean, lat_mean, lon_mean
real(r8) :: u_std, v_std, omega_std, theta_std, lat_std, lon_std

real(r8) :: uflux_mean, vflux_mean
real(r8) :: uflux_std, vflux_std

contains

!==========================================================================

subroutine gw_nlgw_ann_infer(state_in, ptend, lchnk)

  use gw_nlgw_utils, only: flux_to_forcing

  ! inputs
  type(physics_state), intent(in) :: state_in
  integer,             intent(in)    :: lchnk
  ! outputs
  type(physics_ptend), intent(inout) :: ptend

  !---------------------------Local storage-------------------------------
  type(torch_tensor) :: tensor_in(1), tensor_out(1)
  integer :: ninputs = 1, noutputs = 1
  integer, dimension(2) :: layout = [1 , 2]

  ncol = state_in%ncol

  allocate(lat(ncol))
  allocate(lon(ncol))
  allocate(ps(ncol))
  allocate(phis(ncol))
  allocate(u(ncol,pver))
  allocate(v(ncol,pver))
  allocate(t(ncol,pver))
  allocate(pmid(ncol,pver))
  allocate(theta(ncol,pver))
  allocate(omega(ncol,pver))

  allocate(uflux(ncol,pver))
  allocate(vflux(ncol,pver))
  allocate(utgw(ncol,pver))
  allocate(vtgw(ncol,pver))

  allocate(net_inputs(ncol, 4*pver+3))
  allocate(net_outputs(ncol, 2*pver))

  ! dims = (ncol)
  lat = state_in%lat(:ncol)
  lon = state_in%lon(:ncol)
  ps = state_in%ps(:ncol)
  phis = state_in%phis(:ncol)

  ! dims = (ncol, pver)
  u = state_in%u(:ncol,:pver)
  v = state_in%v(:ncol,:pver)
  t = state_in%t(:ncol,:pver)
  pmid = state_in%pmid(:ncol,:pver)
  theta = t * (p0 / pmid) ** cappa
  omega = state_in%omega(:ncol,:pver)

  ! Normalise and construct the input
  call normalise_data()
  call construct_input()

  ! send all columns from this process
  call torch_tensor_from_array(tensor_in(1), net_inputs, layout, torch_kCPU)
  call torch_tensor_from_array(tensor_out(1), net_outputs, layout, torch_kCPU)

  ! Run net forward on data
  call torch_model_forward(nlgw_model, tensor_in, tensor_out)

  ! Extract and denormalise outputs
  call extract_output()
  call denormalise_data()

  call flux_to_forcing(uflux, utgw, pmid, ncol)
  call flux_to_forcing(vflux, vtgw, pmid, ncol)

  ! Write UTGW and VTGW to file
  call outfld('UTGW_NL', utgw, ncol, lchnk)
  call outfld('VTGW_NL', vtgw, ncol, lchnk)

  call outfld('UFLUX_NL', uflux, ncol, lchnk)
  call outfld('VFLUX_NL', vflux, ncol, lchnk)

  ! update the tendencies
  ptend%u(:ncol,:pver) = ptend%u(:ncol,:pver) + utgw(:ncol,:pver)
  ptend%v(:ncol,:pver) = ptend%v(:ncol,:pver) + vtgw(:ncol,:pver)

  ! Clean up the tensors
  call torch_delete(tensor_in)
  call torch_delete(tensor_out)

  deallocate(lat)
  deallocate(lon)
  deallocate(ps)
  deallocate(phis)
  deallocate(u)
  deallocate(v)
  deallocate(t)
  deallocate(pmid)
  deallocate(theta)
  deallocate(omega)

  deallocate(uflux)
  deallocate(vflux)
  deallocate(utgw)
  deallocate(vtgw)

  deallocate(net_inputs)
  deallocate(net_outputs)

end subroutine gw_nlgw_ann_infer


subroutine gw_nlgw_ann_init(model_path)

  character(len=*), intent(in) :: model_path  ! Filepath to PyTorch Torchscript net

  ! Load the convective drag net from TorchScript file
  call torch_model_load(nlgw_model, model_path, device_type=torch_kCPU)
  ! read in normalisation weights
  call read_norms()

  if (masterproc) then
     write(iulog,*)'nlgw model loaded from: ', model_path
  endif

  call addfld('UTGW_NL', (/ 'lev' /), 'A', 'm/s2', 'Nonlinear GW zonal wind tendency')
  call addfld('VTGW_NL', (/ 'lev' /), 'A', 'm/s2', 'Nonlinear GW meridional wind tendency')
  call addfld('UFLUX_NL', (/ 'lev' /), 'A', 'm/s', 'Nonlinear GW zonal wind flux')
  call addfld('VFLUX_NL', (/ 'lev' /), 'A', 'm/s', 'Nonlinear GW meridional wind flux')

end subroutine gw_nlgw_ann_init


subroutine gw_nlgw_ann_finalize()

  deallocate(net_inputs)
  deallocate(net_outputs)
  ! free model memory
  call torch_delete(nlgw_model)

end subroutine gw_nlgw_ann_finalize


subroutine read_norms()

  ! TODO
  ! - replace hardcoded means/std devs with netcdf file?

  lat_mean = 0._r8
  lon_mean = 0._r8
  u_mean = 6.717847278462159_r8
  v_mean = -0.002744777264668839_r8
  theta_mean = 0._r8
  omega_mean = 0.0013401482063147452_r8

  lat_std = 90._r8
  lon_std = 360._r8
  u_std = 20.760385183200206_r8
  v_std = 9.877389116738264_r8
  theta_std = 1000._r8
  omega_std = 0.11202126259282257_r8

  uflux_mean = -0.0004691528666736032_r8
  vflux_mean = -0.0002586195082961397_r8
  uflux_std = 0.032814051953840274_r8
  vflux_std = 0.03024781201672967_r8

end subroutine read_norms

subroutine normalise_data()
  use gw_nlgw_utils, only: cbrt

  ! lat lon are in radians (convert to degrees first)
  lat = lat * 180. / pi
  lon = lon * 180. / pi
  lat = (lat-lat_mean)/lat_std
  lon = (lon-lon_mean)/lon_std
  phis = phis / 50000._r8

  u = (u-u_mean)/(3._r8 * u_std)
  v = (v-v_mean)/(3._r8 * v_std)
  theta = (theta-theta_mean)/theta_std
  omega = (omega-omega_mean)/omega_std
  omega = cbrt(omega)

end subroutine normalise_data

subroutine construct_input()

  integer :: idx_beg, idx_end, i

  net_inputs(:,1) = lat
  net_inputs(:,2) = lon
  net_inputs(:,3) = phis

  idx_end = 3 ! last index written to was phis at position 3
  idx_beg = idx_end + 1
  idx_end = idx_beg + pver - 1
  net_inputs(:,idx_beg:idx_end) = u
  idx_beg = idx_end + 1
  idx_end = idx_beg + pver - 1
  net_inputs(:,idx_beg:idx_end) = v
  idx_beg = idx_end + 1
  idx_end = idx_beg + pver - 1
  net_inputs(:,idx_beg:idx_end) = theta
  idx_beg = idx_end + 1
  idx_end = idx_beg + pver - 1
  net_inputs(:,idx_beg:idx_end) = omega


end subroutine construct_input

subroutine extract_output()

  uflux(:, :) = net_outputs(:,:pver)
  vflux(:, :) = net_outputs(:,pver+1:)

end subroutine extract_output

subroutine denormalise_data()

  uflux = uflux**3._r8 * uflux_std + uflux_mean
  vflux = vflux**3._r8 * vflux_std + vflux_mean

end subroutine denormalise_data

end module gw_nlgw_ann
