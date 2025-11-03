module gw_nlgw_unet

!
! This module predicts gravity wave forcings via PyTorch NNs trained to include non-local gravity wave effects
!

use gw_utils,       only: r8, r4
use ppgrid,         only: pver !vertical levels
use physics_types,  only: physics_state, physics_ptend
use spmd_utils,     only: mpicom, mstrid=>masterprocid, masterproc, mpi_real8, iam
use cam_abortutils, only: endrun
use cam_logfile,    only: iulog
use physconst,      only: cappa
use gw_nlgw_utils,  only: lonlat_vars, nlon, nlat

use ftorch

implicit none

public :: gw_nlgw_unet_init, gw_nlgw_unet_infer, gw_nlgw_unet_finalize, gw_nlgw_unet_update_ptend

real(r8), dimension(:,:,:), allocatable, public :: &
  utgw_allchunk,    &! zonal wind tendency (m/s^2)
  vtgw_allchunk      ! meridional wind tendency (m/s^2)

private

type(torch_model) :: nlgw_model ! pytorch model

real(r4), dimension(:,:,:,:), allocatable, target :: net_inputs
real(r4), dimension(:,:,:,:), allocatable, target :: net_outputs

! normalisation means and std devs
real(r8) :: u_mean, v_mean, w_mean, theta_mean
real(r8) :: u_std, v_std, w_std, theta_std

real(r8) :: uflux_mean, vflux_mean
real(r8) :: uflux_std, vflux_std

contains

!==========================================================================


subroutine gw_nlgw_unet_init(model_path)

  character(len=*), intent(in) :: model_path  ! Filepath to PyTorch Torchscript net
  integer :: device_id

  device_id = 0

  ! Load the convective drag net from TorchScript file
  call torch_model_load(nlgw_model, model_path, device_type=torch_kCUDA, device_index=device_id)
  ! read in normalisation weights
  call read_norms()

  if (masterproc) then
     write(iulog,*)'nlgw model loaded from: ', model_path

    ! UNet will only run on the master process
    ! space for u, v, theta and w
    allocate(net_inputs(1, pver*4, nlat, nlon))
    ! space for uflux and vflux
    allocate(net_outputs(1, pver*2, nlat, nlon))
  endif

end subroutine gw_nlgw_unet_init

subroutine gw_nlgw_unet_infer(gathered_lonlat)

  ! global unet data
  type(lonlat_vars), intent(inout) :: gathered_lonlat

  !---------------------------Local storage-------------------------------
  type(torch_tensor) :: tensor_in(1), tensor_out(1)
  integer :: ninputs = 1, noutputs = 1
  integer, dimension(4) :: layout = [1 , 2, 3, 4]

  integer :: device_id

  device_id = 0

  ! Normalise and construct the input
  call normalise_data(gathered_lonlat)
  call construct_input(gathered_lonlat)

  ! send all columns from this process
  call torch_tensor_from_array(tensor_in(1), net_inputs, layout, torch_kCUDA, device_id)
  call torch_tensor_from_array(tensor_out(1), net_outputs, layout, torch_kCPU)

  ! Run net forward on data
  call torch_model_forward(nlgw_model, tensor_in, tensor_out)

  ! Extract and denormalise outputs
  call extract_output(gathered_lonlat)
  call denormalise_data(gathered_lonlat)

  ! Clean up the tensors
  call torch_delete(tensor_in)
  call torch_delete(tensor_out)

end subroutine gw_nlgw_unet_infer

subroutine gw_nlgw_unet_finalize()

  if (masterproc) then
    deallocate(net_inputs)
    deallocate(net_outputs)
  end if
  ! free model memory
  call torch_delete(nlgw_model)

end subroutine gw_nlgw_unet_finalize

subroutine gw_nlgw_unet_update_ptend(ptend, lchnk, ncol)

  use gw_nlgw_utils, only: flux_to_forcing

  ! inputs
  type(physics_ptend), intent(inout) :: ptend
  integer, intent(in) :: lchnk, ncol

  ! update the tendencies
  ptend%u(:ncol,:pver) = ptend%u(:ncol,:pver) + utgw_allchunk(:ncol,:pver, lchnk)
  ptend%v(:ncol,:pver) = ptend%v(:ncol,:pver) + vtgw_allchunk(:ncol,:pver, lchnk)

end subroutine gw_nlgw_unet_update_ptend

subroutine read_norms()

  ! TODO
  ! - replace hardcoded means/std devs with netcdf file?

  u_mean = 6.717847278462159_r8
  v_mean = -0.002744777264668839_r8
  theta_mean = 0._r8
  w_mean = 0.0013401482063147452_r8

  u_std = 20.760385183200206_r8
  v_std = 9.877389116738264_r8
  theta_std = 1000._r8
  w_std = 0.11202126259282257_r8

  uflux_mean = -0.0004691528666736032_r8
  vflux_mean = -0.0002586195082961397_r8
  uflux_std = 0.032814051953840274_r8
  vflux_std = 0.03024781201672967_r8

end subroutine read_norms

subroutine normalise_data(gathered_lonlat)
  use gw_nlgw_utils, only: cbrt
  type(lonlat_vars), intent(inout) :: gathered_lonlat

  gathered_lonlat%u = (gathered_lonlat%u-u_mean)/(3._r8 * u_std)
  gathered_lonlat%v = (gathered_lonlat%v-v_mean)/(3._r8 * v_std)
  gathered_lonlat%theta = (gathered_lonlat%theta-theta_mean)/theta_std
  gathered_lonlat%w = (gathered_lonlat%w-w_mean)/w_std
  gathered_lonlat%w = cbrt(gathered_lonlat%w)

end subroutine normalise_data

subroutine construct_input(gathered_lonlat)

  type(lonlat_vars), intent(inout) :: gathered_lonlat
  integer :: idx_beg, idx_end, i

  idx_end = 0

  idx_beg = idx_end + 1
  idx_end = idx_end + pver
  net_inputs(1,idx_beg:idx_end,:,:) = reshape(gathered_lonlat%u, shape=[pver, nlat, nlon], order=[3,2,1])
  idx_beg = idx_end + 1
  idx_end = idx_end + pver
  net_inputs(1,idx_beg:idx_end,:,:) = reshape(gathered_lonlat%v, shape=[pver, nlat, nlon], order=[3,2,1])
  idx_beg = idx_end + 1
  idx_end = idx_end + pver
  net_inputs(1,idx_beg:idx_end,:,:) = reshape(gathered_lonlat%theta, shape=[pver, nlat, nlon], order=[3,2,1])
  idx_beg = idx_end + 1
  idx_end = idx_end + pver
  net_inputs(1,idx_beg:idx_end,:,:) = reshape(gathered_lonlat%w, shape=[pver, nlat, nlon], order=[3,2,1])

end subroutine construct_input

subroutine extract_output(gathered_lonlat)

  type(lonlat_vars), intent(inout) :: gathered_lonlat

  gathered_lonlat%uflux(:,:,:) = reshape(net_outputs(1,:pver,:,:), shape=[nlon, nlat, pver], order=[3,2,1])
  gathered_lonlat%vflux(:,:,:) = reshape(net_outputs(1,pver+1:,:,:), shape=[nlon, nlat, pver], order=[3,2,1])

end subroutine extract_output

subroutine denormalise_data(gathered_lonlat)

  type(lonlat_vars), intent(inout) :: gathered_lonlat

  gathered_lonlat%uflux = gathered_lonlat%uflux**3._r8 * uflux_std + uflux_mean
  gathered_lonlat%vflux = gathered_lonlat%vflux**3._r8 * vflux_std + vflux_mean

end subroutine denormalise_data

end module gw_nlgw_unet
