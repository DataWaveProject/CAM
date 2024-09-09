module gw_ml

!
! This module handles machine learnt gravity wave schemes developed by
! the DataWave Project
!

use gw_utils, only: r8
use ppgrid,   only: pver
use spmd_utils,      only: mpicom, mstrid=>masterprocid, masterproc, mpi_real8
use cam_abortutils, only: endrun

use ftorch

implicit none
private
save

public :: gw_drag_convect_dp_ml, gw_drag_convect_dp_ml_init, gw_drag_convect_dp_ml_final

! Neural Net as read in by FTorch
type(torch_model) :: convect_net

! Means for normalisation
real(r8) :: utgw_mean(pver), vtgw_mean(pver)
real(r8) :: u_mean(pver), v_mean(pver)
real(r8) :: t_mean(pver)
real(r8) :: dse_mean(pver)
real(r8) :: nm_mean(pver)
real(r8) :: netdt_mean(pver)
real(r8) :: zm_mean(pver)
real(r8) :: rhoi_mean(pver+1)
real(r8) :: ps_mean
real(r8) :: lat_mean
real(r8) :: lon_mean
! Standard deviations for normalisation
real(r8) :: utgw_std(pver), vtgw_std(pver)
real(r8) :: u_std(pver), v_std(pver)
real(r8) :: t_std(pver)
real(r8) :: dse_std(pver)
real(r8) :: nm_std(pver)
real(r8) :: netdt_std(pver)
real(r8) :: zm_std(pver)
real(r8) :: rhoi_std(pver+1)
real(r8) :: ps_std
real(r8) :: lat_std
real(r8) :: lon_std

contains

!==========================================================================

subroutine gw_drag_convect_dp_ml(ncol, dt, &
                                 u, v, t, dse, nm, netdt, zm, rhoi, ps, lat, lon, &
                                 utgw, vtgw)

  ! Take data from CAM, normalise and concatenate before passing it to the Torch neural
  ! net to calculate u and v tendencies.




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
  real(r8), intent(in) :: netdt(ncol,pver)
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

  integer :: i

  real(r8), dimension(:,:), target :: net_inputs(8*pver+4, ncol)
  real(r8), dimension(:,:), target :: net_outputs(2*pver, ncol)
  type(torch_tensor) :: net_input_tensors(1), net_output_tensors(1)
  integer :: ninputs = 1
  integer :: noutputs = 1
  integer, dimension(1) :: layout = [1]

  ! Normalise and concatenate the data
  call normalise_data(ncol, u, v, t, dse, nm, netdt, zm, rhoi, ps, lat, lon, &
                      net_inputs)

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

  ! Denormalise outputs and extract the data
  call denormalise_data(ncol, utgw, vtgw, net_outputs)

end subroutine gw_drag_convect_dp_ml


subroutine gw_drag_convect_dp_ml_init(neural_net_path, norms_path)

  character(len=132), intent(in) :: neural_net_path  ! Filepath to PyTorch Torchscript net
  character(len=132), intent(in) :: norms_path       ! Filepath to NetCDF normalisation weights

  ! Load the convective drag net from TorchScript file
  call torch_model_load(convect_net, neural_net_path)
  ! read in normalisation weights
  call read_norms(norms_path)

  if (masterproc) then
     write(iulog,*)'gw_convect_net loaded from: ', neural_net_path
     write(iulog,*)'Normalisation weights loaded from: ', norms_path
  endif

end subroutine gw_drag_convect_dp_ml_init


subroutine gw_drag_convect_dp_ml_final()

  ! Destroy the convective drag net
  call torch_delete(convect_net)

end subroutine gw_drag_convect_dp_ml_final


subroutine read_norms(norms_path)

  use netcdf
  use error_messages, only: handle_ncerr

  character(len=132), intent(in) :: norms_path  ! Filepath to NetCDF normalisation weights

  integer :: ncid, varid, retva, ierr
  character(len=*), parameter :: sub = 'gw_ml/F90 read_norms: '

  ! Load normalisation weights from file in master process then broadcast
  if (masterproc) then
    ! Open the NetCDF file
    call handle_ncerr( nf90_open(trim(norms_path), NF90_NOWRITE, ncid), &
                       "Error opening NetCDF norms file in gw_ml.F90")

    ! We do not need to read in dimensions here as we assume inputs match the grid.

    ! Read in variables (means and deviations).
    call handle_ncerr( nf90_inq_varid(ncid, 'U_mean', varid), &
                       "Error getting U_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, u_mean), &
                       "Error getting U_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_inq_varid(ncid, 'U_std', varid), &
                       "Error getting U_std varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, u_std), &
                       "Error getting U_std varid from NetCDF Norms file in gw_ml.F90")

    call handle_ncerr( nf90_inq_varid(ncid, 'V_mean', varid), &
                       "Error getting V_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, v_mean), &
                       "Error getting V_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_inq_varid(ncid, 'V_std', varid), &
                       "Error getting V_std varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, v_std), &
                       "Error getting V_std varid from NetCDF Norms file in gw_ml.F90")

    call handle_ncerr( nf90_inq_varid(ncid, 'T_mean', varid), &
                       "Error getting T_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, t_mean), &
                       "Error getting t_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_inq_varid(ncid, 'T_std', varid), &
                       "Error getting T_std varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, t_std), &
                       "Error getting T_std varid from NetCDF Norms file in gw_ml.F90")

    call handle_ncerr( nf90_inq_varid(ncid, 'DSE_mean', varid), &
                       "Error getting DSE_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, dse_mean), &
                       "Error getting U_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_inq_varid(ncid, 'DSE_std', varid), &
                       "Error getting DSE_std varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, dse_std), &
                       "Error getting DSE_std varid from NetCDF Norms file in gw_ml.F90")

    call handle_ncerr( nf90_inq_varid(ncid, 'NMBV_mean', varid), &
                       "Error getting NMBV_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, nm_mean), &
                       "Error getting NMBV_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_inq_varid(ncid, 'NMBV_std', varid), &
                       "Error getting NMBV_std varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, nm_std), &
                       "Error getting NMBV_std varid from NetCDF Norms file in gw_ml.F90")

    call handle_ncerr( nf90_inq_varid(ncid, 'NETDT_mean', varid), &
                       "Error getting NETDT_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, netdt_mean), &
                       "Error getting NETDT_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_inq_varid(ncid, 'NETDT_std', varid), &
                       "Error getting NETDT_std varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, netdt_std), &
                       "Error getting NETDT_std varid from NetCDF Norms file in gw_ml.F90")

    call handle_ncerr( nf90_inq_varid(ncid, 'Z3_mean', varid), &
                       "Error getting Z3_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, zm_mean), &
                       "Error getting Z3_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_inq_varid(ncid, 'Z3_std', varid), &
                       "Error getting Z3_std varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, zm_std), &
                       "Error getting Z3_std varid from NetCDF Norms file in gw_ml.F90")

    call handle_ncerr( nf90_inq_varid(ncid, 'RHOI_mean', varid), &
                       "Error getting RHOI_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, rhoi_mean), &
                       "Error getting RHOI_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_inq_varid(ncid, 'RHOI_std', varid), &
                       "Error getting RHOI_std varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, rhoi_std), &
                       "Error getting RHOI_std varid from NetCDF Norms file in gw_ml.F90")

    call handle_ncerr( nf90_inq_varid(ncid, 'PS_mean', varid), &
                       "Error getting PS_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, ps_mean), &
                       "Error getting PS_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_inq_varid(ncid, 'PS_std', varid), &
                       "Error getting PS_std varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, ps_std), &
                       "Error getting PS_std varid from NetCDF Norms file in gw_ml.F90")

    call handle_ncerr( nf90_inq_varid(ncid, 'lat_mean', varid), &
                       "Error getting lat_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, lat_mean), &
                       "Error getting lat_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_inq_varid(ncid, 'lat_std', varid), &
                       "Error getting lat_std varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, lat_std), &
                       "Error getting lat_std varid from NetCDF Norms file in gw_ml.F90")

    call handle_ncerr( nf90_inq_varid(ncid, 'lon_mean', varid), &
                       "Error getting lon_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, lon_mean), &
                       "Error getting lon_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_inq_varid(ncid, 'lon_std', varid), &
                       "Error getting lon_std varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, lon_std), &
                       "Error getting lon_std varid from NetCDF Norms file in gw_ml.F90")

    call handle_ncerr( nf90_inq_varid(ncid, 'UTGWSPEC_mean', varid), &
                       "Error getting UTGWSPEC_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, utgw_mean), &
                       "Error getting UTGWSPEC_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_inq_varid(ncid, 'UTGWSPEC_std', varid), &
                       "Error getting UTGWSPEC_std varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, utgw_std), &
                       "Error getting UTGWSPEC_std varid from NetCDF Norms file in gw_ml.F90")

    call handle_ncerr( nf90_inq_varid(ncid, 'VTGWSPEC_mean', varid), &
                       "Error getting VTGWSPEC_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, vtgw_mean), &
                       "Error getting VTGWSPEC_mean varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_inq_varid(ncid, 'VTGWSPEC_std', varid), &
                       "Error getting VTGWSPEC_std varid from NetCDF Norms file in gw_ml.F90")
    call handle_ncerr( nf90_get_var(ncid, varid, vtgw_std), &
                       "Error getting VTGWSPEC_std varid from NetCDF Norms file in gw_ml.F90")

  endif

  ! Broadcast normalisation variables to other processes
  call mpi_bcast(utgw_mean, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: utgw_mean from gw_ml.F90")
  call mpi_bcast(utgw_std, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: utgw_std from gw_ml.F90")

  call mpi_bcast(vtgw_mean, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: vtgw_mean from gw_ml.F90")
  call mpi_bcast(vtgw_std, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: vtgw_std from gw_ml.F90")

  call mpi_bcast(u_mean, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: u_mean from gw_ml.F90")
  call mpi_bcast(u_std, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: u_std from gw_ml.F90")

  call mpi_bcast(v_mean, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: v_mean from gw_ml.F90")
  call mpi_bcast(v_std, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: v_std from gw_ml.F90")

  call mpi_bcast(t_mean, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: t_mean from gw_ml.F90")
  call mpi_bcast(t_std, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: t_std from gw_ml.F90")

  call mpi_bcast(dse_mean, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: dse_mean from gw_ml.F90")
  call mpi_bcast(dse_std, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: dse_std from gw_ml.F90")

  call mpi_bcast(nm_mean, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: nm_mean from gw_ml.F90")
  call mpi_bcast(nm_std, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: nm_std from gw_ml.F90")

  call mpi_bcast(zm_mean, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: zm_mean from gw_ml.F90")
  call mpi_bcast(zm_std, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: zm_std from gw_ml.F90")

  call mpi_bcast(rhoi_mean, pver+1, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: rhoi_mean from gw_ml.F90")
  call mpi_bcast(rhoi_std, pver+1, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: rhoi_std from gw_ml.F90")

  call mpi_bcast(ps_mean, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: ps_mean from gw_ml.F90")
  call mpi_bcast(ps_std, pver, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: ps_std from gw_ml.F90")

  call mpi_bcast(lat_mean, 1, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: lat_mean from gw_ml.F90")
  call mpi_bcast(lat_std, 1, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: lat_std from gw_ml.F90")

  call mpi_bcast(lon_mean, 1, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: lon_mean from gw_ml.F90")
  call mpi_bcast(lon_std, 1, mpi_real8, mstrid, mpicom, ierr)
  if (ierr /= 0) call endrun(sub//": FATAL: mpi_bcast: lon_std from gw_ml.F90")

end subroutine read_norms

subroutine normalise_data(ncol, u, v, t, dse, nm, netdt, zm, rhoi, ps, lat, lon, &
                          nn_input)

  integer, intent(in) :: ncol
  real(r8), intent(in) :: u(ncol,pver), v(ncol,pver)
  real(r8), intent(in) :: t(ncol,pver)
  real(r8), intent(in) :: dse(ncol,pver)
  real(r8), intent(in) :: nm(ncol,pver)
  real(r8), intent(in) :: netdt(:,:)
  real(r8), intent(in) :: zm(ncol,pver)
  real(r8), intent(in) :: rhoi(ncol,pver+1)
  real(r8), intent(in) :: ps(ncol)
  real(r8), intent(in) :: lat(ncol)
  real(r8), intent(in) :: lon(ncol)

  real(r8), intent(out) :: nn_input(8*pver+4, ncol)

  integer :: i

  ! Loop over each column.
  ! Normalise data (subtract mean, divide by deviation), transpose into format
  ! expected by the NN, and concatenate into a single input tensor as expected by the NN.
  do i = 1,ncol

    nn_input(:pver, i)             = (u(i, :) - u_mean(:))/u_std(:)
    nn_input(pver+1:2*pver, i)     = (v(i, :) - v_mean(:))/v_std(:)
    nn_input(2*pver+1:3*pver, i)   = (t(i, :) - t_mean(:))/t_std(:)
    nn_input(3*pver+1:4*pver, i)   = (dse(i, :) - dse_mean(:))/dse_std(:)
    nn_input(4*pver+1:5*pver, i)   = (nm(i, :) - nm_mean(:))/nm_std(:)
    nn_input(5*pver+1:6*pver, i)   = (netdt(i, :) - netdt_mean(:))/netdt_std(:)
    nn_input(6*pver+1:7*pver, i)   = (zm(i, :) - zm_mean(:))/zm_std(:)
    nn_input(7*pver+1:8*pver+1, i) = (rhoi(i, :) - rhoi_mean(:))/rhoi_std(:)
    nn_input(8*pver+2, i)          = (ps(i) - ps_mean)/ps_std
    nn_input(8*pver+3, i)          = (lat(i) - lat_mean)/lat_std
    nn_input(8*pver+4, i)          = (lon(i) - lon_mean)/lon_std

  end do

end subroutine normalise_data

subroutine denormalise_data(ncol, utgw, vtgw, nn_output)

  integer, intent(in) :: ncol
  real(r8), intent(out) :: utgw(ncol,pver), vtgw(ncol,pver)
  real(r8), intent(in) :: nn_output(2*pver, ncol)

  integer :: i

  ! Extract data, denormalise, and deconcatenate from NN output tensor
  do i = 1, ncol
      utgw(i, :) = (nn_output(1:pver, i) * utgw_std(:)) + utgw_mean(:)
      vtgw(i, :) = (nn_output(pver+1:2*pver, i) * vtgw_std(:)) + vtgw_mean(:)
  end do

end subroutine denormalise_data

end module gw_ml
