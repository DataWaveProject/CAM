module gw_nlgw

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
use interpolate_data, only: lininterp
use cam_history,    only: outfld

use ftorch

implicit none

public :: gw_nlgw_dp_ml, gw_nlgw_dp_init, gw_nlgw_dp_finalize

private

integer, parameter :: p0 = 100000 ! 1000 hPa (Pa)

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

integer, parameter :: pver_interp = 137 ! number of levels in ERA5

real(r8), parameter :: era5_ak(137) = [ &
            1.000000000e+00_r8, 2.550781250e+00_r8, 3.884765625e+00_r8, 5.746093750e+00_r8, 8.289062500e+00_r8, 1.167968750e+01_r8, 1.610937500e+01_r8, 2.179687500e+01_r8, &
            2.898437500e+01_r8, 3.793750000e+01_r8, 4.890625000e+01_r8, 6.225000000e+01_r8, 7.818750000e+01_r8, 9.712500000e+01_r8, 1.194375000e+02_r8, 1.453750000e+02_r8, &
            1.752500000e+02_r8, 2.096250000e+02_r8, 2.487500000e+02_r8, 2.930000000e+02_r8, 3.427500000e+02_r8, 3.982500000e+02_r8, 4.600000000e+02_r8, 5.285000000e+02_r8, &
            6.040000000e+02_r8, 6.865000000e+02_r8, 7.770000000e+02_r8, 8.750000000e+02_r8, 9.820000000e+02_r8, 1.097000000e+03_r8, 1.221000000e+03_r8, 1.355000000e+03_r8, &
            1.498000000e+03_r8, 1.651000000e+03_r8, 1.813000000e+03_r8, 1.987000000e+03_r8, 2.170000000e+03_r8, 2.366000000e+03_r8, 2.572000000e+03_r8, 2.788000000e+03_r8, &
            3.018000000e+03_r8, 3.258000000e+03_r8, 3.512000000e+03_r8, 3.776000000e+03_r8, 4.054000000e+03_r8, 4.344000000e+03_r8, 4.644000000e+03_r8, 4.960000000e+03_r8, &
            5.288000000e+03_r8, 5.624000000e+03_r8, 5.976000000e+03_r8, 6.340000000e+03_r8, 6.720000000e+03_r8, 7.112000000e+03_r8, 7.520000000e+03_r8, 7.944000000e+03_r8, &
            8.384000000e+03_r8, 8.840000000e+03_r8, 9.320000000e+03_r8, 9.816000000e+03_r8, 1.032800000e+04_r8, 1.084800000e+04_r8, 1.139200000e+04_r8, 1.193600000e+04_r8, &
            1.248800000e+04_r8, 1.304800000e+04_r8, 1.360000000e+04_r8, 1.416000000e+04_r8, 1.470400000e+04_r8, 1.524000000e+04_r8, 1.576800000e+04_r8, 1.628000000e+04_r8, &
            1.676800000e+04_r8, 1.723200000e+04_r8, 1.768000000e+04_r8, 1.811200000e+04_r8, 1.849600000e+04_r8, 1.886400000e+04_r8, 1.918400000e+04_r8, 1.948800000e+04_r8, &
            1.974400000e+04_r8, 1.995200000e+04_r8, 2.014400000e+04_r8, 2.027200000e+04_r8, 2.036800000e+04_r8, 2.043200000e+04_r8, 2.043200000e+04_r8, 2.040000000e+04_r8, &
            2.030400000e+04_r8, 2.017600000e+04_r8, 1.998400000e+04_r8, 1.974400000e+04_r8, 1.945600000e+04_r8, 1.910400000e+04_r8, 1.870400000e+04_r8, 1.825600000e+04_r8, &
            1.774400000e+04_r8, 1.718400000e+04_r8, 1.657600000e+04_r8, 1.592800000e+04_r8, 1.524800000e+04_r8, 1.453600000e+04_r8, 1.380000000e+04_r8, 1.304800000e+04_r8, &
            1.228800000e+04_r8, 1.152000000e+04_r8, 1.075200000e+04_r8, 9.992000000e+03_r8, 9.248000000e+03_r8, 8.520000000e+03_r8, 7.816000000e+03_r8, 7.136000000e+03_r8, &
            6.488000000e+03_r8, 5.868000000e+03_r8, 5.280000000e+03_r8, 4.724000000e+03_r8, 4.208000000e+03_r8, 3.722000000e+03_r8, 3.274000000e+03_r8, 2.858000000e+03_r8, &
            2.476000000e+03_r8, 2.128000000e+03_r8, 1.810000000e+03_r8, 1.524000000e+03_r8, 1.265000000e+03_r8, 1.035000000e+03_r8, 8.310000000e+02_r8, 6.515000000e+02_r8, &
            4.962500000e+02_r8, 3.635000000e+02_r8, 2.525000000e+02_r8, 1.622500000e+02_r8, 9.243750000e+01_r8, 4.281250000e+01_r8, 1.329687500e+01_r8, 1.878906250e+00_r8, &
            0.000000000e+00_r8]

real(r8), parameter :: era5_bk(137) = [ &
            0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, &
            0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, &
            0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, &
            0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, &
            0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, &
            0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, &
            0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 0.0000000000e+00_r8, 2.9802322387e-06_r8, 1.5974044799e-05_r8, &
            4.1007995605e-05_r8, 8.4996223449e-05_r8, 1.5604496002e-04_r8, 2.6893615722e-04_r8, 4.5108795166e-04_r8, 7.2622299194e-04_r8, 1.1224746704e-03_r8, 1.6717910766e-03_r8, &
            2.4242401123e-03_r8, 3.4141540527e-03_r8, 4.6730041503e-03_r8, 6.2561035156e-03_r8, 8.1939697265e-03_r8, 1.0536193847e-02_r8, 1.3313293457e-02_r8, 1.6571044921e-02_r8, &
            2.0339965820e-02_r8, 2.4658203125e-02_r8, 2.9571533203e-02_r8, 3.5095214843e-02_r8, 4.1290283203e-02_r8, 4.8156738281e-02_r8, 5.5755615234e-02_r8, 6.4086914062e-02_r8, &
            7.3181152343e-02_r8, 8.3129882812e-02_r8, 9.3872070312e-02_r8, 1.0546875000e-01_r8, 1.1798095703e-01_r8, 1.3134765625e-01_r8, 1.4575195312e-01_r8, 1.6101074218e-01_r8, &
            1.7724609375e-01_r8, 1.9458007812e-01_r8, 2.1289062500e-01_r8, 2.3229980468e-01_r8, 2.5268554687e-01_r8, 2.7441406250e-01_r8, 2.9687500000e-01_r8, 3.2080078125e-01_r8, &
            3.4570312500e-01_r8, 3.7133789062e-01_r8, 3.9770507812e-01_r8, 4.2480468750e-01_r8, 4.5214843750e-01_r8, 4.7998046875e-01_r8, 5.0781250000e-01_r8, 5.3564453125e-01_r8, &
            5.6298828125e-01_r8, 5.9033203125e-01_r8, 6.1669921875e-01_r8, 6.4306640625e-01_r8, 6.6796875000e-01_r8, 6.9287109375e-01_r8, 7.1630859375e-01_r8, 7.3876953125e-01_r8, &
            7.6025390625e-01_r8, 7.8076171875e-01_r8, 8.0029296875e-01_r8, 8.1835937500e-01_r8, 8.3544921875e-01_r8, 8.5156250000e-01_r8, 8.6669921875e-01_r8, 8.8085937500e-01_r8, &
            8.9355468750e-01_r8, 9.0576171875e-01_r8, 9.1699218750e-01_r8, 9.2675781250e-01_r8, 9.3652343750e-01_r8, 9.4482421875e-01_r8, 9.5263671875e-01_r8, 9.5996093750e-01_r8, &
            9.6630859375e-01_r8, 9.7216796875e-01_r8, 9.7753906250e-01_r8, 9.8242187500e-01_r8, 9.8632812500e-01_r8, 9.9023437500e-01_r8, 9.9365234375e-01_r8, 9.9609375000e-01_r8, &
            9.9902343750e-01_r8]



contains

!==========================================================================

subroutine gw_nlgw_dp_ml(state_in, ptend)

  ! inputs
  type(physics_state), intent(in) :: state_in
  ! outputs
  type(physics_ptend), intent(inout) :: ptend

  !---------------------------Local storage-------------------------------
  type(torch_tensor) :: tensor_in(1), tensor_out(1)
  integer :: ninputs = 1, noutputs = 1
  integer, dimension(2) :: layout = [1 , 2]

  integer :: device_id

  device_id = mod(iam, 2)

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

  allocate(net_inputs(ncol, 4*pver_interp+3))
  allocate(net_outputs(ncol, 2*pver_interp))

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
  call torch_tensor_from_array(tensor_in(1), net_inputs, layout, torch_kCUDA, device_id)
  call torch_tensor_from_array(tensor_out(1), net_outputs, layout, torch_kCPU)

  ! Run net forward on data
  call torch_model_forward(nlgw_model, tensor_in, tensor_out)

  ! Extract and denormalise outputs
  call extract_output()
  call denormalise_data()

  call flux_to_forcing(uflux, utgw)
  call flux_to_forcing(vflux, vtgw)

  ! Write UTGW and VTGW to file
  call outfld('UTGW_NL', utgw, ncol, pver)
  call outfld('VTGW_NL', vtgw, ncol, pver)


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

end subroutine gw_nlgw_dp_ml


subroutine gw_nlgw_dp_init(model_path)

  use cam_history,    only: addfld

  character(len=*), intent(in) :: model_path  ! Filepath to PyTorch Torchscript net
  integer :: device_id

  device_id = mod(iam, 2)

  ! Load the convective drag net from TorchScript file
  call torch_model_load(nlgw_model, model_path, device_type=torch_kCUDA, device_index=device_id)
  ! read in normalisation weights
  call read_norms()

  if (masterproc) then
     write(iulog,*)'nlgw model loaded from: ', model_path
  endif

  call addfld('UTGW_NL', (/ 'lev' /), 'U', 'm/s2', 'Nonlinear GW zonal wind tendency')
  call addfld('VTGW_NL', (/ 'lev' /), 'U', 'm/s2', 'Nonlinear GW meridional wind tendency')

end subroutine gw_nlgw_dp_init


subroutine gw_nlgw_dp_finalize()

  deallocate(net_inputs)
  deallocate(net_outputs)
  ! free model memory
  call torch_delete(nlgw_model)

end subroutine gw_nlgw_dp_finalize


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

  ! temporary arrays for interpolated values
  real(r8), dimension(:,:), allocatable :: &
    u_interp,       &! zonal wind (m/s)
    v_interp,       &! meridional wind (m/s)
    omega_interp,   &! vertical pressure velocity (Pa/s)
    theta_interp,   &! potential temperature (K)
    pmid_interp      ! midpoint pressure (Pa)

  integer :: idx_beg, idx_end, i

  allocate(pmid_interp(ncol,pver_interp))
  allocate(u_interp(ncol,pver_interp))
  allocate(v_interp(ncol,pver_interp))
  allocate(theta_interp(ncol,pver_interp))
  allocate(omega_interp(ncol,pver_interp))

  do i = 1, ncol
    pmid_interp(i,:) = era5_ak(:) + ps(i) * era5_bk(:)
    call lininterp(u(i,:), pmid(i,:), pver, u_interp(i,:), pmid_interp(i,:), pver_interp)
    call lininterp(v(i,:), pmid(i,:), pver, v_interp(i,:), pmid_interp(i,:), pver_interp)
    call lininterp(theta(i,:), pmid(i,:), pver, theta_interp(i,:), pmid_interp(i,:), pver_interp)
    call lininterp(omega(i,:), pmid(i,:), pver, omega_interp(i,:), pmid_interp(i,:), pver_interp)
  end do

  net_inputs(:,1) = lat
  net_inputs(:,2) = lon
  net_inputs(:,3) = phis

  idx_end = 3 ! last index written to was phis at position 3
  idx_beg = idx_end + 1
  idx_end = idx_beg + pver_interp - 1
  net_inputs(:,idx_beg:idx_end) = u_interp
  idx_beg = idx_end + 1
  idx_end = idx_beg + pver_interp - 1
  net_inputs(:,idx_beg:idx_end) = v_interp
  idx_beg = idx_end + 1
  idx_end = idx_beg + pver_interp - 1
  net_inputs(:,idx_beg:idx_end) = theta_interp
  idx_beg = idx_end + 1
  idx_end = idx_beg + pver_interp - 1
  net_inputs(:,idx_beg:idx_end) = omega_interp

  deallocate(pmid_interp)
  deallocate(u_interp)
  deallocate(v_interp)
  deallocate(theta_interp)
  deallocate(omega_interp)

end subroutine construct_input

subroutine extract_output()

  ! temporary arrays for interpolated values
  real(r8), dimension(:,:), allocatable :: &
    uflux_interp,       &! zonal wind (m/s)
    vflux_interp,       &! meridional wind (m/s)
    pmid_interp          ! midpoint pressure (Pa)

  integer :: idx_beg, idx_end, i

  allocate(pmid_interp(ncol,pver_interp))
  allocate(uflux_interp(ncol,pver_interp))
  allocate(vflux_interp(ncol,pver_interp))

  uflux_interp(:, :) = net_outputs(:,:pver_interp)
  vflux_interp(:, :) = net_outputs(:,pver_interp+1:)

  do i = 1, ncol
    pmid_interp(i,:) = era5_ak(:) + ps(i) * era5_bk(:)
    call lininterp(uflux_interp(i,:), pmid_interp(i,:), pver_interp, uflux(i,:), pmid(i,:), pver)
    call lininterp(vflux_interp(i,:), pmid_interp(i,:), pver_interp, vflux(i,:), pmid(i,:), pver)
  end do

  deallocate(pmid_interp)
  deallocate(uflux_interp)
  deallocate(vflux_interp)

end subroutine extract_output

subroutine denormalise_data()

  uflux = uflux**3._r8 * uflux_std + uflux_mean
  vflux = vflux**3._r8 * vflux_std + vflux_mean

end subroutine denormalise_data

elemental function cbrt(a) result(root)
  real(r8), intent(in) :: a
  real(r8), parameter :: one_third = 1._r8/3._r8
  real(r8) :: root
  root = sign(abs(a)**one_third, a)
end function cbrt

subroutine flux_to_forcing(flux, forcing)

  real(r8), intent(in), dimension(:,:) :: flux
  real(r8), intent(out), dimension(:,:) :: forcing ! forcing = -d(u'\omega')/d(p), units = m/s^2

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

end module gw_nlgw
