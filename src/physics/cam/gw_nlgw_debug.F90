module gw_nlgw_debug

    use gw_utils, only: r8, r4
    use spmd_utils, only: iam

    implicit none
    private

    public :: dump_column_data, check_flux_derivatives, dump_flux_profile

contains

    subroutine dump_column_data(col, net_inputs, net_outputs, utgw, vtgw, pmid, phis, threshold, lchnk, pver)
        use gw_utils, only: r8, r4
        use spmd_utils, only: iam
        implicit none

        integer, intent(in) :: col, lchnk, pver
        real(r4), intent(in) :: net_inputs(:, :), net_outputs(:, :)
        real(r8), intent(in) :: utgw(:, :), vtgw(:, :), pmid(:, :), phis(:)
        real(r8), intent(in) :: threshold

        character(len=128) :: column_filename, summary_filename
        character(len=8) :: date_str
        character(len=10) :: time_str
        integer :: col_unit, summary_unit, ios, i
        integer :: maxlev_u, maxlev_v
        real(r8) :: max_utgw, max_vtgw, max_tend
        real(r8) :: lat_val, lon_val

        ! Original statistics for de-normalization
        real(r8), parameter :: lat_mean = 0.0_r8, lon_mean = 0.0_r8
        real(r8), parameter :: lat_std = 90.0_r8, lon_std = 360.0_r8

        call date_and_time(date=date_str, time=time_str)

        ! Determine max tendencies and levels
        max_utgw = maxval(abs(utgw(col, :)))
        maxlev_u = maxloc(abs(utgw(col, :)), dim=1)

        max_vtgw = maxval(abs(vtgw(col, :)))
        maxlev_v = maxloc(abs(vtgw(col, :)), dim=1)

        max_tend = max(max_utgw, max_vtgw)

        ! Compute lat/lon from normalized inputs
        lat_val = net_inputs(col, 1)*lat_std + lat_mean
        lon_val = net_inputs(col, 2)*lon_std + lon_mean

        ! Construct file names
        write (column_filename, '(A,I3.3,"_chunk",I3.3,"_col",I5.5,"_",A,".txt")') &
            'nlgw_bad_col_', iam, lchnk, col, time_str(1:6)
        write (summary_filename, '(A,I3.3,"_chunk",I3.3,"_summary.txt")') 'nlgw_bad_summary_', iam, lchnk

        ! Write detailed data to per-column file
        open (newunit=col_unit, file=column_filename, status='replace', iostat=ios)
        if (ios /= 0) stop 'Could not open per-column output file'

        write (col_unit, *) '--- Bad column detected ---'
        write (col_unit, *) 'col = ', col
        write (col_unit, *) 'lat = ', lat_val, ', lon = ', lon_val
        write (col_unit, *) 'max_tendency = ', max_tend

        if (max_utgw >= max_vtgw) then
            write (col_unit, *) 'max utgw at level ', maxlev_u, ', pmid = ', pmid(col, maxlev_u), ', value = ', utgw(col, maxlev_u)
        else
            write (col_unit, *) 'max vtgw at level ', maxlev_v, ', pmid = ', pmid(col, maxlev_v), ', value = ', vtgw(col, maxlev_v)
        end if

        write (col_unit, *) 'phis = ', phis(col)
        write (col_unit, *) 'pmid(1:5) = ', pmid(col, 1:5)

        write (col_unit, *) 'net_inputs(col,:) = '
        do i = 1, size(net_inputs, 2), 10
            write (col_unit, '(10f12.6)') net_inputs(col, i:min(i + 9, size(net_inputs, 2)))
        end do

        write (col_unit, *) 'net_outputs(col,:) = '
        do i = 1, size(net_outputs, 2), 10
            write (col_unit, '(10f12.6)') net_outputs(col, i:min(i + 9, size(net_outputs, 2)))
        end do

        write (col_unit, *) 'utgw(pver-9:pver) = ', utgw(col, pver - 9:pver)
        write (col_unit, *) 'vtgw(pver-9:pver) = ', vtgw(col, pver - 9:pver)
        write (col_unit, *) '---------------------------'

        close (col_unit)

        ! Append stats to shared summary file
        open (newunit=summary_unit, file=summary_filename, status='unknown', position='append', iostat=ios)
        if (ios /= 0) stop 'Could not open summary file for bad columns'

        write (summary_unit, '(A,I0,A,I0,A,F12.6,A,A)') &
            'col=', col, ', level=', merge(maxlev_u, maxlev_v, max_utgw >= max_vtgw), ', max_tendency=', max_tend, ', file=', trim(column_filename)

        close (summary_unit)
    end subroutine dump_column_data

    ! This outputs in model space i.e. pver_interp = 137.
    subroutine dump_flux_profile(col, flux, pmid, lchnk, pver)
        use gw_utils, only: r8
        use spmd_utils, only: iam
        implicit none

        integer, intent(in) :: col, lchnk, pver
        real(r4), intent(in) :: flux(:, :)
        real(r8), intent(in) :: pmid(:, :)

        character(len=128) :: filename
        character(len=8) :: date_str
        character(len=10) :: time_str
        integer :: unit, ios, level

        ! Get timestamp
        call date_and_time(date=date_str, time=time_str)

        ! Construct output filename
        write (filename, '(A,I3.3,"_chunk",I3.3,"_col",I5.5,"_",A,".txt")') &
            'nlgw_flux_profile_', iam, lchnk, col, time_str(1:6)

        ! Open file
        open (newunit=unit, file=filename, status='replace', iostat=ios)
        if (ios /= 0) stop 'Could not open flux profile output file'

        write (unit, *) 'Column =', col, ' Chunk =', lchnk
        write (unit, *) 'Level    pmid (Pa)     flux'
        write (unit, *) '-----------------------------'

        ! this is only getting the first set of fluxes
        do level = 1, pver
            write (unit, '(I4, 2X, F12.2, 2X, F12.6)') level, pmid(col, level), flux(col, level)
        end do

        close (unit)
    end subroutine dump_flux_profile

    subroutine check_flux_derivatives(flux, pmid, lchnk, threshold)
        use gw_utils, only: r8
        use spmd_utils, only: iam
        implicit none

        real(r8), intent(in) :: flux(:, :), pmid(:, :)
        integer, intent(in) :: lchnk
        real(r8), intent(in) :: threshold
        integer :: col, level, ncol, pver
        real(r8) :: dflux, dp, derivative
        character(len=128) :: dbgfile
        integer :: dbgunit, ios

        ncol = size(flux, 1)
        pver = size(flux, 2)

        do col = 1, ncol
            do level = 2, pver - 1
                dflux = flux(col, level + 1) - flux(col, level - 1)
                dp = log(pmid(col, level + 1)) - log(pmid(col, level - 1))

                ! if (abs(dp) < 1.0e-6_r8) cycle
                if (abs(dp) < 1.0e-6_r8) then
                    write (dbgunit, *) 'WARNING: Small dp detected!'
                    write (dbgunit, *) 'col=', col, ' level=', level
                    write (dbgunit, *) 'pmid(-1,0,+1)=', pmid(col, level - 1), pmid(col, level), pmid(col, level + 1)
                    write (dbgunit, *) 'dp=', dp
                    write (dbgunit, *) '----------------------------------------'
                    ! cycle  ! Optional: still skip this point in derivative calc
                end if

                derivative = dflux/(pmid(col, level)*dp)

                if (abs(derivative) > threshold) then
                    write (dbgfile, '(A,I3.3,"_chunk",I3.3,"_col",I5.5,"_deriv.txt")') &
                        'nlgw_deriv_', iam, lchnk, col

                    open (newunit=dbgunit, file=dbgfile, status='replace', iostat=ios)
                    if (ios /= 0) stop 'Could not open derivative debug file'

                    write (dbgunit, *) '--- Large flux derivative detected ---'
                    write (dbgunit, *) 'col=', col, ' level=', level
                    write (dbgunit, *) 'flux(-1,0,+1)=', flux(col, level - 1), flux(col, level), flux(col, level + 1)
                    write (dbgunit, *) 'pmid(-1,0,+1)=', pmid(col, level - 1), pmid(col, level), pmid(col, level + 1)
                    write (dbgunit, *) 'dflux=', dflux, ' dp=', dp, ' forcing=', derivative
                    write (dbgunit, *) '----------------------------------------'

                    close (dbgunit)

                    exit  ! optional: stop after first bad level for this column
                end if
            end do
        end do
    end subroutine check_flux_derivatives

end module gw_nlgw_debug
