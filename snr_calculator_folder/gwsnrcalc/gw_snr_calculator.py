"""
Calculate gravitational wave SNRs.

This was used in "Evaluating Black Hole Detectability with LISA" (arXiv:1508.07253),
as a part of the BOWIE package (https://github.com/mikekatz04/BOWIE).

This code is licensed with the GNU public license.

This python code impliments PhenomD waveforms from Husa et al 2016 (arXiv:1508.07250)
and Khan et al 2016 (arXiv:1508.07253).

Please cite all of the arXiv papers above if you use this code in a publication.

"""

import numpy as np

from gwsnrcalc.utils.pyphenomd import PhenomDWaveforms
from gwsnrcalc.utils.csnr import csnr
from gwsnrcalc.utils.sensitivity import SensitivityContainer
from gwsnrcalc.utils.parallel import ParallelContainer


class SNR(SensitivityContainer, ParallelContainer):
    """Main class for SNR calculations.

    This class performs gravitational wave SNR calculations with a matched
    filtering approach. It can generate SNRs for single sources or arrays of sources.
    It can run in parallel or on a single processor.

    Args:
        ecc (bool, optional): If True, use the eccentric SNR calculator. If False,
            use PhenomDWaveforms. (For future usage.) Default is False.
        kwargs (dict): kwargs to be added to ParallelContainer and SensitivityContainer.

    Attributes:
        add_wd_noise (str or bool, optional): Options are `yes`, `no`, `True`, `False`,
            `Both`, True, or False. `yes`, `True`, or True
            will exclusively calculate with the wd noise.
            `no`, `False`, or False will exclusively calculate without the wd noise.
            `Both` will calculate with and without wd noise.
            Default is `Both`.
        wd_noise (str or list, optional): If string,
            read the wd noise from `noise_curves` folder or absolute path to file.
            If list, must be ``[f_n_wd, asd_n_wd]``.
            Default is Hils-Bender estimation (Bender & Hils 1997) by Hiscock et al. 2000.
        signal_type (scalar or list of str, optional): Phase of snr.
            Options are 'all' for all phases;
            'ins' for inspiral; 'mrg' for merger; or 'rd' for ringdown. Default is 'all'.
        prefactor (float, optional): Factor to multiply snr (not snr^2) integral values by.
            Default is 1.0.
        num_points (int, optional): Number of points to use in the waveform.
            The frequency points are log-spaced. Default is 8192.
        noise_type_in (str, optional): Type of noise curve passed in.
            Options are `ASD`, `PSD`, or `char_strain`.
            All sensitivity curves must have same noise type.
            Also, their amplitude column must have this same string as its label.
            Default is `ASD`.
        num_processors (int or None, optional): If None, run on single processor.
            If -1, use ```multiprocessing.cpu_count()`` to determine cpus to use.
            Otherwise, this is the number of processors to use. Default is -1.
        num_splits (int, optional): Number of binaries to run for each process. Default is 1000.
        verbose (int, optional): Notify each time ``verbose`` processes finish.
            If -1, then no notification. Default is -1.
        timer (bool, optional): If True, time the parallel process. Default is False.

    """
    def __init__(self, ecc=False, **kwargs):

        # initialize sensitivity and parallel modules
        SensitivityContainer.__init__(self, **kwargs)
        ParallelContainer.__init__(self, **kwargs)

        # set the SNR function
        if ecc:
            pass
        else:
            self.snr_function = parallel_snr_func

    def run(self, length, *binary_args):
        """Perform the SNR calculation.

        This will take binary inputs and calculate their SNRs.

        Args:
            length (int): Number of binaries to process.
            binary_args (list): List of binary arguments from ``__call__`` for
                input into the SNR function.

        Returns:
            (dict): Dictionary with the SNR output from the calculation.

        """
        # if self.num_processors is None, run on single processor
        if self.num_processors is None:
            func_args = (0,) + binary_args + self.sensitivity_args + (self.verbose,)
            return self.snr_function(*func_args)

        self.prep_parallel(length, binary_args, self.sensitivity_args)
        return self.run_parallel(self.snr_function)

    def __call__(self, m1, m2, z_or_dist, st, et,
                 chi_1=None, chi_2=None, chi=0.8, dist_type='redshift'):
        """Input binary parameters and calculate the SNR

        Binary parameters are read in and adjusted based on shapes. They are then
        fed into ``run`` for calculation of the snr.

        Args:
            m1 (float or 1D array of floats): Mass 1 in Solar Masses. (>0.0)
            m2 (float or 1D array of floats): Mass 2 in Solar Masses. (>0.0)
            z_or_dist (float or 1D array of floats): Distance measure to the binary.
                This can take three forms: redshift (dimensionless, *default*),
                luminosity distance (Mpc), comoving_distance (Mpc).
                The type used must be specified in 'dist_type' parameter. (>0.0)
            st (float or 1D array of floats): Start time of waveform in years before
                end of the merger phase. This is determined using 1 PN order. (>0.0)
            et (float or 1D array of floats): End time of waveform in years before
                end of the merger phase. This is determined using 1 PN order. (>0.0)
            chi_1 (float or 1D array of floats, optional): dimensionless spin of mass 1
                aligned to orbital angular momentum. Default is None (not 0.0). [-1.0, 1.0]
            chi_2 (float or 1D array of floats, optional): dimensionless spin of mass 2
                aligned to orbital angular momentum. Default is None (not 0.0). [-1.0, 1.0]
            chi (float or 1D array of floats, optional): dimensionless spin of mass 1 and mass 2
                aligned to orbital angular momentum. This is used if both chi_1 or chi_2 are None.
                Default is 0.8. [-1.0, 1.0]
            dist_type (str, optional): Which type of distance is used. Default is 'redshift'.

        Returns:
            (dict): Dictionary with the SNR output from the calculation.

        Raises:
            UserError: Supplying chi_1 or chi_2

        """
        # check shapes and spins and cast to the right values
        try:
            len(m1)
            try:
                len(st)
            except TypeError:
                st = np.full(len(m1), st)
                et = np.full(len(m1), et)

        except TypeError:
            pass

        if ((chi_1 is None) & (chi_2 is not None)) or ((chi_1 is not None) & (chi_2 is None)):
            raise UserError("Either supply `chi`, or supply both `chi_1` and `chi_2`."
                            + "You supplied only `chi_1` or `chi_2`.")

        if chi_1 is None:
            if type(chi) == float:
                try:
                    chi = np.full((len(m1),), chi)
                except TypeError:
                    pass

            chi_1 = chi
            chi_2 = chi
        else:
            if type(chi_1) == float:
                try:
                    chi_1 = np.full((len(m1),), chi_1)
                except TypeError:
                    pass

            if type(chi_2) == float:
                try:
                    chi_2 = np.full((len(m1),), chi_2)
                except TypeError:
                    pass

            chi_1 = chi_1
            chi_2 = chi_2

        # check the number of binaries
        # if 1 binary, then cast to arrays and then squeeze output
        try:
            len(m1)
            return self.run(len(m1), m1, m2, z_or_dist, st, et, chi_1, chi_2, dist_type)
        except TypeError:
            m1 = np.array([m1])
            m2 = np.array([m2])
            z_or_dist = np.array([z_or_dist])
            st = np.array([st])
            et = np.array([et])
            chi_1 = np.array([chi_1])
            chi_2 = np.array([chi_2])

            snr_out = self.run(len(m1), m1, m2, z_or_dist, st, et, chi_1, chi_2, dist_type)
            snr_out = {key: float(np.squeeze(snr_out[key])) for key in snr_out}
            return snr_out


def parallel_snr_func(num, m1, m2, z_or_dist, st, et, chi1, chi2, dist_type,
                      noise_interpolants, signal_type,
                      prefactor, num_points, verbose):
    """SNR calulation with PhenomDWaveforms

    Generate PhenomDWaveforms and calculate their SNR against sensitivity curves.

    Args:
        num (int): Process number. If only a single process, num=0.
        m1 (float or 1D array of floats): Mass 1 in Solar Masses. (>0.0)
        m2 (float or 1D array of floats): Mass 2 in Solar Masses. (>0.0)
        z_or_dist (float or 1D array of floats): Distance measure to the binary.
            This can take three forms: redshift (dimensionless, *default*),
            luminosity distance (Mpc), comoving_distance (Mpc).
            The type used must be specified in 'dist_type' parameter. (>0.0)
        st (float or 1D array of floats): Start time of waveform in years before
            end of the merger phase. This is determined using 1 PN order. (>0.0)
        et (float or 1D array of floats): End time of waveform in years before
            end of the merger phase. This is determined using 1 PN order. (>0.0)
        chi1 (float or 1D array of floats): dimensionless spin of mass 1
            aligned to orbital angular momentum. Default is None (not 0.0). [-1.0, 1.0]
        chi2 (float or 1D array of floats): dimensionless spin of mass 2
            aligned to orbital angular momentum. Default is None (not 0.0). [-1.0, 1.0]
        dist_type (str): Which type of distance is used. Default is 'redshift'.
        noise_interpolants (list of obj): List of noise interpolations generated from
            sensitivity module.
        phases (list of str): Phases from which SNR is desired. Generated by sensitivity module.
        prefactor (float): Prefactor to multiply SNR by (not SNR^2).
        num_points (int): Number of points in the generated waveforms. More points
            will asympotically converge to correct SNR value. The waveforms are log-spaced
            to conserve memory.
        verbose (int): Notify each time ``verbose`` processes finish. If -1, then no notification.

    Returns:
        (dict): Dictionary with the SNR output from the calculation.

    """

    wave = PhenomDWaveforms(m1, m2, chi1, chi2, z_or_dist, st, et, dist_type, num_points)

    out_vals = {}
    for key in noise_interpolants:
        hn_vals = noise_interpolants[key](wave.freqs)
        snr_out = csnr(wave.freqs, wave.hc, hn_vals,
                       wave.fmrg, wave.fpeak, prefactor=prefactor)

        if len(signal_type) == 1:
            out_vals[key + '_' + signal_type[0]] = snr_out[signal_type[0]]
        else:
            for phase in signal_type:
                out_vals[key + '_' + phase] = snr_out[phase]
    if verbose > 0 and (num+1) % verbose == 0:
        print('Process ', (num+1), 'is finished.')

    return out_vals


def snr(m1, m2, z_or_dist, st, et, chi=0.8, chi_1=None, chi_2=None, dist_type='redshift', **kwargs):
    """Compute the SNR of binaries.

    snr is a function that takes binary parameters and sensitivity curves as inputs,
    and returns snr for chosen phases.

    ** Warning **: All binary parameters need to have the same shape, either scalar or 1D array.
    Start time (st), end time (et), and/or chi values can be scalars while the rest of
    the binary parameters are arrays.

    Arguments:
        m1 (float or 1D array of floats): Mass 1 in Solar Masses. (>0.0)
        m2 (float or 1D array of floats): Mass 2 in Solar Masses. (>0.0)
        z_or_dist (float or 1D array of floats): Distance measure to the binary.
            This can take three forms: redshift (dimensionless, *default*),
            luminosity distance (Mpc), comoving_distance (Mpc).
            The type used must be specified in 'dist_type' parameter. (>0.0)
        st (float or 1D array of floats): Start time of waveform in years before
            end of the merger phase. This is determined using 1 PN order. (>0.0)
        et (float or 1D array of floats): End time of waveform in years before
            end of the merger phase. This is determined using 1 PN order. (>0.0)
        chi_1 (float or 1D array of floats, optional): dimensionless spin of mass 1
            aligned to orbital angular momentum. Default is None (not 0.0). [-1.0, 1.0]
        chi_2 (float or 1D array of floats, optional): dimensionless spin of mass 2
            aligned to orbital angular momentum. Default is None (not 0.0). [-1.0, 1.0]
        chi (float or 1D array of floats, optional): dimensionless spin of mass 1 and mass 2
            aligned to orbital angular momentum. This is used if both chi_1 or chi_2 are None.
            Default is 0.8. [-1.0, 1.0]
        dist_type (str, optional): Which type of distance is used. Default is 'redshift'.
        sensitivity_curves (scalar or list of str or single or list of lists, optional):
            String that starts the .txt file containing the sensitivity curve in
            folder 'noise_curves/' or list of ``[f_n, asd_n]``
            in terms of an amplitude spectral density. It can be a single one of these
            values or a list of these values. A folder string with absolute path to a sensitivity
            curve .txt file can also be input.
            Default is [`LPA`] (LISA Phase A).
        add_wd_noise (str or bool, optional): Options are `yes`, `no`, `True`, `False`,
            `Both`, True, or False. `yes`, `True`, or True
            will exclusively calculate with the wd noise.
            `no`, `False`, or False will exclusively calculate without the wd noise.
            `Both` will calculate with and without wd noise.
            Default is `Both`.
        wd_noise (str or list, optional): If string,
            read the wd noise from `noise_curves` folder or absolute path to file.
            If list, must be ``[f_n_wd, asd_n_wd]``.
            Default is Hils-Bender estimation (Bender & Hils 1997) by Hiscock et al. 2000.
        signal_type (scalar or list of str, optional): Phase of snr.
            Options are 'all' for all phases;
            'ins' for inspiral; 'mrg' for merger; or 'rd' for ringdown. Default is 'all'.
        prefactor (float, optional): Factor to multiply snr (not snr^2) integral values by.
            Default is 1.0.
        num_points (int, optional): Number of points to use in the waveform.
            The frequency points are log-spaced. Default is 8192.
        noise_type_in (str, optional): Type of noise curve passed in.
            Options are `ASD`, `PSD`, or `char_strain`.
            All sensitivity curves must have same noise type.
            Also, their amplitude column must have this same string as its label.
            Default is `ASD`.
        num_processors (int or None, optional): If None, run on single processor.
            If -1, use ```multiprocessing.cpu_count()`` to determine cpus to use.
            Otherwise, this is the number of processors to use. Default is -1.
        num_splits (int, optional): Number of binaries to run for each process. Default is 1000.
        verbose (int, optional): Notify each time ``verbose`` processes finish.
            If -1, then no notification. Default is -1.
        timer (bool, optional): If True, time the parallel process. Default is False.

    Returns:
        (dict or list of dict): Signal-to-Noise Ratio dictionary for requested phases.

    """

    snr_main = SNR(**kwargs)
    return snr_main(m1, m2, z_or_dist, st, et,
                    chi=chi, chi_1=chi_1,
                    chi_2=chi_2, dist_type=dist_type)
