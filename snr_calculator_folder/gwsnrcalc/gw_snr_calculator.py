"""
Author: Michael Katz guided by lal implimentation of PhenomD. This was used in
"Evaluating Black Hole Detectability with LISA" (arXiv:1508.07253), as a part of the BOWIE package
(https://github.com/mikekatz04/BOWIE).

    This code is licensed with the GNU public license.

    This python code impliments PhenomD waveforms from Husa et al 2016 (arXiv:1508.07250)
    and Khan et al 2016 (arXiv:1508.07253). It wraps the accompanying c code, ``phenomd.c``,
    with ``ctypes``. ``phenomd.c`` is mostly from LALsuite. See ``phenomd.c`` for specifics.

    Please cite all of the arXiv papers above if you use this code in a publication.

"""

import ctypes
from astropy.cosmology import Planck15 as cosmo
import numpy as np
from scipy import interpolate
from astropy.io import ascii
import os


from snr_calculator.utils.readnoisecurves import read_noise_curve
from snr_calculator.utils.pyphenomd import PhenomDWaveforms
from snr_calculator.utils.csnr import csnr
from snr_calculator.utils.sensitivity import SensitivityContainer
from snr_calculator.utils.parallel import ParallelContainer


class SNR(SensitivityContainer, ParallelContainer):

    def __init__(self, ecc=False, **kwargs):
        SensitivityContainer.__init__(self, **kwargs)
        ParallelContainer.__init__(self, **kwargs)

        if ecc:
            pass
        else:
            self.snr_function = parallel_snr_func

    def run(self, length, *binary_args):
        if self.num_processors is None:
            func_args = (0,) + binary_args + self.sensitivity_args + (self.verbose,)
            return self.snr_function(*func_args)

        self._prep_parallel(length, binary_args, self.sensitivity_args, self.verbose)
        return self._run_parallel(self.snr_function)

    def __call__(self, m1, m2, z_or_dist, st, et,
                 chi_1=None, chi_2=None, chi=0.8, dist_type='redshift'):

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
            raise Exception("Either supply `chi`, or supply both `chi_1` and `chi_2`."
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
                      noise_interpolants, phases,
                      prefactor, num_points, verbose):

    wave = PhenomDWaveforms(m1, m2, chi1, chi2, z_or_dist, st, et, dist_type, num_points)

    out_vals = {}
    for key in noise_interpolants:
        hn_vals = noise_interpolants[key](wave.freqs)
        snr_out = csnr(wave.freqs, wave.hc, hn_vals,
                                  wave.fmrg, wave.fpeak, prefactor=prefactor)

        if len(phases) == 1:
            out_vals[key + '_' + phases[0]] = snr_out[phases[0]]
        else:
            for phase in phases:
                out_vals[key + '_' + phase] = snr_out[phase]
    if verbose > 0 and (num+1) % verbose == 0:
        print('Process ', (num+1), 'is finished.')

    return out_vals


def snr(m1, m2, z_or_dist, st, et, chi=0.8, chi_1=None, chi_2=None, dist_type='redshift', **kwargs):

        #, ,
        #sensitivity_curves='LPA', wd_noise=None, phases='all',
        #prefactor=1.0,  num_points=8192, num_procs=None,
        #num_splits=1000, verbose=-1, timer=False):
    """Compute the SNR of binaries.

    # TODO: add parallel capabilities
    # TODO: add list of phases and/or sensitivity curves
    snr is a function that takes binary parameters and a sensitivity curve as inputs,
    and returns snr from the chosen phase.

    ** Warning **: All binary parameters need to have the same shape, either scalar or 1D array.
    Start time (st) and end time (et) can be scalars while the rest of
    the binary parameters are arrays.

    Arguments:
        m1 (float or 1D array of floats): Mass 1 in Solar Masses. (>0.0)
        m2 (float or 1D array of floats): Mass 2 in Solar Masses. (>0.0)
        chi1 (float or 1D array of floats): dimensionless spin of mass 1
            aligned to orbital angular momentum. [-1.0, 1.0]
        chi2 (float or 1D array of floats): dimensionless spin of mass 2
            aligned to orbital angular momentum. [-1.0, 1.0]
        z_or_dist (float or 1D array of floats): Distance measure to the binary.
            This can take three forms: redshift (dimensionless, *default*),
            luminosity distance (Mpc), comoving_distance (Mpc).
            The type used must be specified in 'dist_type' parameter. (>0.0)
        st (float or 1D array of floats): Start time of waveform in years before
            end of the merger phase. This is determined using 1 PN order. (>0.0)
        et (float or 1D array of floats): End time of waveform in years before
            end of the merger phase. This is determined using 1 PN order. (>0.0)
        sensitivity_curve (scalar or list of str or single or list of lists, optional):
            String that starts the .txt file containing the sensitivity curve in
            folder 'noise_curves/' or list of ``[f_n, asd_n]``
            in terms of an amplitude spectral density. It can be a single one of these
            values or a list of these values.
            Default is `LPA`.
        wd_noise (str or bool or list, optional): If True, use the Hils-Bender estimation
            (Bender & Hils 1997) by Hiscock et al. 2000 of the wd background. If string,
            read the wd noise from `noise_curves` folder. If list, must be ``[f_n_wd, asd_n_wd]``.
            Default is None.
        phases (scalar or list of str, optional): Phase of snr. Options are 'all' for all phases;
            'ins' for inspiral; 'mrg' for merger; or 'rd' for ringdown. Default is 'all'.
        prefactor (float, optional): Factor to multiply snr (not snr^2) integral values by.
            Default is 1.0.
        dist_type (str, optional): Which type of distance is used. Default is 'redshift'.
        num_points (int, optional): Number of points to use in the waveform.
            The frequency points are log-spaced. Default is 8192.

    Returns:
        (list or list of dict): Signal-to-Noise Ratio for requested phase.

    """

    snr_main = SNR(**kwargs)
    return snr_main(m1, m2, z_or_dist, st, et, chi=chi, chi_1=chi_1, chi_2=chi_2, dist_type=dist_type)
