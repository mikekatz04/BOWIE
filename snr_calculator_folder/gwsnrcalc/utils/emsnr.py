from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

from astropy.cosmology import Planck15 as cosmo

from .emutils.bandpass import Bandpass
from .emutils.sed import Sed
from .emutils import signaltonoise
from .emutils.photometricparameters import PhotometricParameters
from .emutils.emtelescopesetup import EMTelescope


class EMCalc:
    def __init__(self, **kwargs):
        self.noise_interpolants = EMSNR(**kwargs)


class EMSNR:

    def __init__(self, **kwargs):

        prop_defaults = {
            'base_dir': os.path.dirname(os.path.abspath(__file__)) + '/',
            'filedir': 'emutils/em_files/seds/',
            'stdFilter': 'r',
            'signal_type': ('u', 'g', 'r', 'i', 'z'),
            'source': 'mbh',
            'telescope': None,
        }

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        # stock mbh sed
        if self.source == 'mbh':
            self.sed_filename = 'quasar.gz'

        # stock wd sed
        elif self.source == 'wd':
            self.sed_filename = 'wd.gz'

        # upload filename for sed. must be in sed folder
        else:
            self.sed_filename = self.source

        if self.source is None:
            raise ValueError('source must be provided: mbh, wd, or file name in sed folder.')
        if self.telescope is None:
            raise ValueError('telescope must be provided.')

        self.filterlist = self.signal_type
        self.setup_sed()

    def setup_sed(self):
        self.sed_base = Sed()
        self.sed_base.readSED_flambda(self.base_dir + self.filedir
                                      + self.sed_filename, name=self.sed_filename)
        return


    def __call__(self, mag, z):
        # Now we'll read in each of those individual seds, into a Sed object. We'll also redshift the quasar.
        snr_out = {}
        sed = self.sed_base
        sed.redshiftSED(z)

        fluxNorm = sed.calcFluxNorm(mag, self.telescope.noise_total[self.stdFilter]['total_throughputs'])
        sed.multiplyFluxNorm(fluxNorm)

        # Calculate SNR for all seds in all filters.
        for f in self.filterlist:
            tele_info = self.telescope.noise_total[f]
            key = tele_info['telescope'] + '_' + f
            snr_out[key] = signaltonoise.calcSNR_sed(sed,
                                        tele_info['total_throughputs'], tele_info['neff'],
                                        tele_info['noise_sky_sq'],
                                        tele_info['noise_instr_sq'], self.telescope.photParams,
                                        verbose=False)
        return snr_out


class MBHEddMag:
    def __init__(self, **kwargs):
        prop_defaults = {
            'dist_type': 'redshift',
        }

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

    def _eddington_luminosity(self, mass):
        return 3e4*mass  # mass in solar masses returns solar luminosity unit

    def _relative_mag(self, mass, distance):
        lum = self._eddington_luminosity(mass)
        abs_mag = +4.77 - 2.5*np.log10(lum)  # lum in units of solar luminosity
        rel_mag = abs_mag + 5.*np.log10(distance) - 5.  # distance in pc
        return rel_mag

    def _sanity_check(self):
        """Check if parameters are okay.

        Sanity check makes sure each parameter is within an allowable range.

        Raises:
            ValueError: Problem with a specific parameter.

        """
        if any(self.total_mass < 0.0):
            raise ValueError("Mass 1 is negative.")

        if any(self.z <= 0.0):
            raise ValueError("Redshift is zero or negative.")

        if any(self.q > 1.0000001e4) or any(self.q < 9.999999e-5):
            raise ValueError("Mass Ratio too far from unity.")

        return

    def _broadcast_and_set_attrs(self, local_dict):
        """Cast all inputs to correct dimensions.

        This method fixes inputs who have different lengths. Namely one input as
        an array and others that are scalara or of len-1.

        Raises:
            Value Error: Multiple length arrays of len>1

        """
        del local_dict['self']
        self.remove_axis = False
        max_length = 0
        for key in local_dict:
            try:
                length = len(local_dict[key])
                if length > max_length:
                    max_length = length

            except TypeError:
                pass

        if max_length == 0:
            self.remove_axis = True
            for key in local_dict:
                setattr(self, key, np.array([local_dict[key]]))

        # check for bad length arrays
        else:
            for key in local_dict:
                try:
                    if len(local_dict[key]) < max_length and len(local_dict[key]) > 1:
                        raise ValueError("Casting parameters not correct."
                                         + " Need all at a maximum shape and the rest being"
                                         + "len-1 arrays or scalars")
                except TypeError:
                    pass

            # broadcast arrays
            for key in local_dict:
                try:
                    if len(local_dict[key]) == max_length:
                        setattr(self, key, local_dict[key])
                    elif len(local_dict[key]) == 1:
                        setattr(self, key, np.full((max_length,), local_dict[key][0]))
                except TypeError:
                    setattr(self, key, np.full((max_length,), local_dict[key]))
        return

    def __call__(self, total_mass, q, z_or_dist):
        """Calculate the detectability of EM observable.

        This assumes Eddington luminosity for smaller black hole of the pair.

        """
        self._broadcast_and_set_attrs(locals())

        # based on distance inputs, need to find redshift and luminosity distance.
        if self.dist_type == 'redshift':
            self.z = self.z_or_dist
            self.dist = cosmo.luminosity_distance(self.z).value

        elif self.dist_type == 'luminosity_distance':
            z_in = np.logspace(-3, 3, 10000)
            lum_dis = cosmo.luminosity_distance(z_in).value

            self.dist = self.z_or_dist
            self.z = np.interp(self.dist, lum_dis, z_in)

        elif self.dist_type == 'comoving_distance':
            z_in = np.logspace(-3, 3, 10000)
            lum_dis = cosmo.luminosity_distance(z_in).value
            com_dis = cosmo.comoving_distance(z_in).value

            comoving_distance = self.z_or_dist
            self.z = np.interp(comoving_distance, com_dis, z_in)
            self.dist = np.interp(comoving_distance, com_dis, lum_dis)

        self._sanity_check()
        m2 = self.total_mass*self.q/(1+self.q)
        self.dist = self.dist*1e6
        rel_mag = self._relative_mag(m2, self.dist)
        return rel_mag, self.z


def parallel_em_snr_func(num, binary_args, mbheddmag,
                         noise_interpolants, prefactor, verbose):
    """SNR calulation with eccentric waveforms

    Generate eddington magnitudes for MBHs and calculate their SNR for LSST in
    different bands.

    Args:
        num (int): Process number. If only a single process, num=0.
        binary_args (tuple): Binary arguments for
            :meth:`gwsnrcalc.utils.waveforms.EccentricBinaries.__call__`.
        mbheddmag (obj): Initialized class of
            :class:`gwsnrcalc.utils.lsstsnr.MBHEddMag``.
        signal_type (list of str): List with types of SNR to calculate.
            `all` for quadrature sum of modes or `modes` for SNR from each mode.
            This must be `all` if generating contour data with
            :mod:`gwsnrcalc.generate_contour_data`.
        noise_interpolants (dict): All the noise noise interpolants generated by
            :mod:`gwsnrcalc.utils.sensitivity`.
        prefactor (float): Prefactor to multiply SNR by (not SNR^2).
        verbose (int): Notify each time ``verbose`` processes finish. If -1, then no notification.

    Returns:
        (dict): Dictionary with the SNR output from the calculation.

    """
    mag, z = mbheddmag(*binary_args)

    out_vals = noise_interpolants(mag, z)

    if verbose > 0 and (num+1) % verbose == 0:
        print('Process ', (num+1), 'is finished.')

    return out_vals
