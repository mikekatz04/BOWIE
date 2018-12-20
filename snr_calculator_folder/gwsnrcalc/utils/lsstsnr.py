from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

from astropy.cosmology import Planck15 as cosmo

from gwsnrcalc.utils.lsstutils.Bandpass import Bandpass
from gwsnrcalc.utils.lsstutils.Sed import Sed
from gwsnrcalc.utils.lsstutils import SignalToNoise
from gwsnrcalc.utils.lsstutils.PhotometricParameters import PhotometricParameters


class LSSTCalc:
    def __init__(self, **kwargs):
        self.noise_interpolants = LSSTSNR(**kwargs)


class LSSTSNR:

    def __init__(self, **kwargs):

        prop_defaults = {
            'base_dir': os.path.dirname(os.path.abspath(__file__)) + '/',
            'filenames': ['quasar.gz'],
            'filedir': 'lsstutils/lsst_files/',
            'throughputsDir': 'lsstutils/lsst_files/',
            'atmosDir': 'lsstutils/lsst_files/',
            'stdFilter': 'r',
            'filterlist': ('u', 'g', 'r', 'i', 'z', 'y'),
            'filtercolors': {'u': 'b', 'g': 'c', 'r': 'g', 'i': 'y', 'z': 'r', 'y': 'm'},
        }

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        self.setup_sed()
        self.setup_noise_info()

    def setup_sed(self):
        self.seds = {}
        for s in self.filenames:
            self.seds[s] = Sed()
            self.seds[s].readSED_flambda(self.base_dir + self.filedir + s, name=s)
        return

    def setup_noise_info(self):
        self.lsst_std = {}
        for f in self.filterlist:
            self.lsst_std[f] = Bandpass()
            self.lsst_std[f].readThroughput(os.path.join(self.base_dir + self.throughputsDir, 'total_'+f+'.dat'))

        self.lsst_system = {}
        for f in self.filterlist:
            self.lsst_system[f] = Bandpass()
            self.lsst_system[f].readThroughputList(['detector.dat', 'lens1.dat', 'lens2.dat', 'lens3.dat',
                                               'm1.dat', 'm2.dat', 'm3.dat', 'filter_'+f+'.dat'],
                                                rootDir=self.base_dir + self.throughputsDir)

        atmosphere = Bandpass()
        X = 1.0
        atmosphere.readThroughput(os.path.join(self.base_dir + self.atmosDir, 'atmos_%d.dat' %(X*10)))

        self.lsst_total = {}
        for f in self.filterlist:
            wavelen, sb = self.lsst_system[f].multiplyThroughputs(atmosphere.wavelen, atmosphere.sb)
            self.lsst_total[f] = Bandpass(wavelen=wavelen, sb=sb)

        self.darksky = Sed()
        self.darksky.readSED_flambda(os.path.join(self.base_dir + self.throughputsDir, 'darksky.dat'))

        # Set up the photometric parameters for LSST
        self.photParams = PhotometricParameters(gain=1)
        # Set up the seeing. "seeing" traditional = FWHMgeom in our terms
        #  (i.e. the physical size of a double-gaussian or von Karman PSF)
        # But we use the equivalent FWHM of a single gaussian in the SNR calculation, so convert.
        seeing = 0.7
        self.FWHMeff = SignalToNoise.FWHMgeom2FWHMeff(seeing)

    def __call__(self, mag, z):
        # Now we'll read in each of those individual seds, into a Sed object. We'll also redshift the quasar.
        snr_out = {}
        for s in self.seds:
            sed = self.seds[s]
            trans = []
            for z_i in z:
                sed_trans = copy.deepcopy(sed)
                sed_trans.redshiftSED(z_i)
                trans.append(sed_trans)

            sed = trans
            fluxNorm = [sed_i.calcFluxNorm(mag[i], self.lsst_std[self.stdFilter]) for i, sed_i in enumerate(sed)]
            _ = [sed_i.multiplyFluxNorm(fluxNorm[i]) for i, sed_i in enumerate(sed)]

            # Calculate SNR for all seds in all filters.
            for f in self.filterlist:
                snr_out[s.split('.')[0] + '_' + f] = np.asarray([SignalToNoise.calcSNR_sed(sed_i, self.lsst_total[f], self.darksky, self.lsst_system[f], self.photParams, FWHMeff=self.FWHMeff, verbose=False) for sed_i in sed])
        return snr_out


class MBHEddMag:
    def __init__(self, **kwargs):
        pass

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

    def __call__(self, total_mass, q, z):
        """Calculate the detectability of EM observable.

        This assumes Eddington luminosity for smaller black hole of the pair.

        """
        self._broadcast_and_set_attrs(locals())
        self._sanity_check()
        m2 = self.total_mass*self.q/(1+self.q)
        distance = cosmo.luminosity_distance(self.z).value*1e6
        rel_mag = self._relative_mag(m2, distance)
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
