from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

from astropy.cosmology import Planck15 as cosmo

from .bandpass import Bandpass
from .sed import Sed
from . import signaltonoise
from .photometricparameters import PhotometricParameters


class EMTelescope:

    def __init__(self, telescope, **kwargs):

        prop_defaults = {
            'base_dir': os.path.dirname(os.path.abspath(__file__)) + '/',
            'filedir': 'em_files/seds/',
            'throughputsDir': 'em_files/' + telescope.lower() + '_throughputs/',
            'atmosDir': 'em_files/' + telescope.lower() + '_throughputs/',
            'signal_type': ('u', 'g', 'r', 'i', 'z'),
            'filtercolors': {'u': 'b', 'g': 'c', 'r': 'g', 'i': 'y', 'z': 'r'},
            'seeing': 0.7,
            'total_throughput_file': None,
            'gain': 1.0,
        }

        self.telescope = telescope

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        # add stock defaults if files are not provided
        # assume the total throughputs for all pieces (conservative).
        if self.total_throughput_file is None:
            if self.telescope.lower() == 'lsst':
                self.total_throughput_file = 'total'

            elif self.telescope.lower() == 'sdss':
                self.total_throughput_file = 'doi'

            else:
                raise ValueError("If not providing each file for throughputs,"
                                 + "must provide lsst or sdss as telescope.")

        self.filterlist = self.signal_type
        self.setup_noise_info()

    def setup_noise_info(self):

        # assume dark sky sed from lsst
        self.darksky = Sed()
        self.darksky.readSED_flambda(os.path.join(self.base_dir
                                                  + 'em_files/lsst_throughputs',
                                                  'darksky.dat'))

        # Set up the photometric parameters for noise
        self.photParams = PhotometricParameters(gain=self.gain)
        # Set up the seeing. "seeing" traditional = FWHMgeom in our terms
        #  (i.e. the physical size of a double-gaussian or von Karman PSF)
        # But we use the equivalent FWHM of a single gaussian in the SNR calculation, so convert.
        self.FWHMeff = signaltonoise.FWHMgeom2FWHMeff(self.seeing)

        self.noise_total = {}
        for f in self.filterlist:
            total_throughputs = Bandpass()
            total_throughputs.readThroughput(os.path.join(self.base_dir + self.throughputsDir,
                                             self.total_throughput_file + '_'+f+'.dat'))

            neff, noise_sky_sq, noise_instr_sq = signaltonoise.calcTotalNonSourceNoiseSq(self.darksky, total_throughputs, self.photParams, self.FWHMeff)

            self.noise_total[f] = {'telescope': self.telescope,
                                   'total_throughputs': total_throughputs,
                                   'neff': neff,
                                   'noise_sky_sq': noise_sky_sq,
                                   'noise_instr_sq': noise_instr_sq}

        return


if __name__ == '__main__':
    import pdb
    test = EMTelescope('lsst')
    pdb.set_trace()
