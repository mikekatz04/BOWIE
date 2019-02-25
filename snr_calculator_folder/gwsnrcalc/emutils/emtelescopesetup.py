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

    def __init__(self, **kwargs):

        self.seeing = 0.7
        self.gain = 1.0

        self.set_telescope_name()
        self.set_throughputs_dir()
        self.set_total_throughput_file_string()
        self.set_darksky_path()

        for key, item in kwargs.items():
            setattr(self, key, item)

    def set_throughputs_dir(self, throughputs_dir=None):
        if throughputs_dir is None:
            self.throughputs_dir = self.telescope_name.lower() + '_throughputs/'
        else:
            self.throughputs_dir = throughputs_dir
        return

    def set_telescope_name(self, telescope_name='lsst'):
        self.telescope_name = telescope_name
        return

    def set_darksky_path(self, darksky_path='lsst_throughputs/darksky.dat'):
        self.darksky_path = darksky_path
        return

    def set_total_throughput_file_string(self, total_throughput_file_string=None):
        # add stock defaults if files are not provided
        # assume the total throughputs for all pieces (conservative).
        if total_throughput_file_string is None:
            if self.telescope_name.lower() == 'lsst':
                self.total_throughput_file_string = 'total'

            elif self.telescope_name.lower() == 'sdss':
                self.total_throughput_file_string = 'doi'

            else:
                raise ValueError("If not providing each file for throughputs,"
                                 + "must provide lsst or sdss as telescope.")
        else:
            self.total_throughput_file_string = total_throughput_file_string
        return

    def prep_telescope(self):

        # assume dark sky sed from lsst
        self.darksky = Sed()
        self.darksky.readSED_flambda(self.base_dir + self.darksky_path)

        # Set up the photometric parameters for noise
        self.photParams = PhotometricParameters(gain=self.gain)
        # Set up the seeing. "seeing" traditional = FWHMgeom in our terms
        #  (i.e. the physical size of a double-gaussian or von Karman PSF)
        # But we use the equivalent FWHM of a single gaussian in the SNR calculation, so convert.
        self.FWHMeff = signaltonoise.FWHMgeom2FWHMeff(self.seeing)

        self.telescope = {}
        for f in self.filterlist:
            total_throughputs = Bandpass()
            total_throughputs.readThroughput(os.path.join(self.base_dir + self.throughputs_dir,
                                             self.total_throughput_file_string + '_'+f+'.dat'))

            neff, noise_sky_sq, noise_instr_sq = signaltonoise.calcTotalNonSourceNoiseSq(self.darksky, total_throughputs, self.photParams, self.FWHMeff)

            self.telescope[f] = {'total_throughputs': total_throughputs,
                                 'neff': neff,
                                 'noise_sky_sq': noise_sky_sq,
                                 'noise_instr_sq': noise_instr_sq}

        self.telescope['photParams'] = self.photParams
        self.telescope['name'] = self.telescope_name
        return


if __name__ == '__main__':
    import pdb
    test = EMTelescope('lsst')
    pdb.set_trace()
