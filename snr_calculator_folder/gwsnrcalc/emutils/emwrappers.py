import inspect
import numpy as np
import os
import pdb

from .emtelescopes import LSST, SDSS
from .emsources import MBHEddMag, WDMag, parallel_em_snr_func
from ..utils.baseclass import BaseGenClass


class EMSNRWrapper(BaseGenClass):

    def __init__(self, **kwargs):
        # initialize defaults
        BaseGenClass.__init__(self, **kwargs)

        self.set_signal_type(['r'])
        self.set_telescope_name(**kwargs)
        # 'signal_type': ('u', 'g', 'r', 'i', 'z'),
        # 'filtercolors': {'u': 'b', 'g': 'c', 'r': 'g', 'i': 'y', 'z': 'r'},

        for key, item in kwargs.items():
            if key == 'telescope_name':
                self.set_telescope_name(telescope=item, **kwargs)
                continue
            setattr(self, key, item)

    def run(self):
        self.telescope.prep_telescope()
        return self.__run__(globals()[self.parallel_func_name])

    def set_dist_type(self, dist_type='redshift'):
        self.sources.dist_type = dist_type
        if self.sources.dist_type not in ['redshift', 'luminosity_distance', 'comoving_distance']:
            raise ValueError("dist_type needs to be redshift, comoving_distance,"
                             + "or luminosity_distance")
        return

    def set_telescope_name(self, telescope='LSST', **kwargs):
        telescope_name = kwargs.get('telescope_name', telescope)
        self.telescope_name = telescope
        self.telescope = globals()[self.telescope_name.upper()](**kwargs)
        return
