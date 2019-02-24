import inspect
import numpy as np
import pdb

from .sensitivity import SensitivityContainer
from .waveforms import PhenomDWaveforms, EccentricBinaries, parallel_phenomd
from .baseclass import BaseGenClass


class GWSNRWrapper(BaseGenClass, SensitivityContainer):

    def __init__(self, **kwargs):
        # initialize defaults
        BaseGenClass.__init__(self, **kwargs)
        SensitivityContainer.__init__(self, **kwargs)

        self.set_signal_type(['all'])

        for key, item in kwargs.items():
            setattr(self, key, item)

    def run(self):
        self.prep_noise()
        return self.__run__(globals()[self.parallel_func_name])

    def set_num_waveform_points(self, num_wave_points=4096):
        """Number of log-spaced points for waveforms.

        The number of points of the waveforms with asymptotically increase
        the accuracy, but will lower the speed of generation.

        Args:
            num_wave_points (int): Number of log-spaced points for the waveforms.

        """
        self.sources.num_points = num_wave_points
        return

    def set_dist_type(self, dist_type='redshift'):
        self.sources.dist_type = dist_type
        return

    def set_n_max(self, n_max=20):
        """Maximium modes for eccentric signal.

        The number of modes to be considered when calculating the SNR for an
        eccentric signal. This will only matter when calculating eccentric signals.

        Args:
            n_max (int): Maximium modes for eccentric signal.

        """
        self.n_max = n_max
        return
