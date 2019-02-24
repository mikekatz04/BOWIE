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

        self.set_signal_type()
        self.set_snr_prefactor()
        self.set_n_max()

        for key, item in kwargs.items():
            setattr(self, key, item)

    def run(self):
        self.prep_noise()
        return self.__run__(globals()[self.parallel_func_name])

    def add_params(self, *args, **kwargs):
        if 'broadcast' in kwargs:
            self.set_broadcast(broadcast=kwargs['broadcast'])

        self.output_keys = []

        if self.broadcast == 'mesh':
            if self.return_output == 'file':
                for key, arr in zip(self.args_list, list(args)):
                    if isinstance(arr, np.ndarray):
                        if len(arr) > 1:
                            self.output_keys.append(key)

            self.meshgrid_and_set_attrs({key: value for key, value
                                          in zip(self.args_list, list(args))})

        else:
            if 'x' not in kwargs or 'y' not in kwargs:
                Warning("If using pure broadcasting and you want to"
                              + "read out to a file, provide key mapping to x and y in kwargs.")

            else:
                for key in ['x', 'y']:
                    self.output_keys.append(kwargs[key])

            self.broadcast_and_set_attrs({key: value for key, value
                                          in zip(self.args_list, list(args))})

        self.sources.not_broadcasted = False
        self.params_added = True
        return

    def set_signal_type(self, sig_type=['all']):
        """Set the signal type of interest.

        Sets the signal type for which the SNR is calculated.
        This means inspiral, merger, and/or ringdown.

        Args:
            sig_type (str or list of str): Signal type desired by user.
                Choices are `ins`, `mrg`, `rd`, `all` for circular waveforms created with PhenomD.
                If eccentric waveforms are used, must be `all`.

        """
        if isinstance(sig_type, str):
            sig_type = [sig_type]
        self.signal_type = sig_type
        return

    def set_snr_prefactor(self, factor=1.0):
        """Set the SNR multiplicative factor.

        This factor will be multpilied by the SNR, not SNR^2.
        This involves orientation, sky, and polarization averaging, as well
        as any factors for the configuration.

        For example, for LISA, this would be sqrt(2*16/5). The sqrt(2) is for
        a six-link configuration and the 16/5 represents the averaging factors.

        Args:
            factor (float): Factor to multiply SNR by for averaging.

        """
        self.prefactor = factor
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
