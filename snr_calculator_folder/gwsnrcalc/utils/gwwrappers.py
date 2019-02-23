from .sensitivity import SensitivityContainer
from .waveforms import PhenomDWaveforms, EccentricBinaries
from .baseclass import BaseGenClass


def GWSNR(waveform_class):
    class GWTrans(waveform_class, GWSNRWrapper):
        def __init__(self):
            super(self, waveform_class).__init__()
            super(self, GWSNRWrapper).__init__()

            parallel_func = super(self, waveform_class).instantiate_parallel_func()

    return GWTrans()


class GWSNRWrapper(BaseGenClass, SensitivityContainer):

    def __init__(self, **kwargs):
        # initialize defaults
        super(self, BaseGenClass).__init__()
        super(self, SensitivityContainer).__init__()

        self.set_signal_type()
        self.set_snr_prefactor()
        self.set_n_max()

    def prep(self):
        pass

    def add_params(param_dict, broadcast=None):

        if broadcast is not None:
            super(self, BaseGenClass).set_broadcast(broadcast=broadcast)

        if self.broadcast == 'mesh':
            self._meshgrid_and_set_attrs(param_dict)

        else:
            self._broadcast_and_set_attrs(param_dict)

        

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
