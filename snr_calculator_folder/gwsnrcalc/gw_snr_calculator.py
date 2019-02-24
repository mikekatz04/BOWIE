"""
Calculate gravitational wave SNRs.

This was used in "Evaluating Black Hole Detectability with LISA" (arXiv:1508.07253),
as a part of the BOWIE package (https://github.com/mikekatz04/BOWIE). Please cite this
when using this code.

This code is licensed with the GNU public license.

This python code impliments PhenomD waveforms from Husa et al 2016 (arXiv:1508.07250)
and Khan et al 2016 (arXiv:1508.07253). Please cite these papers if PhenomD waveforms are used.

It can also generate eccentric inspirals according to Peters evolution.

"""

import numpy as np
import inspect

from .utils.waveforms import PhenomDWaveforms, EccentricBinaries
from .utils.csnr import csnr
from .utils.sensitivity import SensitivityContainer
from .utils.parallel import ParallelContainer
from .utils.emsnr import EMCalc, MBHEddMag, parallel_em_snr_func
from .utils.gwwrappers import GWSNRWrapper
from .utils.gensnrclass import SNRGen


class SNR(SensitivityContainer, ParallelContainer, EMCalc):
    """Main class for SNR calculations.

    This class performs gravitational wave SNR calculations with a matched
    filtering approach. It can generate SNRs for single sources or arrays of sources.
    It can run in parallel or on a single processor.

    Args:
        calc_type (str, optional): Options are `circ` for circular orbits
            and use of :class:`gwsnrcalc.utils.waveforms.PhenomDWaveforms`;
            `ecc` eccentric orbits and use of :class:`gwsnrcalc.utils.waveforms.EccentricBinaries`,
             or `em` for em snr calculations for a quasar and use of
             :class:`gwsnrcalc.utils.emsnr.EMSNR`. (For future usage.) Default is `circ`.
        **kwargs (dict): kwargs to be added to :class:`gwsnrcalc.utils.parallel.ParallelContainer`,
            waveform class (:class:`gwsnrcalc.utils.waveforms.PhenomDWaveforms`
            or :class:`gwsnrcalc.utils.waveforms.EccentricBinaries`) , and
            :class:`gwsnrcalc.utils.sensitivity.SensitivityContainer`.

    Keyworkd Arguments:
        signal_type (scalar or list of str, optional): Phase of snr.
            Options are 'all' for all phases;
            'ins' for inspiral; 'mrg' for merger; or 'rd' for ringdown. Default is 'all'.
        prefactor (float, optional): Factor to multiply snr (not snr^2) integral values by.
            Default is 1.0.

    Attributes:
        snr_function (obj): Function object representing the snr function to use.
            This is form future use with other snr calculations.
        phenomdwave (obj, optional): If using circular waveforms,
            :class:`gwsnrcalc.utils.pyphenomd.PhenomDWaveforms`
            object for storing initial kwargs and passing to snr function.
        eccwave (obj, optional): If using eccentric waveforms,
            :class:`gwsnrcalc.utils.waveforms.EccentricBinaries` obhect for storing
            initial kwargs and passing to snr function.
        Note: All kwargs are stored as attributes.
        Note: Attributes are inherited from inherited classes.

    """
    def __init__(self, **kwargs):

        # TODO add reference to this in snr function.
        prop_defaults = {
            'prefactor': 1.0,
            'signal_type': ['all'],
            'calc_type': 'circ',
        }

        for (prop, default) in prop_defaults.items():
                setattr(self, prop, kwargs.get(prop, default))

        # initialize sensitivity and parallel modules
        if self.calc_type != 'em':
            SensitivityContainer.__init__(self, **kwargs)

        else:
            EMCalc.__init__(self, **kwargs)

        ParallelContainer.__init__(self, **kwargs)

        # set the SNR function
        if self.calc_type == 'ecc':
            self.snr_function = parallel_ecc_snr_func
            self.wavegen = EccentricBinaries(**kwargs)

        elif self.calc_type == 'em':
            self.snr_function = parallel_em_snr_func
            self.wavegen = MBHEddMag(**kwargs)

        elif self.calc_type == 'circ':
            self.snr_function = parallel_snr_func
            self.wavegen = PhenomDWaveforms(**kwargs)

        else:
            raise ValueError('calc_type must be either circ, ecc, or em.')

    def __call__(self, *binary_args):
        """Input binary parameters and calculate the SNR

        Binary parameters are read in and adjusted based on shapes. They are then
        fed into ``run`` for calculation of the snr.

        Args:
            *args: Arguments for binary parameters (see `:meth:gwsnrcalc.utils.pyphenomd.__call__`)

        Returns:
            (dict): Dictionary with the SNR output from the calculation.

        """
        # if self.num_processors is None, run on single processor
        if self.calc_type == 'ecc' or self.calc_type == 'circ':
            other_args = (self.wavegen, self.signal_type,  self.noise_interpolants,
                          self.prefactor,  self.verbose)

        else:
            other_args = (self.wavegen, self.noise_interpolants,
                          self.prefactor,  self.verbose)

        if self.num_processors == 0:
            func_args = (0, binary_args) + other_args
            return self.snr_function(*func_args)

        self.prep_parallel(binary_args, other_args)
        return self.run_parallel(self.snr_function)


def parallel_ecc_snr_func(num, binary_args, eccwave, signal_type,
                          noise_interpolants, prefactor, verbose):
    """SNR calulation with eccentric waveforms

    Generate eccentric waveforms and calculate their SNR against sensitivity curves.

    Args:
        num (int): Process number. If only a single process, num=0.
        binary_args (tuple): Binary arguments for
            :meth:`gwsnrcalc.utils.waveforms.EccentricBinaries.__call__`.
        eccwave (obj): Initialized class of
            :class:`gwsnrcalc.utils.waveforms.EccentricBinaries`.
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
    wave = eccwave(*binary_args)

    out_vals = {}

    for key in noise_interpolants:
        hn_vals = noise_interpolants[key](wave.freqs)
        integrand = 1./wave.freqs * (wave.hc**2/hn_vals**2)
        snr_squared_out = np.trapz(integrand, x=wave.freqs)

        if 'modes' in signal_type:
            out_vals[key + '_' + 'modes'] = prefactor*np.sqrt(snr_squared_out)

        if 'all' in signal_type:
            out_vals[key + '_' + 'all'] = prefactor*np.sqrt(np.sum(snr_squared_out, axis=-1))

    if verbose > 0 and (num+1) % verbose == 0:
        print('Process ', (num+1), 'is finished.')

    return out_vals

def snr(*args, **kwargs):
    """Compute the SNR of binaries.

    snr is a function that takes binary parameters and sensitivity curves as inputs,
    and returns snr for chosen phases.

    Warning: All binary parameters must be either scalar, len-1 arrays,
    or arrays of the same length. All of these can be used at once. However,
    you cannot input multiple arrays of different lengths.

    Arguments:
        *args: Arguments for :meth:`gwsnrcalc.utils.pyphenomd.PhenomDWaveforms.__call__`
        **kwargs: Keyword arguments related to
            parallel generation (see :class:`gwsnrcalc.utils.parallel`),
            waveforms (see :class:`gwsnrcalc.utils.pyphenomd`),
            or sensitivity information (see :class:`gwsnrcalc.utils.sensitivity`).

    Returns:
        (dict or list of dict): Signal-to-Noise Ratio dictionary for requested phases.

    """
    prop_defaults = {
        'snr_class': SNRGen,
        'source_class': PhenomDWaveforms,
        'snr_wrapper_class': GWSNRWrapper,
        'return_output': dict,
    }

    snr_class = kwargs.get('snr_class', prop_defaults['snr_class'])
    source_class = kwargs.get('source_class', prop_defaults['source_class'])
    snr_wrapper_class = kwargs.get('snr_wrapper_class', prop_defaults['snr_wrapper_class'])

    try:
        snr_class.instantiated
        instantiated = True

    except AttributeError:
        instantiated = False

    for prop, default in prop_defaults.items():
        if prop in ['snr_class', 'source_class', 'snr_wrapper_class']:
            continue
        kwargs[prop] = kwargs.get(prop, default)

    if instantiated is False:
        snr_class = snr_class(source_class=source_class, snr_wrapper_class=snr_wrapper_class, **kwargs)
        snr_class.add_params(*args, **kwargs)

    if len(snr_class.args_list) != len(args) and len(args) != 0:
        raise ValueError('Arg list should be exact arg list from '
                         + '{} class __call__ function.'.format(waveform_class.__name__))

    if snr_class.params_added:
        snr_class.add_params(*args, **kwargs)

    snr_out = snr_class.run()

    if kwargs['return_output'] == dict:
        squeeze = False
        max_length = 0
        for arg in args:
            try:
                length = len(arg)
                if length > max_length:
                    max_length = length

            except TypeError:
                pass

        if max_length == 0:
            squeeze = True

        if squeeze:
            return {key: snr_out[key].squeeze() for key in snr_out}
    return snr_out
