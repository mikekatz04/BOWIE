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

from .gwutils.waveforms import PhenomDWaveforms
from .gwutils.gwwrappers import GWSNRWrapper
from .utils.gensnrclass import SNRGen


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

    if snr_class.params_added is False:
        snr_class.add_params(*args, **kwargs)

    snr_out = snr_class.run()

    if snr_class.return_output == dict:
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
