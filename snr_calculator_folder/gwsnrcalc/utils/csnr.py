"""
Author: Michael Katz
This was used in "Evaluating Black Hole Detectability with LISA" (arXiv:1508.07253),
as a part of the BOWIE package (https://github.com/mikekatz04/BOWIE).

This code is licensed with the GNU public license.

This python code impliments a fast SNR calculator for binary black hole waveforms in c. It wraps the
accompanying c code, `phenomd/phenomd.c`, with ``ctypes``.
`phenomd/phenomd.c` is mostly from LALsuite. See `phenomd/phenomd.c` for specifics.

Please cite all of the arXiv papers above if you use this code in a publication.

"""

import numpy as np
import ctypes
import os


def csnr(freqs, hc, hn, fmrg, fpeak, prefactor=1.0):
    """Calculate the SNR of a frequency domain waveform.

    SNRCalculation is a function that takes waveforms (frequencies and hcs)
    and a noise curve, and returns SNRs for all binary phases and the whole waveform.

    Arguments:
        freqs (1D or 2D array of floats): Frequencies corresponding to the waveforms.
            Shape is (num binaries, num_points) if 2D.
            Shape is (num_points,) if 1D for one binary.
        hc (1D or 2D array of floats): Characteristic strain of the waveforms.
            Shape is (num binaries, num_points) if 2D.
            Shape is (num_points,) if 1D for one binary.
        fmrg: (scalar float or 1D array of floats): Merger frequency of each binary separating
            inspiral from merger phase. (0.014/M) Shape is (num binaries,)
            if more than one binary.
        fpeak: (scalar float or 1D array of floats): Peak frequency of each binary separating
            merger from ringdown phase. (0.014/M) Shape is (num binaries,)
            if more than one binary.
        hn: (1D or 2D array of floats): Characteristic strain of the noise.
            Shape is (num binaries, num_points) if 2D.
            Shape is (num_points,) if 1D for one binary.
        prefactor (float, optional): Factor to multiply snr (not snr^2) integral values by.
            Default is 1.0.

    Returns:
        (dict): Dictionary with SNRs from each phase.

    """
    cfd = os.path.dirname(os.path.abspath(__file__))
    if 'phenomd.cpython-35m-darwin.so' in os.listdir(cfd):
        exec_call = cfd + '/phenomd.cpython-35m-darwin.so'

    else:
        exec_call = cfd + '/phenomd/phenomd.so'

    c_obj = ctypes.CDLL(exec_call)

    # check dimensionality
    remove_axis = False
    try:
        len(fmrg)
    except TypeError:
        remove_axis = True
        freqs, hc = np.array([freqs]), np.array([hc])
        hn, fmrg, fpeak = np.array([hn]), np.array([fmrg]), np.array([fpeak])

    # this implimentation in ctypes works with 1D arrays
    freqs_in = freqs.flatten()
    hc_in = hc.flatten()
    hn_in = hn.flatten()

    num_binaries, length_of_signal = hc.shape

    # prepare outout arrays
    snr_cast = ctypes.c_double*num_binaries
    snr_all = snr_cast()
    snr_ins = snr_cast()
    snr_mrg = snr_cast()
    snr_rd = snr_cast()

    # find SNR values
    c_obj.SNR_function(ctypes.byref(snr_all), ctypes.byref(snr_ins),
                       ctypes.byref(snr_mrg), ctypes.byref(snr_rd),
                       freqs_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       hc_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       hn_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       fmrg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       fpeak.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       ctypes.c_int(length_of_signal), ctypes.c_int(num_binaries))

    # make into numpy arrays
    snr_all, snr_ins, = np.ctypeslib.as_array(snr_all), np.ctypeslib.as_array(snr_ins)
    snr_mrg, snr_rd = np.ctypeslib.as_array(snr_mrg), np.ctypeslib.as_array(snr_rd)

    # remove axis if one binary
    if remove_axis:
        snr_all, snr_ins, snr_mrg, snr_rd = snr_all[0], snr_ins[0], snr_mrg[0], snr_rd[0]

    # prepare output by multiplying by prefactor
    return ({'all': snr_all*prefactor, 'ins': snr_ins*prefactor,
            'mrg': snr_mrg*prefactor, 'rd': snr_rd*prefactor})
