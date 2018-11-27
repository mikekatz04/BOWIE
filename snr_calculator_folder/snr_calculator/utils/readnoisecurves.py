"""
Author: Michael Katz. This code is a simple auxillary function that returns each of the sensitivity curves that come with the installation. This was used in "Evaluating Black Hole Detectability with LISA" (arXiv:1508.07253), as a part of the BOWIE package (https://github.com/mikekatz04/BOWIE).

    Please cite this paper if you use this code in a publication.

    This code is licensed with the GNU public license.
"""

from astropy.io import ascii
from scipy import interpolate
import os
import numpy as np


def char_strain_to_asd(f, amp):
    return amp/np.sqrt(f)


def asd_to_char_strain(f, amp):
    return amp*np.sqrt(f)


def psd_to_asd(f, amp):
    return np.sqrt(amp)


def asd_to_psd(f, amp):
    return amp**2


def psd_to_char_strain(f, amp):
    return np.sqrt(amp*f)


def char_strain_to_psd(f, amp):
    return amp**2/f


def read_noise_curve(noise_curve, noise_type_in='ASD', noise_type_out='ASD',
                     use_wd_noise=False, wd_noise='HB_wd_noise', wd_noise_type_in='ASD'):
    """
    This is simple auxillary function that can read noise curves accompanying this package. All noise curves are in the form of an amplitude spectral density. Information on each one is found in each specific file.

        Inputs:

            :param noise_curve: (string) - scalar - choices are PL (Proposed LISA), CL (Classic LISA), CLLF (Classic LISA Low Frequency), PLCS (Proposed LISA Constant Slope), or PLHB (Proposed LISA Higher Break). See the arXiv paper above for the meaning behind each choice and a plot with each curve.

            :param wd_noise: (boolean) - scalar - adds the Hiscock et al 2000 approximation of the Hils & Bender 1997 white dwarf background.

            :param noise_type: (string) - scalar - type of noise curve returned. Choices are ASD, PSD, or characteristic_strain. Default is ASD.

        Outputs (attributes):

            tuple of (freqs, amplitude):

                freqs: (float) - 1D array - frequencies corresponding to the noise curve.

                amplitude: (float) - 1D array - amplitude spectral density of the noise.
        """

    possible_noise_types = ['ASD', 'PSD', 'char_strain']
    if noise_type_in not in possible_noise_types:
        raise ValueError('noise_type_in must be either ASD, PSD, or char_strain.')
    if noise_type_out not in possible_noise_types:
        raise ValueError('noise_type_out must be either ASD, PSD, or char_strain.')

    # find the noise curve file
    if noise_curve[-4:] == '.txt':
        noise = ascii.read(noise_curve)
    else:
        cfd = os.path.dirname(os.path.abspath(__file__))
        noise = ascii.read(cfd + '/noise_curves/' + noise_curve + '.txt')

    #read it in
    f_n = np.asarray(noise['f'])
    amp_n = np.asarray(noise[noise_type_in])

    if noise_type_in != noise_type_out:
        exec('amp_n = ' + noise_type_in.lower() + '_to_' + noise_type_out.lower() + '(f_n, amp_n)')

    #add wd_noise if true
    if use_wd_noise:
        if wd_noise_type_in not in possible_noise_types:
            raise ValueError('wd_noise_type_in must be either ASD, PSD, or char_strain.')

        if wd_noise[-4:] == '.txt':
            noise = ascii.read(wd_noise)
        else:
            cfd = os.path.dirname(os.path.abspath(__file__))
            noise = ascii.read(cfd + '/noise_curves/' + wd_noise + '.txt')

        wd_data = ascii.read(file_string)

        f_n_wd = np.asarray(wd_data['f'])
        amp_n_wd = np.asarray(wd_data[wd_noise_type_in])

        if wd_noise_type_in != noise_type_out:
            exec('amp_n_wd = ' + noise_type_in.lower() + '_to_' + noise_type_out.lower() + '(f_n, amp_n_wd)')


        amp_n = combine_with_wd_noise(f_n, amp_n, f_n_wd, amp_n_wd)

    return f_n, amp_n


def combine_with_wd_noise(f_n, amp_n, f_n_wd, amp_n_wd):
    amp_n_wd_interp = interpolate.interp1d(f_n_wd, amp_n_wd, bounds_error=False, fill_value=1e-30)
    amp_n_wd = amp_n_wd_interp(f_n)
    amp_n = amp_n*(amp_n >= amp_n_wd) + amp_n_wd*(amp_n < amp_n_wd)
    return f_n, amp_n

# TODO add function to show noise curve choices
