"""
Author: Michael Katz. This code is a simple auxillary function that returns each of the sensitivity curves that come with the installation. This was used in "Evaluating Black Hole Detectability with LISA" (arXiv:1508.07253), as a part of the BOWIE package (https://github.com/mikekatz04/BOWIE).

	Please cite this paper if you use this code in a publication. 

	This code is licensed with the GNU public license. 
"""

from astropy.io import ascii
from scipy import interpolate
import os
import numpy as np

def read_noise_curve(noise_curve, wd_noise=False, noise_type='ASD'):
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


	# find the noise curve file
	cfd  = os.path.dirname(os.path.abspath(__file__))
	noise = ascii.read(cfd + '/noise_curves/' + noise_curve + '.txt')

	#read it in 
	f_n = np.asarray(noise['f'])
	ASD = np.asarray(noise['ASD'])

	#add wd_noise if true
	if wd_noise:
		cfd  = os.path.dirname(os.path.abspath(__file__))
		file_string = cfd + '/noise_curves/' + 'WDnoise' + '.txt'

		ASD_wd_data = ascii.read(file_string)

		f_n_wd = np.asarray(ASD_wd_data['f'])
		ASD_wd = np.asarray(ASD_wd_data['ASD'])

		ASD_wd_interp = interpolate.interp1d(f_n_wd, ASD_wd, bounds_error=False, fill_value=1e-30)
		ASD_wd = ASD_wd_interp(f_n)

		ASD = ASD*(ASD>=ASD_wd) + ASD_wd*(ASD<ASD_wd)


	#output preferred noise type
	if noise_type == 'ASD':
		out_amp = ASD

	elif noise_type == 'characteristic_strain':
		out_amp = np.sqrt(f_n)*ASD

	elif noise_type == 'PSD':
		out_amp = ASD**2

	else:
		raise Exception('Noise type provided is {}. Must be ASD, PSD, or characteristic_strain.'.format(noise_type))


	return f_n, out_amp



