"""
Author: Michael Katz guided by lal implimentation of PhenomD. This was used in "Evaluating Black Hole Detectability with LISA" (arXiv:1508.07253), as a part of the BOWIE package (https://github.com/mikekatz04/BOWIE).

	This code is licensed with the GNU public license. 

	This python code impliments PhenomD waveforms from Husa et al 2016 (arXiv:1508.07250) and Khan et al 2016 (arXiv:1508.07253). It wraps the accompanying c code, phenomd.c, with ctypes. phenomd.c is mostly from LALsuite. See phenomd.c for specifics. 

	Please cite all of the arXiv papers above if you use this code in a publication. 

	pyphenomd.py has two main classes:

		PhenomDWaveforms: This creates characteristic strain waveforms (amplitude only). It takes binary parameters as inputs and outputs f and hc for the waveform amplitude, and the merger frequency (fmrg) and merger-ringdown initial frequency (fpeak). 

		SNRCalculation: This takes waveforms, noise amplitude, fpeak, and fmrg and outputs the SNR for each phase and for the overall waveform.

	pyphenomd.py has one additional snr calculation function:

		snr: This takes in binary inputs and outputs the snr for the phase of choice for binary black hole coalescence.
"""

import ctypes
from astropy.cosmology import Planck15 as cosmo
import numpy as np
from scipy import interpolate
from astropy.io import ascii
import os

class PhenomDWaveforms:
	def __init__(self, m1, m2, chi1, chi2, z_or_dist, st, et, dist_type='redshift', num_points=8192):

		"""
		PhenomDWaveforms is a class that takes binary parameters as inputs, and returns characteristic strain waveforms. 

		Inputs:

		*** Warning ***: All binary parameters need to have the same shape, either scalar or 1D array. Start time (st) and end time (et) can be scalars while the rest of the binary parameters are arrays. 


			:param m1: (float) - scalar or a 1D array  - mass 1 in Solar Masses. (>0.0)
			:param m2: (float) - scalar or a 1D array - mass 2 in Solar Masses. (>0.0)

			:param chi1: (float) - scalar or a 1D array - dimensionless spin of mass 1 aligned to orbital angular momentum. [-1.0, 1.0]
			:param chi1: (float) - scalar or a 1D array - dimensionless spin of mass 2 aligned to orbital angular momentum. [-1.0, 1.0]

			:param z_or_dist: (float) - scalar or a 1D array - a measure of distance to the binary. This can take three forms: redshift (dimensionless) (default), luminosity distance (Mpc), comoving_distance (Mpc). The type used must be specified in 'dist_type' parameter. (>0.0)

			:param st: (float) - scalar or a 1D array - start time of waveform in years before end of the merger phase. This is determined using 1 PN order. (>0.0)

			:param et: (float) - scalar or a 1D array - end time of waveform in years before end of the merger phase. This is determined using 1 PN order. (>=0.0)

			:param dist_type: (str) - which type of distance is used. Default is 'redshift'. 

			:param num_points: (int) - scalar - number of points to use in the waveform. The frequency points are log-spaced. Default is 8192. 

		Outputs (attributes):

			freqs: (float) - 1D or 2D array - shape (num binaries, num_points) - frequencies corresponding to the waveforms. 

			amplitude: (float) - 1D or 2D array - shape (num binaries, num_points) - amplitudes of the waveforms. 

			fmrg: (float) - scalar or 1D array - shape (num binaries) - merger frequency of each binary separating inspiral from merger phase.

			fpeak: (float) - scalar or 1D array - shape (num binaries) - peak frequency of each binary separating merger from ringdown phase. 

		"""

		

		#check if the binary inputs are scalar or 1D
		self.remove_axis = False
		try:
			len(m1)
			try:
				len(st)
			except TypeError:
				st = np.full(len(m1), st)
				et = np.full(len(m1), et)


		except TypeError:
			self.remove_axis=True
			m1, m2,chi1, chi2, z_or_dist, st, et = np.array([m1]), np.array([m2]), np.array([chi1]), np.array([chi2]), np.array([z_or_dist]), np.array([st]), np.array([et])

		#based on distance inputs, need to find redshift and luminosity distance. 
		if dist_type == 'redshift':
			z = z_or_dist
			dist = cosmo.luminosity_distance(z).value

		elif dist_type == 'luminosity_distance':
			z_in = np.logspace(-3, 3, 10000)
			lum_dis = cosmo.luminosity_distance(z_in).value

			dist = z_or_dist
			z = np.interp(dist, lum_dis, z_in)

		elif dist_type == 'comoving_distance':
			z_in = np.logspace(-3, 3, 10000)
			lum_dis = cosmo.luminosity_distance(z_in).value
			com_dis = cosmo.comoving_distance(z_in).value

			comoving_distance = z_or_dist
			z = np.interp(comoving_distance, com_dis, z_in)
			dist = np.interp(comoving_distance, com_dis, lum_dis)

		else:
			raise ValueError("dist_type needs to be redshift, comoving_distance, or luminosity_distance")

		
		self.m1, self.m2, self.chi1, self.chi2, self.z, self.dist, self.st, self.et = m1, m2, chi1, chi2, z, dist, st, et

		self.sanity_check()

		self.length = len(m1)

		self.num_points = num_points

	def sanity_check(self):
		"""
		Check if parameters are okay.
		"""
		if any(self.m1<0.0):
			raise Exception("Mass 1 is negative.")
		if any(self.m2<0.0):
			raise Exception("Mass 2 is negative.")

		if any(self.chi1<-1.0) or any(self.chi1>1.0):
			raise Exception("Chi 1 is outside [-1.0, 1.0].")

		if any(self.chi2<-1.0) or any(self.chi2>1.0):
			raise Exception("Chi 2 is outside [-1.0, 1.0].")

		if any(self.z<=0.0):
			raise Exception("Redshift is zero or negative.")

		if any(self.dist<=0.0):
			raise Exception("Distance is zero or negative.")

		if any(self.st<0.0):
			raise Exception("Start Time is negative.")

		if any(self.et<0.0):
			raise Exception("End Time is negative.")

		if len(np.where(self.st<self.et)[0]) != 0:
			raise Exception("Start Time is less than End time.")

		if any(self.m1/self.m2 > 1.0000001e4) or any(self.m1/self.m2 < 9.999999e-5):
			raise Exception("Mass Ratio too far from unity.")

		return

	def create_waveforms(self):
		"""
		Method to create waveforms for PhenomDWaveforms class. No inputs. It takes inputs from 'self'. It adds the following attributes to 'self':

			freqs: (float) - 1D or 2D array - shape (num binaries, num_points) - frequencies corresponding to the waveforms. 

			amplitude: (float) - 1D or 2D array - shape (num binaries, num_points) - amplitudes of the waveforms. 

			fmrg: (float) - scalar or 1D array - shape (num binaries) - merger frequency of each binary separating inspiral from merger phase.

			fpeak: (float) - scalar or 1D array - shape (num binaries) - peak frequency of each binary separating merger from ringdown phase. 

		"""

		cfd  = os.path.dirname(os.path.abspath(__file__))
		if 'phenomd.cpython-35m-darwin.so' in os.listdir(cfd):
			exec_call = cfd + '/phenomd.cpython-35m-darwin.so'

		else:
			exec_call = cfd + '/phenomd/phenomd.so'

		c_obj = ctypes.CDLL(exec_call)

		#prepare ctypes arrays
		freq_amp_cast=ctypes.c_double*self.num_points*self.length
		freqs = freq_amp_cast()
		amplitude = freq_amp_cast()

		fmrg_fpeak_cast =ctypes.c_double*self.length
		fmrg = fmrg_fpeak_cast()
		fpeak = fmrg_fpeak_cast()


		


		#Find amplitude
		c_obj.Amplitude(ctypes.byref(freqs), ctypes.byref(amplitude), ctypes.byref(fmrg), ctypes.byref(fpeak), self.m1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.m2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.chi1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.chi2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.st.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.et.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(self.length), ctypes.c_int(self.num_points))

		#turn output into numpy arrays
		self.freqs, self.amplitude, self.fmrg, self.fpeak = np.ctypeslib.as_array(freqs), np.ctypeslib.as_array(amplitude), np.ctypeslib.as_array(fmrg), np.ctypeslib.as_array(fpeak)
		
		#remove an axis if inputs were scalar. 
		if self.remove_axis:
			self.freqs, self.amplitude, = np.squeeze(self.freqs), np.squeeze(self.amplitude)
			self.fmrg, self.fpeak = self.fmrg[0], self.fpeak[0]

		return

class SNRCalculation:
	def __init__(self, freqs, hc, hn, fmrg, fpeak, prefactor=1.0):

		"""
		SNRCalculation is a class that takes waveforms (frequencies and amplitudes) and a noise curve, and returns SNRs for all binary phases and the whole waveform. 

		Inputs:

			:param freqs: (float) - 1D or 2D array - shape (num binaries, num_points) - frequencies corresponding to the waveforms. 
			:param hc: (float) - 1D or 2D array - shape (num binaries, num_points) - amplitudes of the waveforms. 

			:param hn: (float) - 1D or 2D array - shape (num binaries, num_points) - noise amplitudes at frequencies in freqs. 

			:param fmrg: (float) - scalar or 1D array - shape (num binaries) - merger frequency of each binary separating inspiral from merger phase.
			:param fpeak: (float) - scalar or 1D array - shape (num binaries) - peak frequency of each binary separating merger from ringdown phase.

			:param prefactor: (float) - scalar - factor to multiply snr (not snr^2) integral values by 

		Outputs (attributes):

			snr_out_dict - keys -> values:

				'all' -> SNR from full waveform - (float) - scalar or 1D array
				'ins' -> SNR from inspiral portion of waveform - (float) - scalar or 1D array
				'mrg' -> SNR from merger portion of waveform - (float) - scalar or 1D array
				'rd' -> SNR from ringdown portion of waveform - (float) - scalar or 1D array

		"""

		cfd  = os.path.dirname(os.path.abspath(__file__))
		if 'phenomd.cpython-35m-darwin.so' in os.listdir(cfd):
			exec_call = cfd + '/phenomd.cpython-35m-darwin.so'

		else:
			exec_call = cfd + '/phenomd/phenomd.so'

		c_obj = ctypes.CDLL(exec_call)

		self.snr_out_dict = {}

		#check dimensionality
		remove_axis = False
		try:
			len(fmrg)
		except TypeError:
			remove_axis = True
			freqs, hc, hn, fmrg, fpeak = np.array([freqs]), np.array([hc]), np.array([hn]), np.array([fmrg]), np.array([fpeak])


		#this implimentation in ctypes works with 1D arrays
		freqs_in = freqs.flatten()
		hc_in = hc.flatten()
		hn_in = hn.flatten()

		num_binaries, length_of_signal = hc.shape

		#prepare outout arrays
		snr_cast =ctypes.c_double*num_binaries
		snr_all = snr_cast()
		snr_ins = snr_cast()
		snr_mrg = snr_cast()
		snr_rd = snr_cast()

		#find SNR values
		c_obj.SNR_function(ctypes.byref(snr_all), ctypes.byref(snr_ins), ctypes.byref(snr_mrg), ctypes.byref(snr_rd), 
			freqs_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), hc_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
			hn_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
			 fmrg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), fpeak.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(length_of_signal), ctypes.c_int(num_binaries))

		#make into numpy arrays
		snr_all, snr_ins, snr_mrg, snr_rd = np.ctypeslib.as_array(snr_all), np.ctypeslib.as_array(snr_ins), np.ctypeslib.as_array(snr_mrg), np.ctypeslib.as_array(snr_rd)
		
		#remove axis if one binary
		if remove_axis:
			snr_all, snr_ins, snr_mrg, snr_rd = snr_all[0], snr_ins[0], snr_mrg[0], snr_rd[0]

		#prepare output dict by multiplying by prefactor
		self.snr_out_dict['all'] = snr_all*prefactor
		self.snr_out_dict['ins'] = snr_ins*prefactor
		self.snr_out_dict['mrg'] = snr_mrg*prefactor
		self.snr_out_dict['rd'] = snr_rd*prefactor


def snr(m1, m2, chi1, chi2, z_or_dist, st, et, sensitivity_curve='PL', wd_noise=False, phase='all', prefactor=1.0, dist_type='redshift', num_points=8192, custom_noise=None, custom_wd_noise=None):
	"""
	snr is a function that takes binary parameters and a sensitivity curve as inputs, and returns snr from the chosen phase. 

	Inputs:

	*** Warning ***: All binary parameters need to have the same shape, either scalar or 1D array. Start time (st) and end time (et) can be scalars while the rest of the binary parameters are arrays. 


		:param m1: (float) - scalar or a 1D array  - mass 1 in Solar Masses. (>0.0)
		:param m2: (float) - scalar or a 1D array - mass 2 in Solar Masses. (>0.0)

		:param chi1: (float) - scalar or a 1D array - dimensionless spin of mass 1 aligned to orbital angular momentum. [-1.0, 1.0]
		:param chi1: (float) - scalar or a 1D array - dimensionless spin of mass 2 aligned to orbital angular momentum. [-1.0, 1.0]

		:param z_or_dist: (float) - scalar or a 1D array - a measure of distance to the binary. This can take three forms: redshift (dimensionless) (default), luminosity distance (Mpc), comoving_distance (Mpc). The type used must be specified in 'dist_type' parameter. (>0.0)

		:param st: (float) - scalar or a 1D array - start time of waveform in years before end of the merger phase. This is determined using 1 PN order. (>0.0)

		:param et: (float) - scalar or a 1D array - end time of waveform in years before end of the merger phase. This is determined using 1 PN order. (>=0.0)

		:param sensitivity_curve: (string) - string that starts the .txt file containing the sensitivity curve in folder 'noise_curves/'. Default is 'PL' for proposed LISA. 

		:param wd_noise: (boolean) - True/False to use White Dwarf background. Default is False. If True, the Hils-Bender estimation (Bender & Hils 1997) by Hiscock et al. 2000 is used.

		:param phase: (string) - options are 'all' for all phases; 'ins' for inspiral; 'mrg' for merger; or 'rd' for ringdown. Default is 'all'. 

		:param prefactor: (float) - scalar - factor to multiply integral values by 

		:param dist_type: (str) - which type of distance is used. Default is 'redshift'. 

		:param num_points: (int) - scalar - number of points to use in the waveform. The frequency points are log-spaced. Default is 8192. 

		:param custom_noise: (str) - file string with a custom txt file for sensitivity curve. It must be an Amplitude Spectral Density curve (ASD) with column headers 'f' for frequency and 'ASD' for the ASD values. Default is None.

		:param custom_wd_noise: (str) - file string with a custom txt file for wd noise. It must be an Amplitude Spectral Density curve (ASD) with column headers 'f' for frequency and 'ASD' for the ASD values. Default is None.
	"""

	wave = PhenomDWaveforms(m1, m2, chi1, chi2, z_or_dist, st, et, dist_type, num_points)
	wave.create_waveforms()
	cfd  = os.path.dirname(os.path.abspath(__file__))
	
	exec_call = cfd + '/phenomd.cpython-35m-darwin.so'

	if custom_noise == None:
		cfd  = os.path.dirname(os.path.abspath(__file__))
		file_string = cfd + '/noise_curves/' + sensitivity_curve + '.txt'

	else:
		file_string = custom_noise
		
	ASD_data = ascii.read(file_string)

	f_n = ASD_data['f']
	ASD = ASD_data['ASD']

	hn = ASD*np.sqrt(f_n)

	if wd_noise == True or custom_wd_noise != None:
		if custom_wd_noise == None:
			cfd  = os.path.dirname(os.path.abspath(__file__))
			file_string = cfd + '/noise_curves/' + 'WDnoise' + '.txt'

		else:
			file_string = custom_wd_noise

		ASD_wd_data = ascii.read(file_string)

		f_n_wd = ASD_wd_data['f']
		ASD_wd = ASD_wd_data['ASD']

		hn_wd = ASD_wd*np.sqrt(f_n_wd)

		hn_wd_interp = interpolate.interp1d(f_n_wd, hn_wd, bounds_error=False, fill_value=1e-30)
		hn_wd = hn_wd_interp(f_n)

		hn = hn*(hn>=hn_wd) + hn_wd*(hn<hn_wd)

	noise_interp = interpolate.interp1d(f_n, hn, bounds_error=False, fill_value=1e30)

	hn_vals = noise_interp(wave.freqs)

	snr_out = SNRCalculation(wave.freqs, wave.amplitude, hn_vals, wave.fmrg, wave.fpeak, prefactor)

	return snr_out.snr_out_dict[phase]






		
	

