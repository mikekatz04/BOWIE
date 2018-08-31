"""
This module houses the generation class that uses the pyphenomd package to generate snr data in parallel. It is part of the BOWIE analysis tool. Author: Michael Katz. Please cite "Evaluating Black Hole Detectability with LISA" (arXiv:1807.02511) for usage of this code. 

PhenomD waveforms are generated according to Husa et al 2016 (arXiv:1508.07250) and Khan et al 2016 (arXiv:1508.07253). Please cite these papers if the PhenomD waveform is used. 

This code is licensed under the GNU public license. 

"""


import sys
from collections import OrderedDict
import h5py
import json
from multiprocessing import Pool
import numpy as np
import datetime
import time
from scipy.interpolate import interp1d
import scipy.constants as ct
#import pdb

from astropy.cosmology import Planck15 as cosmo
from astropy.io import ascii

from pyphenomd.pyphenomd import PhenomDWaveforms, SNRCalculation


def parallel_snr_func(num_proc, binaries, sig_types, sensitivity_dict):
	"""
	This function returns SNR values for binaries from the suggested signal types. 

	Inputs:

		:param num_proc: (int) - scalar - for keeping track of which processes are running in parallel. 

		:param binaries: (CalculateSignal) - contains all the binaries to have the waveforms created. 

		:param sig_types: (string) - list - contains the signal types desired. Needs to be 'all', 'ins', 'mrg', 'rd'. 

		:param sensitivity_dict: (dict) - contains all of the sensitivity curves to be analyzed in the form of interpolated functions. 

	Outputs:

		snr: (dict) - contains all the SNR values requested. Its keys are the sig_types and sensitivity curves. Its values are the array of snr values corresponding to all the binaries in #parallel_snr_func. 
	"""
	#print(num_proc,'start', len(binaries.M))

	#initialize SNR dict
	snr = OrderedDict()

	#iterate through sensitivity curves and signal types to initialize dict
	for sc in sensitivity_dict.keys():
		for sig_type in sig_types:
			snr[sc + '_' + sig_type] = np.zeros(len(binaries.M))

	#find all the waveforms
	binaries.create_waveforms()

	#find snr
	snr = binaries.find_snr(snr, sig_types, sensitivity_dict)
		
	if num_proc % 50==0:
		print(num_proc, 'end')
	return snr

class CalculateSignal(PhenomDWaveforms):
	def __init__(self, pid, M, q, z_or_dist, chi1, chi2, start_time, end_time, extra_dict={}):
		"""
		Class that carries all of the binaries and calls the methods from PhenomDWaveforms, including SNRCalculation. 

		PhenomDWaveforms is a class that takes binary parameters as inputs, and returns characteristic strain waveforms. 

		Inputs:

			:param m1: (float) - 1D array  - mass 1 in Solar Masses. (>0.0)
			:param m2: (float) - 1D array - mass 2 in Solar Masses. (>0.0)

			:param chi1: (float) - 1D array - dimensionless spin of mass 1 aligned to orbital angular momentum. [-1.0, 1.0]
			:param chi1: (float) - 1D array - dimensionless spin of mass 2 aligned to orbital angular momentum. [-1.0, 1.0]

			:param z_or_dist: (float) - 1D array - a measure of distance to the binary. This can take three forms: redshift (dimensionless) (default), luminosity distance (Mpc), comoving_distance (Mpc). The type used must be specified in 'dist_type' parameter. (>0.0)

			:param st: (float) - 1D array - start time of waveform in years before end of the merger phase. This is determined using 1 PN order. (>0.0)

			:param et: (float) - 1D array - end time of waveform in years before end of the merger phase. This is determined using 1 PN order. (>=0.0)

			:param extra_dict: (dict) - carries extra kwargs for waveform generation and snr calculation. Keys and Values:
				
				dist_type: (str) - which type of distance is used. Default is 'redshift'. Needs to be 'redshift', 'luminosity_distance', or 'comoving_distance'. 

				dist_unit: (str) - the unit of distance. Needs to Mpc or Gpc. 

				num_points: (int) - scalar - number of points to use in the waveform. The frequency points are log-spaced. Default is 8192. 

				averaging_factor: (float) - scalar - factor applied to the amplitudes of the binaries averaging over sky location, orientation, and polarization. 

				snr_factor: (float) - factor to multipy the final snr by. For LISA, this is sqrt(2) due to 6-link configuration. 

		"""


		self.pid = pid

		m1 = M*q/(1.0+q)
		m2 = M/(1.0+q)

		dist_type='redshift'
		if 'dist_type' in extra_dict.keys():
			dist_type=extra_dict['dist_type']

		num_points=8192
		if 'num_points' in extra_dict.keys():
			num_points=extra_dict['num_points']

		if 'dist_unit' in extra_dict.keys():
			if 'dist_unit' == 'Gpc':
				z_or_dist = z_or_dist*1e3 #scale it up to Mpc for PhenomDWaveforms

		PhenomDWaveforms.__init__(self, m1, m2, chi1, chi2, z_or_dist, start_time, end_time, dist_type=dist_type, num_points=num_points)

		self.M, self.q = M, q
		self.extra_dict = extra_dict

		if 'averaging_factor' in extra_dict.keys():
			self.averaging_factor = extra_dict['averaging_factor']

		else:
			self.averaging_factor = 1.0

		if 'snr_factor' in extra_dict.keys():
			self.snr_factor = extra_dict['snr_factor']
		else:
			self.snr_factor = 1.0
		
	def find_snr(self, snr, sig_types, sensitivity_dict):
		"""
		Find the SNR for signal types desired. 

		Inputs:

			:param snr: (dict) - carries all snr values found so far. Takes snr dict created in #parallel_snr_func, adds to it, and then returns it as an output. 

			:param sig_types: (string) - list - contains the signal types desired. Needs to be 'all', 'ins', 'mrg', 'rd'. 

			:param sensitivity_dict: (dict) - contains all of the sensitivity curves to be analyzed in the form of interpolated functions. 

		Outputs:
			snr: (dict) - contains all the SNR values requested. Its keys are the sig_types and sensitivity curves. Its values are the array of snr values corresponding to all the binaries in CalculateSignal. 

		"""
		self.hc = self.amplitude * self.averaging_factor
		for sc in sensitivity_dict:
			snr_class = SNRCalculation(self.freqs, self.hc, sensitivity_dict[sc](self.freqs), self.fmrg, self.fpeak)
			for sig_type in sig_types:
				snr[sc + '_' + sig_type] = snr_class.snr_out_dict[sig_type]*self.snr_factor

		return snr


