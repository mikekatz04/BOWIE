"""
Generate gridded data for contour plots with PhenomD waveform. This module provides the main process class which th emain part of the code is run out of. It is part of the BOWIE analysis tool. Author: Michael Katz. Please cite "Evaluating Black Hole Detectability with LISA" (arXiv:1807.02511) for usage of this code. 

PhenomD waveforms are generated according to Husa et al 2016 (arXiv:1508.07250) and Khan et al 2016 (arXiv:1508.07253). Please cite these papers if the PhenomD waveform is used. 

generate_contour_data produces gridded data sets based on an input dict from a script or .json configuration file. It can take any basic set of parameters for binary black holes and produce waveforms and SNR calculations for each phase of binary black hole coalescence. It reads in sensitivity curves from .txt files. The outputs can either be .txt or .hdf5. It can run in parallel or on a single processor. See generate_contour_data_guide.ipynb for examples on how to use this code. See generate_data_for_paper.ipynb for the generation of the data used in the paper. 

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

from bowie_gencondata.genconutils.readout import FileReadOut
from bowie_gencondata.genconutils.calcsig import parallel_snr_func, CalculateSignal

class GenProcess:
	def __init__(self, pid):
		"""
		Class that carries the input dictionary (pid) and directs the program to accomplish plotting tasks. 

		Inputs:
			:param pid: (dict) - carries all arguments for the program from a dictionary in a script or .json configuration file. 
		"""

		self.pid = pid

		#separate 'generate_info' for ease in code
		self.gid = pid['generate_info']

		self.extra_dict = {}


	def read_in_noise_file(self, file_dict, wd_noise=False):
		"""
		reads in noise file from .txt files with ascii.read from astropy.io. 

		Inputs:

			:param file_dict: (dict) - contains information for readining in the file. Keys/Values:

				freq_column_label: (string) - label for the frequency column of the noise curve. Default is 'f'. 
				amplitude_column_label: (string) - label for the amplitude column of the noise curve. Default is 'ASD'. 

				type: (string) - type of curve. PSD, ASD, or characteristic_strain. Default is ASD. 

			:param wd_noise: (Boolean) - boolean represented if the read in is for the wd_noise. This is necessary because labels for the actual noise curves are added to self.labels. Default is false. 

		"""

		data = ascii.read(self.pid['general']['WORKING_DIRECTORY'] + '/' + self.pid['input_info']['input_folder'] + '/' + file_dict['name'])

		#add label to self.labels if it is not wd noise
		if wd_noise == False:
			self.labels.append(file_dict['name'][0:-4])
	
		f_col_name = 'f'
		if 'freq_column_label' in self.pid['input_info'].keys():
			f_col_name = self.pid['input_info']['freq_column_label']
		if 'freq_column_label' in file_dict.keys():
			f_col_name = file_dict['freq_column_label']

		amp_col_name = 'ASD'
		if 'amplitude_column_label' in self.pid['input_info'].keys():
			amp_col_name = self.pid['input_info']['amplitude_column_label']
		if 'amplitude_column_label' in file_dict.keys():
			amp_col_name = file_dict['amplitude_column_label']

		f		 = np.asarray(data[f_col_name])
		#convert from SA PSD to NSA characteristic strain in noise
		amp		 = np.asarray(data[amp_col_name])

		if 'type' not in file_dict.keys():
			file_dict['type'] = 'ASD'

		if file_dict['type'] == 'PSD':
			amp = np.sqrt(amp)

		if file_dict['type'] == 'PSD' or file_dict['type'] == 'ASD':
			amp = amp*np.sqrt(f)


		#averaging factor for sensitivity curve. For LISA, 3/20
		averaging_factor = 1.0
		if 'sensitivity_averaging_factor' in self.pid['input_info'].keys():
			averaging_factor = self.pid['input_info']['sensitivity_averaging_factor']
		if 'sensitivity_averaging_factor' in file_dict.keys():
			averaging_factor = file_dict['sensitivity_averaging_factor']

		hn = amp*averaging_factor
		
		return f, hn

	def read_in_sensitivity_curves(self):
		"""
		Read in all the sensitivity curves
		"""


		#Sensitivity curve files
		self.sensecurves = self.pid['input_info']['sensitivity_curves']

		#declare dict for noise curve functions
		self.sensitivity_dict = OrderedDict()
		self.labels = []

		#read in Sensitvity data
		for i, file_dict in enumerate(self.sensecurves):
			f, hn = self.read_in_noise_file(file_dict, wd_noise=False)

			#place interpolated functions into dict with second including WD
			if self.pid['general']['add_wd_noise'].lower() == 'yes' or self.pid['general']['add_wd_noise'].lower() == 'both' or self.pid['general']['add_wd_noise'].lower() == 'true':

				#check if wd noise has been read in yet
				try:
					self.wd_noise
				except AttributeError:
					f_wd, hn_wd = self.read_in_noise_file(self.pid['input_info']['Galactic_background'], wd_noise=True)
					self.wd_noise = interp1d(f_wd, hn_wd, bounds_error=False, fill_value=1e-30)

				wd_up = (hn/self.wd_noise(f) <= 1.0)
				wd_down = (hn/self.wd_noise(f) > 1.0)

				self.sensitivity_dict[self.labels[i] + '_wd'] = interp1d(f, hn*wd_down+self.wd_noise(f)*wd_up, bounds_error=False, fill_value=1e30)

				if self.pid['general']['add_wd_noise'].lower() == 'both':
					self.sensitivity_dict[self.labels[i]] = interp1d(f, hn, bounds_error=False, fill_value=1e30)

			else:
				self.sensitivity_dict[self.labels[i]] = interp1d(f, hn, bounds_error=False, fill_value=1e30)

		return

	def set_parameters(self):
		"""
		Setup all the parameters for the binaries to be evaluated. 
		"""
		#dimensions of generation
		self.num_x = int(self.gid['num_x'])
		self.num_y = int(self.gid['num_y'])

		#declare 1D arrays of both paramters
		if self.gid['xscale'] != 'lin':
			self.xvals = np.logspace(np.log10(float(self.gid['x_low'])),np.log10(float(self.gid['x_high'])), self.num_x)

		else:
			self.xvals = np.linspace(float(self.gid['x_low']),float(self.gid['x_high']), self.num_x)

		if self.gid['yscale'] != 'lin':
			self.yvals = np.logspace(np.log10(float(self.gid['y_low'])),np.log10(float(self.gid['y_high'])), self.num_y)

		else:
			self.yvals = np.linspace(float(self.gid['y_low']),float(self.gid['y_high']), self.num_y)

		#other parameters
		par_1 = float(self.gid['fixed_parameter_1'])
		par_2 = float(self.gid['fixed_parameter_2'])
		par_3 = float(self.gid['fixed_parameter_3'])
		par_4 = float(self.gid['fixed_parameter_4'])

		#check if spins are set to the same value. Need to fill par_3 with spin_2 if so. 
		#then use np.meshgrid to create a full grid of parameters. 
		if self.gid['par_5_name'] == 'same_spin':
			self.xvals, self.yvals, par_1, par_2, par_3, par_4, = np.meshgrid(self.xvals, self.yvals, np.array([par_1]), np.array([par_2]), np.array([par_3]), np.array([par_4]))
			self.xvals, self.yvals, par_1, par_2, par_3, par_4 = self.xvals.ravel(),self.yvals.ravel(), par_1.ravel(), par_2.ravel(), par_3.ravel(), par_4.ravel()
			for key, vals in [['xval_name', self.xvals], ['yval_name', self.yvals], ['par_1_name', par_1], ['par_2_name', par_2], ['par_3_name', par_3], ['par_4_name', par_4]]:
				if self.gid[key][0:4] == 'spin':
					par_5 = vals
					self.gid[key] = 'spin_1'
					self.gid['par_5_name'] = 'spin_2'

		

		else:
			par_5 = float(self.gid['fixed_parameter_5'])

			self.xvals, self.yvals, par_1, par_2, par_3, par_4, par_5 = np.meshgrid(self.xvals, self.yvals, np.array([par_1]), np.array([par_2]), np.array([par_3]), np.array([par_4]), np.array([par_5]))
			self.xvals, self.yvals, par_1, par_2, par_3, par_4, par_5 = self.xvals.ravel(),self.yvals.ravel(), par_1.ravel(), par_2.ravel(), par_3.ravel(), par_4.ravel(), par_5.ravel()

		#add parameters to input dict. Names must be 'total_mass', 'mass_ratio', 'redshift' or 'luminosity_distance' or 'comoving distance', 'spin_1', 'spin_2'

		self.input_dict = {self.gid['xval_name']:self.xvals, self.gid['yval_name']:self.yvals, self.gid['par_1_name']:par_1, self.gid['par_2_name']:par_2, self.gid['par_3_name']:par_3, self.gid['par_4_name']:par_4, self.gid['par_5_name']:par_5}
 
		return

	def add_extras(self):
		"""
		Add extras to extra_dict. This includes averaging factors and number of points in the waveforms, signal types, distance key
		"""
		if 'snr_calculation_factors' in self.gid.keys():
			if 'averaging_factor' in self.gid['snr_calculation_factors'].keys():
				self.extra_dict['averaging_factor'] = float(self.gid['snr_calculation_factors']['averaging_factor'])

			if 'snr_factor' in self.gid['snr_calculation_factors'].keys():
				self.extra_dict['snr_factor'] = float(self.gid['snr_calculation_factors']['snr_factor'])

		if 'num_points' in self.gid.keys():
			self.extra_dict['num_points'] = self.gid['num_points']

		self.extra_dict['signal_types'] = self.pid['general']['signal_type']

		#find which type of distance is used
		self.dist_key = 'redshift'
		for key in ['xval_name', 'yval_name', 'par_1_name', 'par_2_name', 'par_3_name']:
			if self.gid[key] == 'luminosity_distance' or self.gid[key] == 'comoving_distance':
				self.dist_key = self.extra_dict['dist_type'] = self.gid[key]
				self.extra_dict['dist_unit'] = self.gid[key[:-4] + 'unit']

		return


	def prep_parallel(self):
		"""
		Prepare the program for generating the waveforms and finding the snr. This divides the data arrays into chunks and loads them into CalculateSignal. This is also used with single generation, running through the chunks in a list.  
		"""
		st = time.time()
		num_splits = 1000

		if 'num_splits' in self.pid['general']:
			num_splits = int(self.pid['general']['num_splits'])

		#set up inputs for each processor
		#based on num_splits which indicates max number of boxes per processor
		split_val = int(np.ceil(len(self.xvals)/num_splits))

		split_inds = [num_splits*i for i in np.arange(1,split_val)]
		array_inds = np.arange(len(self.xvals))
		find_split = np.split(array_inds,split_inds)

		#start time ticker

		self.args = []
		for i, find_part in enumerate(find_split):
			binaries_class = CalculateSignal(self.pid, self.input_dict['total_mass'][find_part], self.input_dict['mass_ratio'][find_part], self.input_dict[self.dist_key][find_part], self.input_dict['spin_1'][find_part], self.input_dict['spin_2'][find_part],  self.input_dict['start_time'], self.input_dict['end_time'], self.extra_dict)
			self.args.append((i, binaries_class,  self.pid['general']['signal_type'], self.sensitivity_dict))

		return

	def run_parallel(self):
		"""
		Runs the generation in parallel.
		"""
		self.num_processors = 4
		
		if 'num_processors' in self.pid['general']:
			self.num_processors = int(self.pid['general']['num_processors'])

		results = []
		with Pool(self.num_processors) as pool:
			print('start pool', 'num process:', len(self.args), '\n')
			results = [pool.apply_async(parallel_snr_func, arg) for arg in self.args]

			out = [r.get() for r in results]

		self.final_dict = OrderedDict()
		for sc in self.sensitivity_dict.keys():
			for sig_type in self.pid['general']['signal_type']:
				self.final_dict[sc + '_' + sig_type] = np.concatenate([r[sc + '_' + sig_type] for r in out])

		return

	def run_single(self):
		"""
		Runs the generation with single process.
		"""

		out = [parallel_snr_func(*arg) for arg in self.args]

		self.final_dict = OrderedDict()
		for sc in self.sensitivity_dict.keys():
			for sig_type in self.pid['general']['signal_type']:
				self.final_dict[sc + '_' + sig_type] = np.concatenate([r[sc + '_' + sig_type] for r in out])

		return
