"""
Generate gridded data for contour plots with PhenomD waveform. It is part of the BOWIE analysis tool. Author: Michael Katz. Paper: (arxiv: ******)

generate_contour_data produces gridded data sets based on an input dict from a script or .json configuration file. It can take any basic set of parameters for binary black holes and produce waveforms and SNR calculations for each phase of binary black hole coalescence. It reads in sensitivity curves from .txt files. The outputs can either be .txt or .hdf5. It can run in parallel or on a single processor. See generate_contour_data_guide.ipynb for examples on how to use this code. See generate_data_for_paper.ipynb for the generation of the data used in the paper. 

This code is licensed under the GNU public license. 

PhenomD waveforms are generated according to Husa et al 2016 (arXiv:1508.07250) and Khan et al 2016 (arXiv:1508.07253). 
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

from astropy.cosmology import Planck13 as cosmo
from astropy.io import ascii

from pyphenomd import PhenomDWaveforms, SNRCalculation

Msun=1.989e30


def parallel_func(num_proc, binaries, sig_types, sensitivity_dict):
	"""
	This function returns SNR values for binaries from the suggested signal types. 

	Inputs:

		:param num_proc: (int) - scalar - for keeping track of which processes are running in parallel. 

		:param binaries: (CalculateSignalClass) - contains all the binaries to have the waveforms created. 

		:param sig_types: (string) - list - contains the signal types desired. Needs to be 'all', 'ins', 'mrg', 'rd'. 

		:param sensitivity_dict: (dict) - contains all of the sensitivity curves to be analyzed in the form of interpolated functions. 

	Outputs:

		snr: (dict) - contains all the SNR values requested. Its keys are the sig_types and sensitivity curves. Its values are the array of snr values corresponding to all the binaries in CalculateSignalClass. 
	"""
	print(num_proc,'start', len(binaries.M))

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
		
	print(num_proc, 'end')
	return snr

class CalculateSignalClass(PhenomDWaveforms):
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

			:param snr: (dict) - carries all snr values found so far. Takes snr dict created in parallel_func, adds to it, and then returns it as an output. 

			:param sig_types: (string) - list - contains the signal types desired. Needs to be 'all', 'ins', 'mrg', 'rd'. 

			:param sensitivity_dict: (dict) - contains all of the sensitivity curves to be analyzed in the form of interpolated functions. 

		Outputs:
			snr: (dict) - contains all the SNR values requested. Its keys are the sig_types and sensitivity curves. Its values are the array of snr values corresponding to all the binaries in CalculateSignalClass. 

		"""
		self.hc = self.amplitude * self.averaging_factor
		for sc in sensitivity_dict:
			snr_class = SNRCalculation(self.freqs, self.hc, sensitivity_dict[sc](self.freqs), self.fmrg, self.fpeak)
			for sig_type in sig_types:
				snr[sc + '_' + sig_type] = snr_class.snr_out_dict[sig_type]*self.snr_factor

		return snr

class file_read_out:


	def __init__(self, pid, file_type, output_string, xvals, yvals, output_dict, num_x, num_y, xval_name, yval_name, par_1_name, par_2_name, par_3_name):
		"""
		Class designed for reading out files in .txt files or hdf5 compressed files. 

		Inputs:

			:param pid: (dict) - contains all of the input parameters from the input dict or .json configuration file. 

			:param file_type: (string) - type of file for read out. Needs to be .txt or .hdf5. 

			:param output_string: (string) - file output path including name (and folder if necessary)

			:param xvals, yvals: (float) - 1D array - all of x,y values making up the grid. 

			:param output_dict: (dict) - all of the snr values for the outputs. 

			:param num_x, num_y: (int) - scalar - number of x and y values. 

			:param xval_name, yval_name: (string) - name of the x, y parameters.

			:param par_1_name, par_2_name, par_3_name: (string) - name of par_1, par_2, and par_3 parameters. 

		"""

		self.pid = pid
		self.file_type, self.output_string, self.xvals, self.yvals, self.output_dict, self.num_x, self.num_y, self.xval_name, self.yval_name, self.par_1_name, self.par_2_name, self.par_3_name = file_type, output_string,  xvals, yvals,output_dict, num_x, num_y, xval_name, yval_name, par_1_name, par_2_name, par_3_name

	def prep_output(self):
		"""
		Prepare the units to be read out and an added note for the file if included. 
		"""
		self.units_dict = {}
		for key in self.pid['generate_info'].keys():
			if key[-4::] == 'unit':
				self.units_dict[key] = self.pid['generate_info'][key]

		self.added_note = ''
		if 'added_note' in self.pid['output_info'].keys():
			self.added_note = self.pid['output_info']['added_note']
		return

	def hdf5_read_out(self):
		"""
		Read out an hdf5 file. 
		"""

		with h5py.File(WORKING_DIRECTORY + '/' + self.output_string + '.' + self.file_type, 'w') as f:

			header = f.create_group('header')
			header.attrs['Title'] = 'Generated SNR Out'
			header.attrs['Author'] = 'Generator by: Michael Katz'
			header.attrs['Date/Time'] = str(datetime.datetime.now())

			header.attrs['xval_name'] = self.xval_name
			header.attrs['num_x_pts'] = self.num_x
			header.attrs['xval_unit'] = self.units_dict['xval_unit']

			header.attrs['yval_name'] = self.yval_name
			header.attrs['num_y_pts'] = self.num_y
			header.attrs['yval_unit'] = self.units_dict['yval_unit']

			header.attrs['par_1_name'] = self.par_1_name
			header.attrs['par_1_unit'] = self.units_dict['par_1_unit']
			header.attrs['par_1_value'] = self.pid['generate_info']['fixed_parameter_1']

			header.attrs['par_2_name'] = self.par_2_name
			header.attrs['par_2_unit'] = self.units_dict['par_2_unit']
			header.attrs['par_2_value'] = self.pid['generate_info']['fixed_parameter_2']

			header.attrs['par_3_name'] = self.par_3_name
			header.attrs['par_3_unit'] = self.units_dict['par_3_unit']
			header.attrs['par_3_value'] = self.pid['generate_info']['fixed_parameter_3']

			if self.added_note != '':
				header.attrs['Added note'] = self.added_note

			data = f.create_group('data')

			#read out x,y values in compressed data set
			x_col_name = self.pid['generate_info']['xval_name']
			if 'x_col_name' in self.pid['output_info'].keys():
				x_col_name = self.pid['output_info']['x_col_name']

			dset = data.create_dataset(x_col_name, data = self.xvals, dtype = 'float64', chunks = True, compression = 'gzip', compression_opts = 9)

			y_col_name = self.pid['generate_info']['yval_name']
			if 'y_col_name' in self.pid['output_info'].keys():
				y_col_name = self.pid['output_info']['y_col_name']

			dset = data.create_dataset(y_col_name, data = self.yvals, dtype = 'float64', chunks = True, compression = 'gzip', compression_opts = 9)

			#read out all datasets
			for key in self.output_dict.keys():
				dset = data.create_dataset(key, data = self.output_dict[key], dtype = 'float64', chunks = True, compression = 'gzip', compression_opts = 9)

	def txt_read_out(self):
		"""
		Read out an txt file. 
		"""

		header = '#Generated SNR Out\n'
		header += '#Generator by: Michael Katz\n'
		header += '#Date/Time: %s\n'%str(datetime.datetime.now())

		header += '#xval_name: %s\n'%self.xval_name
		header += '#num_x_pts: %i\n'%self.num_x
		header += '#xval_unit: %s\n'%self.units_dict['xval_unit']
		
		header += '#yval_name: %s\n'%self.yval_name
		header += '#num_y_pts: %i\n'%self.num_y
		header += '#yval_unit: %s\n'%self.units_dict['yval_unit']

		header += '#par_1_name: %s\n'%self.par_1_name
		header += '#par_1_unit: %s\n'%self.units_dict['par_1_unit']
		header += '#par_1_value: %s\n'%self.pid['generate_info']['fixed_parameter_1']

		header += '#par_2_name: %s\n'%self.par_2_name
		header += '#par_2_unit: %s\n'%self.units_dict['par_2_unit']
		header += '#par_2_value: %s\n'%self.pid['generate_info']['fixed_parameter_2']

		header += '#par_3_name: %s\n'%self.par_3_name
		header += '#par_3_unit: %s\n'%self.units_dict['par_3_unit']
		header += '#par_3_value: %s\n'%self.pid['generate_info']['fixed_parameter_3']

		if self.added_note != '':
			header+= '#Added note: ' + self.added_note + '\n'
		else:
			header+= '#Added note: None\n'

		header += '#--------------------\n'

		x_col_name = self.pid['generate_info']['xval_name']
		if 'x_col_name' in self.pid['output_info'].keys():
			x_col_name = self.pid['output_info']['x_col_name']

		header += x_col_name + '\t'

		y_col_name = self.pid['generate_info']['yval_name']
		if 'y_col_name' in self.pid['output_info'].keys():
			y_col_name = self.pid['output_info']['y_col_name']

		header += y_col_name + '\t'

		for key in self.output_dict.keys():
			header += key + '\t'

		#read out x,y and the data
		x_and_y = np.asarray([self.xvals, self.yvals])
		snr_out = np.asarray([self.output_dict[key] for key in self.output_dict.keys()]).T

		data_out = np.concatenate([x_and_y.T, snr_out], axis=1)

		np.savetxt(WORKING_DIRECTORY + '/' + self.output_string + '.' + self.file_type, data_out, delimiter = '\t',header = header, comments='')
		return

		
class main_process:
	def __init__(self, pid):
		"""
		Class that carries the input dictionary (pid) and directs the program to accomplish generation tasks. 

		Inputs:
			:param pid: (dict) - carries all arguments for the program from a dictionary in a script or .json configuration file. 
		"""

		self.pid = pid

		self.gid = pid['generate_info']

		self.extra_dict = {}

		#Galactic Background Noise --> 'True', 'False', or 'Both'
		if pid['general']['add_wd_noise'] == 'True' or pid['general']['add_wd_noise'] == 'Both':
			self.read_in_wd_noise()

		#Sensitivity curve files
		self.sensecurves = pid['input_info']['sensitivity_curves']

		self.read_in_sensitivity_curves()


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

		data = ascii.read(self.pid['input_info']['input_location'] + '/' + file_dict['name'])

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

		#declare dict for noise curve functions
		self.sensitivity_dict = OrderedDict()
		self.labels = []

		#read in Sensitvity data
		for i, file_dict in enumerate(self.sensecurves):
			f, hn = self.read_in_noise_file(file_dict, wd_noise=False)

			#place interpolated functions into dict with second including WD
			if self.pid['general']['add_wd_noise'] == 'True' or self.pid['general']['add_wd_noise'] == 'Both' or self.pid['general']['add_wd_noise'] == 'both':

				#check if wd noise has been read in yet
				try:
					self.wd_noise
				except AttributeError:
					f_wd, hn_wd = self.read_in_noise_file(self.pid['input_info']['Galactic_background'], wd_noise=True)
					self.wd_noise = interp1d(f_wd, hn_wd, bounds_error=False, fill_value=1e-30)

				wd_up = (hn/self.wd_noise(f) <= 1.0)
				wd_down = (hn/self.wd_noise(f) > 1.0)

				self.sensitivity_dict[self.labels[i] + '_wd'] = interp1d(f, hn*wd_down+self.wd_noise(f)*wd_up, bounds_error=False, fill_value=1e30)

				if self.pid['general']['add_wd_noise'] == 'Both' or self.pid['general']['add_wd_noise'] == 'both':
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

		#check if spins are set to the same value. Need to fill par_3 with spin_2 if so. 
		#then use np.meshgrid to create a full grid of parameters. 
		if self.gid['par_3_name'] == 'same_spin':
			self.xvals, self.yvals, par_1, par_2 = np.meshgrid(self.xvals, self.yvals, np.array([par_1]), np.array([par_2]))
			self.xvals, self.yvals, par_1, par_2 = self.xvals.ravel(),self.yvals.ravel(), par_1.ravel(), par_2.ravel()
			for key, vals in [['xval_name', self.xvals], ['yval_name', self.yvals], ['par_1_name', par_1], ['par_2_name', par_2]]:
				if self.gid[key][0:4] == 'spin':
					par_3 = vals
					self.gid[key] = 'spin_1'
					self.gid['par_3_name'] = 'spin_2'

		else:
			par_3 = float(self.gid['fixed_parameter_3'])

			self.xvals, self.yvals, par_1, par_2, par_3 = np.meshgrid(self.xvals, self.yvals, np.array([par_1]), np.array([par_2]), np.array([par_3]))
			self.xvals, self.yvals, par_1, par_2, par_3 = self.xvals.ravel(),self.yvals.ravel(), par_1.ravel(), par_2.ravel(), par_3.ravel()

		#add parameters to input dict. Names must be 'total_mass', 'mass_ratio', 'redshift' or 'luminosity_distance' or 'comoving distance', 'spin_1', 'spin_2'
		self.input_dict = {self.gid['xval_name']:self.xvals, self.gid['yval_name']:self.yvals, self.gid['par_1_name']:par_1, self.gid['par_2_name']:par_2, self.gid['par_3_name']:par_3, 'start_time': float(self.gid['start_time']), 'end_time':float(self.gid['end_time'])}
 
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
		Prepare the program for generating the waveforms and finding the snr. This divides the data arrays into chunks and loads them into CalculateSignalClass. This is also used with single generation, running through the chunks in a list.  
		"""
		st = time.time()
		num_splits = 100

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
			binaries_class = CalculateSignalClass(self.pid, self.input_dict['total_mass'][find_part], self.input_dict['mass_ratio'][find_part], self.input_dict[self.dist_key][find_part], self.input_dict['spin_1'][find_part], self.input_dict['spin_2'][find_part],  self.input_dict['start_time'], self.input_dict['end_time'], self.extra_dict)
			self.args.append((i, binaries_class,  self.pid['general']['signal_type'], self.sensitivity_dict))

		return

	def run_parallel(self):
		"""
		Runs the generation in parallel.
		"""
		self.num_processors = int(self.pid['general']['num_processors'])

		results = []
		with Pool(self.num_processors) as pool:
			print('start pool', 'num process:', len(self.args), '\n')
			results = [pool.apply_async(parallel_func, arg) for arg in self.args]

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

		out = [parallel_func(*arg) for arg in self.args]

		self.final_dict = OrderedDict()
		for sc in self.sensitivity_dict.keys():
			for sig_type in self.pid['general']['signal_type']:
				self.final_dict[sc + '_' + sig_type] = np.concatenate([r[sc + '_' + sig_type] for r in out])

		return


def generate_contour_data(pid):
	"""
	Main function for this program. 

	Input:
		:param pid: (dict) - contains all of the input information for the generation from a dict in a script or .json configuration file. See BOWIE documentation for all possibilities with the pid. 
	"""

	begin_time = time.time()
	global WORKING_DIRECTORY

	gid = pid['generate_info']

	WORKING_DIRECTORY = '.'

	if 'WORKING_DIRECTORY' in pid['general'].keys():
		WORKING_DIRECTORY = pid['general']['WORKING_DIRECTORY']

	#instantiate
	running_process = main_process(pid)
	running_process.set_parameters()
	running_process.add_extras()

	running_process.prep_parallel()
	if pid['general']['generation_type'] == 'parallel':
		
		running_process.run_parallel()

	else:
		running_process.run_single()

	#read out
	file_out = file_read_out(pid, pid['output_info']['output_file_type'], pid['output_info']['output_folder'] + '/' + pid['output_info']['output_file_name'],  running_process.xvals, running_process.yvals, running_process.final_dict, running_process.num_x, running_process.num_y, gid['xval_name'], gid['yval_name'], gid['par_1_name'], gid['par_2_name'], gid['par_3_name'])

	#adding extras to output info
	file_out.prep_output()

	print('outputing file')
	getattr(file_out, pid['output_info']['output_file_type'] + '_read_out')()

	print(time.time()-begin_time)

if __name__ == '__main__':
	"""
	__main__ function for loading in .json configuration file. 
	"""
	plot_info_dict = json.load(open(sys.argv[1], 'r'))

	generate_contour_data(plot_info_dict)
				
