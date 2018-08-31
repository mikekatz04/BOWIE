"""
Generate gridded data for contour plots with PhenomD waveform. It is part of the BOWIE analysis tool. Author: Michael Katz. Please cite "Evaluating Black Hole Detectability with LISA" (arXiv:1807.02511) for usage of this code. 

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

from bowie_gencondata.genconutils.genprocess import GenProcess
from bowie_gencondata.genconutils.readout import FileReadOut
from bowie_gencondata.genconutils.calcsig import parallel_snr_func, CalculateSignal


Msun=1.989e30

def generate_contour_data(pid):
	"""
	Main function for this program. 

	Input:
		:param pid: (dict) - contains all of the input information for the generation from a dict in a script or .json configuration file. See BOWIE documentation for all possibilities with the pid. 
	"""

	begin_time = time.time()
	global WORKING_DIRECTORY

	gid = pid['generate_info']

	if "output_folder" not in pid['output_info']:
		pid['output_info']['output_folder'] = "."
		
	if "input_folder" not in pid['input_info']:
		pid['input_info']['input_folder'] = "."

	if "output_file_type" not in pid['output_info']:
		pid['output_info']['output_file_type'] = "hdf5"

	#adjust for same spin
	par_3_name = gid['par_3_name']

	WORKING_DIRECTORY = '.'
	if 'WORKING_DIRECTORY' not in pid['general'].keys():
		pid['general']['WORKING_DIRECTORY'] = WORKING_DIRECTORY

	#instantiate
	running_process = GenProcess(pid)
	running_process.read_in_sensitivity_curves()
	running_process.set_parameters()
	running_process.add_extras()

	running_process.prep_parallel()
	if pid['general']['generation_type'] == 'parallel':
		
		running_process.run_parallel()

	else:
		running_process.run_single()

	#read out
	pid['generate_info']['par_3_name'] = par_3_name
	file_out = FileReadOut(pid, pid['output_info']['output_file_type'], WORKING_DIRECTORY + '/' + pid['output_info']['output_folder'] + '/' + pid['output_info']['output_file_name'],  running_process.xvals, running_process.yvals, running_process.final_dict, running_process.num_x, running_process.num_y)

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
				
