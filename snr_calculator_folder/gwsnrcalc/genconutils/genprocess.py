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

from snr_calculator.gw_snr_calculator import snr

from snr_calculator.genconutils.readout import FileReadOut


class GenProcess:
    def __init__(self, pid):
        """
        Class that carries the input dictionary (pid) and directs the program to accomplish plotting tasks.

        Inputs:
            :param pid: (dict) - carries all arguments for the program from a dictionary in a script or .json configuration file.
        """

        self.pid = pid

        # separate 'generate_info' for ease in code
        self.gid = pid['generate_info']

    def set_parameters(self):
        """
        Setup all the parameters for the binaries to be evaluated.
        """
        # dimensions of generation
        self.num_x = int(self.gid['num_x'])
        self.num_y = int(self.gid['num_y'])

        # declare 1D arrays of both paramters
        if self.gid['xscale'] != 'lin':
            self.xvals = np.logspace(np.log10(float(self.gid['x_low'])),np.log10(float(self.gid['x_high'])), self.num_x)

        else:
            self.xvals = np.linspace(float(self.gid['x_low']),float(self.gid['x_high']), self.num_x)

        if self.gid['yscale'] != 'lin':
            self.yvals = np.logspace(np.log10(float(self.gid['y_low'])),np.log10(float(self.gid['y_high'])), self.num_y)

        else:
            self.yvals = np.linspace(float(self.gid['y_low']),float(self.gid['y_high']), self.num_y)

        if 'spin' in [self.gid['xval_name'], self.gid['yval_name']]:
            sub_par_5 = True
        else:
            sub_par_5 = False

        #other parameters
        par_1 = float(self.gid['fixed_parameter_1'])
        par_2 = float(self.gid['fixed_parameter_2'])
        par_3 = float(self.gid['fixed_parameter_3'])
        par_4 = float(self.gid['fixed_parameter_4'])

        if sub_par_5 is False:
            par_5 = float(self.gid['fixed_parameter_5'])
            self.xvals, self.yvals, par_1, par_2, par_3, par_4, par_5 = np.meshgrid(self.xvals, self.yvals, np.array([par_1]), np.array([par_2]), np.array([par_3]), np.array([par_4]), np.array([par_5]))
            self.xvals, self.yvals, par_1, par_2, par_3, par_4, par_5 = self.xvals.ravel(),self.yvals.ravel(), par_1.ravel(), par_2.ravel(), par_3.ravel(), par_4.ravel(), par_5.ravel()

        else:
            self.xvals, self.yvals, par_1, par_2, par_3, par_4 = np.meshgrid(self.xvals, self.yvals, np.array([par_1]), np.array([par_2]), np.array([par_3]), np.array([par_4]))
            self.xvals, self.yvals, par_1, par_2, par_3, par_4 = self.xvals.ravel(),self.yvals.ravel(), par_1.ravel(), par_2.ravel(), par_3.ravel(), par_4.ravel()

            self.gid['par_5_name'] = 'spin_2'
            self.gid['par_5_unit'] = 'None'
            if self.gid['xval_name'] == 'spin':
                self.gid['xval_name'] = 'spin_1'
                par_5 = self.xvals
            if self.gid['yval_name'] == 'spin':
                self.gid['yval_name'] = 'spin_1'
                par_5 = self.yvals


        #add parameters to input dict. Names must be 'total_mass', 'mass_ratio', 'redshift' or 'luminosity_distance' or 'comoving distance', 'spin_1', 'spin_2'

        self.input_dict = {self.gid['xval_name']:self.xvals, self.gid['yval_name']: self.yvals, self.gid['par_1_name']: par_1, self.gid['par_2_name']: par_2, self.gid['par_3_name']: par_3, self.gid['par_4_name']: par_4, self.gid['par_5_name']: par_5}

        for key in ['par_1_name', 'par_2_name', 'par_3_name', 'par_4_name', 'par_5_name', 'xval_name', 'yval_name']:
            if self.gid[key] in ['redshift', 'luminosity_distance', 'comoving_distance']:
                self.input_dict['dist_type'] = self.gid[key]
                self.input_dict['z_or_dist'] = self.input_dict.pop(self.gid[key])

        self.input_dict['chi_1'] = self.input_dict.pop('spin_1')
        self.input_dict['chi_2'] = self.input_dict.pop('spin_2')
        self.input_dict['st'] = self.input_dict.pop('start_time')
        self.input_dict['et'] = self.input_dict.pop('end_time')

        # add m1 and m2
        self.input_dict['m1'] = (self.input_dict['total_mass'] /
                                 (1. + self.input_dict['mass_ratio']))
        self.input_dict['m2'] = (self.input_dict['total_mass']
                                 * self.input_dict['mass_ratio'] /
                                 (1. + self.input_dict['mass_ratio']))

        self.input_dict['prefactor'] = self.gid['prefactor']
        return

    def run_snr(self):
        """
        Add extras to extra_dict. This includes averaging factors and number of points in the waveforms, signal types, distance key
        """
        input_dict = {**self.input_dict, **self.pid['general'], **self.pid['input_info']}
        self.final_dict = snr(**input_dict)
        return
