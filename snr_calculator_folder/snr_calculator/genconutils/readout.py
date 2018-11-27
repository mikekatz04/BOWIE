"""
Read out contour data. It is part of the BOWIE analysis tool. Author: Michael Katz. Please cite "Evaluating Black Hole Detectability with LISA" (arXiv:1807.02511) for usage of this code.

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

class FileReadOut:


    def __init__(self, pid, file_type, output_string, xvals, yvals, output_dict, num_x, num_y):
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
        self.file_type, self.output_string, self.xvals, self.yvals, self.output_dict, self.num_x, self.num_y = file_type, output_string,  xvals, yvals,output_dict, num_x, num_y

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

        with h5py.File(self.pid['general']['WORKING_DIRECTORY'] + '/' + self.output_string, 'w') as f:

            header = f.create_group('header')
            header.attrs['Title'] = 'Generated SNR Out'
            header.attrs['Author'] = 'Generator by: Michael Katz'
            header.attrs['Date/Time'] = str(datetime.datetime.now())

            for which in ['x', 'y']:
                header.attrs[which + 'val_name'] = self.pid['generate_info'][which + 'val_name']
                header.attrs['num_' + which + '_pts'] = self.pid['generate_info']['num_' + which]
                header.attrs[which + 'val_unit'] = self.units_dict[which + 'val_unit']

            for which in np.arange(1,6).astype(str):
                header.attrs['par_' + which + '_name'] = self.pid['generate_info']['par_' + which + '_name']
                header.attrs['par_' + which + '_unit'] = self.units_dict['par_' + which + '_unit']
                header.attrs['par_' + which + '_value'] = self.pid['generate_info']['fixed_parameter_' + which]

            if self.added_note != '':
                header.attrs['Added note'] = self.added_note

            data = f.create_group('data')

            #read out x,y values in compressed data set
            x_col_name = "x"
            if 'x_col_name' in self.pid['output_info'].keys():
                x_col_name = self.pid['output_info']['x_col_name']

            dset = data.create_dataset(x_col_name, data = self.xvals, dtype = 'float64', chunks = True, compression = 'gzip', compression_opts = 9)

            y_col_name = "y"
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

        for which in ['x', 'y']:
            header += '#' + which + 'val_name: %s\n'%self.pid['generate_info'][which + 'val_name']
            header += '#num_' + which + '_pts: %i\n'%self.pid['generate_info']['num_' + which]
            header += '#' + which + 'val_unit: %s\n'%self.units_dict[which + 'val_unit']

        for which in np.arange(1,6).astype(str):
            header += '#par_' + which + '_name: %s\n'%self.pid['generate_info']['par_' + which + '_name']
            header += '#par_' + which + '_unit: %s\n'%self.units_dict['par_' + which + '_unit']
            header += '#par_' + which + '_value: %s\n'%self.pid['generate_info']['fixed_parameter_' + which]

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

        np.savetxt(self.pid['general']['WORKING_DIRECTORY'] + '/' + self.output_string, data_out, delimiter = '\t',header = header, comments='')
        return
