"""
Read out contour data. It is part of the BOWIE analysis tool. Author: Michael Katz.
Please cite "Evaluating Black Hole Detectability with LISA" (arXiv:1807.02511)
for usage of this code.

This code is licensed under the GNU public license.

"""

import h5py
import numpy as np
import datetime


class FileReadOut:
    """
    Class designed for reading out files in .txt files or hdf5 compressed files.

    FileReadOut will export all of the contour data to a file and file type
    of the user's choice. It will include all supplemental information for reference
    back to the file at a later point in time.

    Args:
        xvals/yvals (1D array): The x/y values for the contour data.
        output_dict (dict): The output of the SNR calculations. This is the dictionary
            returned by ``gwsnrcalc.gw_snr_calculator.snr``.
        **kwargs (dict): Combination of the `general`, `output_info`, and `generate_info`
            dictionaries from pid. These kwargs are stored as attributes.

    Keyword Arguments:
        output_file_name (str): Path and name of output file in relation to working directory.
        x_col_name (str, optional): Column label for x column in output file. Default is `x`.
        y_col_name (str, optional): Column label for y column in output file. Default is `y`.
        added_note (str, optional): Add note to output file. Default is ''.

    Attributes:
        output_file_type (str): Type of file. Must be `hdf5` or `txt`.
        Note: All args above are added as attributes.
        Note: All kwargs above are added as attributes.
        Note: kwargs from :class:`gwsnrcalc.genconutils.genprocess.GenProcess`
            are also included for readout information. These are stored as attributes.

    """

    def __init__(self, xvals, yvals, output_dict, **kwargs):

        self.xvals, self.yvals = xvals, yvals
        self.output_dict = output_dict

        for key, value in kwargs.items():
            setattr(self, key, value)

        prop_defaults = {
            'added_note': '',
            'x_col_name': 'x',
            'y_col_name': 'y',
        }

        for (prop, default) in prop_defaults.items():
                setattr(self, prop, kwargs.get(prop, default))

        output_file_type = self.output_file_name.split('.')[-1]

        if output_file_type not in ['hdf5', 'txt']:
            raise ValueError('file_output_type must be hdf5 or txt.')
        self.output_file_type = output_file_type

    def hdf5_read_out(self):
        """Read out an hdf5 file.

        Takes the output of :class:`gwsnrcalc.genconutils.genprocess.GenProcess`
        and reads it out to an HDF5 file.

        """
        with h5py.File(self.WORKING_DIRECTORY + '/' + self.output_file_name, 'w') as f:

            header = f.create_group('header')
            header.attrs['Title'] = 'Generated SNR Out'
            header.attrs['Author'] = 'Generator by: Michael Katz'
            header.attrs['Date/Time'] = str(datetime.datetime.now())

            for which in ['x', 'y']:
                header.attrs[which + 'val_name'] = getattr(self, which + 'val_name')
                header.attrs['num_' + which + '_pts'] = getattr(self, 'num_' + which)

            ecc = 'eccentricity' in self.__dict__
            if ecc:
                name_list = ['observation_time', 'start_frequency', 'start_separation'
                             'eccentricity']
            else:
                name_list = ['spin_1', 'spin_2', 'spin', 'end_time']

            name_list += ['total_mass', 'mass_ratio', 'start_time', 'luminosity_distance',
                          'comoving_distance', 'redshift']

            for name in name_list:
                if name != self.xval_name and name != self.yval_name:
                    try:
                        getattr(self, name)
                        header.attrs[name] = getattr(self, name)
                    except AttributeError:
                        pass

            if self.added_note != '':
                header.attrs['Added note'] = self.added_note

            data = f.create_group('data')

            # read out x,y values in compressed data set
            dset = data.create_dataset(self.x_col_name, data=self.xvals,
                                       dtype='float64', chunks=True,
                                       compression='gzip', compression_opts=9)

            dset = data.create_dataset(self.y_col_name, data=self.yvals,
                                       dtype='float64', chunks=True,
                                       compression='gzip', compression_opts=9)

            # read out all datasets
            for key in self.output_dict.keys():
                dset = data.create_dataset(key, data=self.output_dict[key],
                                           dtype='float64', chunks=True,
                                           compression='gzip', compression_opts=9)

    def txt_read_out(self):
        """Read out txt file.

        Takes the output of :class:`gwsnrcalc.genconutils.genprocess.GenProcess`
        and reads it out to a txt file.

        """

        header = '#Generated SNR Out\n'
        header += '#Generator by: Michael Katz\n'
        header += '#Date/Time: {}\n'.format(datetime.datetime.now())

        for which in ['x', 'y']:
            header += '#' + which + 'val_name: {}\n'.format(getattr(self, which + 'val_name'))
            header += '#num_' + which + '_pts: {}\n'.format(getattr(self, 'num_' + which))

        ecc = 'eccentricity' in self.__dict__
        if ecc:
            name_list = ['observation_time', 'start_frequency', 'start_separation'
                         'eccentricity']
        else:
            name_list = ['spin_1', 'spin_2', 'spin', 'end_time']

        name_list += ['total_mass', 'mass_ratio', 'start_time', 'luminosity_distance',
                      'comoving_distance', 'redshift']

        for name in name_list:
            if name != self.xval_name and name != self.yval_name:
                try:
                    getattr(self, name)
                    header += '#{}: {}\n'.format(name, getattr(self, name))
                except AttributeError:
                    pass

        if self.added_note != '':
            header += '#Added note: ' + self.added_note + '\n'
        else:
            header += '#Added note: None\n'

        header += '#--------------------\n'

        header += self.x_col_name + '\t'

        header += self.y_col_name + '\t'

        for key in self.output_dict.keys():
            header += key + '\t'

        # read out x,y and the data
        x_and_y = np.asarray([self.xvals, self.yvals])
        snr_out = np.asarray([self.output_dict[key] for key in self.output_dict.keys()]).T

        data_out = np.concatenate([x_and_y.T, snr_out], axis=1)

        np.savetxt(self.WORKING_DIRECTORY + '/' + self.output_file_name,
                   data_out, delimiter='\t', header=header, comments='')
        return
