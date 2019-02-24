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

    def __init__(self, fp, **kwargs):

        self.output_file_name = self.return_output = fp
        self.set_y_col_name()
        self.set_x_col_name()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.output_file_type = self.output_file_name.split('.')[-1]

        if self.output_file_type not in ['hdf5', 'txt']:
            raise ValueError('file_output_type must be hdf5 or txt.')

    def hdf5_read_out(self, output_dict):
        """Read out an hdf5 file.

        Takes the output of :class:`gwsnrcalc.genconutils.genprocess.GenProcess`
        and reads it out to an HDF5 file.

        """
        with h5py.File(self.WORKING_DIRECTORY + '/' + self.output_file_name, 'w') as f:

            header = f.create_group('header')
            header.attrs['Title'] = 'Generated SNR Out'
            header.attrs['Author'] = 'Generator by: Michael Katz'
            header.attrs['Date/Time'] = str(datetime.datetime.now())

            import pdb; pdb.set_trace()
            for which in ['x', 'y']:
                header.attrs[which + 'val_name'] = getattr(self, which + 'val_name')
                header.attrs['num_' + which + '_pts'] = getattr(self, 'num_' + which)


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

    def txt_read_out(self, output_dict):
        """Read out txt file.

        Takes the output of :class:`gwsnrcalc.genconutils.genprocess.GenProcess`
        and reads it out to a txt file.

        """
        header = '#Generated SNR Out\n'
        header += '#Generator by: Michael Katz\n'
        header += '#Date/Time: {}\n'.format(datetime.datetime.now())

        i = 1
        while 'dim{}'.format(i) in self.kwargs:
            which = 'dim{}'.format(i)
            header += '#' + which + '_name: {}\n'.format(self.kwargs[which + '_name'])
            i += 1

        """
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
        """

        if 'added_note' in self.__dict__:
            header += '#Added note: ' + self.added_note + '\n'
        else:
            header += '#Added note: None\n'

        header += '#--------------------\n'

        i = 1
        while 'dim{}'.format(i) in self.kwargs:
            which = 'dim{}'.format(i)
            header += self.kwargs[which + '_name'] + '\t'
            i += 1

        out_keys = output_dict.keys()
        for key in out_keys:
            header += key + '\t'

        # read out x,y and the data
        i = 1
        out_list = []
        while 'dim{}'.format(i) in self.kwargs:
            which = 'dim{}'.format(i)
            out_list.append(getattr(self, getattr(self, which)))
            i += 1

        pars = np.asarray(out_list)
        snr_out = np.asarray([output_dict[key] for key in out_keys]).T

        data_out = np.concatenate([pars.T, snr_out], axis=1)

        np.savetxt(self.WORKING_DIRECTORY + '/' + self.output_file_name,
                   data_out, delimiter='\t', header=header, comments='')
        return

    def set_output_file(self, output_file_name):
        """Add information for the ouput file.

        Take information on the output file name, type, and folder.

        Args:
            output_file_name (str): String representing the name of the file
                without the file extension.

        """
        self.output_file_name = output_file_name
        return

    def _set_column_name(self, which, col_name):
        """Set a column name.

        Sets the column name in the output file.

        Args:
            which (str): `x` or `y`.
            col_name (str): Column name to be added.

        """
        setattr(self, which + '_col_name', col_name)
        return

    def set_y_col_name(self, y_col_name='y'):
        """Set y column name.

        Sets the y column name in the output file.

        Args:
            y_col_name (str): y column name to be added.

        """
        self._set_column_name('y', y_col_name)
        return

    def set_x_col_name(self, x_col_name='x'):
        """Set x column name.

        Sets the x column name in the output file.

        Args:
            x_col_name (str): x column name to be added.

        """
        self._set_column_name('x', x_col_name)
        return

    def add_note(self, note):
        """Add a note to output file.

        This will add a note of user input to the output file.

        Args:
            note (str): Note to be added.

        """
        self.added_note = note
        return

    def set_return_output(self, fp, **kwargs):
        if 'dim1' not in kwargs or 'dim2' not in kwargs:
            raise ValueError("If using mesh broadcasting and you want to "
                             + "read out to a file, provide key mapping "
                             + "to dim1 and dim2 in kwargs.")

        self.output_keys = []
        self.output_key_names = {}
        i = 1
        while 'dim{}'.format(i) in kwargs:
            if kwargs['dim{}'.format(i)] in ['z', 'd', 'd_L', 'luminosity_distance', 'redshift']:
                self.output_keys.append('z_or_dist')
                kwargs['dim{}'.format(i)] = 'z_or_dist'

            else:
                self.output_keys.append(kwargs['dim{}'.format(i)])

            if 'dim{}_name'.format(i) in kwargs:
                self.output_key_names['dim{}_name'.format(i)] = kwargs['dim{}_name'.format(i)]
            else:
                self.output_key_names['dim{}_name'.format(i)] = 'dim{}'.format(i)

            i += 1

        self.kwargs = {**kwargs, **self.kwargs}
        print('Will return data in file {}'.format(fp))
        FileReadOut.__init__(self, fp, **self.kwargs)
        return
