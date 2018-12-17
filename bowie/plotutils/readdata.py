import h5py
from astropy.io import ascii
import numpy as np


class PlotVals:
    """
    This class is designed to carry around the data for each plot as an attribute of self.

    Args:
        x_arr_list/y_arr_list/z_arr_list (list of 2D arrays of floats): List of gridded,
            2D datasets representing the x/y/z-values.

    """
    def __init__(self, x_arr_list, y_arr_list, z_arr_list):
        self.x_arr_list, self.y_arr_list, self.z_arr_list = x_arr_list, y_arr_list, z_arr_list
        return


class ReadInData:
    """Read in the data from txt or hdf5.

    This class reads in data. The information is transferred to the read in methods
    that work for .txt, .csv, and .hdf5.

    Keyword Arguments:
        WORKING_DIRECTORY (str): Path to working directory.
        file_name (str): File name for output file.
        label (str): Label for column of data in file that will be the z value (SNR).
        xlims/ylims (len-2 list of floats): x/y min value followed by max value.
            Default is `lin` for linear. If log, the limits should be log of values.
        dx/dy (float): increments in x and y.
        xscale/yscale (string, optional): scaling for axes. Either 'log' or 'lin'.
            Default is `lin`.
        x_column_label/y_column_label (str, optional): Column label within the file for x and y.

    Attributes:
        file_type (str): File extension. Either `hdf5`, `txt`, or `csv`.
        xvals/yvals/zvals (2D array of floats): Values received from files.
        x_append_value/y_append_value/z_append_value (2D array of floats): The value to append
            to data lists. This can be log10 values for x/y if ``xcale/yscale == log10``.
            For z values, it will just be zvals.


    """
    def __init__(self, **kwargs):
        # TODO: Remove 'data' group
        prop_default = {
            'x_column_label': 'x',
            'y_column_label': 'y',
            'xscale': 'lin',
            'yscale': 'lin',
        }

        for key, value in kwargs.items():
            setattr(self, key, kwargs.get(key))

        for prop, default in prop_default.items():
            setattr(self, prop, kwargs.get(prop, default))

        if 'file_name' not in self.__dict__.keys():
            raise Exception("A file name is not provided.")

        # get file type
        self.file_type = self.file_name.split('.')[-1]

        if self.file_type == 'csv':
            self.file_type = 'txt'

        # z column name
        self.z_column_label = self.label

        # Extract data with either txt or hdf5 methods.
        getattr(self, self.file_type + '_read_in')()

        # append x,y values based on xscale, yscale.
        self.x_append_value = self.xvals
        if self.xscale == 'log':
            self.x_append_value = np.log10(self.xvals)

        self.y_append_value = self.yvals
        if self.yscale == 'log':
            self.y_append_value = np.log10(self.yvals)

        self.z_append_value = self.zvals

    def txt_read_in(self):
        """Read in txt files.

        Method for reading in text or csv files. This uses ascii class from astropy.io
        for flexible input. It is slower than numpy, but has greater flexibility with less input.

        """

        # read in
        data = ascii.read(self.WORKING_DIRECTORY + '/' + self.file_name)

        # find number of distinct x and y points.
        num_x_pts = len(np.unique(data[self.x_column_label]))
        num_y_pts = len(np.unique(data[self.y_column_label]))

        # create 2D arrays of x,y,z
        self.xvals = np.reshape(np.asarray(data[self.x_column_label]), (num_y_pts, num_x_pts))
        self.yvals = np.reshape(np.asarray(data[self.y_column_label]), (num_y_pts, num_x_pts))
        self.zvals = np.reshape(np.asarray(data[self.z_column_label]), (num_y_pts, num_x_pts))

        return

    def hdf5_read_in(self):
        """Method for reading in hdf5 files.

        """

        with h5py.File(self.WORKING_DIRECTORY + '/' + self.file_name) as f:

            # read in
            data = f['data']

            # find number of distinct x and y points.
            num_x_pts = len(np.unique(data[self.x_column_label][:]))
            num_y_pts = len(np.unique(data[self.y_column_label][:]))

            # create 2D arrays of x,y,z
            self.xvals = np.reshape(data[self.x_column_label][:], (num_y_pts, num_x_pts))
            self.yvals = np.reshape(data[self.y_column_label][:], (num_y_pts, num_x_pts))
            self.zvals = np.reshape(data[self.z_column_label][:], (num_y_pts, num_x_pts))
        return
