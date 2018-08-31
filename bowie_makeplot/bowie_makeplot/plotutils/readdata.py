"""
This module houses the data read in classes for plotting within the BOWIE package.
	
	It is part of the BOWIE analysis tool. Author: Michael Katz. Please cite "Evaluating Black Hole Detectability with LISA" (arXiv:1807.02511) for usage of this code. 

	This code is licensed under the GNU public license. 
"""

class PlotVals:


	def __init__(self, x_arr_list, y_arr_list, z_arr_list):
		""" 
		This class is designed to carry around the data for each plot as an attribute of self.

		Inputs/Attributes:
			:param x_arr_list: (float) -list of 2D arrays - list of gridded, 2D datasets representing the x-values.
			:param y_arr_list: (float) - list of 2d arrays - list of gridded, 2D datasets representing the y-values.
			:param z_arr_list: (float) - list of 2d arrays - list of gridded, 2D datasets representing the z-values.
		"""

		self.x_arr_list, self.y_arr_list, self.z_arr_list = x_arr_list, y_arr_list, z_arr_list


class ReadInData:


	def __init__(self, pid, file_dict, limits_dict={}):
		"""
		This class reads in data based on the pid and file_dict. The file_dict provides information about the files to read in. This information is transferred to the read in methods that work for .txt, .csv, and .hdf5. 

			Mandatory Inputs:
				pid - dict - plot_info_dict used in main code. It contains information for types of plots created and the general settings in the pid['general'] dict. See documentation for all options. 


				file_dict - dict - contains info about the file to read in. Inputs/keys:
						Mandatory:
						name - string - file name including path extension from WORKING_DIRECTORY.
						label - string - name of column for the z values.

						Optional:
						x_column_label, y_column_label - string - x and y column names in file_dict

			Optional Inputs:
				limits_dict - dict - contains info on scaling of x and y
					xscale, yscale - string - 'log' or 'lin' representing scale of data in x and y. 'lin' is default.

						


			Optional Inputs:
				limits_dict - dict containing axis limits and axes labels information. Inputs/keys:
					xlims, ylims - length 2 list of floats - min followed by max. default is log for x and linear for y. If log, the limits should be log of values.
					dx, dy - float - x-change and y-change
					xscale, yscale - string - scaling for axes. Either 'log' or 'lin'.
		"""

		if 'name' in file_dict.keys():
			self.file_name = file_dict['name']
		elif 'file_name' in pid['general'].keys():
			self.file_name = pid['general']['file_name']
		else:
			raise Exception("A file name is not provided.")

		#get file type
		self.file_type = self.file_name.split('.')[-1]

		if self.file_type == 'csv':
			self.file_type = 'txt'

		#find x,y column names
		self.x_col_name = 'x'
		if 'x_column_label' in file_dict.keys():
			self.x_col_name = file_dict['x_column_label']
		else:
			if 'x_column_label' in pid['general'].keys():
				self.x_col_name = pid['general']['x_column_label']


		self.y_col_name = 'y'
		if 'y_column_label' in file_dict.keys():
			self.y_col_name = file_dict['y_column_label']
		else:
			if 'y_column_label' in pid['general'].keys():
				self.y_col_name = pid['general']['y_column_label']

		#z column name
		self.z_col_name = file_dict['label']

		#Extract data with either txt or hdf5 methods.
		getattr(self, self.file_type + '_read_in')()	

		#append x,y values based on xscale, yscale.
		self.x_append_value = self.xvals
		if 'xscale' in limits_dict.keys():
			if limits_dict['xscale'] =='log':
				self.x_append_value = np.log10(self.xvals)
		else:
			if 'xscale' in pid['general'].keys():
				if pid['general']['xscale'] == 'log':
					self.x_append_value = np.log10(self.xvals)

		self.y_append_value = self.yvals
		if 'yscale' in limits_dict.keys():
			if limits_dict =='log':
				self.y_append_value = np.log10(self.yvals)
		else:
			if 'yscale' in pid['general'].keys():
				if pid['general']['yscale'] == 'log':
					self.y_append_value = np.log10(self.yvals)

		self.z_append_value = self.zvals

	def txt_read_in(self):
		"""
		Method for reading in text or csv files. This uses ascii class from astropy.io for flexible input. It is slower than numpy, but has greater flexibility with less input.
		"""

		#read in
		data = ascii.read(WORKING_DIRECTORY + '/' + self.file_name)

		#find number of distinct x and y points.
		num_x_pts = len(np.unique(data[self.x_col_name]))
		num_y_pts = len(np.unique(data[self.y_col_name]))

		#create 2D arrays of x,y,z
		self.xvals = np.reshape(np.asarray(data[self.x_col_name]), (num_y_pts,num_x_pts))
		self.yvals = np.reshape(np.asarray(data[self.y_col_name]), (num_y_pts,num_x_pts))
		self.zvals = np.reshape(np.asarray(data[self.z_col_name]), (num_y_pts,num_x_pts))

		return

	def hdf5_read_in(self):
		"""
		Method for reading in hdf5 files.

		"""
		
		with h5py.File(WORKING_DIRECTORY + '/' + self.file_name) as f:

			#read in
			data = f['data']

			#find number of distinct x and y points.
			num_x_pts = len(np.unique(data[self.x_col_name][:]))
			num_y_pts = len(np.unique(data[self.y_col_name][:]))

			#create 2D arrays of x,y,z
			self.xvals = np.reshape(data[self.x_col_name][:],
				(num_y_pts,num_x_pts))
			self.yvals = np.reshape(data[self.y_col_name][:],
				(num_y_pts,num_x_pts))
			self.zvals = np.reshape(data[self.z_col_name][:],
				(num_y_pts,num_x_pts))

		return	
