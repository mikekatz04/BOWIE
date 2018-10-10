"""
This module houses the main class for plotting within the BOWIE package. It runs through the whole plot creation process.
	
	It is part of the BOWIE analysis tool. Author: Michael Katz. Please cite "Evaluating Black Hole Detectability with LISA" (arXiv:1807.02511) for usage of this code. 

	This code is licensed under the GNU public license. 
"""

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from bowie_makeplot.plotutils.readdata import PlotVals, ReadInData
from bowie_makeplot.plotutils.plottypes import CreateSinglePlot, Waterfall, Ratio, Horizon, CodetectionPotential, CodetectionPotential2

class MakePlotProcess:
	def __init__(self, pid):
		"""
		Class that carries the input dictionary (pid) and directs the program to accomplish plotting tasks. 

		Inputs:
			:param pid: (dict) - carries all arguments for the program from a dictionary in a script or .json configuration file. 
		"""

		self.pid = pid


	def input_data(self):
		"""
		Function to extract data from files according to pid. 

		"""

		#set control_dict to plot_info part of pid.
		control_dict = self.pid['plot_info']

		ordererd = np.sort(np.asarray(list(control_dict.keys())).astype(int))

		trans_cont_dict = OrderedDict()
		for i in ordererd:
			trans_cont_dict[str(i)] = control_dict[str(i)]

		control_dict = trans_cont_dict
		
		#set empty lists for x,y,z
		x = [[]for i in np.arange(len(control_dict.keys()))]
		y = [[] for i in np.arange(len(control_dict.keys()))]
		z = [[] for i in np.arange(len(control_dict.keys()))]

		#read in base files/data
		for k, axis_string in enumerate(control_dict.keys()):
			limits_dict = {}
			if 'limits' in control_dict[axis_string].keys():
				liimits_dict = control_dict[axis_string]['limits']

			if 'file' not in control_dict[axis_string].keys():
				continue
			for j, file_dict in enumerate(control_dict[axis_string]['file']):
				data_class = ReadInData(self.pid, file_dict, limits_dict)

				x[k].append(data_class.x_append_value)
				y[k].append(data_class.y_append_value)
				z[k].append(data_class.z_append_value)

			#print(axis_string)


		#add data from plots to current plot based on index
		for k, axis_string in enumerate(control_dict.keys()):
			
			#takes first file from plot
			if 'indices' in control_dict[axis_string]:
				if type(control_dict[axis_string]['indices']) == int:
					control_dict[axis_string]['indices'] = [control_dict[axis_string]['indices']]

				for index in control_dict[axis_string]['indices']:

					index = int(index)

					x[k].append(x[index][0])
					y[k].append(y[index][0])
					z[k].append(z[index][0])
		
		#read or append control values for ratio plots
		for k, axis_string in enumerate(control_dict.keys()):
			if 'control' in control_dict[axis_string]:
				if 'name' in control_dict[axis_string]['control']:
					file_dict = control_dict[axis_string]['control']
					if 'limits' in control_dict[axis_string].keys():
						liimits_dict = control_dict[axis_string]['limits']

					data_class = ReadInData(self.pid, file_dict, limits_dict)
					x[k].append(data_class.x_append_value)
					y[k].append(data_class.y_append_value)
					z[k].append(data_class.z_append_value)

				elif 'index' in control_dict[axis_string]['control']:
					index = int(control_dict[axis_string]['control']['index'])

					x[k].append(x[index][0])
					y[k].append(y[index][0])
					z[k].append(z[index][0])

			#print(axis_string)

		#transfer lists in PlotVals class.
		value_classes = []
		for k in range(len(x)):
			value_classes.append(PlotVals(x[k],y[k],z[k]))

		self.value_classes = value_classes
		return 

	def setup_figure(self):
		"""
		Sets up the initial figure on which every plot is added. 
		"""
		#defaults for sharing axes
		sharex = True
		sharey = True

		#if share axes options are in input, change to option
		if 'sharex' in self.pid['general'].keys():
			sharex = self.pid['general']['sharex']

		if 'sharey' in self.pid['general'].keys():
			sharey = self.pid['general']['sharey']

		#declare figure and axes environments
		fig, ax = plt.subplots(nrows = int(self.pid['general']['num_rows']),
			ncols = int(self.pid['general']['num_cols']),
			sharex = sharex, sharey = sharey)

		#set figure size
		figure_width = 8
		if 'figure_width' in self.pid['general'].keys():
			figure_width = self.pid['general']['figure_width']

		figure_height = 8
		if 'figure_height' in self.pid['general'].keys():
			figure_height = self.pid['general']['figure_height']

		fig.set_size_inches(figure_width,figure_height)

		#create list of ax. Catch error if it is a single plot.
		try:
			ax = ax.ravel()
		except AttributeError:
			ax = [ax]

		#create list of plot types
		plot_types = [self.pid['plot_info'][axis_string]['type']
			for axis_string in self.pid['plot_info'].keys()]

		#set subplots_adjust settings
		adjust_bottom = 0.1
		if 'adjust_figure_bottom' in self.pid['general'].keys():
			adjust_bottom = self.pid['general']['adjust_figure_bottom']

		adjust_right = 0.9
		if 'Ratio' in plot_types or 'Waterfall' in plot_types or 'CodetectionPotential' or 'CodetectionPotential2' in plot_types:
			adjust_right = 0.79
		
		if 'adjust_figure_right' in self.pid['general'].keys():
			adjust_right = self.pid['general']['adjust_figure_right']

		adjust_left = 0.12
		if 'adjust_figure_left' in self.pid['general'].keys():
			adjust_left = self.pid['general']['adjust_figure_left']

		adjust_top = 0.85
		if 'adjust_figure_top' in self.pid['general'].keys():
			adjust_top = self.pid['general']['adjust_figure_top']

		#adjust spacing
		adjust_wspace = 0.3
		if 'spacing' in self.pid['general'].keys():
			if self.pid['general']['spacing'] == 'wide':
				pass
			else:
				adjust_wspace = 0.0
		if 'adjust_wspace' in self.pid['general'].keys():
			adjust_wspace = self.pid['general']['adjust_wspace']

		adjust_hspace = 0.3
		if 'spacing' in self.pid['general'].keys():
			if self.pid['general']['spacing'] == 'wide':
				pass
			else:
				adjust_hspace = 0.0
		if 'adjust_hspace' in self.pid['general'].keys():
			adjust_hspace = self.pid['general']['adjust_hspace']

		#adjust figure sizes
		fig.subplots_adjust(left=adjust_left, top=adjust_top, bottom=adjust_bottom, right=adjust_right, wspace=adjust_wspace, hspace=adjust_hspace)


		#labels for figure
		fig_label_fontsize = 20
		if 'fig_label_fontsize' in self.pid['general'].keys():
			fig_label_fontsize = float(self.pid['general']['fig_label_fontsize'])

		if 'fig_x_label' in self.pid['general'].keys():
			fig.text(0.01, 0.51, r'%s'%(self.pid['general']['fig_y_label']),
				rotation = 'vertical', va = 'center', fontsize = fig_label_fontsize)

		if 'fig_y_label' in self.pid['general'].keys():
			fig.text(0.45, 0.02, r'%s'%(self.pid['general']['fig_x_label']),
				ha = 'center', fontsize = fig_label_fontsize)

		fig_title_fontsize = 20
		if 'fig_title_fontsize' in self.pid['general'].keys():
			fig_title_fontsize = float(self.pid['general']['fig_title_fontsize'])

		if 'fig_title' in self.pid['general'].keys():
			fig.suptitle(0.45, 0.02, r'%s'%(self.pid['general']['fig_title']),
				ha = 'center', fontsize = fig_label_fontsize)

		self.fig, self.ax = fig, ax
		return


	def create_plots(self):
		"""
		Creates plots according to each plotting class. 
		"""
		for i, axis in enumerate(self.ax):

			self.setup_trans_dict(i)
			#plot everything. First check general dict for parameters related to plots.		#create plot class

			trans_plot_class_call = globals()[self.trans_dict['type']]
			trans_plot_class = trans_plot_class_call(self.fig, axis,
				self.value_classes[i].x_arr_list,self.value_classes[i].y_arr_list,
				self.value_classes[i].z_arr_list, self.pid['general'],
				self.trans_dict['limits'], self.trans_dict['label'],
				self.trans_dict['extra'], self.trans_dict['legend'])

			#create the plot
			trans_plot_class.make_plot()

			#setup the plot
			trans_plot_class.setup_plot()

			#print("Axis", i, "Complete")

		return

	def setup_trans_dict(self, i):
		"""
		Take necessary parameters from 'general' if they are not in plot specific dictionaries. 
		"""

		trans_dict = self.pid['plot_info'][str(i)]
		for name in ['legend', 'limits', 'label', 'extra']:
			if name not in trans_dict:
				trans_dict[name] = {}

		if 'xlims' in self.pid['general'].keys() and 'xlims' not in trans_dict['limits']:
			trans_dict['limits']['xlims'] = self.pid['general']['xlims']
		if 'dx' in self.pid['general'].keys() and 'dx' not in trans_dict['limits']:
			trans_dict['limits']['dx'] = float(self.pid['general']['dx'])
		if 'xscale' in self.pid['general'].keys() and 'xscale' not in trans_dict['limits']:
			trans_dict['limits']['xscale'] = self.pid['general']['xscale']

		if 'ylims' in self.pid['general'].keys() and 'ylims' not in trans_dict['limits']:
			trans_dict['limits']['ylims'] = self.pid['general']['ylims']
		if 'dy' in self.pid['general'].keys() and 'dy' not in trans_dict['limits']:
			trans_dict['limits']['dy'] = float(self.pid['general']['dy'])
		if 'yscale' in self.pid['general'].keys() and 'yscale' not in trans_dict['limits']:
			trans_dict['limits']['yscale'] = self.pid['general']['yscale']

		trans_dict['extra']['spacing'] = 'tight'
		if 'spacing' in self.pid['general'].keys():
			if self.pid['general']['spacing'] == 'wide':
				trans_dict['extra']['spacing'] = 'wide'

		self.trans_dict = trans_dict

		return