"""
This module houses the main classes for plotting for the BOWIE package.

	It is part of the BOWIE analysis tool. Author: Michael Katz. Please cite "Evaluating Black Hole Detectability with LISA" (arXiv:1807.02511) for usage of this code.

	This code is licensed under the GNU public license.

	CreateSinglePlot is the base class inherited by actual classes that output a plot type.

	The three main classes are plot types: Waterfall, Horizon, and Ratio.

	Waterfall:
		SNR contour plot based on plots from LISA Mission proposal.

	Ratio:
		Comparison plot of the ratio of SNRs for two different inputs. This plot also contains Loss/Gain contours, which describe when sources are gained or lost compared to one another based on a user specified SNR cut. See paper above for further explanation.

	Horizon:
		SNR contour plots comparing multipile inputs. User can specify contour value. The default is the user specified SNR cut.
"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors


class CreateSinglePlot:

    def __init__(self, fig, axis, xvals, yvals, zvals, gen_dict={}, limits_dict={},
                 label_dict={}, extra_dict={}, legend_dict={}):
        """
        This is the base class for the subclasses designed for creating the plots.

                Mandatory Inputs:
                        :param fig: - figure object - Figure environment for the plots.
                        :param axis: - axes object - Axis object representing specific plot.

                        :param xvals, yvals, zvals: (float) - list of 2d arrays - list of x,y,z-value arrays for the plot.

                Optional Inputs (options for the dictionaries will be given in the documentation):
                        gen_dict - dict containing extra kwargs for plot

                        limits_dict - dict containing axis limits and axes labels information. Inputs/keys:

                        label_dict - dict containing label information for x labels, y labels, and title. Inputs/keys:

                        extra_dict - dict containing extra plot information to aid in customization. Inputs/keys:

                        legend_dict - dict describing legend labels and properties. This is used for horizon plots.

        """
        self.gen_dict = gen_dict
        self.fig = fig
        self.axis = axis

        # make sure that xvals, yvals, and zvals of shape (num data sets, num_x, num_y)
        if len(np.shape(zvals)) == 2:
            self.zvals = [zvals]
            len_z = 1

        else:
            self.zvals = zvals
            len_z = len(zvals)

        if len(np.shape(xvals)) == 2:
            self.xvals = [xvals for i in range(len_z)]

        else:
            self.xvals = xvals

        if len(np.shape(yvals)) == 2:
            self.yvals = [yvals for i in range(len_z)]

        else:
            self.yvals = yvals

        self.limits_dict, self.label_dict, self.extra_dict, self.legend_dict = limits_dict, label_dict, extra_dict, legend_dict

    def setup_plot(self):
        """
        Takes an axis and sets up plot limits and labels according to label_dict and limits_dict from input dict or .json file.

        """

        x_tick_label_fontsize = 14
        if 'x_tick_label_fontsize' in self.label_dict.keys():
            x_tick_label_fontsize = self.label_dict['x_tick_label_fontsize']
        elif 'tick_label_fontsize' in self.label_dict.keys():
            x_tick_label_fontsize = self.label_dict['tick_label_fontsize']
        elif 'x_tick_label_fontsize' in self.gen_dict.keys():
            x_tick_label_fontsize = self.gen_dict['x_tick_label_fontsize']
        elif 'tick_label_fontsize' in self.gen_dict.keys():
            x_tick_label_fontsize = self.gen_dict['tick_label_fontsize']

        y_tick_label_fontsize = 14
        if 'y_tick_label_fontsize' in self.label_dict.keys():
            y_tick_label_fontsize = self.label_dict['y_tick_label_fontsize']
        elif 'tick_label_fontsize' in self.label_dict.keys():
            y_tick_label_fontsize = self.label_dict['tick_label_fontsize']
        elif 'y_tick_label_fontsize' in self.gen_dict.keys():
            y_tick_label_fontsize = self.gen_dict['y_tick_label_fontsize']
        elif 'tick_label_fontsize' in self.gen_dict.keys():
            y_tick_label_fontsize = self.gen_dict['tick_label_fontsize']

        # setup xticks and yticks and limits
        # if logspaced, the log values are used.
        xticks = np.arange(float(self.limits_dict['xlims'][0]),
                           float(self.limits_dict['xlims'][1])
                           + float(self.limits_dict['dx']),
                           float(self.limits_dict['dx']))

        yticks = np.arange(float(self.limits_dict['ylims'][0]),
                           float(self.limits_dict['ylims'][1])
                           + float(self.limits_dict['dy']),
                           float(self.limits_dict['dy']))

        xlim = [xticks.min(), xticks.max()]
        if 'reverse_x_axis' in self.limits_dict.keys():
            if self.limits_dict['reverse_x_axis'] == True:
                xticks = xticks[::-1]
                xlim = [xticks.max(), xticks.min()]

        elif 'reverse_x_axis' in self.gen_dict.keys():
            if self.gen_dict['reverse_x_axis'] == True:
                xticks = xticks[::-1]
                xlim = [xticks.max(), xticks.min()]

        ylim = [yticks.min(), yticks.max()]
        if 'reverse_y_axis' in self.limits_dict.keys():
            if self.limits_dict['reverse_y_axis'] == True:
                yticks = yticks[::-1]
                ylim = [yticks.max(), yticks.min()]

        elif 'reverse_y_axis' in self.gen_dict.keys():
            if self.gen_dict['reverse_y_axis'] == True:
                yticks = yticks[::-1]
                ylim = [yticks.max(), yticks.min()]

        self.axis.set_xlim(xlim)
        self.axis.set_ylim(ylim)

        # adjust ticks for spacing. If 'wide' then show all labels, if 'tight' remove end labels.
        if self.extra_dict['spacing'] == 'wide':
            x_inds = np.arange(len(xticks))
            y_inds = np.arange(len(yticks))
        else:
            # remove end labels
            x_inds = np.arange(1, len(xticks)-1)
            y_inds = np.arange(1, len(yticks)-1)

        self.axis.set_xticks(xticks[x_inds])
        self.axis.set_yticks(yticks[y_inds])

        # set tick labels based on scale
        if self.limits_dict['xscale'] == 'log':
            self.axis.set_xticklabels([r'$10^{%i}$' % int(i)
                                       for i in xticks[x_inds]], fontsize=x_tick_label_fontsize)
        else:
            self.axis.set_xticklabels([r'$%.3g$' % (i)
                                       for i in xticks[x_inds]], fontsize=x_tick_label_fontsize)

        if self.limits_dict['yscale'] == 'log':
            self.axis.set_yticklabels([r'$10^{%i}$' % int(i)
                                       for i in yticks[y_inds]], fontsize=y_tick_label_fontsize)
        else:
            self.axis.set_yticklabels([r'$%.3g$' % (i)
                                       for i in yticks[y_inds]], fontsize=y_tick_label_fontsize)

        # add grid
        add_grid = True
        if 'add_grid' in self.extra_dict.keys():
            add_grid = self.extra_dict['add_grid']
        elif 'add_grid' in self.gen_dict.keys():
            add_grid = self.gen_dict['add_grid']

        if add_grid:
            self.axis.grid(True, linestyle='-', color='0.75')

        # add title
        title_fontsize = 20
        if 'title' in self.label_dict.keys():
            if 'title_fontsize' in self.label_dict.keys():
                title_fontsize = float(self.label_dict['title_fontsize'])
            self.axis.set_title(r'%s' % self.label_dict['title'],
                                fontsize=title_fontsize)

        # add x,y labels
        label_fontsize = 20
        if 'xlabel' in self.label_dict.keys():
            if 'xlabel_fontsize' in self.label_dict.keys():
                label_fontsize = float(self.label_dict['xlabel_fontsize'])
            self.axis.set_xlabel(r'%s' % self.label_dict['xlabel'],
                                 fontsize=label_fontsize)

        label_fontsize = 20
        if 'ylabel' in self.label_dict.keys():
            if 'ylabel_fontsize' in self.label_dict.keys():
                label_fontsize = float(self.label_dict['ylabel_fontsize'])
            self.axis.set_ylabel(r'%s' % self.label_dict['ylabel'],
                                 fontsize=label_fontsize)

        return

    def interpolate_data(self):
        #TODO think about getting rid of this or fixing this
        """
        Interpolate data if two data sets have different shapes. This method is mainly used on ratio plots to allow for direct comparison of snrs. The higher resolution array is reduced to the lower resolution.
        """
        # check the number of points in the two arrays and select max and min
        points = [np.shape(x_arr)[0]*np.shape(x_arr)[1]
                  for x_arr in self.xvals]

        min_points = np.argmin(points)
        max_points = np.argmax(points)

        # create new arrays to replace array with more points to interpolated array with less points.
        new_x = np.linspace(self.xvals[min_points].min(),
                            self.xvals[min_points].max(),
                            np.shape(self.xvals[min_points])[1])

        new_y = np.linspace(self.yvals[min_points].min(),
                            self.yvals[min_points].max(),
                            np.shape(self.yvals[min_points])[1])

        # grid new arrays
        new_x, new_y = np.meshgrid(new_x, new_y)

        # use griddate from scipy.interpolate to create new z array
        new_z = griddata((self.xvals[max_points].ravel(),
                          self.yvals[max_points].ravel()),
                         self.zvals[max_points].ravel(),
                         (new_x, new_y), method='linear')

        self.xvals[max_points], self.yvals[max_points], self.zvals[max_points] = new_x, new_y, new_z

        return

    def setup_colorbars(self, colorbar_pos, plot_call_sign, plot_type, colorbar_label, levels=[], ticks_fontsize=17, label_fontsize=20, colorbar_axes=[], colorbar_ticks=[], colorbar_tick_labels=[]):
        """
        Setup colorbars for each type of plot.

        Inputs:
                :param colorbar_pos: (int) - scalar - position of colorbar. Options are 1-5. See documentation for positions and uses.
                :param plot_call_sign: (string) - plot class name
                :colorbar_label: (string) - label of colorbar

        Optional Inputs:
                :param levels: (float) - list or array - levels used in the contouring. This is used for Waterfall plots.
                :param ticks_fontsize: (float) - fontsize for the ticks applied to colorbar
                :param label_fontsize: (float) - fontsize for the label for the colorbar
                :param colorbar_axes: (float) - list of len 4 - allows for custom placement of colorbar. Values must be [0.0, 1.0]. See adding colorbar axes to figures in matplotlib documentation. Structure [horizontal placement, vertical placement, horizontal stretch, vertical stretch]
        """

        # setup tick labels depending on plot
        if plot_type == 'Waterfall':
            ticks = levels
            tick_labels = [int(i) for i in np.delete(levels, -1)]

        elif plot_type == 'Ratio':
            ticks = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
            tick_labels = [r'$10^{%i}$' % i for i in [-2.0, -1.0, 0.0, 1.0, 2.0]]

        elif plot_type == 'CodetectionPotential' or plot_type == 'SingleDetection':
            ticks = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
            tick_labels = [r'$\downarrow 10^{%i}$' % i for i in [4.0, 3.0, 2.0, 1.0]] + [
                r'    $10^{%i}$' % i for i in [0.0]] + [r'$\uparrow 10^{%i}$' % i for i in [1.0, 2.0, 3.0, 4.0]]

        elif plot_type == 'CodetectionPotential1':
            ticks = [0.0, 1.0, 2.0, 3.0, 4.0]
            tick_labels = ['$10^{%i}$' % i for i in [0.0, 1.0, 2.0, 3.0, 4.0]]

            # ticks = np.arange(-8, 10, 2)
            # tick_labels = ['%i'%(-i) for i in ticks if i<0.0] + ['%i'%i for i in ticks if i>=0.0]
        """
		elif plot_type == 'SingleDetection':
			ticks = [-4.0, -3.0,-2.0,-1.0,0.0, 1.0,2.0, 3.0, 4.0]
			tick_labels = [r'$-10^{%i}$'%i for i in [4.0, 3.0, 2.0, 1.0]] + \
			    [r'$10^{%i}$'%i for i in [0.0, 1.0,2.0, 3.0, 4.0]]

			# ticks = np.arange(-4, 5, 1)
			# tick_labels = ['%i'%(-i) for i in ticks if i<0.0] + ['%i'%i for i in ticks if i>=0.0]
		"""

        if colorbar_ticks != []:
            ticks = colorbar_ticks

        if colorbar_tick_labels != []:
            tick_labels = colorbar_tick_labels

        # dict with axes locations
        cbar_axes_dict = {'1': [0.83, 0.49, 0.03, 0.38], '2': [0.83, 0.05, 0.03, 0.38], '3': [
            0.05, 0.92, 0.38, 0.03], '4': [0.47, 0.92, 0.38, 0.03], '5': [0.83, 0.1, 0.03, 0.8]}

        # check if custom colorbar desired
        if colorbar_axes == []:
            cbar_ax_list = cbar_axes_dict[str(colorbar_pos)]
        else:
            cbar_ax_list = colorbar_axes

        cbar_ax = self.fig.add_axes(cbar_ax_list)

        # check if colorbar is horizontal or vertical
        if cbar_ax_list[2] > cbar_ax_list[3]:
            orientation = 'horizontal'
            label_pad = +10
            var = 'x'
        else:
            orientation = 'vertical'
            label_pad = 20
            var = 'y'

        self.fig.colorbar(plot_call_sign, cax=cbar_ax,
                          ticks=ticks, orientation=orientation)

        # setup colorbar ticks
        getattr(cbar_ax, 'set_' + var + 'ticklabels')(tick_labels, fontsize=ticks_fontsize)
        getattr(cbar_ax, 'set_' + var + 'label')(colorbar_label,
                                                 fontsize=label_fontsize, labelpad=label_pad)
        if cbar_ax_list[2] > cbar_ax_list[3]:
            cbar_ax.xaxis.set_ticks_position('top')

        return

    def find_colorbar_information(self, cbar_info_dict, plot_type):
        """
        This method helps configure the colorbar.
        Inputs:
                :param cbar_info_dict: (dict) - contains information set by user for colorbar. Includes, label, ticks_fontsize, label_fontsize, colorbar_axes, pos.
                :param plot_type: (string) - plot class name.

        Outputs:
                colorbar_pos: (int) - scalar - position of colorbar, will be offset by colorbar_axes if specified.
                cbar_label: (string) - label for the colorbar.
                ticks_fontsize: (float) - scalar - colorbar ticks fontsize
                label_fontsize: (float) - scalar - colorbar label fontsize
                colorbar_axes: (float) - list of len 4 - default is []. See setup_colorbars method and documentation.
        """

        cbar_label = "None"
        if 'label' in cbar_info_dict.keys():
            cbar_label = cbar_info_dict['label']

        ticks_fontsize = 15
        if 'ticks_fontsize' in cbar_info_dict.keys():
            ticks_fontsize = cbar_info_dict['ticks_fontsize']

        label_fontsize = 20
        if 'label_fontsize' in cbar_info_dict.keys():
            label_fontsize = cbar_info_dict['label_fontsize']

        colorbar_axes = []
        if 'colorbar_axes' in cbar_info_dict.keys():
            colorbar_axes = cbar_info_dict['colorbar_axes']

        colorbar_ticks = []
        if 'colorbar_ticks' in cbar_info_dict.keys():
            colorbar_ticks = cbar_info_dict['colorbar_ticks']

        colorbar_tick_labels = []
        if 'colorbar_tick_labels' in cbar_info_dict.keys():
            colorbar_tick_labels = cbar_info_dict['colorbar_tick_labels']

        # check if pos is specified. If not set to defaults.
        if 'pos' in cbar_info_dict.keys():
            colorbar_pos = cbar_info_dict['pos']

        elif len(self.fig.axes) == 1 and ((plot_type == 'Waterfall') | (plot_type == 'Ratio')):
            colorbar_pos = 5

        else:
            colorbar_defaults = {'Waterfall': 1, 'Ratio': 2, 'CodetectionPotential': 1,
                                 'SingleDetection': 2, 'CodetectionPotential1': 5}
            colorbar_pos = colorbar_defaults[plot_type]

        return colorbar_pos, cbar_label, ticks_fontsize, label_fontsize, colorbar_axes, colorbar_ticks, colorbar_tick_labels


class Waterfall(CreateSinglePlot):

    def __init__(self, fig, axis, xvals, yvals, zvals, gen_dict={}, limits_dict={}, label_dict={}, extra_dict={}, legend_dict={}):
        """
        Waterfall is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information.

        Waterfall creates an snr filled contour plot similar in style to those seen in the LISA proposal. Contours are displayed at snrs of 10, 20, 50, 100, 200, 500, 1000, and 3000 and above. If lower contours are needed, adjust 'contour_vals' in extra_dict for the specific plot.

                Contour_vals needs to start with zero and end with a higher value than the max in the data. Contour_vals needs to be a list of max length 9 including zero and max value.
        """

        CreateSinglePlot.__init__(self, fig, axis, xvals, yvals, zvals,
                                  gen_dict, limits_dict, label_dict, extra_dict, legend_dict)

    def make_plot(self):
        """
        This methd creates the waterfall plot.
        """

        # sets levels of main contour plot
        colors1 = ['None', 'darkblue', 'blue', 'deepskyblue', 'aqua',
                   'greenyellow', 'orange', 'red', 'darkred']

        levels = np.array([0., 10, 20, 50, 100, 200, 500, 1000, 3000, 1e10])

        if 'contour_vals' in self.extra_dict.keys():
            levels = np.asarray(self.extra_dict['contour_vals'])

        # produce filled contour of SNR
        sc = self.axis.contourf(self.xvals[0], self.yvals[0], self.zvals[0],
                                levels=levels, colors=colors1)

        # check for user desire to show separate contour line
        if 'snr_contour_value' in self.extra_dict.keys():
            contour_val = self.extra_dict['snr_contour_value']
            self.axis.contour(self.xvals[0], self.yvals[0], self.zvals[0], np.array([contour_val]),
                              colors='white', linewidths=1.5, linestyles='dashed')

        cbar_info_dict = {}
        if 'colorbars' in self.gen_dict.keys():
            if 'Waterfall' in self.gen_dict['colorbars'].keys():
                cbar_info_dict = self.gen_dict['colorbars']['Waterfall']

        colorbar_pos, cbar_label, ticks_fontsize, label_fontsize, colorbar_axes, colorbar_ticks, colorbar_tick_labels = super(
            Waterfall, self).find_colorbar_information(cbar_info_dict, 'Waterfall')

        # default label for Waterfall colorbar
        if cbar_label == 'None':
            cbar_label = r"$\rho_i$"

        super(Waterfall, self).setup_colorbars(colorbar_pos, sc, 'Waterfall', cbar_label, levels=levels, ticks_fontsize=ticks_fontsize,
                                               label_fontsize=label_fontsize, colorbar_axes=colorbar_axes, colorbar_ticks=colorbar_ticks, colorbar_tick_labels=colorbar_tick_labels)

        return


class Ratio(CreateSinglePlot):
    #TODO make colormaps adjustable
    def __init__(self, fig, axis, xvals, yvals, zvals, gen_dict={}, limits_dict={},
                 label_dict={}, extra_dict={}, legend_dict={}):
        """
        Ratio is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information.

        Ratio creates a filled contour plot comparing snrs from two different data sets. Typically, it is used to compare sensitivty curves and/or varying binary parameters. It takes the snr of the first dataset and divides it by the snr from the second dataset. Additionally, a Loss/Gain contour is plotted. Loss/Gain contour is based on SNR_CUT but can be overridden with 'snr_contour_value' in extra_dict. A gain indicates the first dataset reaches the snr threshold while the second does not. A loss is the opposite.
        """
        CreateSinglePlot.__init__(self, fig, axis, xvals, yvals, zvals,
                                  gen_dict, limits_dict, label_dict, extra_dict, legend_dict)

    def make_plot(self):
        """
        This methd creates the ratio plot.
        """

        # check to make sure ratio plot has 2 arrays to compare.
        if len(self.zvals) != 2:
            raise Exception("Length of vals not equal to 2. Ratio plots must have 2 inputs.")

        # check and interpolate data so that the dimensions of each data set are the same
        if np.shape(self.xvals[0]) != np.shape(self.xvals[1]):
            self.interpolate_data()

        # sets colormap for ratio comparison plot
        #cmap2 = cm.seismic
        cmap2 = cm.coolwarm


        # set values of ratio comparison contour
        normval2 = 2.0
        num_contours2 = 40  # must be even
        levels2 = np.linspace(-normval2, normval2, num_contours2)
        norm2 = colors.Normalize(-normval2, normval2)

        # find Loss/Gain contour and Ratio contour
        diff_out, loss_gain_contour = self.find_difference_contour()

        cmap2.set_bad(color='white', alpha=0.001)
        # plot ratio contours
        sc3 = self.axis.contourf(self.xvals[0], self.yvals[0], diff_out,
                                 levels=levels2, norm=norm2, extend='both', cmap=cmap2)

        # toggle line contours of orders of magnitude of ratio comparisons
        if 'ratio_contour_lines' in self.extra_dict.keys():
            if self.extra_dict['ratio_contour_lines'] == True:
                self.axis.contour(self.xvals[0], self.yvals[0], diff_out, np.array(
                    [-2.0, -1.0, 1.0, 2.0]), colors='black', linewidths=1.0)

        # plot loss gain contour
        loss_gain_status = True
        if "show_loss_gain" in self.extra_dict.keys():
            loss_gain_status = self.extra_dict['show_loss_gain']

        if loss_gain_status == True:
            # if there is no loss/gain contours, this will produce an error, so we catch the exception.
            try:
                cs = self.axis.contourf(self.xvals[0], self.yvals[0],
                                  loss_gain_contour, levels=[-2, -0.5, 0.5, 2], colors='none',hatches=['x',None, '.'])
                self.axis.contour(self.xvals[0], self.yvals[0],
                                  loss_gain_contour, 3, colors='black', linewidths=2)

                # TODO: add legend ability for Ratio plots
            except ValueError:
                pass



        cbar_info_dict = {}
        if 'colorbars' in self.gen_dict.keys():
            if 'Ratio' in self.gen_dict['colorbars'].keys():
                cbar_info_dict = self.gen_dict['colorbars']['Ratio']

        colorbar_pos, cbar_label, ticks_fontsize, label_fontsize, colorbar_axes, colorbar_ticks, colorbar_tick_labels = super(
            Ratio, self).find_colorbar_information(cbar_info_dict, 'Ratio')

        # default Ratio colorbar label
        if cbar_label == 'None':
            cbar_label = r"$\rho_i/\rho_0$"

        super(Ratio, self).setup_colorbars(colorbar_pos, sc3, 'Ratio', cbar_label, ticks_fontsize=ticks_fontsize,
                                           label_fontsize=label_fontsize, colorbar_axes=colorbar_axes, colorbar_ticks=colorbar_ticks, colorbar_tick_labels=colorbar_tick_labels)

        return

    def find_difference_contour(self):
        """
        This method finds the ratio contour and the Loss/Gain contour values. Its inputs are the two datasets for comparison where the second is the control to compare against the first.

                The input data sets need to be the same shape. CreateSinglePlot.interpolate_data corrects for two datasets of different shape.

                Outputs:
                        loss_gain_contour: (float) - 2D array - values for Loss/Gain contour. Value will be -1, 0, or 1.
                        diff_out: (float) - 2D array - ratio contours.

        """

        # set contour to test and control contour
        zout = self.zvals[0]
        control_zout = self.zvals[1]

        # set comparison value. Default is SNR_CUT
        comparison_value = self.gen_dict['SNR_CUT']
        if 'snr_contour_value' in self.extra_dict.keys():
            comparison_value = self.extra_dict['snr_contour_value']

        ratio_comp_value = comparison_value
        if 'ratio_comp_value' in self.extra_dict.keys():
            ratio_comp_value = self.extra_dict['ratio_comp_value']

        # indices of loss,gained.
        inds_gained = np.where((zout >= comparison_value) & (control_zout < comparison_value))
        inds_lost = np.where((zout < comparison_value) & (control_zout >= comparison_value))



        # set diff to ratio for purposed of determining raito differences
        diff = zout/control_zout

        # flatten arrays
        diff = diff.ravel()

        # the following determines the log10 of the ratio difference. If it is extremely small, we neglect and put it as zero (limits chosen to resemble ratios of less than 1.05 and greater than 0.952)
        diff = np.log10(diff)*(diff >= 1.05) + (-np.log10(1.0/diff)) * \
            (diff <= 0.952) + 0.0*((diff < 1.05) & (diff > 0.952))

        # reshape difference array for dimensions of contour
        diff = np.reshape(diff, np.shape(zout))

        # change inds rid
        # Also set rid for when neither curve measures gets SNR of 1.0.
        #inds_rid = np.where((zout < ratio_comp_value) | (control_zout < ratio_comp_value))
        #diff[inds_rid] = 0.0
        diff = np.ma.masked_where((zout < ratio_comp_value) | (control_zout < ratio_comp_value), diff)

        # initialize loss/gain
        loss_gain_contour = np.zeros(np.shape(zout))

        # fill out loss/gain
        loss_gain_contour[inds_lost] = -1
        loss_gain_contour[inds_gained] = 1


        return diff, loss_gain_contour


class CodetectionPotential1(CreateSinglePlot):

    def __init__(self, fig, axis, xvals, yvals, zvals, gen_dict={}, limits_dict={},
                 label_dict={}, extra_dict={}, legend_dict={}):
        """
        Ratio is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information.

        Ratio creates an filled contour plot comparing snrs from two different data sets. Typically, it is used to compare sensitivty curves and/or varying binary parameters. It takes the snr of the first dataset and divides it by the snr from the second dataset. The log10 of this ratio is ploted. Additionally, a loss/gain contour is plotted. Loss/gain contour is based on SNR_CUT but can be overridden with 'snr_contour_value' in extra_dict. A gain indicates the first dataset reaches the snr threshold while the second does not. A loss is the opposite.
        """
        CreateSinglePlot.__init__(self, fig, axis, xvals, yvals, zvals, gen_dict,
                                  limits_dict, label_dict, extra_dict, legend_dict)

    def make_plot(self):
        """
        This methd creates the ratio plot.
        """
        colormaps = ['Purples', 'Blues', 'Greens', 'Reds']
        cmap_keys = [map_name.lower()[:-1] for map_name in colormaps]

        codet_cmap = 'Greys'
        codet_cmap_key = codet_cmap.lower()[:-1]

        # check to make sure ratio plot has 2 arrays to compare.
        if len(self.xvals) < 2:
            raise Exception("Length of vals less than 2. No need for codetection.")

        if len(self.xvals) > 4:
            raise NotImplementedError

        # check and interpolate data so that the dimensions of each data set are the same
        # check this
        for i in np.arange(len(self.xvals)-1):
            if np.shape(self.xvals[0]) != np.shape(self.xvals[i+1]):
                raise Exception("Need the same shape for this plot.")

        # if np.shape(self.xvals[0]) != np.shape(self.xvals[1]):
        #	self.interpolate_data()

        # set comparison value. Default is SNR_CUT
        comparison_value = self.gen_dict['SNR_CUT']
        if 'snr_contour_value' in self.extra_dict.keys():
            comparison_value = self.extra_dict['snr_contour_value']

        # find loss/gain contour and ratio contour
        codet_pot, singles = self.find_codetection_potential(comparison_value)

        codet_pot = np.ma.masked_where((codet_pot < np.log10(comparison_value)), codet_pot)
        singles = np.ma.masked_where((singles < np.log10(comparison_value)), singles)

        normval = 4.0
        num_contours = 17  # must be even
        levels = np.linspace(0.0, normval, num_contours)
        norm = colors.Normalize(0.0, normval)

        for i, single in enumerate(singles):
            cmap_single = getattr(cm, colormaps[i])
            self.axis.contourf(self.xvals[0], self.yvals[0], single,
                               levels=levels, norm=norm, extend='both', cmap=cmap_single, alpha=1.0)

        cmap_comb = getattr(cm, codet_cmap)
        codet_sc = self.axis.contourf(self.xvals[0], self.yvals[0], codet_pot,
                                      levels=levels, norm=norm, extend='both', cmap=cmap_comb, alpha=1.0)

        # toggle line contours of orders of magnitude of ratio comparisons
        if 'ratio_contour_lines' in self.extra_dict.keys():
            if self.extra_dict['ratio_contour_lines'] == True:
                self.axis.contour(self.xvals[0], self.yvals[0], codet_pot,
                                  np.arange(-8, 9, 1), colors='black', linewidths=1.0)
                self.axis.contour(self.xvals[0], self.yvals[0], single, np.array(
                    [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]), colors='black', linewidths=1.0)

        for i in range(len(self.zvals)):
            self.axis.scatter([-1e10, -1e10], [-1e1, -2e1], color=cmap_keys[i],
                              label=self.legend_dict['labels'][i])

        self.axis.scatter([-1e10, -1e10], [-1e1, -2e1], color=codet_cmap_key, label='Codetection')

        # Add legend. Defaults followed by change.
        loc = 'upper left'
        if 'loc' in self.legend_dict.keys():
            loc = self.legend_dict['loc']

        size = 10
        if 'size' in self.legend_dict.keys():
            size = float(self.legend_dict['size'])

        bbox_to_anchor = None
        if 'bbox_to_anchor' in self.legend_dict.keys():
            bbox_to_anchor = self.legend_dict['bbox_to_anchor']

        ncol = 1
        if 'ncol' in self.legend_dict.keys():
            ncol = int(self.legend_dict['ncol'])

        self.axis.legend(loc=loc, markerscale=3.0, bbox_to_anchor=bbox_to_anchor,
                         ncol=ncol, prop={'size': size})

        cbar_info_dict = {}
        if 'colorbars' in self.gen_dict.keys():
            if 'CodetectionPotential2' in self.gen_dict['colorbars'].keys():
                cbar_info_dict = self.gen_dict['colorbars']['CodetectionPotential2']

        colorbar_pos, cbar_label, ticks_fontsize, label_fontsize, colorbar_axes, colorbar_ticks, colorbar_tick_labels = super(
            CodetectionPotential1, self).find_colorbar_information(cbar_info_dict, 'CodetectionPotential1')

        if cbar_label == 'None':
            cbar_label = r'$\rho$'

        super(CodetectionPotential1, self).setup_colorbars(colorbar_pos, codet_sc, 'CodetectionPotential1', cbar_label, ticks_fontsize=ticks_fontsize,
                                                           label_fontsize=label_fontsize, colorbar_axes=colorbar_axes, colorbar_ticks=colorbar_ticks, colorbar_tick_labels=colorbar_tick_labels)
        """
		cbar_info_dict = {}
		if 'colorbars' in self.gen_dict.keys():
			if 'SingleDetection' in self.gen_dict['colorbars'].keys():
				cbar_info_dict = self.gen_dict['colorbars']['SingleDetection']

		colorbar_pos, cbar_label, ticks_fontsize, label_fontsize, colorbar_axes, colorbar_ticks, colorbar_tick_labels = super(
		    CodetectionPotential2, self).find_colorbar_information(cbar_info_dict, 'SingleDetection')

		if cbar_label == 'None':
			cbar_label = 'Single Detection'

		super(CodetectionPotential2, self).setup_colorbars(colorbar_pos, sc4, 'SingleDetection', cbar_label, ticks_fontsize=ticks_fontsize,
		      label_fontsize=label_fontsize, colorbar_axes=colorbar_axes, colorbar_ticks=colorbar_ticks, colorbar_tick_labels=colorbar_tick_labels)
		"""

        return

    def find_codetection_potential(self, comparison_value):
        """
        This method finds the ratio contour and the loss gain contour values. Its inputs are the two datasets for comparison where the second is the control to compare against the first.

                The input data sets need to be the same shape. CreateSinglePlot.interpolate_data corrects for two datasets of different shape.

                Returns: loss_gain_contour, ratio contour (diff)

        """

        # ONLY FOR TESTING
        comparison_value = self.gen_dict['SNR_CUT']
        compare_snrs = []
        for zval in self.zvals:
            compare_snrs.append(1*(zval >= comparison_value) + 0*(zval < comparison_value))

        compare_snrs = np.asarray(compare_snrs)
        how_many_detectors = np.sum(np.asarray(compare_snrs), axis=0).astype(int)

        inds_codet = np.asarray(how_many_detectors > 1, dtype=int)
        single_det = np.asarray(how_many_detectors == 1, dtype=int)

        snr_single = []
        for i in range(len(self.zvals)):
            snr_single.append(self.zvals[i]*compare_snrs[i]*single_det)

        snr_single = np.asarray(snr_single)

        combined_det = []
        for i in range(len(self.zvals)):
            combined_det.append(self.zvals[i]*compare_snrs[i]*inds_codet)

        combined_snr = np.sqrt(np.sum(np.asarray(combined_det)**2, axis=0))

        return np.log10(combined_snr), np.log10(snr_single)


class CodetectionPotential(CreateSinglePlot):

    def __init__(self, fig, axis, xvals, yvals, zvals, gen_dict={}, limits_dict={},
                 label_dict={}, extra_dict={}, legend_dict={}):
        """
        """
        CreateSinglePlot.__init__(self, fig, axis, xvals, yvals, zvals, gen_dict,
                                  limits_dict, label_dict, extra_dict, legend_dict)

    def find_codetection_potential(self, val1, val2, return_single, comparison_value):
        """
        This method finds the ratio contour and the loss gain contour values. Its inputs are the two datasets for comparison where the second is the control to compare against the first.

                The input data sets need to be the same shape. CreateSinglePlot.interpolate_data corrects for two datasets of different shape.

                Returns: loss_gain_contour, ratio contour (diff)

        """

        zout = val1
        control_zout = val2

        # ONLY FOR TESTING
        # comparison_value = self.gen_dict['SNR_CUT']
        inds_up = ((zout >= comparison_value) & (
            control_zout >= comparison_value) & (zout >= control_zout))
        inds_down = ((zout >= comparison_value) & (
            control_zout >= comparison_value) & (zout < control_zout))

        # out_vals_codect = inds_up*np.log10(np.sqrt(zout**2 + control_zout**2)) + inds_down*-1*np.log10(np.sqrt(zout**2 + control_zout**2))

        if return_single:
            factor=-1
            out_vals_codect = inds_up*(np.log10(np.sqrt(zout**2 + control_zout**2))) + \
                inds_down*factor*(np.log10(np.sqrt(zout**2 + control_zout**2)))
        else:
            factor=1
            out_vals_codect = inds_up*(np.sqrt(zout**2 + control_zout**2)) + \
                inds_down*factor*(np.sqrt(zout**2 + control_zout**2))

        # reshape difference array for dimensions of contour
        out_vals_codect = np.reshape(out_vals_codect, np.shape(zout))

        if return_single:
            # set indices of loss,gained. inds_check tells the ratio calculator not to control_zout if both SNRs are below 1
            inds_up_only = ((zout >= comparison_value) & (control_zout < comparison_value))
            inds_down_only = ((zout < comparison_value) & (control_zout >= comparison_value))

            out_vals_single = inds_up_only*np.log10(zout) + inds_down_only*-1*np.log10(control_zout)
            out_vals_single = np.reshape(out_vals_single, np.shape(zout))

            return out_vals_codect, out_vals_single

        return out_vals_codect


class CodetectionPotential2(CodetectionPotential, CreateSinglePlot):
    # TODO remove need to have list for input files if you are going to specify control or remove control entirely
    # TODO make functions that generate input dicts
    # TODO make axis not visible
    def __init__(self, fig, axis, xvals, yvals, zvals, gen_dict={}, limits_dict={},
                 label_dict={}, extra_dict={}, legend_dict={}):
        """
        """
        CodetectionPotential.__init__(self, fig, axis, xvals, yvals, zvals, gen_dict,
                                      limits_dict, label_dict, extra_dict, legend_dict)

    def make_plot(self):
        """
        """

        # check to make sure ratio plot has 2 arrays to compare.
        if len(self.xvals) != 2:
            raise Exception("Length of vals not equal to 2. Ratio plots must have 2 inputs.")

        # check and interpolate data so that the dimensions of each data set are the same
        if np.shape(self.xvals[0]) != np.shape(self.xvals[1]):
            self.interpolate_data()

        # set comparison value. Default is SNR_CUT
        comparison_value = self.gen_dict['SNR_CUT']
        if 'snr_contour_value' in self.extra_dict.keys():
            comparison_value = self.extra_dict['snr_contour_value']

        # find loss/gain contour and ratio contour
        codect_pot, single = super(CodetectionPotential2, self).find_codetection_potential(
            self.zvals[0], self.zvals[1], True, comparison_value)

        codect_pot = np.ma.masked_where(
            (codect_pot > -np.log10(comparison_value)) & (codect_pot < np.log10(comparison_value)), codect_pot)
        single = np.ma.masked_where((single > -np.log10(comparison_value))
                                    & (single < np.log10(comparison_value)), single)

        # sets colormap for ratio comparison plot
        cmap2 = cm.BrBG
        cmap_single = cm.PiYG

        cmap2.set_bad(color='white', alpha=0.001)
        cmap_single.set_bad(color='white', alpha=0.001)

        normval2 = 4.0
        num_contours2 = 40  # must be even
        levels2 = np.linspace(-normval2, normval2, num_contours2)
        norm2 = colors.Normalize(-normval2, normval2)

        # plot ratio contours
        sc3 = self.axis.contourf(self.xvals[0], self.yvals[0], codect_pot,
                                 levels=levels2, norm=norm2, extend='both', cmap=cmap2, alpha=1.0)

        normval3 = 4.0
        num_contours3 = 40  # must be even
        levels3 = np.linspace(-normval3, normval3, num_contours3)
        norm3 = colors.Normalize(-normval3, normval3)
        sc4 = self.axis.contourf(self.xvals[0], self.yvals[0], single,
                                 levels=levels3, norm=norm3, extend='both', cmap=cmap_single, alpha=1.0)

        # toggle line contours of orders of magnitude of ratio comparisons
        if 'ratio_contour_lines' in self.extra_dict.keys():
            if self.extra_dict['ratio_contour_lines'] == True:
                self.axis.contour(self.xvals[0], self.yvals[0], codect_pot,
                                  np.arange(-8, 9, 1), colors='black', linewidths=1.0)
                self.axis.contour(self.xvals[0], self.yvals[0], single, np.array(
                    [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]), colors='black', linewidths=1.0)

        cbar_info_dict = {}
        if 'colorbars' in self.gen_dict.keys():
            if 'CodetectionPotential' in self.gen_dict['colorbars'].keys():
                cbar_info_dict = self.gen_dict['colorbars']['CodetectionPotential']

        colorbar_pos, cbar_label, ticks_fontsize, label_fontsize, colorbar_axes, colorbar_ticks, colorbar_tick_labels = super(
            CodetectionPotential, self).find_colorbar_information(cbar_info_dict, 'CodetectionPotential')

        if cbar_label == 'None':
            cbar_label = 'Codetection Potential'

        super(CodetectionPotential, self).setup_colorbars(colorbar_pos, sc3, 'CodetectionPotential', cbar_label, ticks_fontsize=ticks_fontsize,
                                                          label_fontsize=label_fontsize, colorbar_axes=colorbar_axes, colorbar_ticks=colorbar_ticks, colorbar_tick_labels=colorbar_tick_labels)

        cbar_info_dict = {}
        if 'colorbars' in self.gen_dict.keys():
            if 'SingleDetection' in self.gen_dict['colorbars'].keys():
                cbar_info_dict = self.gen_dict['colorbars']['SingleDetection']

        colorbar_pos, cbar_label, ticks_fontsize, label_fontsize, colorbar_axes, colorbar_ticks, colorbar_tick_labels = super(
            CodetectionPotential, self).find_colorbar_information(cbar_info_dict, 'SingleDetection')

        if cbar_label == 'None':
            cbar_label = 'Single Detection'

        super(CodetectionPotential, self).setup_colorbars(colorbar_pos, sc4, 'SingleDetection', cbar_label, ticks_fontsize=ticks_fontsize,
                                                          label_fontsize=label_fontsize, colorbar_axes=colorbar_axes, colorbar_ticks=colorbar_ticks, colorbar_tick_labels=colorbar_tick_labels)

        return


class CodetectionPotential3(CodetectionPotential, CreateSinglePlot):

    def __init__(self, fig, axis, xvals, yvals, zvals, gen_dict={}, limits_dict={},
                 label_dict={}, extra_dict={}, legend_dict={}):
        """
        Ratio is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information.

        Ratio creates an filled contour plot comparing snrs from two different data sets. Typically, it is used to compare sensitivty curves and/or varying binary parameters. It takes the snr of the first dataset and divides it by the snr from the second dataset. The log10 of this ratio is ploted. Additionally, a loss/gain contour is plotted. Loss/gain contour is based on SNR_CUT but can be overridden with 'snr_contour_value' in extra_dict. A gain indicates the first dataset reaches the snr threshold while the second does not. A loss is the opposite.
        """
        CodetectionPotential.__init__(self, fig, axis, xvals, yvals, zvals, gen_dict,
                                      limits_dict, label_dict, extra_dict, legend_dict)

    def make_plot(self):
        """
        This methd creates the ratio plot.
        """

        # check to make sure ratio plot has 2 arrays to compare.
        if len(self.xvals) != 3:
            # FIXME: fix this
            raise Exception("Length of vals not equal to 2. Ratio plots must have 2 inputs.")

        # check and interpolate data so that the dimensions of each data set are the same
        if np.shape(self.xvals[0]) != np.shape(self.xvals[1]):
            self.interpolate_data()

        # set comparison value. Default is SNR_CUT
        comparison_value = self.gen_dict['SNR_CUT']
        if 'snr_contour_value' in self.extra_dict.keys():
            comparison_value = self.extra_dict['snr_contour_value']

        # find loss/gain contour and ratio contour
        codect_pots = [super(CodetectionPotential3, self).find_codetection_potential(
            self.zvals[i], self.zvals[2], False, comparison_value) for i in range(2)]

        ratio_plot = Ratio(self.fig, self.axis, self.xvals, self.yvals, codect_pots, gen_dict=self.gen_dict, limits_dict=self.limits_dict,
                           label_dict=self.label_dict, extra_dict=self.extra_dict, legend_dict=self.legend_dict)

        ratio_plot.make_plot()

        """
		# TODO: remove this section
		codect_pot = np.ma.masked_where((codect_pot>-np.log10(comparison_value)) & (codect_pot<np.log10(comparison_value)), codect_pot)
		single = np.ma.masked_where((single>-np.log10(comparison_value)) & (single<np.log10(comparison_value)), single)


		# sets colormap for ratio comparison plot
		cmap2 = cm.BrBG
		cmap_single = cm.PiYG

		cmap2.set_bad(color='white', alpha=0.001)
		cmap_single.set_bad(color='white', alpha=0.001)

		normval2 = 4.0
		num_contours2 = 40 #must be even
		levels2 = np.linspace(-normval2, normval2,num_contours2)
		norm2 = colors.Normalize(-normval2, normval2)

		# plot ratio contours
		sc3=self.axis.contourf(self.xvals[0],self.yvals[0],codect_pot,
			levels = levels2, norm=norm2, extend='both', cmap=cmap2, alpha=1.0)

		normval3 = 4.0
		num_contours3 = 40 #must be even
		levels3 = np.linspace(-normval3, normval3,num_contours3)
		norm3 = colors.Normalize(-normval3, normval3)
		sc4=self.axis.contourf(self.xvals[0],self.yvals[0],single,
			levels = levels3, norm=norm3, extend='both', cmap=cmap_single, alpha=1.0)


		# toggle line contours of orders of magnitude of ratio comparisons
		if 'ratio_contour_lines' in self.extra_dict.keys():
			if self.extra_dict['ratio_contour_lines'] == True:
				self.axis.contour(self.xvals[0],self.yvals[0],codect_pot, np.arange(-8, 9, 1), colors = 'black', linewidths = 1.0)
				self.axis.contour(self.xvals[0],self.yvals[0],single, np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]), colors = 'black', linewidths = 1.0)

		cbar_info_dict = {}
		if 'colorbars' in self.gen_dict.keys():
			if 'CodetectionPotential' in self.gen_dict['colorbars'].keys():
				cbar_info_dict = self.gen_dict['colorbars']['CodetectionPotential']

		colorbar_pos, cbar_label, ticks_fontsize, label_fontsize, colorbar_axes, colorbar_ticks, colorbar_tick_labels = super(CodetectionPotential, self).find_colorbar_information(cbar_info_dict, 'CodetectionPotential')

		if cbar_label == 'None':
			cbar_label = 'Codetection Potential'

		super(CodetectionPotential, self).setup_colorbars(colorbar_pos, sc3, 'CodetectionPotential', cbar_label, ticks_fontsize=ticks_fontsize, label_fontsize=label_fontsize, colorbar_axes=colorbar_axes, colorbar_ticks=colorbar_ticks, colorbar_tick_labels=colorbar_tick_labels)

		cbar_info_dict = {}
		if 'colorbars' in self.gen_dict.keys():
			if 'SingleDetection' in self.gen_dict['colorbars'].keys():
				cbar_info_dict = self.gen_dict['colorbars']['SingleDetection']

		colorbar_pos, cbar_label, ticks_fontsize, label_fontsize, colorbar_axes, colorbar_ticks, colorbar_tick_labels = super(CodetectionPotential, self).find_colorbar_information(cbar_info_dict, 'SingleDetection')

		if cbar_label == 'None':
			cbar_label = 'Single Detection'

		super(CodetectionPotential, self).setup_colorbars(colorbar_pos, sc4, 'SingleDetection', cbar_label, ticks_fontsize=ticks_fontsize, label_fontsize=label_fontsize, colorbar_axes=colorbar_axes, colorbar_ticks=colorbar_ticks, colorbar_tick_labels=colorbar_tick_labels)
		"""

        return


class Horizon(CreateSinglePlot):

    def __init__(self, fig, axis, xvals, yvals, zvals, gen_dict={}, limits_dict={}, label_dict={}, extra_dict={}, legend_dict={}):
        """
        Horizon is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information.

        Horizon plots snr contour lines for a designated SNR value. The defaul is SNR_CUT, but can be overridden with "snr_contour_value" in extra_dict. Horizon can take as many curves as the user prefers and will plot a legend to label those curves. It can only contour one snr value.

        Additional Inputs:

        legend_dict - dict describing legend labels and properties.
        legend_dict inputs/keys:
                labels - list of strings - contains labels for each plot that will appear in the legend.
                loc - string or int - location of legend. Refer to matplotlib documentation for legend placement for choices. Default is 'upper left'.
                size - float - size of legend. Default is 10.
                bbox_to_anchor - list of floats, length 2 or 4 - Places legend in custom location. First two entries represent corner of box is placed. Second two entries (not required) represent how to stretch the legend box from there. See matplotlib documentation on bbox_to_anchor for specifics.
                ncol - int - number of columns in legend. Default is 1.
        """

        CreateSinglePlot.__init__(self, fig, axis, xvals, yvals, zvals, gen_dict,
                                  limits_dict, label_dict, extra_dict, legend_dict)

    def make_plot(self):
        """
        This method adds a horizon plot as desribed in the Horizon class docstring. Can compare up to 7 curves.
        """
        # sets levels of main contour plot
        colors1 = ['blue', 'green', 'red', 'purple', 'orange',
                   'gold', 'magenta']

        # set contour value. Default is SNR_CUT.
        self.contour_val = self.gen_dict['SNR_CUT']
        if 'snr_contour_value' in self.extra_dict.keys():
            self.contour_val = float(self.extra_dict['snr_contour_value'])

        # plot contours
        for j in range(len(self.zvals)):
            hz = self.axis.contour(self.xvals[j], self.yvals[j],
                                   self.zvals[j], np.array([self.contour_val]),
                                   colors=colors1[j], linewidths=1., linestyles='solid')

            # plot invisible lines for purpose of creating a legend
            if self.legend_dict != {}:
                # plot a curve off of the grid with same color for legend label.
                self.axis.plot([0.1, 0.2], [0.1, 0.2], color=colors1[j],
                               label=self.legend_dict['labels'][j])

        if self.legend_dict != {}:
            # Add legend. Defaults followed by change.
            loc = 'upper left'
            if 'loc' in self.legend_dict.keys():
                loc = self.legend_dict['loc']

            size = 10
            if 'size' in self.legend_dict.keys():
                size = float(self.legend_dict['size'])

            bbox_to_anchor = None
            if 'bbox_to_anchor' in self.legend_dict.keys():
                bbox_to_anchor = self.legend_dict['bbox_to_anchor']

            ncol = 1
            if 'ncol' in self.legend_dict.keys():
                ncol = int(self.legend_dict['ncol'])

            self.axis.legend(loc=loc, bbox_to_anchor=bbox_to_anchor,
                             ncol=ncol, prop={'size': size})

        return
