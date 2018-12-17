"""
This module houses the main class for plotting within the BOWIE package.
It runs through the whole plot creation process.

It is part of the BOWIE analysis tool. Author: Michael Katz. Please cite
"Evaluating Black Hole Detectability with LISA" (arXiv:1807.02511) for usage of this code.

This code is licensed under the GNU public license.
"""

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from bowie.plotutils.readdata import PlotVals, ReadInData
from bowie.plotutils.baseplot import FigColorbar
from bowie.plotutils.plottypes import (CreateSinglePlot,
                                                Waterfall,
                                                Ratio,
                                                Horizon)


class MakePlotProcess:
    """Process hub for creating plots.

    Class that carries the input dictionary (pid) and directs
    the program to accomplish plotting tasks.

    Args:
        **kwargs (dict): Combination of all input dictionaries to have their information
            stored as attributes.

    Keyword Arguments:
        plot_info (dict): Dictionary containing all the individual plot info.
        sharex, sharey (bool, optional): Applies sharex, sharey as used in ``plt.subplots()``.
            See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
            Default is True.
        figure_width/figure_height (float, optional): Dimensions of the figure set with
            ``fig.set_size_inches()`` from matplotlib. Default is 8.0 for each.
        bottom (float, optional): Sets position of bottom of subplots in figure coordinates.
            Default is 0.1.
        top (float, optional): Sets position of top of subplots in figure coordinates.
            Default is 0.85.
        left (float, optional): Sets position of left edge of subplots in figure coordinates.
            Default is 0.12.
        right (float, optional): Sets position of right edge of subplots in figure coordinates.
            Default is 0.79.
        wspace (float, optional): The amount of width reserved for space between subplots,
           It is expressed as a fraction of the average axis width. Default is 0.3.
        hspace (float, optional): The amount of height reserved for space between subplots,
           It is expressed as a fraction of the average axis width. Default is 0.3.
        snr_input (dict): kwargs for :class:`gwsnrcalc.gw_snr_calculator.SNR`.


    Attributes:
        total_mass (1D array of floats): Total mass of each binary.
        mass_ratio (1D array of floats): Mass ratio of each binary.
        m1 (1D array of floats): Mass 1 of each binary.
        m2 (1D array of floats): Mass 2 of each binary.
        z_or_dist (1D array of floats): Redshift or distance of each binary.
        spin_1 (1D array of floats): Spin of mass 1 of each binary.
        spin_2 (1D array of floats): Spin of mass 2 of each binary.
        start_time (1D array of floats): Start time in years before merger of each binary.
        total_mass (1D array of floats): End time in years before merger of each binary.
        dist_type (str): Which type of distance measure is used. Options are `redshift`,
            `luminosity_distance`, or `comoving_distance`.
        value_classes (obj): Class :class:`bowie.plotutils.readdata.PlotVals` that
            holds data.
        final_dict (dict): Dictionary with SNR results.
        Note: All kwargs above are added as attributes.

    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        prop_default = {
            'sharex': True,
            'sharey': True,
            'figure_width': 8,
            'figure_height': 8,
            'bottom': 0.1,
            'right': 0.9,
            'left': 0.12,
            'top': 0.85,
            'hspace': 0.3,
            'wspace': 0.3,
            'colorbars': {},
            'subplots_adjust_kwargs': {},
            }

        for prop, default in prop_default.items():
            setattr(self, prop, kwargs.get(prop, default))

    def input_data(self):
        """Function to extract data from files according to pid.

        This function will read in the data with
        :class:`bowie.plotutils.readdata.ReadInData`.

        """
        ordererd = np.sort(np.asarray(list(self.plot_info.keys())).astype(int))

        trans_cont_dict = OrderedDict()
        for i in ordererd:
            trans_cont_dict[str(i)] = self.plot_info[str(i)]

        self.plot_info = trans_cont_dict

        # set empty lists for x,y,z
        x = [[]for i in np.arange(len(self.plot_info.keys()))]
        y = [[] for i in np.arange(len(self.plot_info.keys()))]
        z = [[] for i in np.arange(len(self.plot_info.keys()))]

        # read in base files/data
        for k, axis_string in enumerate(self.plot_info.keys()):

            if 'file' not in self.plot_info[axis_string].keys():
                continue
            for j, file_dict in enumerate(self.plot_info[axis_string]['file']):
                data_class = ReadInData(**{**self.general, **file_dict,
                                           **self.plot_info[axis_string]['limits']})

                x[k].append(data_class.x_append_value)
                y[k].append(data_class.y_append_value)
                z[k].append(data_class.z_append_value)

            # print(axis_string)

        # add data from plots to current plot based on index
        for k, axis_string in enumerate(self.plot_info.keys()):

            # takes first file from plot
            if 'indices' in self.plot_info[axis_string]:
                if type(self.plot_info[axis_string]['indices']) == int:
                    self.plot_info[axis_string]['indices'] = (
                        [self.plot_info[axis_string]['indices']])

                for index in self.plot_info[axis_string]['indices']:

                    index = int(index)

                    x[k].append(x[index][0])
                    y[k].append(y[index][0])
                    z[k].append(z[index][0])

        # read or append control values for ratio plots
        for k, axis_string in enumerate(self.plot_info.keys()):
            if 'control' in self.plot_info[axis_string]:
                if ('name' in self.plot_info[axis_string]['control'] or
                        'label' in self.plot_info[axis_string]['control']):
                    file_dict = self.plot_info[axis_string]['control']
                    if 'limits' in self.plot_info[axis_string].keys():
                        liimits_dict = self.plot_info[axis_string]['limits']

                    data_class = ReadInData(**{**self.general, **file_dict,
                                               **self.plot_info[axis_string]['limits']})
                    x[k].append(data_class.x_append_value)
                    y[k].append(data_class.y_append_value)
                    z[k].append(data_class.z_append_value)

                elif 'index' in self.plot_info[axis_string]['control']:
                    index = int(self.plot_info[axis_string]['control']['index'])

                    x[k].append(x[index][0])
                    y[k].append(y[index][0])
                    z[k].append(z[index][0])

            # print(axis_string)

        # transfer lists in PlotVals class.
        value_classes = []
        for k in range(len(x)):
            value_classes.append(PlotVals(x[k], y[k], z[k]))

        self.value_classes = value_classes
        return

    def setup_figure(self):
        """Sets up the initial figure on to which every plot is added.

        """

        # declare figure and axes environments
        fig, ax = plt.subplots(nrows=int(self.num_rows),
                               ncols=int(self.num_cols),
                               sharex=self.sharex,
                               sharey=self.sharey)

        fig.set_size_inches(self.figure_width, self.figure_height)

        # create list of ax. Catch error if it is a single plot.
        try:
            ax = ax.ravel()
        except AttributeError:
            ax = [ax]

        # create list of plot types
        self.plot_types = [self.plot_info[str(i)]['plot_type'] for i in range(len(ax))]

        if len(self.plot_types) == 1:
            if self.plot_types[0] not in self.colorbars:
                self.colorbars[self.plot_types[0]] = {'cbar_pos': 5}
            else:
                if 'cbar_pos' not in self.colorbars[self.plot_types[0]]:
                    self.colorbars[self.plot_types[0]]['cbar_pos'] = 5

        # prepare colorbar classes
        self.colorbar_classes = {}
        for plot_type in self.plot_types:
            if plot_type in self.colorbar_classes:
                continue
            if plot_type == 'Horizon':
                self.colorbar_classes[plot_type] = None

            elif plot_type in self.colorbars:
                self.colorbar_classes[plot_type] = FigColorbar(fig, plot_type,
                                                               **self.colorbars[plot_type])

            else:
                self.colorbar_classes[plot_type] = FigColorbar(fig, plot_type)

        # set subplots_adjust settings
        if 'Ratio' in self.plot_types or 'Waterfall':
            self.subplots_adjust_kwargs['right'] = 0.79

        # adjust figure sizes
        fig.subplots_adjust(**self.subplots_adjust_kwargs)

        if 'fig_y_label' in self.__dict__.keys():
            fig.text(self.fig_y_label_x,
                     self.fig_y_label_y,
                     r'{}'.format(self.fig_y_label),
                     **self.fig_y_label_kwargs)

        if 'fig_x_label' in self.__dict__.keys():
            fig.text(self.fig_x_label_x,
                     self.fig_x_label_y,
                     r'{}'.format(self.fig_x_label),
                     **self.fig_x_label_kwargs)

        if 'fig_title' in self.__dict__.keys():
            fig.text(self.fig_title_kwargs['x'],
                     self.fig_title_kwargs['y'],
                     r'{}'.format(self.fig_title),
                     **self.fig_title_kwargs)

        self.fig, self.ax = fig, ax
        return

    def create_plots(self):
        """Creates plots according to each plotting class.

        """
        for i, axis in enumerate(self.ax):
            # plot everything. First check general dict for parameters related to plots.
            trans_plot_class_call = globals()[self.plot_types[i]]
            trans_plot_class = trans_plot_class_call(self.fig, axis,
                                                     self.value_classes[i].x_arr_list,
                                                     self.value_classes[i].y_arr_list,
                                                     self.value_classes[i].z_arr_list,
                                                     colorbar=(
                                                        self.colorbar_classes[self.plot_types[i]]),
                                                     **{**self.general,
                                                        **self.figure,
                                                        **self.plot_info[str(i)],
                                                        **self.plot_info[str(i)]['limits'],
                                                        **self.plot_info[str(i)]['label'],
                                                        **self.plot_info[str(i)]['extra'],
                                                        **self.plot_info[str(i)]['legend']})

            # create the plot
            trans_plot_class.make_plot()

            # setup the plot
            trans_plot_class.setup_plot()

            # print("Axis", i, "Complete")
        return
