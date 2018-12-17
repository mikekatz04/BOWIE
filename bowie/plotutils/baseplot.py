import numpy as np


class CreateSinglePlot:
    """Base class for the subclasses designed for creating the plots.

    All of the plot types inherit this class and its methods. The __init__ method
    is the same for all plots. Each plot_type in :mod:`bowie.plotutils.plottypes`
    has its own ``make_plot`` method and associated methods for creating the plot.

    Args:
        fig (obj): Figure environment for the plots.
        axis (obj): Axis object representing specific plot.
        xvals/yvals/zvals (list of 2D arrays of floats): List of x,y,z-value
            arrays for the plot.

    Keyword Arguments:
        plot_type (str): String representing the plot type.
        SNR_CUT (float): SNR cut for detectability or another desired value.
        xlims/ylims (len-2 list of floats): Limits for the x/y axis. If
            ``xscale/yscale == 'log'``, these values are the log10 of the limits.
        dx/dy (float): Amount to increment for tick labels. If log10 of values,
            then dx/dy repesents increments in log10.
        xscale/yscale (str): `lin` for linear scale. `log` for log10 scale.
        title/xlabel/ylabel (str, optional): String for title/xlabel/ylabel for specific plot.
        title_kwargs/xlabel_kwargs/ylabel_kwargs: Kwargs for call to add string labelling to plot.
        x_tick_label_fontsize/y_tick_label_fontsize (int, optional): Tick label fontsize
            for x/y axis. Default is 14.
        tick_label_fontsize (int, optional): Tick label fontsize for both x and y axes.
            If not None, this will set both axis in one go to the same value.
            Default is None.
        reverse_x_axis/reverse_y_axis (bool, optional): If True, reverse the order
            of the ticks along each axis from ascending to descending. Default is False.
        spacing (str, optional): Spacing of plots. `wide` indicates spaced out plots
            (``hspace = 0.3`` and ``wspace = 0.3`` in subplots_adjust).
            All the tick labels will show. If `tight`, no space between plots
            in the figure (``hspace = 0.0`` and ``wspace = 0.0`` in subplots_adjust).
            In this case, the edge tick labels are cut off for clarity. Default is `wide`.
        add_grid (bool, optional): If True, add a grid to the specific plot. Default is True.
        colorbar (obj, optional): Object of type
            :class:`bowie.plotutils.baseplot.FigColorbar`. This houses information for the
            colorbar for the associated plot type. Default is None.
        colormap (str, optional): String represented the the colormap choice through python's
            predefined colormaps. This is for ratio plots. Default is `coolwarm`.
        loss_gain_status (bool, optional): If True, show the loss gain contours.
            This only applies to :class:`bowie.plotutils.plottypes.Ratio` plot.
            Default is True.
        snr_contour_value (float, optional): This will set the value for contours in plots.
            In :class:`bowie.plotutils.plottypes.Ratio` plots, this sets the value for
            loss/gain contours and minimal value for ratio contours (overrides SNR_CUT). For
            :class:`bowie.plotutils.plottypes.Horizon` plots this sets the value
            for the horizon contours. For :class:`bowie.plotutils.plottypes.Waterfall`
            plots, this will add an additional contour in white. Default is None.
        order_contour_lines (bool, optional): Show dashed lines at each order of magnitude
            within the ratio contours. Default is False.
        ratio_comp_value (float, optional): If not None, this value is used as the minimum
            value for the ratio contours. If None, this value will be `SNR_CUT` or
            `snr_contour_value` if given. In other words, it will be the
            same as loss/gain contour maximum. Default is None.
        legend_kwargs (dict, optional): kwargs for call to ax.legend(). Default is {}.
        legend_labels (list of str, optional): Labels for the curves in the legend.
            If calling legend, these are required. Default is [].
        add_legend (bool, optional): If True, add legend.
        contour_vals (list or array of floats): Contour values to use for
            :class:`bowie.plotutils.plottypes.Waterfall`. Default is
            np.array([0., 10, 20, 50, 100, 200, 500, 1000, 3000, 1e10]).
            Default kwarg value is [].

    Attributes:
        comparison_value (float): Value to use for contour comparisons. This will either
            be SNR_CUT or snr_contour_value.
        Note: all args and kwargs become attributes.

    """
    def __init__(self, fig, axis, xvals, yvals, zvals, **kwargs):

        prop_default = {
            'x_tick_label_fontsize': 14,
            'y_tick_label_fontsize': 14,
            'tick_label_fontsize': None,
            'reverse_x_axis': False,
            'reverse_y_axis': False,
            'spacing': 'wide',
            'add_grid': True,
            'colorbar': None,
            'colormap': 'coolwarm',
            'loss_gain_status': True,
            'snr_contour_value': None,
            'order_contour_lines': False,
            'ratio_comp_value': None,
            'legend_kwargs': {},
            'legend_labels': [],
            'add_legend': False,
            'contour_vals':  np.array([0., 10, 20, 50, 100, 200, 500, 1000, 3000, 1e10]),
        }

        for key, value in kwargs.items():
            setattr(self, key, value)

        for prop, default in prop_default.items():
            setattr(self, prop, kwargs.get(prop, default))

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

        if self.plot_type not in ['Horizon', 'Waterfall']:
            # check that the dimensions of each data set are the same
            for i in np.arange(len(self.xvals)-1):
                if np.shape(self.xvals[0]) != np.shape(self.xvals[i+1]):
                    raise Exception("All data must be the same shape for"
                                    + " {} plots.".format(self.plot_type))

            for i in np.arange(len(self.xvals)-1):
                if (len(np.where(self.xvals[0] != self.xvals[i+1])[0]) != 0
                        or len(np.where(self.yvals[0] != self.yvals[i+1])[0]) != 0):
                    raise Exception("All x and y values from each dataset must be the same for"
                                    + " {} plots.".format(self.plot_type))

        # check input data sizes.
        if len(self.zvals) != 2 and self.plot_type == 'Ratio':
            raise Exception("Length of vals not equal to 2. Ratio plots must have 2 inputs.")

        self.comparison_value = (self.snr_contour_value if self.snr_contour_value is not
                                 None else self.SNR_CUT)

    def setup_plot(self):
        """Set up limits and labels.

        For all plot types, this method is used to setup the basic features of each plot.

        """
        if self.tick_label_fontsize is not None:
            self.x_tick_label_fontsize = self.tick_label_fontsize
            self.y_tick_label_fontsize = self.tick_label_fontsize

        # setup xticks and yticks and limits
        # if logspaced, the log values are used.
        xticks = np.arange(float(self.xlims[0]),
                           float(self.xlims[1])
                           + float(self.dx),
                           float(self.dx))

        yticks = np.arange(float(self.ylims[0]),
                           float(self.ylims[1])
                           + float(self.dy),
                           float(self.dy))

        xlim = [xticks.min(), xticks.max()]
        ylim = [yticks.min(), yticks.max()]

        if self.reverse_x_axis:
                xticks = xticks[::-1]
                xlim = [xticks.max(), xticks.min()]

        if self.reverse_y_axis:
                    yticks = yticks[::-1]
                    ylim = [yticks.max(), yticks.min()]

        self.axis.set_xlim(xlim)
        self.axis.set_ylim(ylim)

        # adjust ticks for spacing. If 'wide' then show all labels, if 'tight' remove end labels.
        if self.spacing == 'wide':
            x_inds = np.arange(len(xticks))
            y_inds = np.arange(len(yticks))
        else:
            # remove end labels
            x_inds = np.arange(1, len(xticks)-1)
            y_inds = np.arange(1, len(yticks)-1)

        self.axis.set_xticks(xticks[x_inds])
        self.axis.set_yticks(yticks[y_inds])

        # set tick labels based on scale
        if self.xscale == 'log':
            self.axis.set_xticklabels([r'$10^{%i}$' % int(i)
                                      for i in xticks[x_inds]], fontsize=self.x_tick_label_fontsize)
        else:
            self.axis.set_xticklabels([r'$%.3g$' % (i)
                                      for i in xticks[x_inds]], fontsize=self.x_tick_label_fontsize)

        if self.yscale == 'log':
            self.axis.set_yticklabels([r'$10^{%i}$' % int(i)
                                      for i in yticks[y_inds]], fontsize=self.y_tick_label_fontsize)
        else:
            self.axis.set_yticklabels([r'$%.3g$' % (i)
                                      for i in yticks[y_inds]], fontsize=self.y_tick_label_fontsize)

        # add grid
        if self.add_grid:
            self.axis.grid(True, linestyle='-', color='0.75')

        # add title
        if 'title' in self.__dict__.keys():
            self.axis.set_title(r'{}'.format(self.title), **self.title_kwargs)

        if 'xlabel' in self.__dict__.keys():
            self.axis.set_xlabel(r'{}'.format(self.xlabel), **self.xlabel_kwargs)

        if 'ylabel' in self.__dict__.keys():
            self.axis.set_ylabel(r'{}'.format(self.ylabel), **self.ylabel_kwargs)
        return


class FigColorbar:
    """Create colorbars

    This class holds and creates colorbars for the plots shown. It has default values
    for each plot that can be replaced by passing kwargs.

    Args:
        fig (obj): Figure object for adding the colorbar.
        plot_type (str): String representing the plot type for this colorbar.

    Keyword Arguments:
        cbar_ticks (list or array of floats, optional): List or array representing
            the ticks to be shown on the colorbar. Default kwarg is []. If ``cbar_ticks == []``,
            the default for each plot type is used. For Waterfall it is
            [0., 10, 20, 50, 100, 200, 500, 1000, 3000, 1e10]. For Ratio plots,
            it is [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0].
        cbar_tick_labels (list of str, optional): List of strings corresponding to the labels
            for the cbar_ticks. Default is []. In this case, default for each plot type is used.
            Waterfall plots default to [0., 10, 20, 50, 100, 200, 500, 1000, 3000].
            Ratio plots default to [-2.0, -1.0, 0.0, 1.0, 2.0].
        cbar_ticks_fontsize (int, optional): Fontsize for colorbar ticks. Default is 15.
        cbar_label_fontsize (int, optional): Label fontsize for the colorbar. Default is 20.
        cbar_pos (int, optional): Describes defined positions (axes) for the colorbar.
            Default positions are 1-5. {'1': [0.83, 0.49, 0.03, 0.38],
            '2': [0.83, 0.05, 0.03, 0.38], '3': [0.05, 0.92, 0.38, 0.03],
            '4': [0.47, 0.92, 0.38, 0.03], '5': [0.83, 0.1, 0.03, 0.8]}. Default is 1 for Waterfall
            and 2 for Ratio. Default value for kwarg is 'use_default'.
        cbar_axes (len-4 list of floats, optional): Custom axes for colorbar. This overrides any
            entry to ``cbar_pos``. Default is []. In this case, it will default to
            axes for ``cbar_pos`` value.
        cbar_orientation (str, optional): Vertical or horizontal orientation for the colorbar.
            Default is vertical if the bar is thinner than it is tall.
            If it is wider than it is tall, default is horizontal. Default value for kwarg is None.
        cbar_label_pad (float, optional): Label padding of colorbar label. Default is None.
            This means -60 for horizontal and 10 for vertical. # TODO check these numbers.

    Attributes:
        cbar_ax (obj): Axes object on the figure for the colorbar.
        cbar_var (str): String representing the axis of the colorbar to add labels to.
            For horizontal this is 'x' and for vertical it is 'y'.
        Note: all args and kwargs are added as attributes.

    """
    def __init__(self, fig, plot_type, **kwargs):
        # setup tick labels depending on plot
        self.fig = fig
        self.plot_type = plot_type
        if plot_type == 'Waterfall':
            self.cbar_ticks = np.array([0., 10, 20, 50, 100, 200, 500, 1000, 3000, 1e10])
            self.cbar_tick_labels = [int(i) for i in np.delete(self.cbar_ticks, -1)]

        elif plot_type == 'Ratio':
            self.cbar_ticks = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
            self.cbar_tick_labels = [r'$10^{%i}$' % i for i in self.cbar_ticks[1:-1]]

        prop_default = {
            'cbar_label': None,
            'cbar_ticks_fontsize': 15,
            'cbar_label_fontsize': 20,
            'cbar_axes': [],
            'cbar_ticks': [],
            'cbar_tick_labels': [],
            'cbar_pos': 'use_default',
            'cbar_orientation': None,
            'cbar_label_pad': None,
            'cbar_var': None,
        }

        for prop, default in prop_default.items():
            if prop not in self.__dict__:
                setattr(self, prop, kwargs.get(prop, default))

        if self.cbar_label is None:
            cbar_label_defaults = {'Waterfall': r'$\rho$',
                                   'Ratio': r"$\rho_1/\rho_2$"}

            self.cbar_label = cbar_label_defaults[plot_type]

        # dict with axes locations
        if self.cbar_axes == []:
            cbar_pos_defaults = {'Waterfall': 1,
                                 'Ratio': 2}

            cbar_axes_defaults = {'1': [0.83, 0.49, 0.03, 0.38],
                                  '2': [0.83, 0.05, 0.03, 0.38],
                                  '3': [0.05, 0.92, 0.38, 0.03],
                                  '4': [0.47, 0.92, 0.38, 0.03],
                                  '5': [0.83, 0.1, 0.03, 0.8]}

            if self.cbar_pos == 'use_default':
                self.cbar_pos = cbar_pos_defaults[plot_type]
            self.cbar_axes = cbar_axes_defaults[str(self.cbar_pos)]

        self.cbar_ax = self.fig.add_axes(self.cbar_axes)

        # check if colorbar is horizontal or vertical
        if self.cbar_axes[2] > self.cbar_axes[3]:
            self.cbar_orientation = ('horizontal' if self.cbar_orientation is None
                                     else self.cbar_orientation)
            self.cbar_label_pad = -60 if self.cbar_label_pad is None else self.cbar_label_pad

        else:
            self.cbar_orientation = ('vertical' if self.cbar_orientation is None
                                     else self.cbar_orientation)
            self.cbar_label_pad = 10 if self.cbar_label_pad is None else self.cbar_label_pad

        if self.cbar_orientation == 'horizontal':
            self.cbar_var = 'x'
            self.cbar_ax.xaxis.set_ticks_position('top')
        else:
            self.cbar_var = 'y'

    def setup_colorbars(self, plot_call_sign):
        """Setup colorbars for each type of plot.

        Take all of the optional performed during ``__init__`` method and makes the colorbar.

        Args:
            plot_call_sign (obj): Plot instance of ax.contourf with colormapping to
                add as a colorbar.

        """
        self.fig.colorbar(plot_call_sign, cax=self.cbar_ax,
                          ticks=self.cbar_ticks, orientation=self.cbar_orientation)
        # setup colorbar ticks
        (getattr(self.cbar_ax, 'set_' + self.cbar_var + 'ticklabels')
            (self.cbar_tick_labels, fontsize=self.cbar_ticks_fontsize))
        (getattr(self.cbar_ax, 'set_' + self.cbar_var + 'label')
            (self.cbar_label, fontsize=self.cbar_label_fontsize, labelpad=self.cbar_label_pad))

        return
