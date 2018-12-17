"""
Documentation is provided for creating the main input dictionary for ``make_plot.py``.
A dictionary is passed into the main function of the code, providing the preferences of the user.
A dictionary in python is contained within {} (see 5.5 Dictionaries at
https://docs.python.org/3/tutorial/datastructures.html).

This documentation will direct the user to all choices related to this dictionary
included required and optional parameters.

These dictionaries can be used in a python code or jupyter notebook by importing the main functions
from each script. They can also be implemented with .json files similar to those included
in the repository. If .json is used, the function call is
``python make_plot.py make_plot_config.json``.

This code is part of the BOWIE analysis tool. Author: Michael Katz. Please see
"Evaluating Black Hole Detectability with LISA" (arXiv:1807.02511) for example usage.
Please cite arXiv:1807.02511 for its usage.

This code is shared under the GNU Public License.

**NOTE**: This code associates classes with a name (e.g. Label) and a Container
    class for storing its information (e.g. LabelContainer). This is due to the need to inherit
    methods in a separate way from where the information is actually stored.

    Therefore, when reading the documentation, the main classes (without Container)
    will describe the methods for adding information for plotting. This includes
    required values and default values for optional Args **related to the methods
    themselves and not the defaults and required values within the actual plotting module code.**

    The Container classes have descriptions of their attributes. Their attributes involve
    the same requirements and default/optional Args as the plotting code itself.

"""

import warnings
import inspect


class Label:
    """Label contains the methods inherited by SinglePlot.

    These methods are used by SinglePlot class to add configuration
    information pertaining to the labels dict involved in plot creation.

    All attributes are appended to the LabelContainer class.
    The LabelContainer class is contained in the label attribute
    in the SinglePlot class.

    """
    def _set_label(self, which, label, **kwargs):
        """Private method for setting labels.

        Args:
            which (str): The indicator of which part of the plots
                to adjust. This currently handles `xlabel`/`ylabel`,
                and `title`.
            label (str): The label to be added.
            fontsize (int, optional): Fontsize for associated label. Default
                is None.

        """
        prop_default = {
            'fontsize': 18,
        }

        for prop, default in prop_default.items():
            kwargs[prop] = kwargs.get(prop, default)

        setattr(self.label, which, label)
        setattr(self.label, which + '_kwargs', kwargs)
        return

    def set_xlabel(self, label, **kwargs):
        """Set xlabel for plot.

        Similar to matplotlib, this will set the xlabel for the specific plot.

        Args:
            label (str): The label to be added.

        Keyword Arguments:
            fontsize (int, optional): Fontsize for associated label. Default
                is None.
            Note: Other kwargs are available. See:
                https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html

        """
        self._set_label('xlabel', label, **kwargs)
        return

    def set_ylabel(self, label, **kwargs):
        """Set ylabel for plot.

        Similar to matplotlib, this will set the ylabel for the specific plot.

        Args:
            label (str): The label to be added.

        Keyword Arguments:
            fontsize (int, optional): Fontsize for associated label. Default
                is None.
            Note: Other kwargs are available. See:
                https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html

        """
        self._set_label('ylabel', label, **kwargs)
        return

    def set_title(self, title, **kwargs):
        """Set title for plot.

        Similar to matplotlib, this will set the title for the specific plot.

        Args:
            title (str): The title to be added.

        Keyword Arguments:
            fontsize (int, optional): Fontsize for associated title. Default
                is None.
            Note: Other kwargs available. See:
                https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_title.html

        """
        self._set_label('title', title, **kwargs)
        return

    def _set_tick_label_fontsize(self, which, fontsize):
        """Private method for setting tick label fontsize.

        Args:
            which (str): The indicator of which part of the plots
                to adjust. This currently handles `x` and `y`.
            fontsize (int): Fontsize for associated label.

        """
        setattr(self.label, which + '_tick_label_fontsize', fontsize)
        return

    def set_x_tick_label_fontsize(self, fontsize):
        """Set x_tick_label_fontsize for plot.

        This will set the tick label fontsize for the x axis
        for the specific plot. This will override a call
        to set_tick_label_fontsize.

        Args:
            fontsize (int): Fontsize for x axis tick labels.

        """
        self._set_tick_label_fontsize('x', fontsize)
        return

    def set_y_tick_label_fontsize(self, fontsize):
        """Set y_tick_label_fontsize for plot.

        This will set the tick label fontsize for the y axis
        for the specific plot. This will override a call
        to set_tick_label_fontsize.

        Args:
            fontsize (int): Fontsize for y axis tick labels.

        """
        self._set_tick_label_fontsize('y', fontsize)
        return

    def set_tick_label_fontsize(self, fontsize):
        """Set x_tick_label_fontsize and y_tick_label_fontsize for plot.

        This will set the tick label fontsize for both the x and y axes
        for the specific plot.

        Args:
            fontsize (int): Fontsize for x and y axes tick labels.

        """
        self._set_tick_label_fontsize('x', fontsize)
        return


class LabelContainer:
    """Holds all of the attributes related to the label dictionary.

    This class is used to store the information when methods from Label class
    are called by the SinglePlot class.

    Attributes:
        xlabel/ylabel (str, optional): x,y label for specific plot.
        title (str, optional): Title for specific plot.
        xlabel_fontsize/ylabel_fontsize (float, optional): Sets fontsize for x,y label for specific
            plot. Default is 20.
        title_fontsize (float, optional): Sets fontsize for y label for specific
            plot. Default is 20.
        x_tick_label_fontsize/y_tick_label_fontsize (float, optional): Sets fontsize for x,y tick
            labels for specific
            plot. Default is 14.

    """
    def __init__(self):
        pass


class Limits:
    """ Limits contains the methods inherited by SinglePlot.

    These methods are used by SinglePlot class to add configuration
    information pertaining to the limits dict involved in plot creation.

    All attributes are appended to the LimitsContainer class.
    The LimitsContainer class is contained in the limits attribute
    in the SinglePlot class.

    """
    def _set_axis_limits(self, which, lims, d, scale, reverse=False):
        """Private method for setting axis limits.

        Sets the axis limits on each axis for an individual plot.

        Args:
            which (str): The indicator of which part of the plots
                to adjust. This currently handles `x` and `y`.
            lims (len-2 list of floats): The limits for the axis.
            d (float): Amount to increment by between the limits.
            scale (str): Scale of the axis. Either `log` or `lin`.
            reverse (bool, optional): If True, reverse the axis tick marks. Default is False.

        """
        setattr(self.limits, which + 'lims', lims)
        setattr(self.limits, 'd' + which, d)
        setattr(self.limits, which + 'scale', scale)

        if reverse:
            setattr(self.limits, 'reverse_' + which + '_axis', True)
        return

    def set_xlim(self, xlims, dx, xscale, reverse=False):
        """Set x limits for plot.

        This will set the limits for the x axis
        for the specific plot.

        Args:
            xlims (len-2 list of floats): The limits for the axis.
            dx (float): Amount to increment by between the limits.
            xscale (str): Scale of the axis. Either `log` or `lin`.
            reverse (bool, optional): If True, reverse the axis tick marks. Default is False.

        """
        self._set_axis_limits('x', xlims, dx, xscale, reverse)
        return

    def set_ylim(self, xlims, dx, xscale, reverse=False):
        """Set y limits for plot.

        This will set the limits for the y axis
        for the specific plot.

        Args:
            ylims (len-2 list of floats): The limits for the axis.
            dy (float): Amount to increment by between the limits.
            yscale (str): Scale of the axis. Either `log` or `lin`.
            reverse (bool, optional): If True, reverse the axis tick marks. Default is False.

        """
        self._set_axis_limits('y', xlims, dx, xscale, reverse)
        return


class LimitsContainer:
    """Holds all of the attributes related to the limits dictionary.

    This class is used to store the information when methods from Limits class
    are called by the SinglePlot class.

    Attributes:
        xlims,ylims (len-2 list of floats): REQUIRED. x,y limits for specific plot.
            If ``xscale,yscale == log``, the xlims/ylims must be log10 of the actual desired values.
            Ex. for 1e4 to 1e8, xlims would be ``[4.0, 8.0]``.
        dx,dy (float): Required. Spacing of x and y ticks.
            If ``xscale,yscale == log``, then dx,dy is a log10 value. See examples.
        xscale,yscale (str): Choices are `lin` for linear spacing or `log`
            for log (base 10) spacing.
            Default is `lin`.
        reverse_x_axis,reverse_y_axis (bool, optional): Reverses the tick
            marks on the x,y axis.
            Default is ``False``.

    """
    def __init__(self):
        pass


class Legend:
    """Legend contains the methods inherited by SinglePlot.

    These methods are used by SinglePlot class to add configuration
    information pertaining to the legend dict involved in plot creation.

    All attributes are appended to the LegendContainer class.
    The LegendContainer class is contained in the legend attribute
    in the SinglePlot class.

    """
    def add_legend(self, labels=None, **kwargs):
        """Specify legend for a plot.

        Adds labels and basic legend specifications for specific plot.

        For the optional Args, refer to
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
        for more information.

        # TODO: Add legend capabilities for Loss/Gain plots. This is possible
            using the return_fig_ax kwarg in the main plotting function.

        Args:
            labels (list of str): String representing each item in plot that
                will be added to the legend.

        Keyword Arguments:
            loc (str, int, len-2 list of floats, optional): Location of
                legend. See matplotlib documentation for more detail.
                Default is None.
            bbox_to_anchor (2-tuple or 4-tuple of floats, optional): Specify
                position and size of legend box. 2-tuple will specify (x,y)
                coordinate of part of box specified with `loc` kwarg.
                4-tuple will specify (x, y, width, height). See matplotlib
                documentation for more detail.
                Default is None.
            size (float, optional): Set size of legend using call to `prop`
                dict in legend call. See matplotlib documentaiton for more
                detail. Default is None.
            ncol (int, optional): Number of columns in the legend.
            Note: Other kwargs are available. See:
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html

        """
        if 'size' in kwargs:
            if 'prop' not in kwargs:
                kwargs['prop'] = {'size': kwargs['size']}
            else:
                kwargs['prop']['size'] = kwargs['size']
            del kwargs['size']
        self.legend.add_legend = True
        self.legend.legend_labels = labels
        self.legend.legend_kwargs = kwargs
        return


class LegendContainer:
    """Holds all of the attributes related to the add_legend dictionary.

    This class is used to store the information when methods from Legend class
    are called by the SinglePlot class.

    Attributes:
        labels (list of str): String representing each item in plot that
            will be added to the legend.
        legend_kwargs (dict): Stores kwargs for ``ax.legend()``.

    """
    def __init__(self):
        pass


class Extra:
    """ Extra contains the methods inherited by SinglePlot.

    These methods are used by SinglePlot class to add configuration
    information pertaining to the extra dict involved in plot creation.

    All attributes are appended to the ExtraContainer class.
    The ExtraContainer class is contained in the extra attribute
    in the SinglePlot class.

    The associated methods in Extra class as well as the attributes
    in the ExtraContainer class detail the options related to
    extra customization of plots.

    """
    def grid(self, grid=True):
        """Add gridlines to plot.

        Adds gridlines for specific plot.

        Args:
            grid (bool, optional): If True, add grid to specific plot.
                Default is True.

        """
        self.extra.add_grid = grid
        return

    def set_contour_vals(self, vals):
        """Set contour values related to Waterfall plots.

        Customize contour values related to Waterfall plots. This functionality
        may be added to the other plots in the future, but, so far, it seems
        as if that is not needed.

        Args:
            vals (list of floats): Set contour values to this list.
                Default is [0.,10,20,50,100,200,500,1000,3000,1e10].

        """
        self.extra.contour_vals = vals
        return

    def set_snr_contour_value(self, val):
        """Set snr value for which singular contours are shown.

        Set a specific contour value to plot. In Waterfall plots this will
        add a line contour in white. For Horizon and Ratio plots, this value
        will override the SNR_CUT option and use this value.

        Args:
            val (float): Set snr contour value specific to each style of plot.
                See above.

        """
        self.extra.snr_contour_value = val
        return

    def set_order_contour_lines(self, lines=True):
        """Toggle showing order of magnitude lines in Ratio plots.

        This will show dashed lines for each order of magnitude contour in Ratio plots.

        Args:
            lines (bool, optional): Add order of magnitude contour lines to Ratio
                plots. Default is True. Specifically, when calling this function,
                the default is True because the assumption is that the user
                desires these lines if they call this function.
                Without calling this function, the default is False.

        """
        self.extra.order_contour_lines = lines
        return

    def show_loss_gain(self, show=True):
        """Toggle showing loss/gain plots.

        This will keep or remove the loss/gain plots when viewing a
        ratio plot.

        Args:
            show (bool, optional): If True, keep loss/gain. Default is True.

        """
        self.extra.show_loss_gain = show
        return

    def ratio_comp_value(self, val):
        """Change the comparison value limit for Ratio plots.

        The ratio plots are only shown when both configurations are above
        a detectable limit set by SNR_CUT. This function will override that value.

        Args:
            val (float): Sets minimum snr where Ratio contours are shown.

        """
        self.extra.ratio_comp_value = val
        return


class ExtraContainer:
    """Holds all of the attributes related to the extra dictionary.

    This class is used to store the information when methods from Extra class
    are called by the SinglePlot class.

    Attributes:
        add_grid (bool, optional): Adds gridlines to plots. Overrides setting from "general".
            Default is True.
        contour_vals (list of floats, optional): Provides contour values for Waterfall plot only.
            Default is [0.,10,20,50,100,200,500,1000,3000,1e10].
        snr_contour_value (float, optional): Adds contour for a specific value for
            Waterfall plot. Color is white.
            For Ratio and Horizon plots, this overrides the SNR_CUT for a custom value.
        order_contour_lines (bool, optional): This only applies to ratio plots.
            This will show contour lines at each order of magnitude ratio. Default is False.
        show_loss_gain (bool, optional): Toggle loss/gain contours on and off.
            Only applies to Ratio plots. Default is True.
        ratio_comp_value (float, optional): This sets the minimum SNR necessary
            for both configurationsin a ratio plot to show the ratio contours.
            (rho_1>ratio_comp_value & rho_2>ratio_comp_value) in order to display the contour.
            Default is same value as loss/gain contour (which defaults to SNR_CUT).

    """
    def __init__(self):
        pass


class DataImport:
    """ DataImport contains the methods inherited by SinglePlot.

    These methods are used by SinglePlot class to add data for each plot.
    This class allows for file import as well as indexing to files that
    have already been imported.

    """
    def add_dataset(self, name=None, label=None,
                    x_column_label=None, y_column_label=None, index=None, control=False):
        """Add a dataset to a specific plot.

        This method adds a dataset to a plot. Its functional use is imperative
        to the plot generation. It handles adding new files as well
        as indexing to files that are added to other plots.

        All Args default to None. However, these are note the defaults
        in the code. See DataImportContainer attributes for defaults in code.

        Args:
            name (str, optional): Name (path) for file.
                Required if reading from a file (at least one).
                Required if file_name is not in "general". Must be ".txt" or ".hdf5".
                Can include path from working directory.
            label (str, optional): Column label in the dataset corresponding to desired SNR value.
                Required if reading from a file (at least one).
            x_column_label/y_column_label (str, optional): Column label from input file identifying
                x/y values. This can override setting in "general". Default
                is `x`/`y`.
            index (int, optional): Index of plot with preloaded data.
                Required if not loading a file.
            control (bool, optional): If True, this dataset is set to the control.
                This is needed for Ratio plots. It sets
                the baseline. Default is False.

        Raises:
            ValueError: If no options are passes. This means no file indication
                nor index.

        """
        if name is None and label is None and index is None:
            raise ValueError("Attempting to add a dataset without"
                             + "supplying index or file information.")

        if index is None:
            trans_dict = DataImportContainer()
            if name is not None:
                trans_dict.file_name = name

            if label is not None:
                trans_dict.label = label

            if x_column_label is not None:
                trans_dict.x_column_label = x_column_label

            if y_column_label is not None:
                trans_dict.y_column_label = y_column_label

            if control:
                self.control = trans_dict
            else:
                # need to append file to file list.
                if 'file' not in self.__dict__:
                    self.file = []
                self.file.append(trans_dict)
        else:
            if control:
                self.control = DataImportContainer()
                self.control.index = index

            else:
                # need to append index to index list.
                if 'indices' not in self.__dict__:
                    self.indices = []

                self.indices.append(index)
        return


class DataImportContainer:
    """Holds all of the attributes related to a dataset.

    This class holds file information for each file imported as an added
    dataset. It does not take index information, which is added to the GeneralContainer class.

    The attributes associated with DataImportContainer can override GeneralContainer values.
    However, the GeneralContainer is used to set these values if it applies to
    all or most of the plots.

    Attributes:
        file_name (str, optional): Name (path) for file.
            Required if reading from a file (at least one).
            Required if file_name is not in "general". Must be ".txt" or ".hdf5".
            Can include path from working directory.
        label (str, optional): Column label in the dataset corresponding to desired SNR value.
            Required if reading from a file (at least one).
        x_column_label/y_column_label (str, optional): Column label from input file identifying
            x/y values. This can override setting in "general". Default
            is `x`/`y`.

    """
    def __init__(self):
        pass


class SinglePlot(Label, Limits, Legend, Extra, DataImport):
    """Add information for each singular plot.

    Each plot has special information stored in its dictionary.
    See above documentation for this information.

    SinglePlot inherits methods from Label, Limits, Legend, Extra, and DataImport.
    It uses the methods to store values in the respective Container classes.

    Attributes:
        type (str): Which type of plot. Currently supporting `Ratio`, `Watefall`,
            `Horizon`.
        file (list of obj): List of DataImportContainer objects with information on which
            dataset to use.
        label (obj): LabelContainer holding informaiton pertaining to labels of the plot.
            Default is empty container or ``{}``.
        limits (obj): LimitsContainer holding information pertaining to limits of the plot.
            Default is empty container or ``{}``.
        legend (obj): LegendContainer holding information pertaining to legend for the plot.
            Default is empty container or ``{}``.
        extra (obj): ExtraContainer holding information pertaining to extra
            information for the plot.
            Default is empty container or ``{}``.

    """
    def __init__(self):
        self.label = LabelContainer()
        self.limits = LimitsContainer()
        self.legend = LegendContainer()
        self.extra = ExtraContainer()

    def set_plot_type(self, plot_type):
        """Set type for each plot.

        This is a mandatory call for each plot to tell the program which plot is desired.

        Args:
            plot_type (str): Type of specific plot.
                Currently supporting `Ratio`, `Watefall`, `Horizon`.

        """
        self.plot_type = plot_type
        return

    def set_colormap(self, cmap):
        """Set colormap for Ratio plots.

        Change the colormap.

        Arguments:
            cmap (str): String representing the colormap from predefined
                python colormaps.

        """
        self.colormap = cmap
        return

    def add_comparison(self, comp_list):
        if 'comparisons' not in self.__dict__:
            self.comparisons = []

        self.comparisons.append(comp_list)


class Figure:
    def savefig(self, output_path, **kwargs):
        """Save figure during generation.

        This method is used to save a completed figure during the main function run.
        It represents a call to ``matplotlib.pyplot.fig.savefig``.

        # TODO: Switch to kwargs for matplotlib.pyplot.savefig

        Args:
            output_path (str): Relative path to the WORKING_DIRECTORY to save the figure.

        Keyword Arguments:
            dpi (int, optional): Dots per inch of figure. Default is 200.
            Note: Other kwargs are available. See:
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html

        """
        self.figure.save_figure = True
        self.figure.output_path = output_path
        self.figure.savefig_kwargs = kwargs
        return

    def show(self):
        """Show figure after generation.

        This method is used to show a completed figure after the main function run.
        It represents a call to ``matplotlib.pyplot.show``.

        """
        self.figure.show_figure = True
        return

    def set_fig_size(self, width, height=None):
        """Set the figure size in inches.

        Sets the figure size with a call to fig.set_size_inches.
        Default in code is 8 inches for each.

        Args:
            width (float): Dimensions for figure width in inches.
            height (float, optional): Dimensions for figure height in inches. Default is None.

        """
        self.figure.figure_width = width
        self.figure.figure_height = height
        return

    def set_spacing(self, space):
        """Set the figure spacing.

        Sets whether in general there is space between subplots.
        If all axes are shared, this can be `tight`. Default in code is `wide`.

        The main difference is the tick labels extend to the ends if space==`wide`.
        If space==`tight`, the edge tick labels are cut off for clearity.

        Args:
            space (str): Sets spacing for subplots. Either `wide` or `tight`.

        """
        self.figure.spacing = space
        if 'subplots_adjust_kwargs' not in self.figure.__dict__:
            self.figure.subplots_adjust_kwargs = {}
        if space == 'wide':
            self.figure.subplots_adjust_kwargs['hspace'] = 0.3
            self.figure.subplots_adjust_kwargs['wspace'] = 0.3
        else:
            self.figure.subplots_adjust_kwargs['hspace'] = 0.0
            self.figure.subplots_adjust_kwargs['wspace'] = 0.0

        return

    def subplots_adjust(self, **kwargs):
        """Adjust subplot spacing and dimensions.

        Adjust bottom, top, right, left, width in between plots, and height in between plots
        with a call to ``plt.subplots_adjust``.

        See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
        for more information.

        Keyword Arguments:
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

        """
        prop_default = {
            'bottom': 0.1,
            'top': 0.85,
            'right': 0.9,
            'left': 0.12,
            'hspace': 0.3,
            'wspace': 0.3,
        }

        if 'subplots_adjust_kwargs' in self.figure.__dict__:
            for key, value in self.figure.subplots_adjust_kwargs.items():
                prop_default[key] = value

        for prop, default in prop_default.items():
            kwargs[prop] = kwargs.get(prop, default)

        self.figure.subplots_adjust_kwargs = kwargs
        return

    def _set_fig_label(self, which, label, x, y, **kwargs):
        setattr(self.figure, 'fig_' + which + '_label', label)
        setattr(self.figure, 'fig_' + which + '_label_x', x)
        setattr(self.figure, 'fig_' + which + '_label_y', y)
        setattr(self.figure, 'fig_' + which + '_label_kwargs', kwargs)

    def set_fig_x_label(self, xlabel, **kwargs):
        """Set overall figure x.

        Set label for x axis on overall figure. This is not for a specific plot.
        It will place the label on the figure at the left with a call to ``fig.text``.

        Args:
            xlabel (str): xlabel for entire figure.

        Keyword Arguments:
            x/y (float, optional): The x/y location of the text in figure coordinates.
                Defaults are 0.01 for x and 0.51 for y.
            horizontalalignment/ha (str, optional): The horizontal alignment of
                the text relative to (x, y). Optionas are 'center', 'left', or 'right'.
                Default is 'center'.
            verticalalignment/va (str, optional): The vertical alignment of the text
                relative to (x, y). Optionas are 'top', 'center', 'bottom',
                or 'baseline'. Default is 'center'.
            fontsize/size (int): The font size of the text. Default is 20.
            rotation (float or str): Rotation of label. Options are angle in degrees,
                `horizontal`, or `vertical`. Default is `vertical`.
            Note: Other kwargs are available.
                See https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.figtext

        """
        prop_default = {
            'x': 0.01,
            'y': 0.51,
            'fontsize': 20,
            'rotation': 'vertical',
            'va': 'center',
        }

        for prop, default in prop_default.items():
            kwargs[prop] = kwargs.get(prop, default)

        self._set_fig_label('x', xlabel, **kwargs)
        return

    def set_fig_y_label(self, ylabel, **kwargs):
        """Set overall figure y.

        Set label for y axis on overall figure. This is not for a specific plot.
        It will place the label on the figure at the left with a call to ``fig.text``.

        Args:
            ylabel (str): ylabel for entire figure.

        Keyword Arguments:
            x/y (float, optional): The x/y location of the text in figure coordinates.
                Defaults are 0.45 for x and 0.02 for y.
            horizontalalignment/ha (str, optional): The horizontal alignment of
                the text relative to (x, y). Optionas are 'center', 'left', or 'right'.
                Default is 'center'.
            verticalalignment/va (str, optional): The vertical alignment of the text
                relative to (x, y). Optionas are 'top', 'center', 'bottom',
                or 'baseline'. Default is 'top'.
            fontsize/size (int): The font size of the text. Default is 20.
            rotation (float or str): Rotation of label. Options are angle in degrees,
                `horizontal`, or `vertical`. Default is `horizontal`.
            Note: Other kwargs are available.
                See https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.figtext

        """
        prop_default = {
            'x': 0.45,
            'y': 0.02,
            'fontsize': 20,
            'rotation': 'horizontal',
            'ha': 'center',
        }

        for prop, default in prop_default.items():
            kwargs[prop] = kwargs.get(prop, default)

        self._set_fig_label('y', ylabel, **kwargs)
        return

    def set_fig_title(self, title, **kwargs):
        """Set overall figure title.

        Set title for overall figure. This is not for a specific plot.
        It will place the title at the top of the figure with a call to ``fig.suptitle``.

        Args:
            title (str): Figure title.

        Keywork Arguments:
            x/y (float, optional): The x/y location of the text in figure coordinates.
                Defaults are 0.5 for x and 0.98 for y.
            horizontalalignment/ha (str, optional): The horizontal alignment of
                the text relative to (x, y). Optionas are 'center', 'left', or 'right'.
                Default is 'center'.
            verticalalignment/va (str, optional): The vertical alignment of the text
                relative to (x, y). Optionas are 'top', 'center', 'bottom',
                or 'baseline'. Default is 'top'.
            fontsize/size (int, optional): The font size of the text. Default is 20.

        """
        prop_default = {
            'fontsize': 20,
        }

        for prop, default in prop_default.items():
            kwargs[prop] = kwargs.get(prop, default)

        self.figure.fig_title = title
        self.figure.fig_title_kwargs = kwargs
        return

    def set_colorbar(self, plot_type, **kwargs):
        """Setup colorbar for specific type of plot.

        Specify a plot type to customize its corresponding colorbar in the figure.

        See the ColorbarContainer class attributes for more specific explanations.

        Args:
            plot_type (str): Type of plot to adjust. e.g. `Ratio`
            label (str, optional): Label for colorbar. Default is None.
            label_fontsize (int, optional): Fontsize for colorbar label. Default is None.
            ticks_fontsize (int, optional): Fontsize for colorbar tick labels. Default is None.
            pos (int, optional): Set a position for colorbar based on defaults. Default is None.
            colorbar_axes (len-4 list of floats): List for custom axes placement of the colorbar.
                See fig.add_axes from matplotlib.
                url: https://matplotlib.org/2.0.0/api/figure_api.html

        Raises:
            UserWarning: User calls set_colorbar without supplying any Args.
                This will not stop the code.

        """
        prop_default = {
            'cbar_label': None,
            'cbar_ticks_fontsize': 15,
            'cbar_label_fontsize': 20,
            'cbar_axes': [],
            'cbar_ticks': [],
            'cbar_tick_labels': [],
            'cbar_pos': 'use_default',
            'cbar_label_pad': None,
        }

        for prop, default in prop_default.items():
            kwargs[prop] = kwargs.get(prop[5:], default)

            if prop[5:] in kwargs:
                del kwargs[prop[5:]]

        if 'colorbars' not in self.figure.__dict__:
            self.figure.colorbars = {}

        self.figure.colorbars[plot_type] = kwargs
        return


class FigureContainer:
    """Holds all of the attributes related to the figure dictionary.

    This class is used to store the information when methods from Figure class
    are called by the MainContainer class.

    Attributes:
        show_figure (bool, optional): Use ``plt.show()`` function from matplotlib to show plot.
            Do not use this in Jupyter Notebook. Use the magic command: ``%matplotlib inline``.
            Default is False.
        save_figure (bool, optional): Use ``fig.savefig()`` function from matplotlib
            to save figure. Default is False.
        output_path (str, optional): Path from the working directory to save the figure,
            including file name and extension. Required if ``save_figure == True``.
        savefig_kwargs (dict, optional): Additional kwargs for ``fig.savefig()``.
        subplots_adjust_kwargs (dict, optional): Kwargs for sizing with ``plt.subplots_adjust()``.
        spacing (str, optional): This sets the general spacing of the plot configuration.
            Choices are `wide` or `tight` (``hspace = wspace = 0.0``). Tight spacing will cut
            off the outer labels on each axis due to axes contacting each other.
            Default is `wide`.
        fig_x_label/fig_y_label (str, optional): Overall figure label on the left and bottom.
            Called with ``fig.text()`` from matplotlib.
        fig_x_label_kwargs, fig_y_label_kwargs (dict, optional):
            Kwargs for ``fig.text()``.
        fig_title (str, optional): Overall figure title. Produced with ``fig.suptitle()``
            from matplotlib.
        fig_title_kwargs (dict, optional): Kwargs for `fig.suptitle``.
        figure_width/figure_height (float, optional): Dimensions of the figure set with
        ``fig.set_size_inches()`` from matplotlib. Default is 8.0 for each.
        colorbars (obj): ColorbarContainer object carrying information for the colorbars.

    """
    def __init__(self):
        pass


class General:
    """ General contains the methods inherited by MainContainer.

    These methods are used by MainContainer class to add information that applies
    to all plots. Many of its settings pertaining to plots
    can be overriden with methods in SinglePlot.

    """
    def set_snr_cut(self, cut_val):
        """Set the SNR_CUT value.

        Sets the SNR cut value for detectability. In code, it defaults to 5.
        This can also be viewed as the contour value desired if it is not a cut.

        Args:
            cut_val (float): Sets SNR_CUT variable.

        """
        self.general.SNR_CUT = cut_val
        return

    def set_working_directory(self, wd):
        """Set the WORKING_DIRECTORY variable.

        Sets the WORKING_DIRECTORY. The code will then use all paths as relative paths
        to the WORKING_DIRECTORY. In code default is current directory.

        Args:
            wd (str): Absolute or relative path to working directory.

        """
        self.general.WORKING_DIRECTORY = wd
        return

    def switch_backend(self, string='agg'):
        """Switch the matplotlib backend.

        Changes the backend for matplotlib. See https://matplotlib.org/faq/usage_faq.html
        for more infomation.
        This is specifcally important if generating plots in parallel or on devices without
        a graphical user interface.

        Args:
            string (str, optional): String indicating the new backend. Default is `agg`.

        """
        self.general.switch_backend = string
        return

    def set_all_file_name(self, name):
        """Add general file name.

        This sets the file name for all the plots. This can be overriden
        by specific plots.

        Args:
            name (str): String indicating the relative path to the file.

        """
        self.general.file_name = name
        return

    def set_all_file_column_labels(self, xlabel=None, ylabel=None):
        """Indicate general x,y column labels.

        This sets the general x and y column labels into data files for all plots.
        It can be overridden for specific plots.

        Args:
            xlabel/ylabel (str, optional): String indicating column label for x,y values
                into the data files. Default is None.

        Raises:
            UserWarning: If xlabel and ylabel are both not specified,
                The user will be alerted, but the code will not stop.

        """
        if xlabel is not None:
            self.general.x_column_label = xlabel
        if ylabel is not None:
            self.general.y_column_label = ylabel
        if xlabel is None and ylabel is None:
            warnings.warn("is not specifying x or y lables even"
                          + "though column labels function is called.", UserWarning)
        return

    def _set_all_lims(self, which, lim, d, scale, fontsize=None):
        """Set limits and ticks for an axis for whole figure.

        This will set axis limits and tick marks for the entire figure.
        It can be overridden in the SinglePlot class.

        Args:
            which (str): The indicator of which part of the plots
                to adjust. This currently handles `x` and `y`.
            lim (len-2 list of floats): The limits for the axis.
            d (float): Amount to increment by between the limits.
            scale (str): Scale of the axis. Either `log` or `lin`.
            fontsize (int, optional): Set fontsize for associated axis tick marks.
                Default is None.

        """

        setattr(self.general, which + 'lims', lim)
        setattr(self.general, 'd' + which, d)
        setattr(self.general, which + 'scale', scale)

        if fontsize is not None:
            setattr(self.general, which + '_tick_label_fontsize', fontsize)
        return

    def set_all_xlims(self, xlim, dx, xscale, fontsize=None):
        """Set limits and ticks for x axis for whole figure.

        This will set x axis limits and tick marks for the entire figure.
        It can be overridden in the SinglePlot class.

        Args:
            xlim (len-2 list of floats): The limits for the axis.
            dx (float): Amount to increment by between the limits.
            xscale (str): Scale of the axis. Either `log` or `lin`.
            fontsize (int, optional): Set fontsize for x axis tick marks.
                Default is None.

        """
        self._set_all_lims('x', xlim, dx, xscale, fontsize)
        return

    def set_all_ylims(self, ylim, dy, yscale, fontsize=None):
        """Set limits and ticks for y axis for whole figure.

        This will set y axis limits and tick marks for the entire figure.
        It can be overridden in the SinglePlot class.

        Args:
            ylim (len-2 list of floats): The limits for the axis.
            dy (float): Amount to increment by between the limits.
            yscale (str): Scale of the axis. Either `log` or `lin`.
            fontsize (int, optional): Set fontsize for y axis tick marks.
                Default is None.

        """
        self._set_all_lims('y', ylim, dy, yscale, fontsize)
        return

    def set_all_ticklabel_fontsize(self, fontsize):
        """Set tick label fontsize for both axis in entire figure.

        This will set x and y axis tick label fontsize for the entire figure.
        It can be overridden in the SinglePlot class.

        Args:
            fontsize (int): Set fontsize for x and y axis tick marks.

        """
        self.general.tick_label_fontsize = fontsize
        return

    def set_grid(self, grid=True):
        """Add gridlines to all plots in figure.

        Adds gridlines for all plots in figure. Can be overridden with SinglePlot class.

        Args:
            grid (bool, optional): If True, add grid to specific plot.
                Default is True.

        """
        self.general.add_grid = grid
        return

    def reverse_axis(self, axis_to_reverse):
        """Reverse an axis in all figure plots.

        This will reverse the tick marks on an axis for each plot in the figure.
        It can be overridden in SinglePlot class.

        Args:
            axis_to_reverse (str): Axis to reverse. Supports `x` and `y`.

        Raises:
            ValueError: The string representing the axis to reverse is not `x` or `y`.

        """
        if axis_to_reverse.lower() == 'x':
            self.general.reverse_x_axis = True
        if axis_to_reverse.lower() == 'y':
            self.general.reverse_y_axis = True
        if axis_to_reverse.lower() != 'x' or axis_to_reverse.lower() != 'y':
            raise ValueError('Axis for reversing needs to be either x or y.')
        return


class GeneralContainer:
    """Holds all of the attributes related to the general dictionary.

    This class is used to store the information when methods from General class
    are called by the MainContainer class.

    Args:
        nrows, ncols (int): Number of rows/columns of plots present in the figure.
            These determine the number of plots to create.
        sharex, sharey (bool, optional): Applies sharex, sharey as used in ``plt.subplots()``.
            See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
            Default is True.

    Attributes:
        num_rows, num_cols (int): Number of rows/columns of plots present in the figure.
        xlims/ylims (len-2 list of floats): Sets the x,y limits of the plots.
            Can be overridden for specific plots. If xscale/yscale == log,
            the xlims/ylims must be log10 of the actual desired values.
            Ex. for 1e4 to 1e8, xlims would be [4.0, 8.0].
            Required if xlims/ylims are not specified for every specific plot.
        dx, dy (float): Spacing of x and y ticks. If xscale/yscale == log,
            the dx/dy is a log10 value. See examples. Required if limits are not
            set for each specific plot.
        xscale/yscale (str, optional): Choices are lin for linear spacing or log for
            log (base 10) spacing. Default is `lin`.
        tick_label_fontsize (float, optional): Sets fontsize for both x and y
            tick labels on the plots. This can be overridden for individual plots.
            Default is 14.
        x_tick_label_fontsize/y_tick_label_fontsize (float, optional): Sets fontsize fot
            x/y tick labels. This overrides `tick_label_fontsize` and
            can be overridden for specific plots. Default is 14.
        WORKING_DIRECTORY (str, optional): Working directory for file export and retrieval.
            Default is ``'.'``.
        SNR_CUT (float, optional): The SNR cut for a detectable signal. Usually between 1-10.
            Default is 5.0.
        switch_backend (str, optional): Use for switching backend of matplotlib.
            Use if running codes in parallel. Typical string value is "agg".
        file_name (str, optional): File name for input SNR grids. Can be overridden by
            specific plot. Required if no file names are given in "plot_info" dictionaries.
        x_column_label/y_column_label (str, optional): Column label from input file
            identifying x/y values. This can be overrided in the file_dicts for specific files.
            Default is `x`/`y`.
        sharex, sharey (bool, optional): Applies sharex, sharey as used in ``plt.subplots()``.
            See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
            Default is True.
        add_grid (bool, optional): Adds gridlines to all plots. Can be overridden
            for specific plots. Default is True.
        reverse_x_axis, reverse_y_axis (bool, optional): Reverses the tick marks on the x/y axis.
            Can be overridden for specific plots. Default is False.

    """
    def __init__(self, nrows, ncols, sharex=True, sharey=True):
        self.num_rows = nrows
        self.num_cols = ncols
        self.sharex = sharex
        self.sharey = sharey


class MainContainer(General, Figure):
    """Main class for creating input dictionary to ``make_plot.py``.

    This class creates a pythonic way to add information to the input dictionary
    to ``make_plot.py``. It creates and can read out this dictionary.

    MainContainer inherits methods from General class so that it can add
    general dictionary information to GeneralContainer class

    The ``__init__`` function is similar to a call to ``plt.subplots()``.

    # TODO: make default on sharing to False?

    Args:
        nrows/ncols (int): Number of rows/columns of plots present in the figure.
            This is passed into the general dictionary. These determine the number of
            plots to create.
        sharex/sharey (bool, optional): Applies sharex, sharey as used in ``plt.subplots()``.
            See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
            Default is True.
        print_input (bool, optional): If True, print the dictionary created by MainContainer
            class after it is completed.

    Attributes:
        total_plots (int): `nrows`*`ncols`.
        general (obj): GeneralContainer class for holding information for general dictionary.
        ax (obj or list of objects): SinglePlot object for holding information pertaining
            to specific plots. If there is more than one plot in the figure, this will
            be a list of SinglePlot objects of ``len=total_plots``.
        print_input (bool, optional): If True, print the dictionary created by MainContainer
            class after it is completed.

    """
    def __init__(self, nrows, ncols, sharex=True, sharey=True, print_input=False):
        self.print_input = print_input
        self.total_plots = nrows*ncols

        self.general = GeneralContainer(nrows, ncols, sharex, sharey)
        self.figure = FigureContainer()

        if self.total_plots > 1:
            self.ax = [SinglePlot() for i in range(self.total_plots)]
        else:
            self.ax = SinglePlot()

    def return_dict(self):
        """Output dictionary for ``make_plot.py`` input.

        Iterates through the entire MainContainer class turning its contents
        into dictionary form. This dictionary becomes the input for ``make_plot.py``.

        If `print_input` attribute is True, the entire dictionary will be printed
        prior to returning the dicitonary.

        Returns:
            - **output_dict** (*dict*): Dicitonary for input into ``make_plot.py``.

        """
        output_dict = {}
        output_dict['general'] = self._iterate_through_class(self.general.__dict__)
        output_dict['figure'] = self._iterate_through_class(self.figure.__dict__)

        if self.total_plots > 1:
            trans_dict = ({
                           str(i): self._iterate_through_class(axis.__dict__) for i, axis
                          in enumerate(self.ax)})
            output_dict['plot_info'] = trans_dict

        else:
            output_dict['plot_info'] = {'0': self._iterate_through_class(self.ax.__dict__)}

        if self.print_input:
            print(output_dict)
        return output_dict

    def _iterate_through_class(self, class_dict):
        """Recursive function for output dictionary creation.

        Function will check each value in a dictionary to see if it is a
        class, list, or dictionary object. The idea is to turn all class objects into
        dictionaries. If it is a class object it will pass its ``class.__dict__``
        recursively through this function again. If it is a dictionary,
        it will pass the dictionary recursively through this functin again.

        If the object is a list, it will iterate through entries checking for class
        or dictionary objects and pass them recursively through this function.
        This uses the knowledge of the list structures in the code.

        Args:
            class_dict (obj): Dictionary to iteratively check.

        Returns:
            Dictionary with all class objects turned into dictionaries.

        """
        output_dict = {}
        for key in class_dict:
            val = class_dict[key]
            try:
                val = val.__dict__
            except AttributeError:
                pass

            if type(val) is dict:
                val = self._iterate_through_class(val)

            if type(val) is list:
                temp_val = []
                for val_i in val:
                    try:
                        val_i = val_i.__dict__
                    except AttributeError:
                        pass

                    if type(val_i) is dict:
                        val_i = self._iterate_through_class(val_i)
                    temp_val.append(val_i)
                val = temp_val

            output_dict[key] = val

        return output_dict
