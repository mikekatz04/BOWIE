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
    """ Label contains the methods inherited by SinglePlot.

    These methods are used by SinglePlot class to add configuration
    information pertaining to the labels dict involved in plot creation.

    All attributes are appended to the LabelContainer class.
    The LabelContainer class is contained in the label attribute
    in the SinglePlot class.

    """

    def _set_labels(self, which, label, fontsize=None):
        """Private method for setting labels.

        Args:
            which (str): The indicator of which part of the plots
                to adjust. This currently handles `xlabel`/`ylabel`,
                and `title`.
            label (str): The label to be added.
            fontsize (int, optional): Fontsize for associated label. Default
                is None.

        """
        setattr(self.label, which, label)

        if fontsize is not None:
            setattr(self.label, which + '_fontsize', fontsize)
        return

    def set_xlabel(self, label, fontsize=None):
        """Set xlabel for plot.

        Similar to matplotlib, this will set the xlabel for the specific plot.

        Args:
            label (str): The label to be added.
            fontsize (int, optional): Fontsize for associated label. Default
                is None.

        """
        self._set_label('xlabel', label, fontsize)
        return

    def set_ylabel(self, label, fontsize=None):
        """Set ylabel for plot.

        Similar to matplotlib, this will set the ylabel for the specific plot.

        Args:
            label (str): The label to be added.
            fontsize (int, optional): Fontsize for associated label. Default
                is None.

        """
        self._set_labels('ylabel', label, fontsize)
        return

    def set_title(self, title, fontsize=None):
        """Set title for plot.

        Similar to matplotlib, this will set the title for the specific plot.

        Args:
            title (str): The title to be added.
            fontsize (int, optional): Fontsize for associated title. Default
                is None.

        """
        self._set_labels('title', title, fontsize)
        return

    def set_labels_fontsize(self, fontsize):
        """Set fontsize for both x and y labels.

        This will adjust the font size of both x and y labels for a
        specific plot.

        Args:
            fontsize (int): Fontsize for associated label.

        """
        for which in ['xlabel', 'ylabel']:
            setattr(self.label, which + '_fontsize', fontsize)
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
        x_tick_label_fontsize/y_tick_label_fontsize (float, optional): Sets fontsize for x,y tick labels for specific
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
    """ Legend contains the methods inherited by SinglePlot.

    These methods are used by SinglePlot class to add configuration
    information pertaining to the legend dict involved in plot creation.

    All attributes are appended to the LegendContainer class.
    The LegendContainer class is contained in the legend attribute
    in the SinglePlot class.

    """
    def legend(self, labels, loc=None, bbox_to_anchor=None, size=None):
        """Specify legend for a plot.

        Adds labels and basic legend specifications for specific plot.

        For the optional Args, refer to
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
        for more information.

        # TODO: Add legend capabilities for Loss/Gain plots. This is possible
            using the return_fig_ax kwarg in the main plotting function.

        # TODO: Allow the call to legend to take all legend kwargs, therefore
            just passing a dictionary of kwargs straight to legend.

        Args:
            labels (list of str): String representing each item in plot that
                will be added to the legend.

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

        """
        self.legend.labels = labels
        if loc is not None:
            self.legend.loc = loc

        if bbox_to_anchor is not None:
            self.legend.bbox_to_anchor = bbox_to_anchor

        if size is not None:
            self.legend.size = size
        return


class LegendContainer:
    """Holds all of the attributes related to the legend dictionary.

    This class is used to store the information when methods from Legend class
    are called by the SinglePlot class.

    Attributes:
        labels (list of str): String representing each item in plot that
            will be added to the legend.
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
            detail. Default is 10.

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

    def ratio_contour_lines(self, lines=True):
        """Toggle showing order of magnitude lines in Ratio plots.

        This will show dashed lines for each order of magnitude contour in Ratio plots.

        Args:
            lines (bool, optional): Add order of magnitude contour lines to Ratio
                plots. Default is True. Specifically, when calling this function,
                the default is True because the assumption is that the user
                desires these lines if they call this function.
                Without calling this function, the default is False.

        """
        self.extra.ratio_contour_lines = lines
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
        ratio_contour_lines (bool, optional): This only applies to ratio plots.
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
    def add_dataset(self, name=None, label=None, x_column_label=None, y_column_label=None, index=None, control=False):
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
                This is needed for Ratio and Codetection plots. It sets
                the baseline. Default is False.

        Raises:
            ValueError: If no options are passes. This means no file indication
                nor index.

        """
        if name is None and label is None and index is None:
            raise ValueError("Attempting to add a dataset without supplying index or file information.")

        if index is None:
            trans_dict = DataImportContainer()
            if name is not None:
                trans_dict.name = name

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
        name (str, optional): Name (path) for file.
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
            `Horizon`, `Codetection1`, `Codetection2`, `Codetection3`.
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

    def set_type(self, plot_type):
        """Set type for each plot.

        This is a mandatory call for each plot to tell the program which plot is desired.

        Args:
            plot_type (str): Type of specific plot.
                Currently supporting `Ratio`, `Watefall`,
                `Horizon`, `Codetection1`, `Codetection2`, `Codetection3`.

        """
        self.type = plot_type
        return


class General:
    """ General contains the methods inherited by MainContainer.

    These methods are used by MainContainer class to add information that applies
    to all plots or the figure as a whole. Many of its settings pertaining to plots
    can be overriden with methods in SinglePlot.

    """
    def savefig(self, output_path, dpi=200):
        """Save figure during generation.

        This method is used to save a completed figure during the main function run.
        It represents a call to ``matplotlib.pyplot.fig.savefig``.

        # TODO: Switch to kwargs for matplotlib.pyplot.savefig

        Args:
            output_path (str): Relative path to the WORKING_DIRECTORY to save the figure.
            dpi (int, optional): Dots per inch of figure. Default is 200.

        """
        self.general.save_figure = True
        self.general.output_path = output_path
        self.general.dpi = dpi
        return

    def show(self):
        """Show figure after generation.

        This method is used to show a completed figure after the main function run.
        It represents a call to ``matplotlib.pyplot.show``.

        """
        self.general.show_figure = True
        return

    def snr_cut(self, cut_val):
        """Set the SNR_CUT value.

        Sets the SNR cut value for detectability. In code, it defaults to 5.
        This can also be viewed as the contour value desired if it is not a cut.

        Args:
            cut_val (float): Sets SNR_CUT variable.

        """
        self.general.SNR_CUT = cut_val
        return

    def working_directory(self, wd):
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

    def file_name(self, name):
        """Add general file name.

        This sets the file name for all the plots. This can be overriden
        by specific plots.

        Args:
            name (str): String indicating the relative path to the file.

        """
        self.general.file_name = name
        return

    def file_column_labels(self, xlabel=None, ylabel=None):
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
            warnings.warn("is not specifying x or y lables even though column labels function is called.", UserWarning)
        return

    def set_fig_size(self, width, height):
        """Set the figure size in inches.

        Sets the figure size with a call to fig.set_size_inches.
        Default in code is 8 inches for each.

        Args:
            width, height (float): Dimensions for figure size in inches.

        """
        self.general.figure_width = width
        self.general.figure_height = height
        return

    def spacing(self, space):
        """Set the figure spacing.

        Sets whether in general there is space between subplots.
        If all axes are shared, this can be `tight`. Default in code is `wide`.

        The main difference is the tick labels extend to the ends if space==`wide`.
        If space==`tight`, the edge tick labels are cut off for clearity.

        Args:
            space (str): Sets spacing for subplots. Either `wide` or `tight`.

        """
        self.general.spacing = space
        return

    def subplots_adjust(self, bottom=None, top=None, right=None, left=None, hspace=None, wspace=None):
        """Adjust subplot spacing and dimensions.

        Adjust bottom, top, right, left, width in between plots, and height in between plots
        with a call to ``plt.subplots_adjust``.

        See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
        for more information.

        Args:
            bottom (float, optional): Sets position of bottom of subplots in figure coordinates.
                Default is None.
            top (float, optional): Sets position of top of subplots in figure coordinates.
                Default is None.
            left (float, optional): Sets position of left edge of subplots in figure coordinates.
                Default is None.
            right (float, optional): Sets position of right edge of subplots in figure coordinates.
                Default is None.
            wspace (float, optional): The amount of width reserved for space between subplots,
               It is expressed as a fraction of the average axis width. Default is None.
            hspace (float, optional): The amount of height reserved for space between subplots,
               It is expressed as a fraction of the average axis width. Default is None.

        Raises:
            UserWarning: User calls subplots_adjust without supplying any Args.
                This will not stop the code.

        """
        if bottom is not None:
            self.general.adjust_figure_bottom = bottom
        if top is not None:
            self.general.adjust_figure_top = top
        if right is not None:
            self.general.adjust_figure_right = right
        if left is not None:
            self.general.adjust_figure_left = left
        if wspace is not None:
            self.general.adjust_figure_wspace = wspace
        if hspace is not None:
            self.general.adjust_figure_hspace = hspace

        if bottom is None and top is None and right is None and left is None and hspace is None and wspace is None:
            warnings.warn("Not specifying figure adjustments even though figure adjustment function is called.", UserWarning)

        return

    def set_fig_labels(self, xlabel=None, ylabel=None, fontsize=None):
        """Set overall figure x and y labels.

        Set label for x and y axis on overall figure. This is not for a specific plot.
        It will place the label on the figure at the bottom/left with a call to ``fig.text``.

        Args:
            xlabel/ylabel (str, optional): x/y label to placed for entire figure.
                Default is None.
            fontsize (int, optional): Set fontsize for associated label provided.
                Default is None.
        Raises:
            UserWarning: User calls set_fig_labels without supplying any Args.
                This will not stop the code.

        """
        if xlabel is not None:
            self.general.fig_x_label = xlabel
        if ylabel is not None:
            self.general.fig_y_label = ylabel
        if fontsize is not None:
            self.general.fig_label_fontsize = fontsize

        if xlabel is None and ylabel is None:
            warnings.warn("Not specifying x or y lables even though figure labels function is called.", UserWarning)
        return

    def set_fig_title(self, title, fontsize=None):
        """Set overall figure title.

        Set title for overall figure. This is not for a specific plot.
        It will place the title at the top of the figure with a call to ``fig.suptitle``.

        Args:
            title (str): Figure title.
            fontsize (int, optional): Set fontsize for figure title.
                Default is None.

        """
        self.general.fig_title = title
        if fontsize is not None:
            self.general.fig_title_fontsize = fontsize
        return

    def _set_fig_lims(self, which, lim, d, scale, fontsize=None):
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

    def set_fig_xlims(self, xlim, dx, xscale, fontsize=None):
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
        self._set_fig_lims('x', xlim, dx, xscale, fontsize)
        return

    def set_fig_ylims(self, ylim, dy, yscale, fontsize=None):
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
        self._set_fig_lims('y', ylim, dy, yscale, fontsize)
        return

    def set_ticklabel_fontsize(self, fontsize):
        """Set tick label fontsize for both axis in entire figure.

        This will set x and y axis tick label fontsize for the entire figure.
        It can be overridden in the SinglePlot class.

        Args:
            fontsize (int): Set fontsize for x and y axis tick marks.

        """
        self.general.tick_label_fontsize = fontsize
        return

    def grid(self, grid=True):
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

    def set_colorbar(self, plot_type, label=None, label_fontsize=None, ticks_fontsize=None, pos=None, colorbar_axes=None):
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
        if 'colorbars' not in self.general.__dict__:
            self.general.colorbars = {}

        trans_dict = ColorbarContainer()
        # TODO: Check this. only problem for codet
        if label is not None:
            trans_dict.label = label

        if label_fontsize is not None:
            trans_dict.label_fontsize = label_fontsize

        if ticks_fontsize is not None:
            trans_dict.ticks_fontsize = ticks_fontsize

        if pos is not None:
            trans_dict.pos = pos

        if colorbar_axes is not None:
            trans_dict.colorbar_axes = colorbar_axes

        self.general.colorbars[plot_type] = trans_dict

        if label is None and label_fontsize is None and ticks_fontsize is None and pos is None and colorbar_axes is None:
            warnings.warn("Call to set_colorbar without supplying Args.", UserWarning)
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
        show_figure (bool, optional): Use ``plt.show()`` function from matplotlib to show plot.
            Do not use this in Jupyter Notebook. Use the magic command: ``%matplotlib inline``.
            Default is False.
        save_figure (bool, optional): Use ``fig.savefig()`` function from matplotlib
            to save figure. Default is False.
        dpi (float, optional): dots per inch (dpi) for output image. Default is 200.
        output_path (str, optional): Path from the working directory to save the figure,
            including file name and extension. Required if ``save_figure == True``.
        file_name (str, optional): File name for input SNR grids. Can be overridden by
            specific plot. Required if no file names are given in "plot_info" dictionaries.
        x_column_label/y_column_label (str, optional): Column label from input file
            identifying x/y values. This can be overrided in the file_dicts for specific files.
            Default is `x`/`y`.
        sharex, sharey (bool, optional): Applies sharex, sharey as used in ``plt.subplots()``.
            See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
            Default is True.
        figure_width,figure_height (float, optional): Dimensions of the figure set with
        ``fig.set_size_inches()`` from matplotlib. Defaul is 8.0 for each.
        spacing (str, optional): This sets the general spacing of the plot configuration.
            Choices are `wide` or `tight` (``hspace = wspace = 0.0``). Tight spacing will cut
            off the outer labels on each axis due to axes contacting each other.
            Default is `wide`.
        adjust_figure_bottom, adjust_figure_top, adjust_figure_left, adjust_figure_right,
        adjust_wspace, adjust_hspace (float, optional): Adjust figure dimensions using
            ``plt.subplots_adjust()`` from matplotlib. See the matplotlib url for more info:
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
            Defaults are 0.1, 0.85, 0.12, 0.85, 0.3, 0.3, respectively.
        fig_x_label, fig_y_label (str, optional): Overall figure label on the left and bottom.
            Called with ``fig.text()`` from matplotlib.
        fig_title (str, optional): Overall figure title. Produced with ``fig.suptitle()``
            from matplotlib.
        fig_label_fontsize, fig_title_fontsize (float, optional): Fontsize corresponding
            to fig_x_label/fig_y_label and fig_title. Default is 20.
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


class ColorbarContainer:
    """Holds all of the attributes related to the colorbar dictionary.

    This class is used to store the information when ``set_colorbar`` is called
    by the MainContainer class.

    Attributes:
        label (str, optional): Label for the colorbar. Default for Waterfall is `rho_i`.
            Default for Ratio: `rho_i/rho_0`.
        ticks_fontsize (float, optional): Fontsize for tick marks on colorbar.
            The ticks are set based on the plot type. Default is 17.
        label_fontsize (float, optional): Colorbar label fontsize. Default is 20.
        pos (int, optional): Preset positions for the colorbars. 1 - top right, 2 - lower right,
            3 - top left (horizontal), 4 - top right (horizontal),
            5 - stretched to fill right side (effectively 1 & 2 combined).
            Defaults are Waterfall-1, Ratio -2. If plot is alone on figure, default is 5.
        colorbar_axes (len-4 list of floats, optional): List for custom axes placement
            of the colorbar. See ``fig.add_axes`` from matplotlib.
            url: https://matplotlib.org/2.0.0/api/figure_api.html
            Default is placement based on `pos` attribute.

    """
    def __init__(self):
        pass


class MainContainer(General):
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

        if self.total_plots > 1:
            self.ax = [SinglePlot() for i in range(self.total_plots)]
        else:
            self.ax = SinglePlot()

    def return_overall_dictionary(self):
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

        if self.total_plots > 1:
            trans_dict = {str(i): self._iterate_through_class(axis.__dict__) for i, axis in enumerate(self.ax)}
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

            if val != {}:
                output_dict[key] = val
        return output_dict
