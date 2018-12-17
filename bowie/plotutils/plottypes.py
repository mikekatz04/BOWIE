import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.patches import Patch

from bowie.plotutils.baseplot import CreateSinglePlot

# TODO: remove requirement for axis limits


class Waterfall(CreateSinglePlot):
    """Create Waterfall plot.

    Waterfall is a subclass of :class:`bowie.plotutils.baseplot.CreateSinglePlot`.

    Waterfall creates an snr filled contour plot similar in style to those seen in
    the LISA proposal. Contours are displayed at snrs of 10, 20, 50, 100, 200,
    500, 1000, and 3000 and above. If lower contours are needed, adjust 'contour_vals' kwarg.

    Contour_vals needs to start with zero and end with a higher value than the max in the data.
    Contour_vals needs to be a list of max length 9 including zero and max value.

    """
    def make_plot(self):
        """This method creates the waterfall plot.

        """

        # sets levels of main contour plot
        colors1 = ['None', 'darkblue', 'blue', 'deepskyblue', 'aqua',
                   'greenyellow', 'orange', 'red', 'darkred']

        if len(self.contour_vals) > len(colors1) + 1:
            raise AttributeError("Reduce number of contours.")

        # produce filled contour of SNR
        sc = self.axis.contourf(self.xvals[0], self.yvals[0], self.zvals[0],
                                levels=np.asarray(self.contour_vals), colors=colors1)

        self.colorbar.setup_colorbars(sc)

        # check for user desire to show separate contour line
        if self.snr_contour_value is not None:
            self.axis.contour(self.xvals[0], self.yvals[0], self.zvals[0],
                              np.array([self.snr_contour_value]),
                              colors='white', linewidths=1.5, linestyles='dashed')

        return


class Ratio(CreateSinglePlot):
    """Create a ratio plot.

    Ratio is a subclass of :class:`bowie.plotutils.baseplot.CreateSinglePlot`.

    Ratio creates a filled contour plot comparing snrs from two different data sets.
    Typically, it is used to compare sensitivty curves and/or varying binary parameters.
    It takes the snr of the first dataset and divides it by the snr from the second dataset.
    Additionally, a Loss/Gain contour is plotted. Loss/Gain contour is based on SNR_CUT
    but can be overridden with 'snr_contour_value' kwarg. A gain indicates
    the first dataset reaches the snr threshold while the second does not.
    A loss is the opposite.

    Attributes:
        comp1/comp2 (2D array of floats): Comparison and control datasets respectivley.

    """
    # TODO make colormaps adjustable
    def make_plot(self):
        """Creates the ratio plot.

        """
        # sets colormap for ratio comparison plot
        cmap = getattr(cm, self.colormap)

        # set values of ratio comparison contour
        normval = 2.0
        num_contours = 40  # must be even
        levels = np.linspace(-normval, normval, num_contours)
        norm = colors.Normalize(-normval, normval)

        # find Loss/Gain contour and Ratio contour
        self.set_comparison()
        diff_out, loss_gain_contour = self.find_difference_contour()

        cmap.set_bad(color='white', alpha=0.001)
        # plot ratio contours

        sc = self.axis.contourf(self.xvals[0], self.yvals[0], diff_out,
                                levels=levels, norm=norm,
                                extend='both', cmap=cmap)

        self.colorbar.setup_colorbars(sc)

        # toggle line contours of orders of magnitude of ratio comparisons
        if self.order_contour_lines:
                self.axis.contour(self.xvals[0], self.yvals[0], diff_out, np.array(
                    [-2.0, -1.0, 1.0, 2.0]), colors='black', linewidths=1.0)

        # plot loss gain contour
        if self.loss_gain_status is True:
            # if there is no loss/gain contours, this will produce an error,
            # so we catch the exception.
            try:
                # make hatching
                cs = self.axis.contourf(self.xvals[0], self.yvals[0],
                                        loss_gain_contour, levels=[-2, -0.5, 0.5, 2], colors='none',
                                        hatches=['x', None, '+'])
                # make loss/gain contour outline
                self.axis.contour(self.xvals[0], self.yvals[0],
                                  loss_gain_contour, 3, colors='black', linewidths=2)

            except ValueError:
                pass

        if self.add_legend:
            loss_patch = Patch(fill=None, label='Loss', hatch='x', linestyle='--', linewidth=2)
            gain_patch = Patch(fill=None, label='Gain', hatch='+', linestyle='-', linewidth=2)
            legend = self.axis.legend(handles=[loss_patch, gain_patch], **self.legend_kwargs)

        return

    def set_comparison(self):
        """Defines the comparison values for the ratio.

        This function is added for easier modularity.

        """
        self.comp1 = self.zvals[0]
        self.comp2 = self.zvals[1]
        return

    def find_difference_contour(self):
        """Find the ratio and loss/gain contours.

        This method finds the ratio contour and the Loss/Gain contour values.
        Its inputs are the two datasets for comparison where the second is the control
        to compare against the first.

        The input data sets need to be the same shape.

        Returns:
            2-element tuple containing
                - **diff** (*2D array of floats*): Ratio contour values.
                - **loss_gain_contour** (*2D array of floats*): loss/gain contour values.

        """
        # set contour to test and control contour
        self.ratio_comp_value = (self.comparison_value if self.ratio_comp_value is None
                                 else self.ratio_comp_value)

        # indices of loss,gained.
        inds_gained = np.where((self.comp1 >= self.comparison_value)
                               & (self.comp2 < self.comparison_value))
        inds_lost = np.where((self.comp1 < self.comparison_value)
                             & (self.comp2 >= self.comparison_value))

        self.comp1 = np.ma.masked_where(self.comp1 < self.ratio_comp_value, self.comp1)
        self.comp2 = np.ma.masked_where(self.comp2 < self.ratio_comp_value, self.comp2)

        # set diff to ratio for purposed of determining raito differences
        diff = self.comp1/self.comp2

        # the following determines the log10 of the ratio difference.
        # If it is extremely small, we neglect and put it as zero
        # (limits chosen to resemble ratios of less than 1.05 and greater than 0.952)
        diff = (np.log10(diff)*(diff >= 1.05)
                + (-np.log10(1.0/diff)) * (diff <= 0.952)
                + 0.0*((diff < 1.05) & (diff > 0.952)))

        # initialize loss/gain
        loss_gain_contour = np.zeros(np.shape(self.comp1))

        # fill out loss/gain
        loss_gain_contour[inds_lost] = -1
        loss_gain_contour[inds_gained] = 1

        return diff, loss_gain_contour


class Horizon(CreateSinglePlot):
    """Create Horizon plots.

    Horizon is a subclass of :class:`bowie.plotutils.baseplot.CreateSinglePlot`.

    Horizon plots snr contour lines for a designated SNR value. The defaul is SNR_CUT,
    but can be overridden with `snr_contour_value`. Horizon can make as many
    curves as the user prefers and will plot a legend to label those curves.
    It can only contour one snr value.

    """
    def make_plot(self):
        """Make the horizon plot.

        """

        self.get_contour_values()
        # sets levels of main contour plot
        colors1 = ['blue', 'green', 'red', 'purple', 'orange',
                   'gold', 'magenta']

        # set contour value. Default is SNR_CUT.
        self.snr_contour_value = (self.SNR_CUT if self.snr_contour_value is None
                                  else self.snr_contour_value)

        # plot contours
        for j in range(len(self.zvals)):
            hz = self.axis.contour(self.xvals[j], self.yvals[j],
                                   self.zvals[j], np.array([self.snr_contour_value]),
                                   colors=colors1[j], linewidths=1., linestyles='solid')

            # plot invisible lines for purpose of creating a legend
            if self.legend_labels != []:
                # plot a curve off of the grid with same color for legend label.
                self.axis.plot([0.1, 0.2], [0.1, 0.2], color=colors1[j],
                               label=self.legend_labels[j])

        if self.add_legend:
            self.axis.legend(**self.legend_kwargs)

        return

    def get_contour_values(self):
        """Get values for contours.

        This function is added for modularity.
        """

        return
