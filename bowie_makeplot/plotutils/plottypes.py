import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.patches import Patch

from bowie_makeplot.plotutils.baseplot import CreateSinglePlot

# TODO: remove requirement for axis limits


class Waterfall(CreateSinglePlot):
    """Create Waterfall plot.

    Waterfall is a subclass of :class:`bowie_makeplot.plotutils.baseplot.CreateSinglePlot`.

    Waterfall creates an snr filled contour plot similar in style to those seen in
    the LISA proposal. Contours are displayed at snrs of 10, 20, 50, 100, 200,
    500, 1000, and 3000 and above. If lower contours are needed, adjust 'contour_vals' kwarg.

    Contour_vals needs to start with zero and end with a higher value than the max in the data.
    Contour_vals needs to be a list of max length 9 including zero and max value.

    """
    def make_plot(self):
        """This method creates the waterfall plot.

        """

        prop_default = {
            'contour_vals': np.array([0., 10, 20, 50, 100, 200, 500, 1000, 3000, 1e10])
        }

        for prop, default in prop_default.items():
            setattr(self, prop, self.__dict__.get(prop, default))

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

    Ratio is a subclass of :class:`bowie_makeplot.plotutils.baseplot.CreateSinglePlot`.

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

    Horizon is a subclass of :class:`bowie_makeplot.plotutils.baseplot.CreateSinglePlot`.

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


class CodetectionPotential1(CreateSinglePlot):
    """
    Ratio is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information.

    Ratio creates an filled contour plot comparing snrs from two different data sets. Typically, it is used to compare sensitivty curves and/or varying binary parameters. It takes the snr of the first dataset and divides it by the snr from the second dataset. The log10 of this ratio is ploted. Additionally, a loss/gain contour is plotted. Loss/gain contour is based on SNR_CUT but can be overridden with 'snr_contour_value' in extra_dict. A gain indicates the first dataset reaches the snr threshold while the second does not. A loss is the opposite.
    """
    # TODO: remove requirement of legend
    def make_plot(self):
        """
        This methd creates the ratio plot.
        """
        colormaps = ['Purples', 'Blues', 'Greens', 'Reds']
        cmap_keys = [map_name.lower()[:-1] for map_name in colormaps]

        codet_cmap = 'Greys'
        codet_cmap_key = codet_cmap.lower()[:-1]

        # find loss/gain contour and ratio contour
        codet_pot, singles = self.find_codetection_potential()

        codet_pot = np.ma.masked_where((codet_pot < np.log10(self.comparison_value)), codet_pot)
        singles = np.ma.masked_where((singles < np.log10(self.comparison_value)), singles)

        if self.solid_single_detection_contour:
            contour_kwargs = {
                'levels': [-1.0, 0.0, 1e8]
            }

        else:
            normval = 4.0
            num_contours = 17
            contour_kwargs = {
                'levels': np.linspace(0.0, normval, num_contours),
                'norm': colors.Normalize(0.0, normval),
                'extend': 'max'
            }

        # setup single detections
        for i, single in enumerate(singles):
            cmap_single = getattr(cm, colormaps[i])
            sdet = self.axis.contourf(self.xvals[0], self.yvals[0], single,
                                      cmap=cmap_single, **contour_kwargs)

            self.axis.scatter([-1e10, -1e10], [-1e1, -2e1], color=cmap_keys[i],
                              label=self.legend_labels[i])

            # toggle line contours of orders of magnitude of ratio comparisons
            if self.order_contour_lines and self.solid_single_detection_contour is not True:
                self.axis.contour(self.xvals[0], self.yvals[0], single, levels=np.array(
                    [1.0, 2.0, 3.0]), colors='black', linewidths=1.0)

        if self.solid_single_detection_contour is False:
            self.colorbar.setup_colorbars(sdet)

        # add codetections
        if self.solid_codetection_contour:
            contour_kwargs = {
                'levels': [-1.0, 0.0, 1e8]
            }

        else:
            normval = 4.0
            num_contours = 17
            contour_kwargs = {
                'levels': np.linspace(0.0, normval, num_contours),
                'norm': colors.Normalize(0.0, normval),
                'extend': 'max'
            }

        cmap_comb = getattr(cm, codet_cmap)
        codet_sc = self.axis.contourf(self.xvals[0], self.yvals[0], codet_pot,
                                      cmap=cmap_comb, **contour_kwargs)
        if self.solid_codetection_contour is not True:
            # black colorbar
            self.colorbar.setup_colorbars(codet_sc)

        if self.order_contour_lines and self.solid_codetection_contour is not True:
            self.axis.contour(self.xvals[0], self.yvals[0], levels=np.array(
                [1.0, 2.0, 3.0]), colors='black', linewidths=1.0)

        self.axis.scatter([-1e10, -1e10], [-1e1, -2e1], color=codet_cmap_key, label='Codetection')



        self.axis.legend(markerscale=3.0, **self.legend_kwargs)
        return

    def find_codetection_potential(self):
        """
        This method finds the ratio contour and the loss gain contour values. Its inputs are the two datasets for comparison where the second is the control to compare against the first.

                The input data sets need to be the same shape. CreateSinglePlot.interpolate_data corrects for two datasets of different shape.

                Returns: loss_gain_contour, ratio contour (diff)

        """

        compare_snrs = []
        for zval in self.zvals:
            compare_snrs.append(1*(zval >= self.comparison_value) + 0*(zval < self.comparison_value))

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


class CodetectionPotential6(CreateSinglePlot):
    """
    Ratio is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information.

    Ratio creates an filled contour plot comparing snrs from two different data sets. Typically, it is used to compare sensitivty curves and/or varying binary parameters. It takes the snr of the first dataset and divides it by the snr from the second dataset. The log10 of this ratio is ploted. Additionally, a loss/gain contour is plotted. Loss/gain contour is based on SNR_CUT but can be overridden with 'snr_contour_value' in extra_dict. A gain indicates the first dataset reaches the snr threshold while the second does not. A loss is the opposite.
    """
    # TODO: remove requirement of legend
    def make_plot(self):
        """
        This methd creates the ratio plot.
        """

        normval2 = 4.0
        num_contours2 = 40  # must be even
        levels2 = np.linspace(-normval2, normval2, num_contours2)
        norm2 = colors.Normalize(0.0, normval2)

        sc = self.axis.contourf(self.xvals[0], self.yvals[0], np.log10(self.zvals[0]),
                                levels=np.array([-50.0, np.log10(8.0), 1.5, 2., 2.5, 3., 3.5, 4.]), norm=norm2, cmap=cm.viridis, extend='max')

        levels = np.array([-np.inf, np.log10(8.0), 2., 3., 4., 10.0])
        line_contours = self.axis.contour(self.xvals[0], self.yvals[0], np.log10(self.zvals[1]),
                                levels=levels,
                                colors='white')

        strs = [r'$10^{%i}$'%val for val in line_contours.levels[1:]]
        strs = ['%i'%(8.0)] + strs
        fmt = {}
        for l, s in zip(line_contours.levels, strs):
            fmt[l] = s

        self.axis.clabel(line_contours, line_contours.levels, fontsize=14, inline=1, inline_spacing=1, fmt=fmt)


        self.colorbar.setup_colorbars(sc)

        return

    def find_codetection_potential(self):
        """
        This method finds the ratio contour and the loss gain contour values. Its inputs are the two datasets for comparison where the second is the control to compare against the first.

                The input data sets need to be the same shape. CreateSinglePlot.interpolate_data corrects for two datasets of different shape.

                Returns: loss_gain_contour, ratio contour (diff)

        """

        compare_snrs = []
        for zval in self.zvals:
            compare_snrs.append(1*(zval >= self.comparison_value) + 0*(zval < self.comparison_value))

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

        codet = 1.0*(combined_snr>0.0)
        return codet, np.log10(snr_single)

class CodetectionPotential5(CreateSinglePlot):
    """
    Ratio is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information.

    Ratio creates an filled contour plot comparing snrs from two different data sets. Typically, it is used to compare sensitivty curves and/or varying binary parameters. It takes the snr of the first dataset and divides it by the snr from the second dataset. The log10 of this ratio is ploted. Additionally, a loss/gain contour is plotted. Loss/gain contour is based on SNR_CUT but can be overridden with 'snr_contour_value' in extra_dict. A gain indicates the first dataset reaches the snr threshold while the second does not. A loss is the opposite.
    """
    # TODO: remove requirement of legend
    def make_plot(self):
        """
        This methd creates the ratio plot.
        """

        #waterfall = Waterfall(**self.__dict__)
        #waterfall.make_plot()

        colormaps = ['Reds', 'Blues', 'Greens', 'Purples']
        cmap_keys = [map_name.lower()[:-1] for map_name in colormaps]

        codet_cmap = 'Greys'
        codet_cmap_key = codet_cmap.lower()[:-1]

        # find loss/gain contour and ratio contour
        codet_pot, singles = self.find_codetection_potential()

        #codet_pot = np.ma.masked_where((codet_pot < np.log10(self.comparison_value)), codet_pot)
        #singles = np.ma.masked_where((singles < np.log10(self.comparison_value)), singles)

        if self.solid_single_detection_contour:
            contour_kwargs = {
                'levels': [-1.0, 0.0, 1e8]
            }

        else:
            normval = 4.0
            num_contours = 17
            contour_kwargs = {
                'levels': np.linspace(0.0, normval, num_contours),
                'norm': colors.Normalize(0.0, normval),
                'extend': 'max'
            }

        normval = 4.0
        num_contours = 17  # must be even
        levels_vals = np.linspace(0.0, normval, num_contours)
        norm = colors.Normalize(0.0, normval)

        # setup single detections
        for i, single in enumerate(singles):
            contour = np.ma.masked_where((self.zvals[i] < self.SNR_CUT), self.zvals[i])
            cmap_single = getattr(cm, colormaps[i])

            if i == 0:
                #levels = [-1.0, 0.0, 10.0]
                levels = levels_vals
            else:
                levels = levels_vals
            sdet = self.axis.contourf(self.xvals[0], self.yvals[0], np.log10(contour),
                               levels=levels, norm=norm, extend='max', cmap=cmap_single, alpha=1.0)


        if self.solid_single_detection_contour is False:
            self.colorbar.setup_colorbars(sdet)

        cmap_comb = getattr(cm, codet_cmap)
        codet_sc = self.axis.contour(self.xvals[0], self.yvals[0], codet_pot, np.array([0.5]),
                                     colors='black', linewidths=3, linestyles='--')
        #import pdb; pdb.set_trace()
        if self.solid_codetection_contour is not True:
            # black colorbar
            self.colorbar.setup_colorbars(codet_sc)

        if self.order_contour_lines and self.solid_codetection_contour is not True:
            self.axis.contour(self.xvals[0], self.yvals[0], levels=np.array(
                [1.0, 2.0, 3.0]), colors='black', linewidths=1.0)

        #self.axis.scatter([-1e10, -1e10], [-1e1, -2e1], color=codet_cmap_key, label='Codetection')



        #self.axis.legend(markerscale=3.0, **self.legend_kwargs)
        return

    def find_codetection_potential(self):
        """
        This method finds the ratio contour and the loss gain contour values. Its inputs are the two datasets for comparison where the second is the control to compare against the first.

                The input data sets need to be the same shape. CreateSinglePlot.interpolate_data corrects for two datasets of different shape.

                Returns: loss_gain_contour, ratio contour (diff)

        """

        compare_snrs = []
        for zval in self.zvals:
            compare_snrs.append(1*(zval >= self.comparison_value) + 0*(zval < self.comparison_value))

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

        codet = 1.0*(combined_snr>0.0)
        return codet, np.log10(snr_single)


class CodetectionPotential4(CreateSinglePlot):
    """
    Ratio is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information.

    Ratio creates an filled contour plot comparing snrs from two different data sets. Typically, it is used to compare sensitivty curves and/or varying binary parameters. It takes the snr of the first dataset and divides it by the snr from the second dataset. The log10 of this ratio is ploted. Additionally, a loss/gain contour is plotted. Loss/gain contour is based on SNR_CUT but can be overridden with 'snr_contour_value' in extra_dict. A gain indicates the first dataset reaches the snr threshold while the second does not. A loss is the opposite.
    """
    def make_plot(self):
        """
        This methd creates the ratio plot.
        """
        colormaps = ['Purples', 'Blues', 'Greens', 'Reds']
        cmap_keys = [map_name.lower()[:-1] for map_name in colormaps]

        codet_cmap = 'Greys'
        codet_cmap_key = codet_cmap.lower()[:-1]

        # find loss/gain contour and ratio contour
        codet_pot, singles = self.find_codetection_potential()

        codet_pot = np.ma.masked_where((codet_pot < np.log10(self.comparison_value)), codet_pot)
        singles = np.ma.masked_where((singles < np.log10(self.comparison_value)), singles)

        normval = 4.0
        num_contours = 17  # must be even
        levels = np.linspace(0.0, normval, num_contours)
        norm = colors.Normalize(0.0, normval)

        # setup single detections
        for i, single in enumerate(singles):
            cmap_single = getattr(cm, colormaps[i])
            sdet = self.axis.contourf(self.xvals[0], self.yvals[0], single,
                               levels=levels, norm=norm, extend='max', cmap=cmap_single, alpha=1.0)

            if i == 0:
                sdet_keep = sdet

            self.axis.scatter([-1e10, -1e10], [-1e1, -2e1], color=cmap_keys[i],
                              label=self.legend_labels[i])

            # toggle line contours of orders of magnitude of ratio comparisons
            if self.order_contour_lines:
                self.axis.contour(self.xvals[0], self.yvals[0], single, levels=np.array(
                    [1.0, 2.0, 3.0]), colors='black', linewidths=1.0)

        # add codetections
        cmap_comb = getattr(cm, codet_cmap)
        #codet_sc = self.axis.contourf(self.xvals[0], self.yvals[0], codet_pot,
        #                              levels=levels, norm=norm, extend='max', cmap=cmap_comb, alpha=1.0)

        self.axis.contourf(self.xvals[0], self.yvals[0], codet_pot,
                                      levels=[-1.0, 5.0, 1e8], norm=norm, extend='max', cmap=cmap_comb, alpha=1.0)
        #if self.order_contour_lines:
        #    self.axis.contour(self.xvals[0], self.yvals[0], levels=np.array(
        #        [1.0, 2.0, 3.0]), colors='black', linewidths=1.0)

        self.axis.scatter([-1e10, -1e10], [-1e1, -2e1], color=codet_cmap_key, label='Codetection')

        # black colorbar
        self.colorbar.setup_colorbars(sdet_keep)

        self.axis.legend(markerscale=3.0, **self.legend_kwargs)
        return

    def find_codetection_potential(self):
        """
        This method finds the ratio contour and the loss gain contour values. Its inputs are the two datasets for comparison where the second is the control to compare against the first.

                The input data sets need to be the same shape. CreateSinglePlot.interpolate_data corrects for two datasets of different shape.

                Returns: loss_gain_contour, ratio contour (diff)

        """

        compare_snrs = []
        for zval in self.zvals:
            compare_snrs.append(1*(zval >= self.comparison_value) + 0*(zval < self.comparison_value))

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

    def find_codetection_potential(self, val1, val2, return_single):
        """
        This method finds the ratio contour and the loss gain contour values. Its inputs are the two datasets for comparison where the second is the control to compare against the first.

                The input data sets need to be the same shape. CreateSinglePlot.interpolate_data corrects for two datasets of different shape.

                Returns: loss_gain_contour, ratio contour (diff)

        """
        inds_up = ((val1 >= self.comparison_value) & (
            val2 >= self.comparison_value) & (val1 >= val2))
        inds_down = ((val1 >= self.comparison_value) & (
            val2 >= self.comparison_value) & (val1 < val2))

        if return_single:
            factor = -1
            out_vals_codect = inds_up*(np.log10(np.sqrt(val1**2 + val2**2))) + \
                inds_down*factor*(np.log10(np.sqrt(val1**2 + val2**2)))
        else:
            factor = 1
            out_vals_codect = inds_up*(np.sqrt(val1**2 + val2**2)) + \
                inds_down*factor*(np.sqrt(val1**2 + val2**2))

        # reshape difference array for dimensions of contour
        out_vals_codect = np.reshape(out_vals_codect, np.shape(val1))

        if return_single:
            # set indices of loss,gained. inds_check tells the ratio calculator not to val2 if both SNRs are below 1
            inds_up_only = ((val1 >= self.comparison_value) & (val2 < self.comparison_value))
            inds_down_only = ((val1 < self.comparison_value) & (val2 >= self.comparison_value))

            out_vals_single = inds_up_only*np.log10(val1) + inds_down_only*-1*np.log10(val2)
            out_vals_single = np.reshape(out_vals_single, np.shape(val1))

            return out_vals_codect, out_vals_single

        return out_vals_codect


class CodetectionPotential2(CodetectionPotential, CreateSinglePlot):
    # TODO remove need to have list for input files if you are going to specify control or remove control entirely
    # TODO make axis not visible

    def make_plot(self):
        """
        """
        # find loss/gain contour and ratio contour
        codect_pot, single = super(CodetectionPotential2, self).find_codetection_potential(
            self.zvals[0], self.zvals[1], True)

        codect_pot = np.ma.masked_where(
            (codect_pot > -np.log10(self.comparison_value)) & (codect_pot < np.log10(self.comparison_value)), codect_pot)
        single = np.ma.masked_where((single > -np.log10(self.comparison_value))
                                    & (single < np.log10(self.comparison_value)), single)

        # sets colormap for ratio comparison plot
        cmap_codect = cm.BrBG
        cmap_single = cm.PiYG

        cmap_codect.set_bad(color='white', alpha=0.001)
        cmap_single.set_bad(color='white', alpha=0.001)

        normval = 4.0
        num_contours = 40  # must be even
        levels = np.linspace(-normval, normval, num_contours)
        norm = colors.Normalize(-normval, normval)

        # plot ratio contours
        sc_codect = self.axis.contourf(self.xvals[0], self.yvals[0], codect_pot,
                                 levels=levels, norm=norm, extend='both', cmap=cmap_codect, alpha=1.0)

        self.colorbar[0].setup_colorbars(sc_codect)

        sc_single = self.axis.contourf(self.xvals[0], self.yvals[0], single,
                                 levels=levels, norm=norm, extend='both', cmap=cmap_single, alpha=1.0)

        self.colorbar[1].setup_colorbars(sc_single)

        # toggle line contours of orders of magnitude of ratio comparisons
        if self.order_contour_lines:
            self.axis.contour(self.xvals[0], self.yvals[0], codect_pot,
                                  np.arange(-3, 4, 1), colors='black', linewidths=1.0)
            self.axis.contour(self.xvals[0], self.yvals[0], single, np.arange(-3, 4, 1), colors='black', linewidths=1.0)

        return


class CodetectionPotential3(Horizon, CodetectionPotential, CreateSinglePlot):
    """
    Ratio is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information.

    Ratio creates an filled contour plot comparing snrs from two different data sets. Typically, it is used to compare sensitivty curves and/or varying binary parameters. It takes the snr of the first dataset and divides it by the snr from the second dataset. The log10 of this ratio is ploted. Additionally, a loss/gain contour is plotted. Loss/gain contour is based on SNR_CUT but can be overridden with 'snr_contour_value' in extra_dict. A gain indicates the first dataset reaches the snr threshold while the second does not. A loss is the opposite.
    """
    def get_contour_values(self):
        codet = []
        for comp in self.comparisons:
            trans_cont_vals = [self.zvals[i] for i in comp]
            codet.append(self.find_codetection_potential(trans_cont_vals))

        self.zvals = codet
        return

    def find_codetection_potential(self, comp_vals):
        """
        This method finds the ratio contour and the loss gain contour values. Its inputs are the two datasets for comparison where the second is the control to compare against the first.

                The input data sets need to be the same shape. CreateSinglePlot.interpolate_data corrects for two datasets of different shape.

                Returns: loss_gain_contour, ratio contour (diff)

        """

        compare_snrs = []
        for zval in comp_vals:
            compare_snrs.append(1*(zval >= self.comparison_value) + 0*(zval < self.comparison_value))

        compare_snrs = np.asarray(compare_snrs)
        how_many_detectors = np.sum(np.asarray(compare_snrs), axis=0).astype(int)

        codet = np.asarray(how_many_detectors == len(comp_vals), dtype=int)*100.0

        return codet


class CodetPot(CreateSinglePlot):
    """
    Ratio is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information.

    Ratio creates a filled contour plot comparing snrs from two different data sets. Typically, it is used to compare sensitivty curves and/or varying binary parameters. It takes the snr of the first dataset and divides it by the snr from the second dataset. Additionally, a Loss/Gain contour is plotted. Loss/Gain contour is based on SNR_CUT but can be overridden with 'snr_contour_value' in extra_dict. A gain indicates the first dataset reaches the snr threshold while the second does not. A loss is the opposite.
    """
    # TODO make colormaps adjustable
    def make_plot(self):
        """
        This methd creates the ratio plot.
        """
        # sets colormap for ratio comparison plot
        cmap = cm.viridis

        import matplotlib as mpl
        mpl.rcParams['hatch.linewidth'] = 0.25
        # set values of ratio comparison contour
        normval2 = 5.0
        num_contours2 = 11  # must be even
        levels2 = np.linspace(0.0, normval2, num_contours2)
        norm2 = colors.Normalize(0.0, normval2)

        # find Loss/Gain contour and Ratio contour
        pot, loss_gain_contour = self.find_difference_contour()

        cmap.set_bad(color='white', alpha=0.001)
        # plot ratio contours

        sc = self.axis.contourf(self.xvals[0], self.yvals[0], pot,
                                levels=levels2, norm=norm2,
                                extend='max', cmap=cmap)

        self.colorbar.setup_colorbars(sc)

        # toggle line contours of orders of magnitude of ratio comparisons
        if self.order_contour_lines:
                self.axis.contour(self.xvals[0], self.yvals[0], diff_out, np.array(
                    [-2.0, -1.0, 1.0, 2.0]), colors='black', linewidths=1.0)

        # plot loss gain contour
        if self.loss_gain_status is True:
            # if there is no loss/gain contours, this will produce an error, so we catch the exception.
            try:
                cs = self.axis.contourf(self.xvals[0], self.yvals[0],
                                  loss_gain_contour, levels=[-2, -0.5, 0.5, 2], colors='none', edgecolor='grey', hatches=['\\',None, '/'])
                self.axis.contour(self.xvals[0], self.yvals[0],
                                  loss_gain_contour, 3, colors='black', linewidths=1.5)

            except ValueError:
                pass

        if self.add_legend:
            loss_patch = Patch(fill=None, label=self.legend_labels[0], hatch='\\', linestyle='--', linewidth=2)
            gain_patch = Patch(fill=None, label=self.legend_labels[1], hatch='/', linestyle='-', linewidth=2)
            legend = self.axis.legend(handles=[loss_patch, gain_patch], **self.legend_kwargs)

        return

    def find_difference_contour(self):
        """
        This method finds the ratio contour and the Loss/Gain contour values. Its inputs are the two datasets for comparison where the second is the control to compare against the first.

                The input data sets need to be the same shape. CreateSinglePlot.interpolate_data corrects for two datasets of different shape.

                Outputs:
                        loss_gain_contour: (float) - 2D array - values for Loss/Gain contour. Value will be -1, 0, or 1.
                        diff_out: (float) - 2D array - ratio contours.

        """
        self.comp1 = self.zvals[0]
        self.comp2 = self.zvals[1]

        # set contour to test and control contour
        self.ratio_comp_value = (self.comparison_value if self.ratio_comp_value is None
                                 else self.ratio_comp_value)

        # indices of loss,gained.
        inds_1 = np.where((self.comp1 >= self.comparison_value) & (self.comp2 < self.comp1))
        inds_2 = np.where((self.comp2 >= self.comparison_value) & (self.comp1 < self.comp2))

        self.comp1 = np.ma.masked_where((self.comp1 < self.ratio_comp_value) | (self.comp2 < self.ratio_comp_value), self.comp1)
        self.comp2 = np.ma.masked_where((self.comp1 < self.ratio_comp_value) | (self.comp2 < self.ratio_comp_value), self.comp2)

        # set diff to ratio for purposed of determining raito differences
        ratio = self.comp1/self.comp2 * (self.comp1 < self.comp2) + self.comp2/self.comp1 * (self.comp1 > self.comp2)
        #pot = (np.log10(self.comp1)**(1/5) + np.log10(self.comp2)**(1/5))**(2)
        outer_exp = 2
        pot = 1./(1/np.log10(self.comp1) + 1/np.log10(self.comp2))
        pot = pot/0.45154499349597177#pot.min()
        #pot = np.log10(ratio * (self.comp1**(1/outer_exp) + self.comp1**(1/outer_exp))**(outer_exp))
        # the following determines the log10 of the ratio difference. If it is extremely small, we neglect and put it as zero (limits chosen to resemble ratios of less than 1.05 and greater than 0.952)

        # initialize loss/gain
        loss_gain_contour = np.zeros(np.shape(self.comp1))

        # fill out loss/gain
        loss_gain_contour[inds_1] = -1
        loss_gain_contour[inds_2] = 1

        return pot, loss_gain_contour
