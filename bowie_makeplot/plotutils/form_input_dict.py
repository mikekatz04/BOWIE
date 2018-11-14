import warnings


class SinglePlot:

    def __init__(self, index):
        self.index = index
        self.plot_info = {'label':{}, 'limits':{}, 'legend':{}, 'extra':{}}

    def add_dataset(self, name=None, label=None, x_column_label=None, y_column_label=None, index=None, control=False):

        if index is not None:
            if 'file' not in self.plot_info and control != True:
                self.plot_info['file'] = []

            trans_dict = {}
            if name is not None:
                trans_dict['name'] = name

            if label is not None:
                trans_dict['label'] = label

            if x_column_label is not None:
                trans_dict['x_column_label'] = x_column_label

            if y_column_label is not None:
                y_column_label['name'] = y_column_label

            if control:

            else:
                self.plot_info['control'] = trans_dict

        else:
            if control:
                self.plot_info['control'] = {'index':index}

            else:
                if 'indices' not in self.plot_info:
                    self.plot_info['indices'] = []

                self.plot_info['indices'].append(index)

        return

    def define_type(self, plot_type):
        self.plot_info['type'] = plot_type
        return

    def set_labels(self, which, label, fontsize):

        self.plot_info['label'][which] = label

        if fontsize is not None:
            self.plot_info['label'][which + '_fontsize'] = fontsize
        return

    def set_xlabel(self, label, fontsize=None):
        self.set_label('xlabel', label, fontsize)
        return

    def set_ylabel(self, label, fontsize=None):
        self.set_labels('ylabel', label, fontsize)
        return

    def set_title(self, title, fontsize=None):
        self.set_labels('title', title, fontsize)
        return

    def set_labels_fontsize(self, fontsize):
        for which in ['xlabel', 'ylabel']:
            self.plot_info['label'][which + '_fontsize'] = fontsize

    def set_axis_limits(self, which, lims, d, scale, reverse=False):

        self.plot_info['limits'][which + 'lims'] = lims
        self.plot_info['limits']['d' + which] = d
        self.plot_info['limits'][which + 'scale'] = scale

        if reverse:
            self.plot_info['limits']['reverse_' + which + '_axis'] = True

        return

    def set_xlim(self, xlims, dx, xscale, reverse=False):
        self.set_axis_limits('x', xlims, dx, xscale, reverse)
        return

    def set_ylim(self, xlims, dx, xscale, reverse=False):
        self.set_axis_limits('y', xlims, dx, xscale, reverse)
        return

    def legend(labels, loc=None, bbox_to_anchor=None, size=None):
        #TODO: try to pass kwargs
        self.plot_info['legend']['labels'] = labels
        if loc is not None:
            self.plot_info['legend']['loc'] = loc

        if bbox_to_anchor is not None:
            self.plot_info['legend']['bbox_to_anchor'] = bbox_to_anchor

        if size is not None:
            self.plot_info['legend']['size'] = size
        return

    def grid(self, grid=True):
        self.plot_info['extra']['add_grid'] = grid
        return

    def set_contour_vals(self, vals):
        self.plot_info['extra']['contour_vals'] = vals
        return

    def set_snr_contour_value(self, val):
        self.plot_info['extra']['snr_contour_value'] = val
        return

    def ratio_contour_lines(self, lines=True):
        self.plot_info['extra']['ratio_contour_lines'] = lines
        return

    def show_loss_gain(self, show=True):
        self.plot_info['extra']['show_loss_gain'] = show
        return

    def ratio_comp_value(self, val):
        self.plot_info['extra']['ratio_comp_value'] = val
        return


class GeneralDict:

    def __init__(self, nrows, ncols, sharex=False, sharey=False):
        self.general_dict = {'num_rows':nrows, 'num_cols':ncols, 'sharex':sharex, 'sharey':sharey}

    def savefig(self, output_path, dpi=200):
        self.general_dict.save_figure = True
        self.general_dict['output_path'] = output_path
        self.general_dict['dpi'] = dpi
        return

    def show(self):
        self.general_dict['show_figure'] = True
        return

    def snr_cut(self, cut_val):
        self.general_dict['SNR_CUT'] = cut_val
        return

    def working_directory(self, wd):
        self.general_dict['WORKING_DIRECTORY'] = wd
        return

    def switch_backend(self, string='agg'):
        self.general_dict['switch_backend'] = string
        return

    def file_name(self, name):
        self.general_dict['file_name'] = name
        return

    def column_labels(self, xlabel=None, ylabel=None):
        if xlabel is not None:
            self.general_dict['x_column_label'] = xlabel
        if ylabel is not None:
            self.general_dict['y_column_label'] = ylabel
        if xlabel == None and ylabel == None:
            warnings.warn("is not specifying x or y lables even though column labels function is called.", UserWarning)
        return

    def set_fig_size(self, width, height):
        self.general_dict['figure_width'] = width
        self.general_dict['figure_height'] = height
        return

    def spacing(self, space):
        self.general_dict['spacing'] = space
        return

    def subplots_adjust(self, bottom=None, top=None, right=None, left=None, hspace=None, wspace=None):
        if bottom is not None:
            self.general_dict['adjust_figure_bottom'] = bottom
        if top is not None:
            self.general_dict['adjust_figure_top'] = top
        if right is not None:
            self.general_dict['adjust_figure_right'] = right
        if left is not None:
            self.general_dict['adjust_figure_left'] = left
        if wspace is not None:
            self.general_dict['adjust_figure_wspace'] = wspace
        if hspace is not None:
            self.general_dict['adjust_figure_hspace'] = hspace

        if bottom == None and top == None and right == None and left == None and hspace == None and wspace == None:
            warnings.warn("is not specifying figure adjustments even though figure adjustment function is called.", UserWarning)

        return

    def set_fig_labels(self, xlabel=None, ylabel=None, fontsize=None):
        if xlabel is not None:
            self.general_dict['fig_x_label'] = xlabel
        if ylabel is not None:
            self.general_dict['fig_y_label'] = ylabel
        if fontsize is not None:
            self.general_dict['fig_label_fontsize'] = fontsize

        if xlabel == None and ylabel == None:
            warnings.warn("is not specifying x or y lables even though figure labels function is called.", UserWarning)
        return

    def set_fig_title(self, title, fontsize=None):
        self.general_dict['fig_title'] = title
        if fontsize is not None:
            self.general_dict['fig_title_fontsize'] = fontsize
        return

    def set_fig_xlims(self, xlim, dx, scale, fontsize=None):
        self.general_dict['xlims'] = xlim
        self.general_dict['dx'] = dx
        self.general_dict['scale'] = scale
        if tick_fontsize is not None:
            self.general_dict['x_tick_label_fontsize'] = fontsize
        return

    def set_fig_ylims(self, ylim, dy, scale, fontsize=None):
        self.general_dict['ylims'] = ylim
        self.general_dict['dy'] = dy
        self.general_dict['scale'] = scale
        if tick_fontsize is not None:
            self.general_dict['x_tick_label_fontsize'] = fontsize
        return

    def set_ticklabel_fontsize(self, fontsize):
        self.general_dict['tick_label_fontsize'] = fontsize
        return

    def grid(self, grid=True):
        self.general_dict['add_grid'] = grid
        return

    def reverse_axis(self, axis_to_reverse):
        if axis_to_reverse.lower() == 'x':
            self.general_dict['reverse_x_axis'] = True
        if axis_to_reverse.lower() == 'y':
            self.general_dict['reverse_y_axis'] = True
        if axis_to_reverse.lower() != 'x' or axis_to_reverse.lower() != 'y':
            raise Exception('Axis for reversing needs to be either x or y.')
        return

    def set_colorbar(self, plot_type, label=None, label_fontsize=None, ticks_fontsize=None, pos=None, colorbar_axes=None):
        if 'colorbars' is not in self.general_dict:
            self.general_dict['colorbars'] = {}

        self.general_dict['colorbars'][plot_type] = {}

        if label_fontsize is not None:
            self.general_dict['colorbars'][plot_type]['label_fontsize'] = label_fontsize

        if ticks_fontsize is not None:
            self.general_dict['ticks_fontsize'] = ticks_fontsize

        if pos is not None:
            self.general_dict['pos'] = pos

        if colorbar_axes is not None:
            self.general_dict['colorbar_axes'] = colorbar_axes

        return



class GeneralContainer(GeneralDict, Plot):
    def __init__(self, nrows, ncols, sharex=False, sharey=False, print_input=False):
        self.print_output = print_input

        GeneralDict.__init__(self, nrows, ncols, sharex=False, sharey=False)
        self.ax = [SinglePlot(i) for i in range(nrows*ncols)]

    def return_overall_dictionary(self):
        plot_dicts = [str(i):self.ax[i].plot_info for i in range(nrows*ncols)]
        return {'general': self.general_dict, 'plot_info': self.plot_dict}
