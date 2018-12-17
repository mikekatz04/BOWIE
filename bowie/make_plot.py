"""
make_plot turns gridded datasets into helpful plots. It is designed for
LISA Signal-to-Noise (SNR) comparisons across sennsitivity curves and parameters,
but is flexible to other needs. It is part of the BOWIE analysis tool.
Author: Michael Katz.

Please cite "Evaluating Black Hole Detectability with LISA" (arXiv:1807.02511)
for usage of this code.

This code is licensed under the GNU public license.

TThe plotting classes are also importable for customization.
See BOWIE_basic_examples.ipynb for examples on how to use this code.
See paper_plots.ipynb for the plots shown in the paper.

The three main classes are plot types: Waterfall, Horizon, and Ratio.

Waterfall:
    SNR contour plot based on plots from LISA Mission proposal.

Ratio:
    Comparison plot of the ratio of SNRs for two different inputs.
    This plot also contains Loss/Gain contours, which describe when sources
    are gained or lost compared to one another based on a user specified SNR cut.
    See paper above for further explanation.

Horizon:
    SNR contour plots comparing multipile inputs. User can specify contour value.
    The default is the user specified SNR cut.

"""


import json
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt

from bowie.plotutils.makeprocess import MakePlotProcess
from bowie.plotutils.forminput import MainContainer as PlotInput

SNR_CUT = 5.0


def plot_main(pid, return_fig_ax=False):
    """Main function for creating these plots.

    Reads in plot info dict from json file or dictionary in script.

    Args:
        return_fig_ax (bool, optional): Return figure and axes objects.

    Returns:
        2-element tuple containing
            - **fig** (*obj*): Figure object for customization outside of those in this program.
            - **ax** (*obj*): Axes object for customization outside of those in this program.

    """

    global WORKING_DIRECTORY, SNR_CUT

    if isinstance(pid, PlotInput):
        pid = pid.return_dict()

    WORKING_DIRECTORY = '.'
    if 'WORKING_DIRECTORY' not in pid['general'].keys():
        pid['general']['WORKING_DIRECTORY'] = '.'

    SNR_CUT = 5.0
    if 'SNR_CUT' not in pid['general'].keys():
        pid['general']['SNR_CUT'] = SNR_CUT

    if "switch_backend" in pid['general'].keys():
        plt.switch_backend(pid['general']['switch_backend'])

    running_process = MakePlotProcess(
        **{**pid, **pid['general'], **pid['plot_info'], **pid['figure']})

    running_process.input_data()
    running_process.setup_figure()
    running_process.create_plots()

    # save or show figure
    if 'save_figure' in pid['figure'].keys():
        if pid['figure']['save_figure'] is True:
            running_process.fig.savefig(
                pid['general']['WORKING_DIRECTORY'] + '/' + pid['figure']['output_path'],
                **pid['figure']['savefig_kwargs'])

    if 'show_figure' in pid['figure'].keys():
        if pid['figure']['show_figure'] is True:
            plt.show()

    if return_fig_ax is True:
        return running_process.fig, running_process.ax

    return


if __name__ == '__main__':
    # read in json
    plot_info_dict = json.load(open(sys.argv[1], 'r'), object_pairs_hook=OrderedDict)
    plot_main(plot_info_dict)
