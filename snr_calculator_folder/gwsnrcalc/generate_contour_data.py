"""
Generate gridded data for contour plots with PhenomD waveform.
It is part of the BOWIE analysis tool.
Author: Michael Katz. Please cite "Evaluating Black Hole Detectability with LISA" (arXiv:1807.02511)
for usage of this code.

PhenomD waveforms are generated according to Husa et al 2016 (arXiv:1508.07250) and Khan et al 2016
(arXiv:1508.07253). Please cite these papers if the PhenomD waveform is used.

Eccentric inspiral waveforms are also possible according to Peters evolution.

generate_contour_data produces gridded data sets based on an input class or dictionary.
It can take any basic set of parameters for binary black holes and produce waveforms
and SNR calculations for each phase of binary black hole coalescence
(only inspiral for eccentric binaries). It reads in sensitivity curves from .txt files.
The outputs can either be .txt or .hdf5.
It can run in parallel or on a single processor. See the example notebooks for usage of this module.

This code is licensed under the GNU public license.

"""

import sys
import json
import time

from gwsnrcalc.genconutils.genprocess import GenProcess
from gwsnrcalc.genconutils.readout import FileReadOut
from gwsnrcalc.genconutils.forminput import MainContainer as GenInput


def generate_contour_data(pid):
    """
    Main function for this program.

    This will read in sensitivity_curves and binary parameters; calculate snrs
    with a matched filtering approach; and then read the contour data out to a file.

    Args:
        pid (obj or dict): GenInput class or dictionary containing all of the input information for
            the generation. See BOWIE documentation and example notebooks for usage of
            this class.

    """
    # check if pid is  dicionary or GenInput class
    # if GenInput, change to dictionary
    if isinstance(pid, GenInput):
        pid = pid.return_dict()

    begin_time = time.time()

    WORKING_DIRECTORY = '.'
    if 'WORKING_DIRECTORY' not in pid['general'].keys():
        pid['general']['WORKING_DIRECTORY'] = WORKING_DIRECTORY

    # Generate the contour data.
    running_process = GenProcess(**{**pid, **pid['generate_info']})
    running_process.set_parameters()
    running_process.run_snr()

    # Read out
    file_out = FileReadOut(running_process.xvals, running_process.yvals,
                           running_process.final_dict,
                           **{**pid['general'], **pid['generate_info'], **pid['output_info']})

    print('outputing file:', pid['general']['WORKING_DIRECTORY'] + '/'
          + pid['output_info']['output_file_name'])
    getattr(file_out, file_out.output_file_type + '_read_out')()

    print(time.time()-begin_time)
    return


if __name__ == '__main__':
    plot_info_dict = json.load(open(sys.argv[1], 'r'))

    generate_contour_data(plot_info_dict)
