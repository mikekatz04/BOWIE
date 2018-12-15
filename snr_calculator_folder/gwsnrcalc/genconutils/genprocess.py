"""
Generate gridded data for contour plots with PhenomD waveform.

This module provides the main process class which the main part of the code is run out of.
It is part of the BOWIE analysis tool.

Author: Michael Katz.
Please cite "Evaluating Black Hole Detectability with LISA" (arXiv:1807.02511)
for usage of this code.

PhenomD waveforms are generated according to Husa et al 2016 (arXiv:1508.07250) and
Khan et al 2016 (arXiv:1508.07253). Please cite these papers if the PhenomD waveform is used.

It can take any basic set of parameters for binary black holes and produce waveforms and
SNR calculations for each phase of binary black hole coalescence. It reads in sensitivity curves
from .txt files. The outputs can either be .txt or .hdf5. It can run in parallel or
on a single processor.

This code is licensed under the GNU public license.

"""
import numpy as np

from gwsnrcalc.gw_snr_calculator import snr
from gwsnrcalc.genconutils.readout import FileReadOut


class GenProcess:
    """Direct the contour generation process.

    Class that carries the input information and directs the program to accomplish generation tasks.

    Args:
        **kwargs (dict): Combination of all input dictionaries to have their information
            stored as attributes.

    Keyword Arguments:
        xscale/yscale (str): `lin` for linear scale or `log` for log10 scaling.
        xlow/xhigh/ylow/yhigh (float): Low/high value for x/y.
        num_x/num_y (int): Number of x/y points in contour.
        xval_name/yval_name (str): Name of the x/y quantity.
        xval_unit/yval_unit(str): Units for x/y quantity.
        parameters: All the parameters (args) for the specific waveform generator.
            For both circular and eccentric, this will include start_time, m1, m2,
            redshift (or luminosity_distance or comoving_distance). For circular with will
            also include spin_1, spin_2, and end_time. For eccentric this will
            additionally include eccentricity and observation time.
        sensitivty_input (dict): kwargs for
            :class:`gwsnrcalc.utils.sensitivity.SensitivityContainer`.
        snr_input (dict): kwargs for :class:`gwsnrcalc.gw_snr_calculator.SNR`.

    Attributes:
        final_dict (dict): Dictionary with SNR results.
        Note: All kwargs above are added as attributes.

    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        prop_default = {
            'ecc': False,
        }

        for prop, default in prop_default.items():
            setattr(self, prop, kwargs.get(prop, default))

    def set_parameters(self):
        """Setup all the parameters for the binaries to be evaluated.

        Grid values and store necessary parameters for input into the SNR function.

        """

        # declare 1D arrays of both paramters
        if self.xscale != 'lin':
            self.xvals = np.logspace(np.log10(float(self.x_low)),
                                     np.log10(float(self.x_high)),
                                     self.num_x)

        else:
            self.xvals = np.linspace(float(self.x_low),
                                     float(self.x_high),
                                     self.num_x)

        if self.yscale != 'lin':
            self.yvals = np.logspace(np.log10(float(self.y_low)),
                                     np.log10(float(self.y_high)),
                                     self.num_y)

        else:
            self.yvals = np.linspace(float(self.y_low),
                                     float(self.y_high),
                                     self.num_y)

        self.xvals, self.yvals = np.meshgrid(self.xvals, self.yvals)
        self.xvals, self.yvals = self.xvals.ravel(), self.yvals.ravel()

        for which in ['x', 'y']:
            setattr(self, getattr(self, which + 'val_name'), getattr(self, which + 'vals'))

        self.ecc = 'eccentricity' in self.__dict__
        if self.ecc:
            if 'observation_time' not in self.__dict__:
                if 'start_time' not in self.__dict__:
                    raise ValueError('If no observation time is provided, the time before'
                                     + 'merger must be the inital starting condition.')
                self.observation_time = self.start_time  # small number so it is not zero
        else:
            if 'spin' in self.__dict__:
                self.spin_1 = self.spin
                self.spin_2 = self.spin

        for key in ['redshift', 'luminosity_distance', 'comoving_distance']:
            if key in self.__dict__:
                self.dist_type = key
                self.z_or_dist = getattr(self, key)

            if self.ecc:
                for key in ['start_frequency', 'start_time', 'start_separation']:
                    if key in self.__dict__:
                        self.initial_cond_type = key.split('_')[-1]
                        self.initial_point = getattr(self, key)

        # add m1 and m2
        self.m1 = (self.total_mass / (1. + self.mass_ratio))
        self.m2 = (self.total_mass * self.mass_ratio / (1. + self.mass_ratio))
        return

    def run_snr(self):
        """Run the snr calculation.

        Takes results from ``self.set_parameters`` and other inputs and inputs these
        into the snr calculator.

        """

        if self.ecc:
            required_kwargs = {'dist_type': self.dist_type,
                               'initial_cond_type': self.initial_cond_type,
                               'ecc': True}
            input_args = [self.m1, self.m2, self.z_or_dist, self.initial_point,
                          self.eccentricity, self.observation_time]

        else:
            required_kwargs = {'dist_type': self.dist_type}
            input_args = [self.m1, self.m2, self.spin_1, self.spin_2,
                          self.z_or_dist, self.start_time, self.end_time]

        input_kwargs = {**required_kwargs,
                        **self.general,
                        **self.sensitivity_input,
                        **self.snr_input,
                        **self.parallel_input}

        self.final_dict = snr(*input_args, **input_kwargs)
        return
