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
        fixed_parameter_1/fixed_parameter_1/fixed_parameter_1/fixed_parameter_1 (float):
            Value for each fixed parameter.
        par_1_unit/par_2_unit/par_3_unit/par_4_unit (str): Unit for fixed parameters.
        par_1_name/par_2_name/par_3_name/par_4_name (str): Name of fixed parameters.
        fixed_parameter_5 (float): Value of fixed parameter 5.
            This is only optional if ``xval_name`` or ``yval_name`` is 'spin'.
        par_5_name (str): Name of fixed parameter 5.
            This is only optional if ``xval_name`` or ``yval_name`` is 'spin'.
        par_5_name (str): Unit of fixed parameter 5.
            This is only optional if ``xval_name`` or ``yval_name`` is 'spin'.
        sensitivty_input (dict): kwargs for
            :class:`gwsnrcalc.utils.sensitivity.SensitivityContainer`.
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

        trans = np.meshgrid(self.xvals, self.yvals)
        self.xvals, self.yvals = [tran.ravel() for tran in trans]

        key_list = ['par_1_name', 'par_2_name', 'par_3_name',
                    'par_4_name', 'xval_name', 'yval_name']

        for key in key_list:
            if getattr(self, key) == 'eccentricity':
                    self.ecc = True

        if self.ecc is False:
            key_list = key_list + ['par_5_name']
            num_pars = 5
        else:
            num_pars = 4

        # if testing spin, we do not want to set the second spin to a constant value,
        # so we assume spins are the same.
        if 'spin' in [self.xval_name, self.yval_name] or self.ecc:
            sub_par_5 = True
        else:
            sub_par_5 = False

        # ensure these other parameters are floats
        self.fixed_parameter_1 = float(self.fixed_parameter_1)
        self.fixed_parameter_2 = float(self.fixed_parameter_2)
        self.fixed_parameter_3 = float(self.fixed_parameter_3)
        self.fixed_parameter_4 = float(self.fixed_parameter_4)

        # meshrid to get 1d arrays for each parameter

        if sub_par_5 is False:
            self.fixed_parameter_5 = float(self.fixed_parameter_5)

        elif self.ecc is False:
            self.par_5_name = 'spin_2'
            self.par_5_unit = 'None'
            if self.xval_name == 'spin':
                self.xval_name = 'spin_1'
                self.fixed_parameter_5 = self.xvals
            if self.yval_name == 'spin':
                self.yval_name = 'spin_1'
                self.fixed_parameter_5 = self.yvals

        # add parameters to self. Names must be 'total_mass', 'mass_ratio', 'redshift' or
        # 'luminosity_distance' or 'comoving distance', 'spin_1', 'spin_2'
        for which in ['x', 'y']:
            setattr(self, getattr(self, which + 'val_name'), getattr(self, which + 'vals'))

        for which in np.arange(1, num_pars + 1).astype(str):
            setattr(self, getattr(self, 'par_' + which + '_name'),
                    getattr(self, 'fixed_parameter_' + which))


        for key in key_list:
            if getattr(self, key) in ['redshift', 'luminosity_distance', 'comoving_distance']:
                self.dist_type = getattr(self, key)
                self.z_or_dist = getattr(self, getattr(self, key))

            if self.ecc:
                if getattr(self, key) in ['start_frequency', 'start_time', 'start_separation']:
                    self.initial_cond_type = getattr(self, key).split('_')[-1]
                    self.initial_point = getattr(self, getattr(self, key))

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
