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


class GenerateContainer:
    """Holds all of the attributes related to the generate_info dictionary.

    This class is used to store the information when methods from Generate class
    are called by the MainContainer class.

    Attributes:
        Note: Attributes represent the kwargs from
            :class:`gwsnrcalc.genconutils.genprocess.GenProcess`.

    """
    def __init__(self):
        pass


class Generate:
    """ Generate contains the methods inherited by MainContainer.

    These methods are used by MainContainer class to add generation specific
    information to the generate_info dict.

    All attributes are appended to the GenerateContainer class.
    The GenerateContainer class is contained in the generate_info attribute
    in the MainContainer class.

    """
    def _set_grid_info(self, which, low, high, num, scale, name):
        """Set the grid values for x or y.

        Create information for the grid of x and y values.

        Args:
            which (str): `x` or `y`.
            low/high (float): Lowest/highest value for the axis.
            num (int): Number of points on axis.
            scale (str): Scale of the axis. Choices are 'log' or 'lin'.
            name (str): Name representing the axis. See GenerateContainer documentation
                for options for the name.
            unit (str): Unit for this axis quantity. See GenerateContainer documentation
                for options for the units.

        Raises:
            ValueError: If scale is not 'log' or 'lin'.

        """
        setattr(self.generate_info, which + '_low', low)
        setattr(self.generate_info, which + '_high', high)
        setattr(self.generate_info, 'num_' + which, num)
        setattr(self.generate_info, which + 'val_name', name)

        if scale not in ['lin', 'log']:
            raise ValueError('{} scale must be lin or log.'.format(which))
        setattr(self.generate_info, which + 'scale', scale)
        return

    def set_y_grid_info(self, y_low, y_high, num_y, yscale, yval_name):
        """Set the grid values for y.

        Create information for the grid of y values.

        Args:
            num_y (int): Number of points on axis.
            y_low/y_high (float): Lowest/highest value for the axis.
            yscale (str): Scale of the axis. Choices are 'log' or 'lin'.
            yval_name (str): Name representing the axis. See GenerateContainer documentation
                for options for the name.

        """
        self._set_grid_info('y', y_low, y_high, num_y, yscale, yval_name)
        return

    def set_x_grid_info(self, x_low, x_high, num_x, xscale, xval_name):
        """Set the grid values for x.

        Create information for the grid of x values.

        Args:
            num_x (int): Number of points on axis.
            x_low/x_high (float): Lowest/highest value for the axis.
            xscale (str): Scale of the axis. Choices are 'log' or 'lin'.
            xval_name (str): Name representing the axis. See GenerateContainer documentation
                for options for the name.

        """
        self._set_grid_info('x', x_low, x_high, num_x, xscale, xval_name)
        return

    def add_fixed_parameter(self, val, name):
        """Add the fixed parameters for SNR calculation.

        The fixed parameters represent those that a fixed for the entire 2D grid.

        Args:
            val (float): Value of parameter.
            name (str): Name representing the axis. See GenerateContainer documentation
                for options for the name.

        """
        setattr(self.generate_info, name, val)
        return


class SensitivityInputContainer:
    """Holds all of the attributes related to the sensitivity_input dictionary.

    This class is used to store the information when methods from Input class
    are called by the MainContainer class.

    Attributes:
        Note: Attributes represent the kwargs from
            :class:`gwsnrcalc.utils.sensitivity.SensitivityContainer`.

    """
    def __init__(self):
        pass


class SensitivityInput:
    """ Input contains the methods inherited by MainContainer.

    These methods are used by MainContainer class to add noise curve
    information.

    All attributes are appended to the SensitivityInputContainer class.
    The SensitivityInputContainer class is contained in the sensitivty_input attribute
    in the MainContainer class.

    """
    def add_noise_curve(self, name, noise_type='ASD', is_wd_background=False):
        """Add a noise curve for generation.

        This will add a noise curve for an SNR calculation by appending to the sensitivity_curves
        list within the sensitivity_input dictionary.

        The name of the noise curve prior to the file extension will appear as its
        label in the final output dataset. Therefore, it is recommended prior to
        running the generator that file names are renamed to simple names
        for later reference.

        Args:
            name (str): Name of noise curve including file extension inside input_folder.
            noise_type (str, optional): Type of noise. Choices are `ASD`, `PSD`, or `char_strain`.
                Default is ASD.
            is_wd_background (bool, optional): If True, this sensitivity is used as the white dwarf
                background noise. Default is False.

        """
        if is_wd_background:
            self.sensitivity_input.wd_noise = name
            self.sensitivity_input.wd_noise_type_in = noise_type

        else:
            if 'sensitivity_curves' not in self.sensitivity_input.__dict__:
                self.sensitivity_input.sensitivity_curves = []
            if 'noise_type_in' not in self.sensitivity_input.__dict__:
                self.sensitivity_input.noise_type_in = []

            self.sensitivity_input.sensitivity_curves.append(name)
            self.sensitivity_input.noise_type_in.append(noise_type)
        return

    def set_wd_noise(self, wd_noise):
        """Add White Dwarf Background Noise

        This adds the White Dwarf (WD) Background noise. This can either do calculations with,
        without, or with and without WD noise.

        Args:
            wd_noise (bool or str, optional): Add or remove WD background noise. First option is to
                have only calculations with the wd_noise. For this, use `yes` or True.
                Second option is no WD noise. For this, use `no` or False. For both calculations
                with and without WD noise, use `both`.

        Raises:
            ValueError: Input value is not one of the options.

        """
        if isinstance(wd_noise, bool):
            wd_noise = str(wd_noise)

        if wd_noise.lower() == 'yes' or wd_noise.lower() == 'true':
            wd_noise = 'True'
        elif wd_noise.lower() == 'no' or wd_noise.lower() == 'false':
            wd_noise = 'False'
        elif wd_noise.lower() == 'both':
            wd_noise = 'Both'
        else:
            raise ValueError('wd_noise must be yes, no, True, False, or Both.')

        self.sensitivity_input.add_wd_noise = wd_noise
        return


class OutputContainer:
    """Holds all of the attributes related to the output_info dictionary.

    This class is used to store the information when methods from Output class
    are called by the MainContainer class.

    Attributes:
        Note: Attributes represent the kwargs from
            :class:`gwsnrcalc.genconutils.readout.FileReadOut`.

    """
    def __init__(self):
        pass


class Output:
    """ Output contains the methods inherited by MainContainer.

    These methods are used by MainContainer class to add
    information pertaining to the output of the grid creation.

    All attributes are appended to the OutputContainer class.
    The OutputContainer class is contained in the output_info attribute
    in the SinglePlot class.

    """

    def set_output_file(self, output_file_name):
        """Add information for the ouput file.

        Take information on the output file name, type, and folder.

        Args:
            output_file_name (str): String representing the name of the file
                without the file extension.

        """
        self.output_info.output_file_name = output_file_name
        return

    def _set_column_name(self, which, col_name):
        """Set a column name.

        Sets the column name in the output file.

        Args:
            which (str): `x` or `y`.
            col_name (str): Column name to be added.

        """
        setattr(self.output_info, which + '_col_name', col_name)
        return

    def set_y_col_name(self, y_col_name):
        """Set y column name.

        Sets the y column name in the output file.

        Args:
            y_col_name (str): y column name to be added.

        """
        self._set_column_name('y', y_col_name)
        return

    def set_x_col_name(self, x_col_name):
        """Set x column name.

        Sets the x column name in the output file.

        Args:
            x_col_name (str): x column name to be added.

        """
        self._set_column_name('x', x_col_name)
        return

    def add_note(self, note):
        """Add a note to output file.

        This will add a note of user input to the output file.

        Args:
            note (str): Note to be added.

        """
        self.output_info.added_note = note
        return


class ParallelInputContainer:
    """Holds all of the attributes related to the parallel_input dictionary.

    This class is used to store the information when methods from ParallelInput class
    are called by the MainContainer class.

    Attributes:
        Note: Attributes represent the kwargs from
            :class:`gwsnrcalc.utils.parallel.ParallelContainer`.

    """
    def __init__(self):
        pass


class ParallelInput:
    """General contains the methods inherited by MainContainer.

    These methods are used by MainContainer class to add information that applies
    to parallel generation. This information is stored in a ParallelInputContainer
    class object.

    """
    def set_generation_type(self, num_processors=-1, num_splits=1000, verbose=-1):
        """Change generation type.

        Choose weather to generate the data in parallel or on a single processor.

        Args:
            num_processors (int or None, optional): Number of parallel processors to use.
                If ``num_processors==-1``, this will use multiprocessing module and use
                available cpus. If single generation is desired, num_processors is set
                to ``None``. Default is -1.
            num_splits (int, optional): Number of binaries to run during each process.
                Default is 1000.
            verbose (int, optional): Describes the notification of when parallel processes
                are finished. Value describes cadence of process completion notifications.
                If ``verbose == -1``, no notifications are given. Default is -1.

        """
        self.parallel_input.num_processors = num_processors
        self.parallel_input.num_splits = num_splits
        self.parallel_input.verbose = verbose
        return


class SNRInputContainer:
    """Holds all of the attributes related to the snr_input dictionary.

    This class is used to store the information when methods from SNRInput class
    are called by the MainContainer class.

    Attributes:
        Note: Attributes represent the kwargs from :class:`gwsnrcalc.gw_snr_calculator.SNR`.

    """
    def __init__(self):
        pass


class SNRInput:
    """ General contains the methods inherited by MainContainer.

    These methods are used by MainContainer class to add information that applies
    to all plots or the figure as a whole. Many of its settings pertaining to plots
    can be overriden with methods in SinglePlot.

    """

    def set_signal_type(self, sig_type):
        """Set the signal type of interest.

        Sets the signal type for which the SNR is calculated.
        This means inspiral, merger, and/or ringdown.

        Args:
            sig_type (str or list of str): Signal type desired by user.
                Choices are `ins`, `mrg`, `rd`, `all` for circular waveforms created with PhenomD.
                If eccentric waveforms are used, must be `all`.

        """
        if isinstance(sig_type, str):
            sig_type = [sig_type]
        self.snr_input.signal_type = sig_type
        return

    def set_snr_prefactor(self, factor):
        """Set the SNR multiplicative factor.

        This factor will be multpilied by the SNR, not SNR^2.
        This involves orientation, sky, and polarization averaging, as well
        as any factors for the configuration.

        For example, for LISA, this would be sqrt(2*16/5). The sqrt(2) is for
        a six-link configuration and the 16/5 represents the averaging factors.

        Args:
            factor (float): Factor to multiply SNR by for averaging.

        """
        self.snr_input.prefactor = factor
        return

    def set_num_waveform_points(self, num_wave_points):
        """Number of log-spaced points for waveforms.

        The number of points of the waveforms with asymptotically increase
        the accuracy, but will lower the speed of generation.

        Args:
            num_wave_points (int): Number of log-spaced points for the waveforms.

        """
        self.snr_input.num_points = num_wave_points
        return

    def set_n_max(self, n_max):
        """Maximium modes for eccentric signal.

        The number of modes to be considered when calculating the SNR for an
        eccentric signal. This will only matter when calculating eccentric signals.

        Args:
            n_max (int): Maximium modes for eccentric signal.

        """
        self.snr_input.n_max = n_max
        return


class GeneralContainer:
    """Holds all of the attributes related to the general dictionary.

    This class is used to store the information when methods
    are called by the MainContainer class.

    Attributes:
        WORKING_DIRECTORY (str, optional): Working directory for file export and retrieval.
            Default is ``'.'``.

    """
    def __init__(self):
        pass


class MainContainer(Generate, SensitivityInput, SNRInput, ParallelInput, Output):
    """Main class for creating input dictionary to ``generate_contour_data.py``.

    This class creates a pythonic way to add information to the input dictionary
    to ``generate_contour_data.py``. It creates and can read out this dictionary.

    MainContainer inherits methods from Generate, SensitivityInput, SNRInput,
    ParallelInput, and Output classes so that it can
    add respective dictionary information to MainContainer class.

    Args:
        print_input (bool, optional): If True, print the dictionary created by MainContainer
            class after it is completed.

    Attributes:
        general (obj): GeneralContainer class for holding information for general dictionary.
        generate_info (obj): GenerateContainer class for holding information
            for generate_info dictionary.
        sensitivity_input (obj): InputContainer class for holding information for
            sensitivty_input dictionary.
        snr_input (obj): SNRInputContainer class for holding information for
            snr_input dictionary.
        parallel_input (obj): ParallelInputContainer class for holding information for
            parallel_input dictionary.
        output_info (obj): OutputContainer class for holding information for output_info dictionary.
        print_input (bool, optional): If True, print the dictionary created by MainContainer
            class after it is completed.

    """
    def __init__(self, print_input=False):
        Generate.__init__(self)
        self.general = GeneralContainer()
        self.generate_info = GenerateContainer()
        self.sensitivity_input = SensitivityInputContainer()
        self.snr_input = SNRInputContainer()
        self.parallel_input = ParallelInputContainer()
        self.output_info = OutputContainer()
        self.print_input = print_input

    def set_working_directory(self, wd):
        """Set the WORKING_DIRECTORY variable.

        Sets the WORKING_DIRECTORY. The code will then use all paths as relative paths
        to the WORKING_DIRECTORY. In code default is current directory.

        Args:
            wd (str): Absolute or relative path to working directory.

        """
        self.general.WORKING_DIRECTORY = wd
        return

    def return_dict(self):
        """Output dictionary for :mod:`gwsnrcalc.generate_contour_data` input.

        Iterates through the entire MainContainer class turning its contents
        into dictionary form. This dictionary becomes the input for
        :mod:`gwsnrcalc.generate_contour_data`.

        If `print_input` attribute is True, the entire dictionary will be printed
        prior to returning the dicitonary.

        Returns:
            - output_dict: Dicitonary for input into
                :mod:`gwsnrcalc.generate_contour_data`.

        """
        output_dict = {}
        output_dict['general'] = self._iterate_through_class(self.general.__dict__)
        output_dict['generate_info'] = self._iterate_through_class(self.generate_info.__dict__)
        output_dict['sensitivity_input'] = (self._iterate_through_class(
            self.sensitivity_input.__dict__))
        output_dict['snr_input'] = self._iterate_through_class(self.snr_input.__dict__)
        output_dict['parallel_input'] = self._iterate_through_class(self.parallel_input.__dict__)
        output_dict['output_info'] = self._iterate_through_class(self.output_info.__dict__)

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
