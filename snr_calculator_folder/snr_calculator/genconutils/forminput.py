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


class Generate:
    """ Generate contains the methods inherited by MainContainer.

    These methods are used by MainContainer class to add generation specific
    information to the generate_info dict.

    All attributes are appended to the GenerateContainer class.
    The GenerateContainer class is contained in the generate_info attribute
    in the MainContainer class.

    Args:
        fixed_par_limit (int, optional): Maximum number of fixed parameters allowed.
            Default is 5. This should not be changed unless code is changed.

    Attributes:
        fixed_par_limit (int): Maximum number of fixed parameters allowed.
        par_num (int): Current value of fixed parameter being added.

    """
    def __init__(self, fixed_par_limit=5):
        self.fixed_par_limit = fixed_par_limit
        self.par_num = 0

    def _set_grid_info(self, which, low, high, num, scale, name, unit):
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
        setattr(self.generate_info, which + 'val_unit', unit)

        if scale not in ['lin', 'log']:
            raise ValueError('{} scale must be lin or log.'.format(which))
        setattr(self.generate_info, which + 'scale', scale)
        return

    def set_y_for_grid(self, y_low, y_high, num_y, yscale, yval_name, yval_unit):
        """Set the grid values for y.

        Create information for the grid of y values.

        Args:
            num_y (int): Number of points on axis.
            y_low/y_high (float): Lowest/highest value for the axis.
            yscale (str): Scale of the axis. Choices are 'log' or 'lin'.
            yval_name (str): Name representing the axis. See GenerateContainer documentation
                for options for the name.
            yval_unit (str): Unit for this axis quantity. See GenerateContainer documentation
                for options for the units.

        """
        self._set_grid_info('y', y_low, y_high, num_y, yscale, yval_name, yval_unit)
        return

    def set_x_for_grid(self, x_low, x_high, num_x, xscale, xval_name, xval_unit):
        """Set the grid values for x.

        Create information for the grid of x values.

        Args:
            num_x (int): Number of points on axis.
            x_low/x_high (float): Lowest/highest value for the axis.
            xscale (str): Scale of the axis. Choices are 'log' or 'lin'.
            xval_name (str): Name representing the axis. See GenerateContainer documentation
                for options for the name.
            xval_unit (str): Unit for this axis quantity. See GenerateContainer documentation
                for options for the units.

        """
        self._set_grid_info('x', x_low, x_high, num_x, xscale, xval_name, xval_unit)
        return

    def add_fixed_parameter(self, val, name, unit):
        """Add the fixed parameters for SNR calculation.

        The fixed parameters represent those that a fixed for the entire 2D grid.

        # TODO: remove same spin

        Args:
            val (float): Value of parameter.
            name (str): Name representing the axis. See GenerateContainer documentation
                for options for the name.
            unit (str): Unit for this axis quantity. See GenerateContainer documentation
                for options for the units.

        """
        self.par_num += 1
        par_num = str(self.par_num)
        setattr(self.generate_info, 'fixed_parameter_' + par_num, val)
        setattr(self.generate_info, 'par_' + par_num + '_name', name)
        setattr(self.generate_info, 'par_' + par_num + '_unit', unit)

        print(par_num, 'of', self.fixed_par_limit, 'is set to', val, 'for', name)
        return

    def set_num_waveform_points(self, num_wave_points):
        """Number of log-spaced points for waveforms.

        The number of points of the waveforms with asymptotically increase
        the accuracy, but will lower the speed of generation.

        Args:
            num_wave_points (int): Number of log-spaced points for the waveforms.

        """
        self.generate_info.num_points = num_wave_points
        return

    def set_snr_factor(self, factor):
        """Set the SNR multiplicative factor.

        This factor will be multpilied by the SNR, not SNR^2.
        This involves orientation, sky, and polarization averaging, as well
        as any factors for the configuration.

        For example, for LISA, this would be sqrt(2*16/5). The sqrt(2) is for
        a six-link configuration and the 16/5 represents the averaging factors.

        Args:
            factor (float): Factor to multiply SNR by for averaging.

        """
        self.generate_info.prefactor = factor
        return


class GenerateContainer:
    """Holds all of the attributes related to the generate_info dictionary.

    This class is used to store the information when methods from Generate class
    are called by the MainContainer class.

    Attributes:
        num_x/num_y (int): Number of points on x/y axis.
        x_low/x_high/y_low/y_high (float): Lowest/highest value for the x/y axis.
        xscale/yscale (str): Scale of the x/y axis. Choices are 'log' or 'lin'.
        xval_unit/yval_unit/par_1_unit/par_2_unit/par_3_unit/par_4_unit/par_5_unit (str):
            This is for export to a file. It keeps track of quantities for future use of the file.
            Start time and end time must be given in years. Mass must be given in solar masses.
            If "luminosity_distance" or "comoving_distance" is used, then the units matter.
            Default is "Mpc". However, "Gpc" can be given.
        xval_name/yval_name/par_1_name/par_2_name/par_3_name/par_4_name/par_5_name (str):
            Names of the x/y values or fixed parameters being tested.
            Choices are "total_mass", "mass_ratio", "luminosity_distance",
            "comoving_distance", "redshift", "spin_1", "spin_2",
            "start_time", "end_time".
        fixed_parameter_1/fixed_parameter_2/fixed_parameter_3/fixed_parameter_4/fixed_parameter_5
            (float): Value for the parameters fixed for each binary.
        num_points (int, optional): Number of log-spaced points for the waveforms.
        factor (float, optional): Factor to multiply SNR by for averaging. Default is 1.0.

    """
    def __init__(self):
        pass


class Input:
    """ Input contains the methods inherited by MainContainer.

    These methods are used by MainContainer class to add noise curve
    information.

    All attributes are appended to the InputContainer class.
    The InputContainer class is contained in the legend attribute
    in the MainContainer class.

    """
    def add_noise_curve(self, name, noise_type='ASD', is_wd_background=False):
        """Add a noise curve for generation.

        This will add a noise curve for an SNR calculation by appending to the sensitivity_curves
        list within the input_info dictionary.

        The name of the noise curve prior to the file extension will appear as its
        label in the final output dataset. Therefore, it is recommended prior to
        running the generator that file names are renamed to simple names
        for later reference.

        Args:
            name (str): Name of noise curve including file extension inside input_folder.
            noise_type (str, optional): Type of noise. Choices are `ASD`, `PSD`, or `char_strain`.
                Default is ASD.

        """
        if is_wd_background:
            self.input_info.wd_noise = name
            self.input_info.wd_noise_type_in = noise_type

        else:
            if 'sensitivity_curves' not in self.input_info.__dict__:
                self.input_info.sensitivity_curves = []
            if 'noise_type_in' not in self.input_info.__dict__:
                self.input_info.noise_type_in = []

            self.input_info.sensitivity_curves.append(name)
            self.input_info.noise_type_in.append(noise_type)
        return


class InputContainer:
    """Holds all of the attributes related to the input_info dictionary.

    This class is used to store the information when methods from Input class
    are called by the MainContainer class.

    Attributes:
        sensitivity_curves (list of obj): List of noise curve information held
            in the NoiseCurveContainer class. This sets up the calculation for
            the sensitivty curves desired by the user.
            This is either a provided curve or a file path to use.
        noise_type_in (list of obj): Type of curve provided. Can be `ASD`, `PSD`,
            or `char_strain`. The amplitude column of the file must be named
            this same string.
        wd_noise (str, optional): List of noise curve information held
            in the NoiseCurveContainer class. This sets up the calculation for
            the sensitivty curves desired by the user.
        wd_noise_type_in (str, optional): Type of noise curve provided. Can be `ASD`, `PSD`,
            or `char_strain`. The amplitude column of the file must be named
            this same string.

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
            output_file_type (str, optional): File extension. Choices are `hdf5` or `txt`.
                Default is `hdf5`.

        Raises:
            ValueError: File type is not txt or hdf5.

        """
        self.output_info.output_file_name = output_file_name

        output_file_type = output_file_name.split('.')[-1]

        if output_file_type not in ['hdf5', 'txt']:
            raise ValueError('file_output_type must be hdf5 or txt.')
        self.output_info.output_file_type = output_file_type
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

    def added_note(self, note):
        """Add a note to output file.

        This will add a note of user input to the output file.

        Args:
            note (str): Note to be added.

        """
        self.input_info.added_note = note
        return


class OutputContainer:
    """Holds all of the attributes related to the output_info dictionary.

    This class is used to store the information when methods from Output class
    are called by the MainContainer class.

    Attributes:
        output_file_name (str): String representing the name of the file
            without the file extension.
        output_file_type (str, optional): File extension. Choices are `hdf5` or `txt`.
            Default is `hdf5`.
        output_folder (str, optional): Output folder in relation to WORKING_DIRECTORY.
            Default is '.'.
        x_col_name/y_col_name (st, optional): x/y column name to be added.
            Default is `x`/`y`.
        added_note (str, optional): Note to be added. Default is None.

    """
    def __init__(self):
        pass


class General:
    """ General contains the methods inherited by MainContainer.

    These methods are used by MainContainer class to add information that applies
    to all plots or the figure as a whole. Many of its settings pertaining to plots
    can be overriden with methods in SinglePlot.

    """
    def working_directory(self, wd):
        """Set the WORKING_DIRECTORY variable.

        Sets the WORKING_DIRECTORY. The code will then use all paths as relative paths
        to the WORKING_DIRECTORY. In code default is current directory.

        Args:
            wd (str): Absolute or relative path to working directory.

        """
        self.general.WORKING_DIRECTORY = wd
        return

    def set_signal_type(self, sig_type):
        """Set the signal type of interest.

        Sets the signal type for which the SNR is calculated.
        This means inspiral, merger, and/or ringdown.

        Args:
            sig_type (str or list of str): Signal type desired by user.
                Choices are `ins`, `mrg`, `rd`, `all`.

        """
        if isinstance(sig_type, str):
            sig_type = [sig_type]
        self.general.signal_type = sig_type
        return

    def set_generation_type(self, num_processors=-1, num_splits=1000):
        """Change generation type.

        Choose weather to generate the data in parallel or on a single processor.

        Args:
            num_processors (int or None, optional): Number of parallel processors to use.
                If ``num_processors==-1``, this will use multiprocessing module and use
                available cpus. If single generation is desired, num_processors is set
                to ``None``. Default is -1.
            num_splits (int, optional): Number of binaries to run during each process.
                Default is 1000.

        """
        self.general.num_processors = num_processors
        self.general.num_splits = num_splits
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
            wd_noise = str(bool)

        elif wd_noise.lower() == 'yes' or wd_noise == 'True':
            wd_noise = 'True'
        elif wd_noise.lower() == 'no' or wd_noise == 'True':
            wd_noise = 'False'
        elif wd_noise.lower() == 'both':
            wd_noise = 'Both'
        else:
            raise ValueError('wd_noise must be yes, no, True, False, or Both.')

        self.general.add_wd_noise = wd_noise
        return


class GeneralContainer:
    """Holds all of the attributes related to the general dictionary.

    This class is used to store the information when methods from General class
    are called by the MainContainer class.

    Attributes:
        signal_type (str or list of str): Signal type desired by user.
            Choices are `ins`, `mrg`, `rd`, `all`.
        WORKING_DIRECTORY (str, optional): Working directory for file export and retrieval.
            Default is ``'.'``.
        add_wd_noise (bool or str, optional): Add or remove WD background noise. First option is to
            have only calculations with the wd_noise. For this, use `yes` or True.
            Second option is no WD noise. For this, use `no` or False. For both calculations
            with and without WD noise, use `both`. Default is False.
        num_processors (int or None, optional): Number of parallel processors to use.
            If ``num_processors==-1``, this will use multiprocessing module and use
            available cpus. If single generation is desired, num_processors is set
            to ``None``. Default is -1.
        num_splits (int, optional): Number of binaries to run during each process.
            Default is 1000.
        wd (str): Absolute or relative path to working directory.

    """
    def __init__(self):
        pass


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


class MainContainer(General, Input, Output, Generate):
    """Main class for creating input dictionary to ``generate_contour_data.py``.

    This class creates a pythonic way to add information to the input dictionary
    to ``generate_contour_data.py``. It creates and can read out this dictionary.

    MainContainer inherits methods from General, Input, Output, and Generate classes
    so that it can add respective dictionary information to MainContainer class.

    Args:
        print_input (bool, optional): If True, print the dictionary created by MainContainer
            class after it is completed.

    Attributes:
        general (obj): GeneralContainer class for holding information for general dictionary.
        generate_info (obj): GenerateContainer class for holding information
            for generate_info dictionary.
        input_info (obj): InputContainer class for holding information for input_info dictionary.
        output_info (obj): OutputContainer class for holding information for output_info dictionary.
        print_input (bool, optional): If True, print the dictionary created by MainContainer
            class after it is completed.

    """
    def __init__(self, print_input=False):
        Generate.__init__(self)
        self.generate_info = GenerateContainer()
        self.general = GeneralContainer()
        self.input_info = InputContainer()
        self.output_info = OutputContainer()
        self.print_input = print_input

    def return_overall_dictionary(self):
        """Output dictionary for ``generate_contour_data.py`` input.

        Iterates through the entire MainContainer class turning its contents
        into dictionary form. This dictionary becomes the input for ``generate_contour_data.py``.

        If `print_input` attribute is True, the entire dictionary will be printed
        prior to returning the dicitonary.

        Returns:
            - **output_dict** (*dict*): Dicitonary for input into ``generate_contour_data.py``.

        """
        output_dict = {}
        output_dict['general'] = self._iterate_through_class(self.general.__dict__)
        output_dict['generate_info'] = self._iterate_through_class(self.generate_info.__dict__)
        output_dict['input_info'] = self._iterate_through_class(self.input_info.__dict__)
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
