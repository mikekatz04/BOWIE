import numpy as np
from scipy import interpolate
from gwsnrcalc.utils.readnoisecurves import read_noise_curve, combine_with_wd_noise

# TODO: Change wd_background to background


class SensitivityContainer:
    """Sensitivity curve analysis

    This prepares the sensitivity side of the SNR calculation.
    It takes all the inputs and converts them to characterstic strain.

    Keyword Arguments:
        sensitivity_curves (scalar or list of str or single or list of lists, optional):
            String that starts the .txt file containing the sensitivity curve in
            folder 'noise_curves/' or list of ``[f_n, asd_n]``
            in terms of an amplitude spectral density. It can be a single one of these
            values or a list of these values. A folder string with absolute path to a sensitivity
            curve .txt file can also be input.
            Default is [`LPA`] (LISA Phase A).
        add_wd_noise (str or bool, optional): Options are `yes`, `no`, `True`, `False`,
            `Both`, True, or False. `yes`, `True`, or True
            will exclusively calculate with the wd noise.
            `no`, `False`, or False will exclusively calculate without the wd noise.
            `Both` will calculate with and without wd noise.
            Default is `Both`.
        wd_noise (str or list, optional): If string,
            read the wd noise from `noise_curves` folder or absolute path to file.
            If list, must be ``[f_n_wd, asd_n_wd]``.
            Default is Hils-Bender estimation (Bender & Hils 1997) by Hiscock et al. 2000.
        noise_type_in (str, optional): Type of noise curve passed in.
            Options are `ASD`, `PSD`, or `char_strain`.
            All sensitivity curves must have same noise type.
            Also, their amplitude column must have this same string as its label.
            Default is `ASD`.
        wd_noise_type_in (str, optional): Type of wd noise curve passed in.
            Options are `ASD`, `PSD`, or `char_strain`.
            Amplitude column must have this same string as its label.
            Default is `ASD`.

    Attributes:
        noise_interpolants (dict): Dictionary carrying scipy.interpolate.interp1d objects.
            These are the interpolations used to calculate the SNR.
        Note: All Keyword Arguments are added as attributes.

    """
    def __init__(self, **kwargs):

        # TODO: add 'all' and 'full' capabilities

        # set defaults on WD noise
        self.add_noise_curve('HB_wd_noise', noise_type='ASD', is_wd_background=True)
        self.set_wd_noise('both')

        for key, item in kwargs.items():
            setattr(self, key, item)

    def prep_noise(self):
        try:
            self.sensitivity_curves
        except AttributeError:
            self.sensitivity_curves = ['LPA']
            self.noise_type_in = ['ASD']

        self._prep_noise_interpolants()

    def _prep_noise_interpolants(self):
        """Construct interpolated sensitivity curves

        This will construct the interpolated sensitivity curves
        using scipy.interpolate.interp1d. It will add wd noise
        if that is requested.

        Raises:
            ValueError: ``len(noise_type_in) != len(sensitivity_curves)``
            ValueError: Issue with sensitivity curve type provided.

        """
        noise_lists = {}
        self.noise_interpolants = {}

        if isinstance(self.sensitivity_curves, str):
            self.sensitivity_curves = [self.sensitivity_curves]

        if isinstance(self.noise_type_in, list):
            if len(self.noise_type_in) != len(self.sensitivity_curves):
                raise ValueError('noise_type_in must have same shape as sensitivity_curves if it is'
                                 + 'provided as a list.'
                                 + 'If all curves are of the same type, provide a string.')

        else:
            assert isinstance(self.noise_type_in, str)
            self.noise_type_in = [self.noise_type_in for _ in self.sensitivity_curves]

        if isinstance(self.signal_type, str):
            self.signal_type = [self.signal_type]

        # read in all the noise curves
        for num, sc in enumerate(self.sensitivity_curves):
            if isinstance(sc, str):
                f, h_n = read_noise_curve(sc, noise_type_in=self.noise_type_in[num],
                                          noise_type_out='char_strain')
                if sc[-4:] == '.txt':
                    key = sc.split('/')[-1].split('.')[0]
                else:
                    key = sc
            elif isinstance(sc, list):
                # TODO: add to docs if inputing special noise curve, make sure its char_strain
                f, h_n = sc
                key = str(num)
            else:
                raise ValueError('Sensitivity curves must either be string'
                                 + 'or list containing f_n and asd_n.')

            noise_lists[key] = [f, h_n]

        # add wd noise
        if str(self.add_wd_noise).lower() in ['true', 'both', 'yes']:
            if isinstance(self.wd_noise, str):
                f_n_wd, h_n_wd = read_noise_curve(self.wd_noise,
                                                  noise_type_in=self.wd_noise_type_in,
                                                  noise_type_out='char_strain')
            elif isinstance(self, wd_noise, list):
                f_n_wd, h_n_wd = self.wd_noise

            trans_dict = {}
            for sc in noise_lists.keys():
                f_n, h_n = noise_lists[sc]
                if str(self.add_wd_noise).lower() == 'both':
                    trans_dict[sc] = [f_n, h_n]

                f_n, h_n = combine_with_wd_noise(f_n, h_n, f_n_wd, h_n_wd)
                trans_dict[sc + '_wd'] = [f_n, h_n]

            noise_lists = trans_dict

        # interpolate
        for sc in noise_lists:
            f_n, h_n = noise_lists[sc]
            self.noise_interpolants[sc] = (interpolate.interp1d(f_n, h_n,
                                           bounds_error=False, fill_value=1e30))
        return

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
            self.wd_noise = name
            self.wd_noise_type_in = noise_type

        else:
            if 'sensitivity_curves' not in self.__dict__:
                self.sensitivity_curves = []
            if 'noise_type_in' not in self.__dict__:
                self.noise_type_in = []

            self.sensitivity_curves.append(name)
            self.noise_type_in.append(noise_type)
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

        self.add_wd_noise = wd_noise
        return
