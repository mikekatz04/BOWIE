import numpy as np
from scipy import interpolate
from gwsnrcalc.utils.readnoisecurves import read_noise_curve, combine_with_wd_noise


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
        prop_defaults = {
            'sensitivity_curves': ['LPA'],
            'add_wd_noise': 'Both',
            'wd_noise': 'HB_wd_noise',
            # TODO: add 'all' and 'full' capabilities
            'noise_type_in': 'ASD',
            'wd_noise_type_in': 'ASD',
        }

        for (prop, default) in prop_defaults.items():
                setattr(self, prop, kwargs.get(prop, default))

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
                    key = sc.split('.')[0].split('/')[-1]
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
                if self.add_wd_noise.lower() == 'both':
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
