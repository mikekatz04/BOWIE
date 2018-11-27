import numpy as np
from scipy import interpolate
from snr_calculator.utils.readnoisecurves import read_noise_curve, combine_with_wd_noise

class SensitivityContainer:

    def __init__(self, **kwargs):

        prop_defaults = {
            'sensitivity_curves': ['LPA'],
            'add_wd_noise': 'Both',
            'wd_noise': 'HB_wd_noise',
            'phases': 'all',
            # TODO: add 'all' and 'full' capabilities
            'prefactor': 1.0,
            'num_points': 8192,
            'noise_type_in': 'ASD',
            'wd_noise_type_in': 'ASD',
        }

        for (prop, default) in prop_defaults.items():
                setattr(self, prop, kwargs.get(prop, default))

        self._prep_noise_interpolants()
        self.sensitivity_args = (self.noise_interpolants, self.phases,
                                 self.prefactor, self.num_points)

    def _prep_noise_interpolants(self):
        noise_lists = {}
        self.noise_interpolants = {}

        if isinstance(self.sensitivity_curves, str):
            self.sensitivity_curves = [self.sensitivity_curves]
        #elif isinstance(self.sensitivity_curves, list):
        #    if len(np.shape(self.sensitivity_curves)) == 1:
        #        self.sensitivity_curves = [self.sensitivity_curves]

        if isinstance(self.noise_type_in, list):
            if len(self.noise_type_in) != len(self.sensitivity_curves):
                raise ValueError('noise_type_in must have same shape as sensitivity_curves if it is'
                                 + 'provided as a list.'
                                 + 'If all curves are of the same type, provide a string.')

        else:
            assert isinstance(self.noise_type_in, str)
            self.noise_type_in = [self.noise_type_in for _ in self.sensitivity_curves]

        if isinstance(self.phases, str):
            self.phases = [self.phases]

        for num, sc in enumerate(self.sensitivity_curves):
            if isinstance(sc, str):
                f, h_n = read_noise_curve(sc, noise_type_in=self.noise_type_in[num],
                                          noise_type_out='char_strain')
                key = sc
            elif isinstance(sc, list):
                # TODO: add to docs if inputing special noise curve, make sure its char_strain/ use converter provided
                f, h_n = sc
                key = str(num)
            else:
                raise ValueError('Sensitivity curves must either be string'
                                 + 'or list containing f_n and asd_n.')

            noise_lists[key] = [f, h_n]

        if self.add_wd_noise.lower() == 'true' or self.add_wd_noise.lower() == 'both' or self.add_wd_noise.lower() == 'yes':
            if isinstance(self.wd_noise, str):
                f_n_wd, h_n_wd = read_noise_curve(self.wd_noise, noise_type_in=self.wd_noise_type_in, noise_type_out='char_strain')
            elif isinstance(self,wd_noise, list):
                f_n_wd, h_n_wd = self.wd_noise

            trans_dict = {}
            for sc in noise_lists.keys():
                f_n, h_n = noise_lists[sc]
                if self.add_wd_noise.lower() == 'both':
                    trans_dict[sc] = [f_n, h_n]

                f_n, h_n = combine_with_wd_noise(f_n, h_n, f_n_wd, h_n_wd)
                trans_dict[sc + '_wd'] = [f_n, h_n]

            noise_lists = trans_dict

        for sc in noise_lists:
            f_n, h_n = noise_lists[sc]
            self.noise_interpolants[sc] = (interpolate.interp1d(f_n, h_n,
                                            bounds_error=False, fill_value=1e30))

        return

    def adjust_noise_type(self, f, amp):
        if self.noise_type == 'ASD':
            return np.sqrt(f)*amp
        elif self.noise_type == 'PSD':
            return np.sqrt(f*amp)
        else:
            raise Exception('Noise type provided is {}. Must be ASD, PSD, or char_strain.'.format(self.noise_type))
