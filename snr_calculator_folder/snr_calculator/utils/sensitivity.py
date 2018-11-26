import numpy as np
from scipy import interpolate
from pyphenomd.utils.readnoisecurves import read_noise_curve

class SensitivityContainer:

    def __init__(self, **kwargs):

        prop_defaults = {
            'sensitivity_curves': ['LPA'],
            'wd_noise': None,
            'phases': 'all',
            # TODO: add 'all' and 'full' capabilities
            'prefactor': 1.0,
            'num_points': 8192,
            'noise_type': 'char_strain',
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

        if isinstance(self.phases, str):
            self.phases = [self.phases]

        for num, sc in enumerate(self.sensitivity_curves):
            if isinstance(sc, str):
                f, amp = read_noise_curve(sc, noise_type='char_strain')
                key = sc
            elif isinstance(sc, list):
                f, amp = sc
                key = str(num)
            else:
                raise ValueError('Sensitivity curves must either be string'
                                 + 'or list containing f_n and asd_n.')
            if self.noise_type != 'char_strain':
                h_n = self.adjust_noise_type(f, amp)
            else:
                h_n = amp

            noise_lists[key] = [f, h_n]

        if self.wd_noise is not None:
            if self.wd_noise is True:
                f_n_wd, h_n_wd = read_noise_curve('HB_wd_noise', noise_type=self.noise_type)
            elif isinstance(wd_noise, str):
                f_n_wd, h_n_wd = read_noise_curve(wd_noise, noise_type='char_strain')
            elif isinstance(wd_noise, list):
                f_n_wd, amp_n_wd = wd_noise[0], wd_noise[1]
                if self.wd_noise_type != 'char_strain':
                    h_n_wd = self.adjust_noise_type(f, amp_n_wd)

            for sc in noise_lists:
                f_n, h_n = noise_lists[sc]
                f_n, h_n = add_wd_noise(f_n, h_n, f_n_wd, h_n_wd)
                noise_lists[sc] = [f_n, asd_n]

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
