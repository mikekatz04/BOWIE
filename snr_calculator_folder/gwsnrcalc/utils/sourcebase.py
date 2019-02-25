from scipy import constants as ct
import numpy as np
from astropy.cosmology import Planck15 as cosmo

class SourceBase:
    def __init__(self, **kwargs):
        self.set_dist_type()

        for key, item in kwargs.items():
            setattr(key, item)

    def set_dist_type(self, dist_type='redshift'):
        self.dist_type = dist_type
        if self.dist_type not in ['redshift', 'luminosity_distance', 'comoving_distance']:
            raise ValueError("dist_type needs to be redshift, comoving_distance,"
                             + "or luminosity_distance")
        return

    def set_distance_unit(self, dist_unit='Mpc'):
        self.unit_in = dist_unit
        return

    def check_if_broadcasted(self, locals):
        # cast binary inputs to same shape
        if self.not_broadcasted:
            self.broadcast_and_set_attrs(locals)

        else:
            for key, arr in locals.items():
                if key == 'self':
                    continue
                setattr(self, key, arr)
        return

    def adjust_distances(self):

        unit_conv_to_Mpc = {'Gpc': 1e-3,
                            'Mpc': 1.0,
                            'kpc': 1e3,
                            'pc': 1e6,
                            'm': 1e6*ct.parsec}

        unit_conv_in = unit_conv_to_Mpc[self.unit_in]
        unit_conv_out = unit_conv_to_Mpc[self.unit_out]

        # based on distance inputs, need to find redshift and luminosity distance.
        if self.dist_type == 'redshift':
            self.z = self.z_or_dist
            self.dist = cosmo.luminosity_distance(self.z).value

        elif self.dist_type == 'luminosity_distance':
            z_in = np.logspace(-3, 3, 10000)
            lum_dis = cosmo.luminosity_distance(z_in).value

            self.dist = self.z_or_dist/unit_conv_in
            self.z = np.interp(self.dist, lum_dis, z_in)

        elif self.dist_type == 'comoving_distance':
            z_in = np.logspace(-3, 3, 10000)
            lum_dis = cosmo.luminosity_distance(z_in).value
            com_dis = cosmo.comoving_distance(z_in).value

            comoving_distance = self.z_or_dist/unit_conv_in
            self.z = np.interp(comoving_distance, com_dis, z_in)
            self.dist = np.interp(comoving_distance, com_dis, lum_dis)

        self.dist = self.dist*unit_conv_out  # based on source class
        return
