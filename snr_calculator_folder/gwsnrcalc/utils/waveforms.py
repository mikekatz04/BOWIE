"""
Author: Michael Katz guided by lal implimentation of PhenomD.
This was used in "Evaluating Black Hole Detectability with LISA" (arXiv:1508.07253),
as a part of the BOWIE package (https://github.com/mikekatz04/BOWIE).

This code is licensed with the GNU public license.

This python code impliments PhenomD waveforms from Husa et al 2016 (arXiv:1508.07250)
and Khan et al 2016 (arXiv:1508.07253). It wraps the accompanying c code, `phenomd/phenomd.c`,
with ``ctypes``. `phenomd/phenomd.c` is mostly from LALsuite. See `phenomd/phenomd.c` for specifics.

Please cite all of the arXiv papers above if you use this code in a publication.

"""

import ctypes
from astropy.cosmology import Planck15 as cosmo
import numpy as np
import os
import scipy.constants as ct
from scipy.special import jv  # bessel function of the first kind

M_sun = 1.989e30  # kg


class PhenomDWaveforms:
    """Generate phenomd waveforms

    PhenomDWaveforms is a class that takes binary parameters as inputs, and adds
    characteristic strain waveforms as attributes to self.

    Keyword Arguments:
        dist_type (str, optional): Which type of distance is used. Default is 'redshift'.
            This is stored as an attributed.
        num_points (int, optional): Number of points to use in the waveform.
            The frequency points are log-spaced. Default is 8192. This is stored as an attributed.

    Attributes:
        freqs (1D or 2D array of floats): Frequencies corresponding to the waveforms.
            Shape is (num binaries, num_points) if 2D.
            Shape is (num_points,) if 1D for one binary.
        hc (1D or 2D array of floats): Characteristic strain of the waveforms.
            Shape is (num binaries, num_points) if 2D.
            Shape is (num_points,) if 1D for one binary.
        fmrg: (scalar float or 1D array of floats): Merger frequency of each binary separating
            inspiral from merger phase. (0.014/M) Shape is (num binaries,) if more than one binary.
        fpeak: (scalar float or 1D array of floats): Peak frequency of each binary separating
            merger from ringdown phase. (0.014/M) Shape is (num binaries,) if more than one binary.
        z (float or 1D array of floats): Redshift equivalent of the z_or_dist given.
        dist (float or 1D array of floats): Luminosity distance equivalent of the z_or_dist given.
        remove_axis (bool): Remove axis based on if it is one or more than one binary.
        length (int): Number of binaries.
        Note: All args from :meth:`gwsnrcalc.utils.pyphenomd.PhenomDWaveforms.__call__`
                are stored as attributes.
        Note: All kwargs from above are stored as attributes.

    Raises:
        ValueError: dist_type is not one of the three options.

    """

    def __init__(self, **kwargs):
        prop_defaults = {
            # TODO: add 'all' and 'full' capabilities
            'num_points': 8192,
            'dist_type': 'redshift'
        }

        for (prop, default) in prop_defaults.items():
                setattr(self, prop, kwargs.get(prop, default))

        # get path to c code
        cfd = os.path.dirname(os.path.abspath(__file__))
        if 'phenomd.cpython-35m-darwin.so' in os.listdir(cfd):
            self.exec_call = cfd + '/phenomd.cpython-35m-darwin.so'

        else:
            self.exec_call = cfd + '/phenomd/phenomd.so'

    def _sanity_check(self):
        """Check if parameters are okay.

        Sanity check makes sure each parameter is within an allowable range.

        Raises:
            ValueError: Problem with a specific parameter.

        """
        if any(self.m1 < 0.0):
            raise ValueError("Mass 1 is negative.")
        if any(self.m2 < 0.0):
            raise ValueError("Mass 2 is negative.")

        if any(self.chi_1 < -1.0) or any(self.chi_1 > 1.0):
            raise ValueError("Chi 1 is outside [-1.0, 1.0].")

        if any(self.chi_2 < -1.0) or any(self.chi_2 > 1.0):
            raise ValueError("Chi 2 is outside [-1.0, 1.0].")

        if any(self.z <= 0.0):
            raise ValueError("Redshift is zero or negative.")

        if any(self.dist <= 0.0):
            raise ValueError("Distance is zero or negative.")

        if any(self.st < 0.0):
            raise ValueError("Start Time is negative.")

        if any(self.et < 0.0):
            raise ValueError("End Time is negative.")

        if len(np.where(self.st < self.et)[0]) != 0:
            raise ValueError("Start Time is less than End time.")

        if any(self.m1/self.m2 > 1.0000001e4) or any(self.m1/self.m2 < 9.999999e-5):
            raise ValueError("Mass Ratio too far from unity.")

        return

    def _broadcast_and_set_attrs(self, local_dict):
        """Cast all inputs to correct dimensions.

        This method fixes inputs who have different lengths. Namely one input as
        an array and others that are scalara or of len-1.

        Raises:
            Value Error: Multiple length arrays of len>1

        """
        del local_dict['self']
        self.remove_axis = False
        max_length = 0
        for key in local_dict:
            try:
                length = len(local_dict[key])
                if length > max_length:
                    max_length = length

            except TypeError:
                pass

        if max_length == 0:
            self.remove_axis = True
            for key in local_dict:
                setattr(self, key, np.array([local_dict[key]]))

        # check for bad length arrays
        else:
            for key in local_dict:
                try:
                    if len(local_dict[key]) < max_length and len(local_dict[key]) > 1:
                        raise ValueError("Casting parameters not correct."
                                         + " Need all at a maximum shape and the rest being"
                                         + "len-1 arrays or scalars")
                except TypeError:
                    pass

            # broadcast arrays
            for key in local_dict:
                try:
                    if len(local_dict[key]) == max_length:
                        setattr(self, key, local_dict[key])
                    elif len(local_dict[key]) == 1:
                        setattr(self, key, np.full((max_length,), local_dict[key][0]))
                except TypeError:
                    setattr(self, key, np.full((max_length,), local_dict[key]))
        return

    def _create_waveforms(self):
        """Create frequency domain waveforms.

        Method to create waveforms for PhenomDWaveforms class.
        It adds waveform information in the form of attributes.

        """

        c_obj = ctypes.CDLL(self.exec_call)

        # prepare ctypes arrays
        freq_amp_cast = ctypes.c_double*self.num_points*self.length
        freqs = freq_amp_cast()
        hc = freq_amp_cast()

        fmrg_fpeak_cast = ctypes.c_double*self.length
        fmrg = fmrg_fpeak_cast()
        fpeak = fmrg_fpeak_cast()

        # Find hc
        c_obj.Amplitude(ctypes.byref(freqs), ctypes.byref(hc), ctypes.byref(fmrg),
                        ctypes.byref(fpeak),
                        self.m1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        self.m2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        self.chi_1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        self.chi_2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        self.dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        self.z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        self.st.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        self.et.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        ctypes.c_int(self.length), ctypes.c_int(self.num_points))

        # turn output into numpy arrays
        self.freqs, self.hc = np.ctypeslib.as_array(freqs), np.ctypeslib.as_array(hc)
        self.fmrg, self.fpeak = np.ctypeslib.as_array(fmrg), np.ctypeslib.as_array(fpeak)

        # remove an axis if inputs were scalar.
        if self.remove_axis:
            self.freqs, self.hc, = np.squeeze(self.freqs), np.squeeze(self.hc)
            self.fmrg, self.fpeak = self.fmrg[0], self.fpeak[0]

        return

    def __call__(self, m1, m2, chi_1, chi_2, z_or_dist, st, et):
        """Run the waveform creator.

        Create phenomd waveforms in the amplitude frequency domain.

        **Warning**: Binary parameters have to one of three shapes: scalar, len-1 array,
        or array of len MAX. All scalar quantities or len-1 arrays are cast to len-MAX arrays.
        If arrays of different lengths (len>1) are given, a ValueError will be raised.

        Arguments:
            m1 (float or 1D array of floats): Mass 1 in Solar Masses. (>0.0)
            m2 (float or 1D array of floats): Mass 2 in Solar Masses. (>0.0)
            chi_1 (float or 1D array of floats): dimensionless spin of mass 1
                aligned to orbital angular momentum. Default is None (not 0.0). [-1.0, 1.0]
            chi_2 (float or 1D array of floats): dimensionless spin of mass 2
                aligned to orbital angular momentum. Default is None (not 0.0). [-1.0, 1.0]
            z_or_dist (float or 1D array of floats): Distance measure to the binary.
                This can take three forms: redshift (dimensionless, *default*),
                luminosity distance (Mpc), comoving_distance (Mpc).
                The type used must be specified in 'dist_type' parameter. (>0.0)
            st (float or 1D array of floats): Start time of waveform in years before
                end of the merger phase. This is determined using 1 PN order. (>0.0)
            et (float or 1D array of floats): End time of waveform in years before
                end of the merger phase. This is determined using 1 PN order. (>0.0)

        """
        # cast binary inputs to same shape
        self._broadcast_and_set_attrs(locals())

        # based on distance inputs, need to find redshift and luminosity distance.
        if self.dist_type == 'redshift':
            self.z = self.z_or_dist
            self.dist = cosmo.luminosity_distance(self.z).value

        elif self.dist_type == 'luminosity_distance':
            z_in = np.logspace(-3, 3, 10000)
            lum_dis = cosmo.luminosity_distance(z_in).value

            self.dist = self.z_or_dist
            self.z = np.interp(self.dist, lum_dis, z_in)

        elif self.dist_type == 'comoving_distance':
            z_in = np.logspace(-3, 3, 10000)
            lum_dis = cosmo.luminosity_distance(z_in).value
            com_dis = cosmo.comoving_distance(z_in).value

            comoving_distance = self.z_or_dist
            self.z = np.interp(comoving_distance, com_dis, z_in)
            self.dist = np.interp(comoving_distance, com_dis, lum_dis)

        else:
            raise ValueError("dist_type needs to be redshift, comoving_distance,"
                             + "or luminosity_distance")

        self.length = len(self.m1)
        self._sanity_check()
        self._create_waveforms()
        return self


class EccentricBinaries:
    def __init__(self, **kwargs):
        prop_default = {
            'dist_type': 'redshift',
            'initial_cond_type': 'frequency',
            'num_points': 1024,
            'n_max': 100,
        }

        for prop, default in prop_default.items():
            setattr(self, prop, kwargs.get(prop, default))

    def _broadcast_and_set_attrs(self, local_dict):
        """Cast all inputs to correct dimensions.

        This method fixes inputs who have different lengths. Namely one input as
        an array and others that are scalara or of len-1.

        Raises:
            Value Error: Multiple length arrays of len>1

        """
        del local_dict['self']
        self.remove_axis = False
        max_length = 0
        for key in local_dict:
            try:
                length = len(local_dict[key])
                if length > max_length:
                    max_length = length

            except TypeError:
                pass

        if max_length == 0:
            self.remove_axis = True
            for key in local_dict:
                setattr(self, key, np.array([local_dict[key]]))

        # check for bad length arrays
        else:
            for key in local_dict:
                try:
                    if len(local_dict[key]) < max_length and len(local_dict[key]) > 1:
                        raise ValueError("Casting parameters not correct."
                                         + " Need all at a maximum shape and the rest being"
                                         + "len-1 arrays or scalars")
                except TypeError:
                    pass

            # broadcast arrays
            for key in local_dict:
                try:
                    if len(local_dict[key]) == max_length:
                        setattr(self, key, local_dict[key])
                    elif len(local_dict[key]) == 1:
                        setattr(self, key, np.full((max_length,), local_dict[key][0]))
                except TypeError:
                    setattr(self, key, np.full((max_length,), local_dict[key]))
        return

    def _sanity_check(self):
        """Check if parameters are okay.

        Sanity check makes sure each parameter is within an allowable range.

        Raises:
            ValueError: Problem with a specific parameter.

        """
        if any(self.m1 < 0.0):
            raise ValueError("Mass 1 is negative.")
        if any(self.m2 < 0.0):
            raise ValueError("Mass 2 is negative.")

        if any(self.z <= 0.0):
            raise ValueError("Redshift is zero or negative.")

        if any(self.dist <= 0.0):
            raise ValueError("Distance is zero or negative.")

        if any(self.initial_point < 0.0):
            raise ValueError("initial_point is negative.")

        if any(self.t_obs < 0.0):
            raise ValueError("t_obs is negative.")

        return

    def _convert_units(self):
        self.m1 = self.m1*M_sun*ct.G/ct.c**2
        self.m2 = self.m2*M_sun*ct.G/ct.c**2
        initial_cond_type_conversion = {
            'time': ct.c*ct.Julian_year,
            'frequency': 1./ct.c,
            'separation': ct.parsec,
        }

        self.initial_point = self.initial_point*initial_cond_type_conversion[self.initial_cond_type]

        self.t_obs = self.t_obs*ct.c*ct.Julian_year
        return

    def _find_integrand(self, e):
        return e**(29./19.)*(1.+(121./304.)*e**2.)**(1181./2299.)/(1.-e**2)**(3./2.)

    def _f_e(self, e):
        return e**(12./19.)*(1.+(121./304.)*e**2.)**(870./2299.)/(1.-e**2)

    def _c0_func(self, a0, e0):
        return a0*(1.-e0**2)/e0**(12./19.) * (1.+(121./304.)*e0**2)**(-870./2299.)

    def _t_of_e(self, a0=None, t_start=None, f0=None, ef=None, t_obs=5.0):

        if ef is None:
            ef = np.ones_like(self.e0)*0.0000001

        beta = 64.0/5.0*self.m1*self.m2*(self.m1+self.m2)

        e_vals = np.asarray([np.linspace(ef[i], self.e0[i], self.num_points) for i in range(len(self.e0))])
        integrand = self._find_integrand(e_vals)
        integral = np.asarray([np.trapz(integrand[:,i:], x=e_vals[:,i:]) for i in range(e_vals.shape[1])]).T

        if a0 is None and f0 is None:

            a0 = (19./12.*t_start*beta*1/integral[:,0])**(1./4.) * self._f_e(e_vals[:,-1])

        elif a0 is None:
            a0 = ((self.m1 + self.m2)/self.f0**2)**(1./3.)

        c0 = self._c0_func(a0, self.e0)

        a_vals = c0[:, np.newaxis]*self._f_e(e_vals)

        delta_t = 12./19*c0[:, np.newaxis]**4/beta[:, np.newaxis]*integral

        return e_vals, a_vals, delta_t

    def _chirp_mass(self):
        return (self.m1*self.m2)**(3./5.)/(self.m1+self.m2)**(1./5.)

    def _f_func(self):
        return (1.+(73./24.)*self.e_vals**2.+(37./96.)*self.e_vals**4.)/(1.-self.e_vals**2.)**(7./2.)

    def _g_func(self):
        return self.n**4./32. * ((jv(self.n-2., self.n*self.e_vals) - 2.*self.e_vals*jv(self.n-1., self.n*self.e_vals) + 2./self.n*jv(self.n, self.n*self.e_vals) + 2.*self.e_vals*jv(self.n+1., self.n*self.e_vals) - jv(self.n+2., self.n*self.e_vals))**2. + (1.-self.e_vals**2.)*(jv(self.n-2., self.n*self.e_vals) - 2.*jv(self.n, self.n*self.e_vals) + jv(self.n+2., self.n*self.e_vals))**2. + 4./(3.*self.n**2.)*(jv(self.n, self.n*self.e_vals))**2.)

    def _dEndfr(self):
        #eq 4 from orazio and samsing
        # takes f in rest frame
        Mc = self._chirp_mass()
        return np.pi**(2./3.)*Mc**(5./3.)/(3.*(1.+self.z)**(1./3.)*(self.freqs_orb/(1.+self.z))**(1./3.))*(2./self.n)**(2./3.)*self._g_func()/self._f_func()

    def _hcn_func(self):
        self.hc = 1./(np.pi*self.dist)*np.sqrt(2.*self._dEndfr())
        return

    def _create_waveforms(self):

        e_vals, a_vals, t_vals = self._t_of_e(a0=self.a0, f0=self.f0, t_start=self.t_start, ef=None, t_obs=self.t_obs)

        ind_start = np.asarray([np.where(t_vals[i]<self.t_obs[i])[0][0] for i in range(len(t_vals))])

        self.ef = np.asarray([e_vals[i][ind] for i, ind in enumerate(ind_start)])

        self.e_vals, self.a_vals, self.t_vals = self._t_of_e(a0=a_vals[:,-1], ef=self.ef, t_obs=self.t_obs)

        self.freqs_orb = np.sqrt((self.m1[:, np.newaxis]+self.m2[:, np.newaxis])/self.a_vals**3)

        for attr in ['e_vals', 'a_vals', 't_vals', 'freqs_orb']:
            arr = getattr(self, attr)
            new_arr = np.flip(np.tile(arr, self.n_max).reshape(len(arr)*self.n_max, len(arr[0])), -1)
            setattr(self, attr, new_arr)

        for attr in ['m1', 'm2', 'z', 'dist']:
            arr = getattr(self, attr)
            new_arr = np.repeat(arr, self.n_max)[:, np.newaxis]
            setattr(self, attr, new_arr)

        self.n = np.tile(np.arange(1, self.n_max + 1), self.length)[:, np.newaxis]

        self._hcn_func()

        # reshape hc
        self.hc = self.hc.reshape(self.length, self.n_max, self.hc.shape[-1])
        self.freqs = np.reshape(self.n*self.freqs_orb/(1+self.z)*ct.c, (self.length, self.n_max, self.freqs_orb.shape[-1]))

        self.hc, self.freqs = np.squeeze(self.hc), np.squeeze(self.freqs)
        return

    def __call__(self, m1, m2, z_or_dist, initial_point, e0, t_obs):

        # cast binary inputs to same shape
        self._broadcast_and_set_attrs(locals())

        self.f0 = None
        self.t_start = None
        self.a0 = None

        self._convert_units()

        initial_cond_set_attr = {
            'time': 't_start',
            'frequency': 'f0',
            'separation': 'a0'
        }

        setattr(self, initial_cond_set_attr[self.initial_cond_type], self.initial_point)

        # based on distance inputs, need to find redshift and luminosity distance.
        if self.dist_type == 'redshift':
            self.z = self.z_or_dist
            self.dist = cosmo.luminosity_distance(self.z).value*ct.parsec*1e6

        elif self.dist_type == 'luminosity_distance':
            z_in = np.logspace(-3, 3, 10000)
            lum_dis = cosmo.luminosity_distance(z_in).value

            self.dist = self.z_or_dist*ct.parsec*1e6
            self.z = np.interp(self.dist, lum_dis, z_in)

        elif self.dist_type == 'comoving_distance':
            z_in = np.logspace(-3, 3, 10000)
            lum_dis = cosmo.luminosity_distance(z_in).value
            com_dis = cosmo.comoving_distance(z_in).value

            comoving_distance = self.z_or_dist
            self.z = np.interp(comoving_distance, com_dis, z_in)
            self.dist = np.interp(comoving_distance, com_dis, lum_dis)*ct.parsec*1e6

        else:
            raise ValueError("dist_type needs to be redshift, comoving_distance,"
                             + "or luminosity_distance")

        self.length = len(self.m1)
        self._sanity_check()
        self._create_waveforms()
        return self
