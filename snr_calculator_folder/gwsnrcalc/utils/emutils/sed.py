#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011, 2012 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

"""
sed -

Class data:
wavelen (nm)
flambda (ergs/cm^2/s/nm)
fnu (Jansky)
zp  (basically translates to units of fnu = -8.9 (if Janskys) or 48.6 (ergs/cm^2/s/hz))
the name of the sed file

It is important to note the units are NANOMETERS, not ANGSTROMS. It is possible to rig this so you can
use angstroms instead of nm, but you should know what you're doing and understand the wavelength grid
limits applied here and in Bandpass.py.

Methods:
 Because of how these methods will be applied for catalog generation, (taking one base SED and then
  applying various dust extinctions and redshifts), many of the methods will either work on,
  and update self, OR they can be given a set of lambda/flambda arrays and then will return
  new versions of these arrays. In general, the methods will not explicitly set flambda or fnu to
  something you (the user) did not specify - so, for example, when calculating magnitudes (which depend on
  a wavelength/fnu gridded to match the given bandpass) the wavelength and fnu used are temporary copies
  and the object itself is not changed.
 In general, the philosophy of Sed.py is to not define the wavelength grid for the object until necessary
  (so, not until needed for the magnitude calculation or resampleSED is called). At that time the min/max/step
  wavelengths or the bandpass wavelengths are used to define a new wavelength grid for the sed object.
 When considering whether to use the internal wavelen/flambda (self) values, versus input values:
  For consistency, anytime self.wavelen/flambda is used, it will be updated if the values are changed
  (except in the special case of calculating magnitudes), and if self.wavelen/flambda is updated,
  self.fnu will be set to None. This is because many operations are typically chained together
  which alter flambda -- so it is more efficient to wait and recalculate fnu at the end, plus it
  avoids possible de-synchronization errors (flambda reflecting the addition of dust while fnu does
  not, for example). If arrays are passed into a method, they will not be altered and the arrays
  which are returned will be allocated new memory.
 Another general philosophy for Sed.py is use separate methods for items which only need to be generated once
  for several objects (such as the dust A_x, b_x arrays). This allows the user to optimize their code for
  faster operation, depending on what their requirements are (see example_SedBandpass_star.py and
  exampleSedBandpass_galaxy for examples).

Method include:
  setSED / setFlatSED / readSED_flambda / readSED_fnu -- to input information into Sed wavelen/flambda.
  getSED_flambda / getSED_fnu -- to return wavelen / flambda or fnu to the user.
  clearSED -- set everything to 0.
  synchronizeSED -- to calculate wavelen/flambda/fnu on the desired grid and calculate fnu.
  _checkUseSelf/needResample -- not expected to be useful to the user, rather intended for internal use.
  resampleSED -- primarily internal use, but may be useful to user. Resamples SED onto specified grid.
  flambdaTofnu / fnuToflambda -- conversion methods, does not affect wavelen gridding.
  redshiftSED -- redshifts the SED, optionally adding dimmingx
  (setupODonnell_ab or setupCCM_ab) / addDust -- separated into two components, so that a_x/b_x can be reused between SEDS
if the wavelength range and grid is the same for each SED (calculate a_x/b_x with either setupODonnell_ab
or setupCCM_ab).
  multiplySED -- multiply two SEDS together.
  calcADU / calcMag / calcFlux -- with a Bandpass, calculate the ADU/magnitude/flux of a SED.
  calcFluxNorm / multiplyFluxNorm -- handle fluxnorm parameters (from UW LSST database) properly.
     These methods are intended to give a user an easy way to scale an SED to match an expected magnitude.
  renormalizeSED  -- intended for rescaling SEDS to a common flambda or fnu level.
  writeSED -- keep a file record of your SED.
  setPhiArray -- given a list of bandpasses, sets up the 2-d phiArray (for manyMagCalc) and dlambda value.
  manyMagCalc -- given 2-d phiArray and dlambda, this will return an array of magnitudes (in the same
order as the bandpasses) of this SED in each of those bandpasses.

"""

from __future__ import with_statement
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
import warnings
import numpy as np
import sys
import time
import scipy.interpolate as interpolate
import gzip
import pickle
import os
from .physicalparameters import PhysicalParameters
import warnings
try:
    from lsst.utils import getPackageDir
except:
    pass


# since Python now suppresses DeprecationWarnings by default
warnings.filterwarnings("default", category=DeprecationWarning, module='lsst.sims.photUtils.Sed')


__all__ = ["Sed", "cache_LSST_seds", "read_close_Kurucz"]


class Sed(object):
    """Class for holding and utilizing spectral energy distributions (SEDs)"""
    def __init__(self, wavelen=None, flambda=None, fnu=None, badval=np.NaN, name=None):
        """
        Initialize sed object by giving filename or lambda/flambda array.

        Note that this does *not* regrid flambda and leaves fnu undefined.
        """
        self.fnu = None
        self.wavelen = None
        self.flambda = None
        # self.zp = -8.9  # default units, Jansky.
        self.zp = -2.5*np.log10(3631)
        self.name = name
        self.badval = badval

        self._physParams = PhysicalParameters()

        # If init was given data to initialize class, use it.
        if (wavelen is not None) and ((flambda is not None) or (fnu is not None)):
            if name is None:
                name = 'FromArray'
            self.setSED(wavelen, flambda=flambda, fnu=fnu, name=name)
        return

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if self.zp != other.zp:
            return False
        if not np.isnan(self.badval):
            if self.badval != other.badval:
                return False
        else:
            if not np.isnan(other.badval):
                return False
        if self.fnu is not None and other.fnu is None:
            return False
        if self.fnu is None and other.fnu is not None:
            return False
        if self.fnu is not None:
            try:
                np.testing.assert_array_equal(self.fnu, other.fnu)
            except:
                return False

        if self.flambda is None and other.flambda is not None:
            return False
        if other.flambda is not None and self.flambda is None:
            return False
        if self.flambda is not None:
            try:
                np.testing.assert_array_equal(self.flambda, other.flambda)
            except:
                return False

        if self.wavelen is None and other.wavelen is not None:
            return False
        if self.wavelen is not None and other.wavelen is None:
            return False
        if self.wavelen is not None:
            try:
                np.testing.assert_array_equal(self.wavelen, other.wavelen)
            except:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    # Methods for getters and setters.

    def setFlatSED(self, wavelen_min=None,
                   wavelen_max=None,
                   wavelen_step=None, name='Flat'):
        """
        Populate the wavelength/flambda/fnu fields in sed according to a flat fnu source.
        """
        if wavelen_min is None:
            wavelen_min = self._physParams.minwavelen

        if wavelen_max is None:
            wavelen_max = self._physParams.maxwavelen

        if wavelen_step is None:
            wavelen_step = self._physParams.wavelenstep

        self.wavelen = np.arange(wavelen_min, wavelen_max+wavelen_step, wavelen_step, dtype='float')
        self.fnu = np.ones(len(self.wavelen), dtype='float') * 3631  # jansky
        self.fnuToflambda()
        self.name = name
        return

    def readSED_flambda(self, filename, name=None):
        """
        Read a file containing [lambda Flambda] (lambda in nm) (Flambda erg/cm^2/s/nm).

        Does not resample wavelen/flambda onto grid; leave fnu=None.
        """
        dtype = np.dtype([('wavelen', float), ('flambda', float)])
        data = np.genfromtxt(filename, dtype=dtype)

        sourcewavelen = data['wavelen']
        sourceflambda = data['flambda']

        self.wavelen = sourcewavelen
        self.flambda = sourceflambda
        self.fnu = None
        if name is None:
            self.name = filename
        else:
            self.name = name
        return

    # Utilities common to several later methods.

    def _checkUseSelf(self, wavelen, flux):
        """
        Simple utility to check if should be using self's data or passed arrays.

        Also does data integrity check on wavelen/flux if not self.
        """
        update_self = False
        if (wavelen is None) or (flux is None):
            # Then one of the arrays was not passed - check if this is true for both arrays.
            if (wavelen is not None) or (flux is not None):
                # Then one of the arrays was passed - raise exception.
                raise ValueError("Must either pass *both* wavelen/flux pair, or use defaults.")
            update_self = True
        else:
            # Both of the arrays were passed in - check their validity.
            if (isinstance(wavelen, np.ndarray) is False) or (isinstance(flux, np.ndarray) is False):
                raise ValueError("Must pass wavelen/flux as np arrays.")
            if len(wavelen) != len(flux):
                raise ValueError("Must pass equal length wavelen/flux arrays.")
        return update_self

    def _needResample(self, wavelen_match=None, wavelen=None,
                      wavelen_min=None, wavelen_max=None, wavelen_step=None):
        """
        Check if wavelen or self.wavelen matches wavelen or wavelen_min/max/step grid.
        """
        # Check if should use self or passed wavelen.
        if wavelen is None:
            wavelen = self.wavelen
        # Check if wavelength arrays are equal, if wavelen_match passed.
        if wavelen_match is not None:
            if np.shape(wavelen_match) != np.shape(wavelen):
                need_regrid = True
            else:
                # check the elements to see if any vary
                need_regrid = np.any(abs(wavelen_match-wavelen) > 1e-10)
        else:
            need_regrid = True
            # Check if wavelen_min/max/step are set - if ==None, then return (no regridding).
            # It's possible (writeSED) to call this routine, even with no final grid in mind.
            if ((wavelen_min is None) and (wavelen_max is None) and (wavelen_step is None)):
                need_regrid = False
            else:
                # Okay, now look at comparison of wavelen to the grid.
                wavelen_max_in = wavelen[len(wavelen)-1]
                wavelen_min_in = wavelen[0]
                # First check match to minimum/maximum :
                if ((wavelen_min_in == wavelen_min) and (wavelen_max_in == wavelen_max)):
                    # Then check on step size in wavelength array.
                    stepsize = np.unique(np.diff(wavelen))
                    if (len(stepsize) == 1) and (stepsize[0] == wavelen_step):
                        need_regrid = False
        # At this point, need_grid=True unless it's proven to be False, so return value.
        return need_regrid

    def resampleSED(self, wavelen=None, flux=None, wavelen_match=None,
                    wavelen_min=None, wavelen_max=None, wavelen_step=None, force=False):
        """
        Resample flux onto grid defined by min/max/step OR another wavelength array.

        Give method wavelen/flux OR default to self.wavelen/self.flambda.
        Method either returns wavelen/flambda (if given those arrays) or updates wavelen/flambda in self.
         If updating self, resets fnu to None.
         Method will first check if resampling needs to be done or not, unless 'force' is True.
        """
        # Check if need resampling:
        if force or (self._needResample(wavelen_match=wavelen_match, wavelen=wavelen, wavelen_min=wavelen_min,
                                        wavelen_max=wavelen_max, wavelen_step=wavelen_step)):
            # Is method acting on self.wavelen/flambda or passed in wavelen/flux arrays?
            update_self = self._checkUseSelf(wavelen, flux)
            if update_self:
                wavelen = self.wavelen
                flux = self.flambda
                self.fnu = None
            # Now, on with the resampling.
            # Set up gridded wavelength or copy of wavelen array to match.
            if wavelen_match is None:
                if ((wavelen_min is None) and (wavelen_max is None) and (wavelen_step is None)):
                    raise ValueError('Must set either wavelen_match or wavelen_min/max/step.')
                wavelen_grid = np.arange(wavelen_min, wavelen_max+wavelen_step,
                                            wavelen_step, dtype='float')
            else:
                wavelen_grid = np.copy(wavelen_match)
            # Check if the wavelength range desired and the wavelength range of the object overlap.
            # If there is any non-overlap, raise warning.
            # if (wavelen.max() < wavelen_grid.max()) or (wavelen.min() > wavelen_grid.min()):
            #    warnings.warn('There is an area of non-overlap between desired wavelength range '
            #                  + ' (%.2f to %.2f)' % (wavelen_grid.min(), wavelen_grid.max())
            #                  + 'and sed %s (%.2f to %.2f)' % (self.name, wavelen.min(), wavelen.max()))
            # Do the interpolation of wavelen/flux onto grid. (type/len failures will die here).
            try:
                flux_grid = np.asarray([interpolate.interp1d(wave, fl, bounds_error=False,
                                        fill_value=0.0)(wavelen_grid) for wave, fl
                                        in zip(wavelen, flux)])
            except TypeError:
                flux_grid = interpolate.interp1d(wavelen, flux, bounds_error=False,
                                                 fill_value=0.0)(wavelen_grid)

            flux_grid[np.isnan(flux_grid)] = 0.0

            # Update self values if necessary.
            if update_self:
                self.wavelen = wavelen_grid
                self.flambda = flux_grid
                return
            return wavelen_grid, flux_grid
        else:  # wavelength grids already match.
            update_self = self._checkUseSelf(wavelen, flux)
            if update_self:
                return
            return wavelen, flux

    def flambdaTofnu(self, wavelen=None, flambda=None):
        """
        Convert flambda into fnu.

        This routine assumes that flambda is in ergs/cm^s/s/nm and produces fnu in Jansky.
        Can act on self or user can provide wavelen/flambda and get back wavelen/fnu.
        """
        # Change Flamda to Fnu by multiplying Flambda * lambda^2 = Fv
        # Fv dv = Fl dl .. Fv = Fl dl / dv = Fl dl / (dl*c/l/l) = Fl*l*l/c
        # Check - Is the method acting on self.wavelen/flambda/fnu or passed wavelen/flambda arrays?
        update_self = self._checkUseSelf(wavelen, flambda)
        if update_self:
            wavelen = self.wavelen
            flambda = self.flambda
            self.fnu = None
        # Now on with the calculation.
        # Calculate fnu.
        fnu = flambda * wavelen * wavelen * self._physParams.nm2m / self._physParams.lightspeed
        fnu = fnu * self._physParams.ergsetc2jansky
        # If are using/updating self, then *all* wavelen/flambda/fnu will be gridded.
        # This is so wavelen/fnu AND wavelen/flambda can be kept in sync.
        if update_self:
            self.wavelen = wavelen
            self.flambda = flambda
            self.fnu = fnu
            return
        # Return wavelen, fnu, unless updating self (then does not return).
        return wavelen, fnu

    def fnuToflambda(self, wavelen=None, fnu=None):
        """
        Convert fnu into flambda.

        Assumes fnu in units of Jansky and flambda in ergs/cm^s/s/nm.
        Can act on self or user can give wavelen/fnu and get wavelen/flambda returned.
        """
        # Fv dv = Fl dl .. Fv = Fl dl / dv = Fl dl / (dl*c/l/l) = Fl*l*l/c
        # Is method acting on self or passed arrays?
        update_self = self._checkUseSelf(wavelen, fnu)
        if update_self:
            wavelen = self.wavelen
            fnu = self.fnu
        # On with the calculation.
        # Calculate flambda.
        flambda = fnu / wavelen / wavelen * self._physParams.lightspeed / self._physParams.nm2m
        flambda = flambda / self._physParams.ergsetc2jansky
        # If updating self, then *all of wavelen/fnu/flambda will be updated.
        # This is so wavelen/fnu AND wavelen/flambda can be kept in sync.
        if update_self:
            self.wavelen = wavelen
            self.flambda = flambda
            self.fnu = fnu
            return
        # Return wavelen/flambda.
        return wavelen, flambda

    # methods to alter the sed

    def redshiftSED(self, redshift, dimming=False, wavelen=None, flambda=None):
        """
        Redshift an SED, optionally adding cosmological dimming.

        Pass wavelen/flambda or redshift/update self.wavelen/flambda (unsets fnu).
        """
        # Updating self or passed arrays?
        update_self = self._checkUseSelf(wavelen, flambda)
        if update_self:
            wavelen = self.wavelen
            flambda = self.flambda
            self.fnu = None
        else:
            # Make a copy of input data, because will change its values.
            wavelen = np.copy(wavelen)
            flambda = np.copy(flambda)
        # Okay, move onto redshifting the wavelen/flambda pair.
        # Or blueshift, as the case may be.
        wavelen = ((wavelen[np.newaxis, :] / (1.0-redshift[:, np.newaxis]))*(redshift[:, np.newaxis] < 0.0)
                   + wavelen[np.newaxis, :] * (1.0+redshift[:, np.newaxis])*(redshift[:, np.newaxis] >= 0.0))
        # Flambda now just has different wavelength for each value.
        # Add cosmological dimming if required.
        if dimming:
            flambda = ((flambda[np.newaxis, :] / (1.0-redshift[:, np.newaxis]))*(redshift[:, np.newaxis] < 0.0)
                       + flambda[np.newaxis, :] * (1.0+redshift[:, np.newaxis])*(redshift[:, np.newaxis] >= 0.0))
        # Update self, if required - but just flambda (still no grid required).
        if update_self:
            self.wavelen = wavelen
            self.flambda = flambda
            return
        return wavelen, flambda

    def multiplySED(self, other_sed, wavelen_step=None):
        """
        Multiply two SEDs together - flambda * flambda - and return a new sed object.

        Unless the two wavelength arrays are equal, returns a SED gridded with stepsize wavelen_step
        over intersecting wavelength region. Does not alter self or other_sed.
        """

        if wavelen_step is None:
            wavelen_step = self._physParams.wavelenstep

        # Check if the wavelength arrays are equal (in which case do not resample)
        if (np.all(self.wavelen == other_sed.wavelen)):
            flambda = self.flambda * other_sed.flambda
            new_sed = Sed(self.wavelen, flambda=flambda)
        else:
            # Find overlapping wavelength region.
            wavelen_max = min(self.wavelen.max(), other_sed.wavelen.max())
            wavelen_min = max(self.wavelen.min(), other_sed.wavelen.min())
            if wavelen_max < wavelen_min:
                raise Exception('The two SEDS do not overlap in wavelength space.')
            # Set up wavelen/flambda of first object, on grid.
            wavelen_1, flambda_1 = self.resampleSED(self.wavelen, self.flambda,
                                                    wavelen_min=wavelen_min,
                                                    wavelen_max=wavelen_max,
                                                    wavelen_step=wavelen_step)
            # Set up wavelen/flambda of second object, on grid.
            wavelen_2, flambda_2 = self.resampleSED(wavelen=other_sed.wavelen, flux=other_sed.flambda,
                                                    wavelen_min=wavelen_min, wavelen_max=wavelen_max,
                                                    wavelen_step = wavelen_step)
            # Multiply the two flambda together.
            flambda = flambda_1 * flambda_2
            # Instantiate new sed object. wavelen_1 == wavelen_2 as both are on grid.
            new_sed = Sed(wavelen_1, flambda)
        return new_sed

    # routines related to magnitudes and fluxes

    def calcADU(self, bandpass, photParams, wavelen=None, fnu=None):
        """
        Calculate the number of adu from camera, using sb and fnu.

        Given wavelen/fnu arrays or use self. Self or passed wavelen/fnu arrays will be unchanged.
        Calculating the AB mag requires the wavelen/fnu pair to be on the same grid as bandpass;
         (temporary values of these are used).

        @param [in] bandpass is an instantiation of the Bandpass class

        @param [in] photParams is an instantiation of the
        PhotometricParameters class that carries details about the
        photometric response of the telescope.

        @param [in] wavelen (optional) is the wavelength grid in nm

        @param [in] fnu (optional) is the flux in Janskys

        If wavelen and fnu are not specified, this will just use self.wavelen and
        self.fnu

        """

        use_self = self._checkUseSelf(wavelen, fnu)
        # Use self values if desired, otherwise use values passed to function.
        if use_self:
            # Calculate fnu if required.
            if self.fnu is None:
                # If fnu not present, calculate. (does not regrid).
                self.flambdaTofnu()
            wavelen = self.wavelen
            fnu = self.fnu
        # Make sure wavelen/fnu are on the same wavelength grid as bandpass.
        wavelen, fnu = self.resampleSED(wavelen, fnu, wavelen_match=bandpass.wavelen)
        # Calculate the number of photons.
        dlambda = wavelen[1] - wavelen[0]
        # Nphoton in units of 10^-23 ergs/cm^s/nm.
        nphoton = (fnu / wavelen[np.newaxis, :] * bandpass.sb[np.newaxis, :]).sum(1)
        adu = nphoton * (photParams.exptime * photParams.nexp * photParams.effarea/photParams.gain) * \
              (1/self._physParams.ergsetc2jansky) * \
              (1/self._physParams.planck) * dlambda
        return adu

    def fluxFromMag(self, mag):
        """
        Convert a magnitude back into a flux (implies knowledge of the zeropoint, which is
        stored in this class)
        """

        return np.power(10.0, -0.4*(mag + self.zp))

    def magFromFlux(self, flux):
        """
        Convert a flux into a magnitude (implies knowledge of the zeropoint, which is stored
        in this class)
        """

        return -2.5*np.log10(flux) - self.zp

    def calcFlux(self, bandpass, wavelen=None, fnu=None):
        """
        Integrate the specific flux density of the object over the normalized response
        curve of a bandpass, giving a flux in Janskys (10^-23 ergs/s/cm^2/Hz) through
        the normalized response curve, as detailed in Section 4.1 of the LSST design
        document LSE-180 and Section 2.6 of the LSST Science Book
        (http://ww.lsst.org/scientists/scibook).  This flux in Janskys (which is usually
        though of as a unit of specific flux density), should be considered a weighted
        average of the specific flux density over the normalized response curve of the
        bandpass.  Because we are using the normalized response curve (phi in LSE-180),
        this quantity will depend only on the shape of the response curve, not its
        absolute normalization.

        Note: the way that the normalized response curve has been defined (see equation
        5 of LSE-180) is appropriate for photon-counting detectors, not calorimeters.

        Passed wavelen/fnu arrays will be unchanged, but if uses self will check if fnu is set.

        Calculating the AB mag requires the wavelen/fnu pair to be on the same grid as bandpass;
           (temporary values of these are used).
        """
        # Note - the behavior in this first section might be considered a little odd.
        # However, I felt calculating a magnitude should not (unexpectedly) regrid your
        # wavelen/flambda information if you were using self., as this is not obvious from the "outside".
        # To preserve 'user logic', the wavelen/flambda of self are left untouched. Unfortunately
        # this means, this method can be used inefficiently if calculating many magnitudes with
        # the same sed and same bandpass region - in that case, use self.synchronizeSED() with
        # the wavelen min/max/step set to the bandpass min/max/step first ..
        # then you can calculate multiple magnitudes much more efficiently!
        use_self = self._checkUseSelf(wavelen, fnu)
        # Use self values if desired, otherwise use values passed to function.
        if use_self:
            # Calculate fnu if required.
            if self.fnu is None:
                self.flambdaTofnu()
            wavelen = self.wavelen
            fnu = self.fnu
        # Go on with magnitude calculation.
        wavelen, fnu = self.resampleSED(wavelen, fnu, wavelen_match=bandpass.wavelen)
        # Calculate bandpass phi value if required.
        if bandpass.phi is None:
            bandpass.sbTophi()
        # Calculate flux in bandpass and return this value.
        dlambda = wavelen[1] - wavelen[0]
        flux = (fnu*bandpass.phi).sum(1) * dlambda
        return flux

    def calcMag(self, bandpass, wavelen=None, fnu=None):
        """
        Calculate the AB magnitude of an object using the normalized system response (phi from Section
        4.1 of the LSST design document LSE-180).

        Can pass wavelen/fnu arrays or use self. Self or passed wavelen/fnu arrays will be unchanged.
        Calculating the AB mag requires the wavelen/fnu pair to be on the same grid as bandpass;
         (but only temporary values of these are used).
         """
        flux = self.calcFlux(bandpass, wavelen=wavelen, fnu=fnu)
        #if flux < 1e-300:
        #    raise Exception("This SED has no flux within this bandpass.")
        mag = self.magFromFlux(flux)
        return mag

    def calcFluxNorm(self, magmatch, bandpass, wavelen=None, fnu=None):
        """
        Calculate the fluxNorm (SED normalization value for a given mag) for a sed.

        Equivalent to adjusting a particular f_nu to Jansky's appropriate for the desired mag.
        Can pass wavelen/fnu or apply to self.
        """
        use_self = self._checkUseSelf(wavelen, fnu)
        if use_self:
            # Check possibility that fnu is not calculated yet.
            if self.fnu is None:
                self.flambdaTofnu()
            wavelen = self.wavelen
            fnu = self.fnu
        # Fluxnorm gets applied to f_nu (fluxnorm * SED(f_nu) * PHI = mag - 8.9 (AB zeropoint).
        # FluxNorm * SED => correct magnitudes for this object.
        # Calculate fluxnorm.
        curmag = self.calcMag(bandpass, wavelen, fnu)
        #if curmag == self.badval:
        #    return self.badval
        dmag = magmatch - curmag
        fluxnorm = np.power(10, (-0.4*dmag))
        return fluxnorm

    def multiplyFluxNorm(self, fluxNorm, wavelen=None, fnu=None):
        """
        Multiply wavelen/fnu (or self.wavelen/fnu) by fluxnorm.

        Returns wavelen/fnu arrays (or updates self).
        Note that multiplyFluxNorm does not regrid self.wavelen/flambda/fnu at all.
        """
        # Note that fluxNorm is intended to be applied to f_nu,
        # so that fluxnorm*fnu*phi = mag (expected magnitude).
        update_self = self._checkUseSelf(wavelen, fnu)
        if update_self:
            # Make sure fnu is defined.
            if self.fnu is None:
                self.flambdaTofnu()
            wavelen = self.wavelen
            fnu = self.fnu
        else:
            # Require new copy of the data for multiply.
            wavelen = np.copy(wavelen)
            fnu = np.copy(fnu)
        # Apply fluxnorm.
        fnu = fnu * fluxNorm[:, np.newaxis]
        # Update self.
        if update_self:
            self.wavelen = wavelen
            self.fnu = fnu
            # Update flambda as well.
            self.fnuToflambda()
            return
        # Else return new wavelen/fnu pairs.
        return wavelen, fnu
