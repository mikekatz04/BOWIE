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
bandpass -

Class data:
 wavelen (nm)
 sb  (Transmission, 0-1)
 phi (Normalized system response)
  wavelen/sb are guaranteed gridded.
  phi will be None until specifically needed;
     any updates to wavelen/sb within class will reset phi to None.
 the name of the bandpass file

Note that Bandpass objects are required to maintain a uniform grid in wavelength, rather than
being allowed to have variable wavelength bins. This is because of the method used in 'Sed' to
calculate magnitudes, but is simpler to enforce here.

Methods:
 __init__ : pass wavelen/sb arrays and set values (on grid) OR set data to None's
 setWavelenLimits / getWavelenLimits: set or get the wavelength limits of bandpass
 setBandpass: set bandpass using wavelen/sb input values
 getBandpass: return copies of wavelen/sb values
 imsimBandpass : set up a bandpass which is 0 everywhere but one wavelength
                 (this can be useful for imsim magnitudes)
 readThroughput : set up a bandpass by reading data from a single file
 readThroughtputList : set up a bandpass by reading data from many files and multiplying
                       the individual throughputs
 resampleBandpass : use linear interpolation to resample wavelen/sb arrays onto a regular grid
                    (grid is specified by min/max/step size)
 sbTophi : calculate phi from sb - needed for calculating magnitudes
 multiplyThroughputs : multiply self.wavelen/sb by given wavelen/sb and return
                       new wavelen/sb arrays (gridded like self)
 calcZP_t : calculate instrumental zeropoint for this bandpass
 calcEffWavelen: calculate the effective wavelength (using both Sb and Phi) for this bandpass
 writeThroughput : utility to write bandpass information to file

"""
from __future__ import print_function
from builtins import range
from builtins import object
import os
import warnings
import numpy
import scipy.interpolate as interpolate
import gzip
from .physicalparameters import PhysicalParameters
from .sed import Sed  # For ZP_t and M5 calculations. And for 'fast mags' calculation.

__all__ = ["Bandpass"]


class Bandpass(object):
    """
    Class for holding and utilizing telescope bandpasses.
    """
    def __init__(self, wavelen=None, sb=None,
                 wavelen_min=None, wavelen_max=None, wavelen_step=None):
        """
        Initialize bandpass object, with option to pass wavelen/sb arrays in directly.

        Also can specify wavelength grid min/max/step or use default - sb and wavelen will
        be resampled to this grid. If wavelen/sb are given, these will be set, but phi
        will be set to None.
        Otherwise all set to None and user should call readThroughput, readThroughputList,
        or imsimBandpass to populate bandpass data.
        """

        self._physParams = PhysicalParameters()

        if wavelen_min is None:
            if wavelen is None:
                wavelen_min = self._physParams.minwavelen
            else:
                wavelen_min = wavelen.min()
        if wavelen_max is None:
            if wavelen is None:
                wavelen_max = self._physParams.maxwavelen
            else:
                wavelen_max = wavelen.max()
        if wavelen_step is None:
            if wavelen is None:
                wavelen_step = self._physParams.wavelenstep
            else:
                wavelen_step = numpy.diff(wavelen).min()
        self.setWavelenLimits(wavelen_min, wavelen_max, wavelen_step)
        self.wavelen=None
        self.sb=None
        self.phi=None
        self.bandpassname = None
        if (wavelen is not None) and (sb is not None):
            self.setBandpass(wavelen, sb, wavelen_min, wavelen_max, wavelen_step)

        return

    ## getters and setters
    def setWavelenLimits(self, wavelen_min, wavelen_max, wavelen_step):
        """
        Set internal records of wavelen limits, _min, _max, _step.
        """
        # If we've been given values for wavelen_min, _max, _step, set them here.
        if wavelen_min is not None:
            self.wavelen_min = wavelen_min
        if wavelen_max is not None:
            self.wavelen_max = wavelen_max
        if wavelen_step is not None:
            self.wavelen_step = wavelen_step
        return

    def getWavelenLimits(self, wavelen_min, wavelen_max, wavelen_step):
        """
        Return appropriate wavelen limits (_min, _max, _step) if passed None values.
        """
        if wavelen_min is None:
            wavelen_min = self.wavelen_min
        if wavelen_max is None:
            wavelen_max = self.wavelen_max
        if wavelen_step is None:
            wavelen_step = self.wavelen_step
        return wavelen_min, wavelen_max, wavelen_step

    def setBandpass(self, wavelen, sb,
                    wavelen_min=None, wavelen_max=None, wavelen_step=None):
        """
        Populate bandpass data with wavelen/sb arrays.

        Sets self.wavelen/sb on a grid of wavelen_min/max/step. Phi set to None.
        """
        self.setWavelenLimits(wavelen_min, wavelen_max, wavelen_step)
        # Check data type.
        if (isinstance(wavelen, numpy.ndarray)==False) or (isinstance(sb, numpy.ndarray)==False):
            raise ValueError("Wavelen and sb arrays must be numpy arrays.")
        # Check data matches in length.
        if (len(wavelen)!=len(sb)):
                raise ValueError("Wavelen and sb arrays must have the same length.")
        # Data seems ok then, make a new copy of this data for self.
        self.wavelen = numpy.copy(wavelen)
        self.phi = None
        self.sb = numpy.copy(sb)
        # Resample wavelen/sb onto grid.
        self.resampleBandpass(wavelen_min=wavelen_min, wavelen_max=wavelen_max, wavelen_step=wavelen_step)
        self.bandpassname = 'FromArrays'
        return

    def readThroughput(self, filename, wavelen_min=None, wavelen_max=None, wavelen_step=None):
        """
        Populate bandpass data with data (wavelen/sb) read from file, resample onto grid.

        Sets wavelen/sb, with grid min/max/step as optional parameters. Does NOT set phi.
        """
        self.setWavelenLimits(wavelen_min, wavelen_max, wavelen_step)
        # Set self values to None in case of file read error.
        self.wavelen = None
        self.phi = None
        self.sb = None
        # Check for filename error.
        # If given list of filenames, pass to (and return from) readThroughputList.
        if isinstance(filename, list):
            warnings.warn("Was given list of files, instead of a single file. Using readThroughputList instead")
            self.readThroughputList(componentList=filename,
                                    wavelen_min=self.wavelen_min, wavelen_max=self.wavelen_max,
                                    wavelen_step=self.wavelen_step)
        # Filename is single file, now try to open file and read data.
        try:
            if filename.endswith('.gz'):
                f = gzip.open(filename, 'rt')
            else:
                f = open(filename, 'r')
        except IOError:
            try:
                if filename.endswith('.gz'):
                    f = open(filename[:-3], 'r')
                else:
                    f = gzip.open(filename+'.gz', 'rt')
            except IOError:
                raise IOError('The throughput file %s does not exist' %(filename))
        # The throughput file should have wavelength(A), throughput(Sb) as first two columns.
        wavelen = []
        sb = []
        for line in f:
            if line.startswith("#") or line.startswith('$') or line.startswith('!'):
                continue
            values = line.split()
            if len(values)<2:
                continue
            if (values[0] == '$') or (values[0] =='#') or (values[0] =='!'):
                continue
            wavelen.append(float(values[0]))
            sb.append(float(values[1]))
        f.close()
        self.bandpassname = filename
        # Set up wavelen/sb.
        self.wavelen = numpy.array(wavelen, dtype='float')
        self.sb = numpy.array(sb, dtype='float')
        # Check that wavelength is monotonic increasing and non-repeating in wavelength. (Sort on wavelength).
        if len(self.wavelen) != len(numpy.unique(self.wavelen)):
            raise ValueError('The wavelength values in file %s are non-unique.' %(filename))
        # Sort values.
        p = self.wavelen.argsort()
        self.wavelen = self.wavelen[p]
        self.sb = self.sb[p]
        # Resample throughput onto grid.
        if self.needResample():
            self.resampleBandpass()
        if self.sb.sum() < 1e-300:
            raise Exception("Bandpass data from %s has no throughput in "
                            "desired grid range %f, %f" %(filename, wavelen_min, wavelen_max))
        return

    def readThroughputList(self, componentList=['detector.dat', 'lens1.dat',
                                                'lens2.dat', 'lens3.dat',
                                                'm1.dat', 'm2.dat', 'm3.dat',
                                                'atmos_std.dat'],
                           rootDir = '.',
                           wavelen_min=None, wavelen_max=None, wavelen_step=None):
        """
        Populate bandpass data by reading from a series of files with wavelen/Sb data.

        Multiplies throughputs (sb) from each file to give a final bandpass throughput.
        Sets wavelen/sb, with grid min/max/step as optional parameters.  Does NOT set phi.
        """
        # ComponentList = names of files in that directory.
        # A typical component list of all files to build final component list, including filter, might be:
        #   componentList=['detector.dat', 'lens1.dat', 'lens2.dat', 'lens3.dat',
        #                 'm1.dat', 'm2.dat', 'm3.dat', 'atmos_std.dat', 'ideal_g.dat']
        #
        # Set wavelen limits for this object, if any updates have been given.
        self.setWavelenLimits(wavelen_min, wavelen_max, wavelen_step)
        # Set up wavelen/sb on grid.
        self.wavelen = numpy.arange(self.wavelen_min, self.wavelen_max+self.wavelen_step/2., self.wavelen_step,
                                    dtype='float')
        self.phi = None
        self.sb = numpy.ones(len(self.wavelen), dtype='float')
        # Set up a temporary bandpass object to hold data from each file.
        tempbandpass = Bandpass(wavelen_min=self.wavelen_min, wavelen_max=self.wavelen_max,
                                wavelen_step=self.wavelen_step)
        for component in componentList:
            # Read data from file.
            tempbandpass.readThroughput(os.path.join(rootDir, component))
            # Multiply self by new sb values.
            self.sb = self.sb * tempbandpass.sb
        self.bandpassname = ''.join(componentList)
        return

    def getBandpass(self):
        wavelen = numpy.copy(self.wavelen)
        sb = numpy.copy(self.sb)
        return wavelen, sb

    ## utilities

    def checkUseSelf(self, wavelen, sb):
        """
        Simple utility to check if should be using self.wavelen/sb or passed arrays.

        Useful for other methods in this class.
        Also does data integrity check on wavelen/sb if not self.
        """
        update_self = False
        if (wavelen is None) or (sb is None):
            # Then one of the arrays was not passed - check if true for both.
            if (wavelen is not None) or (sb is not None):
                # Then only one of the arrays was passed - raise exception.
                raise ValueError("Must either pass *both* wavelen/sb pair, or use self defaults")
            # Okay, neither wavelen or sb was passed in - using self only.
            update_self = True
        else:
            # Both of the arrays were passed in - check their validity.
            if (isinstance(wavelen, numpy.ndarray)==False) or (isinstance(sb, numpy.ndarray)==False):
                raise ValueError("Must pass wavelen/sb as numpy arrays")
            if len(wavelen)!=len(sb):
                raise ValueError("Must pass equal length wavelen/sb arrays")
        return update_self

    def needResample(self, wavelen=None,
                     wavelen_min=None, wavelen_max=None, wavelen_step=None):
        """
        Return true/false of whether wavelen need to be resampled onto a grid.

        Given wavelen OR defaults to self.wavelen/sb - return True/False check on whether
        the arrays need to be resampled to match wavelen_min/max/step grid.
        """
        # Thought about adding wavelen_match option here (to give this an array to match to, rather than
        # the grid parameters .. but then thought bandpass always needs to be on a regular grid (because
        # of magnitude calculations). So, this will stay match to the grid parameters only.
        # Check wavelength limits.
        wavelen_min, wavelen_max, wavelen_step = self.getWavelenLimits(wavelen_min, wavelen_max, wavelen_step)
        # Check if method acting on self or other data (here, using data type checks primarily).
        update_self = self.checkUseSelf(wavelen, wavelen)
        if update_self:
            wavelen = self.wavelen
        wavelen_max_in = wavelen[len(wavelen)-1]
        wavelen_min_in = wavelen[0]
        wavelen_step_in = wavelen[1]-wavelen[0]
        # Start check if data is already gridded.
        need_regrid=True
        # First check minimum/maximum and first step in array.
        if ((wavelen_min_in == wavelen_min) and (wavelen_max_in == wavelen_max)):
            # Then check on step size.
            stepsize = numpy.unique(numpy.diff(wavelen))
            if (len(stepsize) == 1) and (stepsize[0] == wavelen_step):
                need_regrid = False
        # At this point, need_grid=True unless it's proven to be False, so return value.
        return need_regrid

    def resampleBandpass(self, wavelen=None, sb=None,
                         wavelen_min=None, wavelen_max=None, wavelen_step=None):
        """
        Resamples wavelen/sb (or self.wavelen/sb) onto grid defined by min/max/step.

        Either returns wavelen/sb (if given those arrays) or updates wavelen / Sb in self.
        If updating self, resets phi to None.
        """
        # Check wavelength limits.
        wavelen_min, wavelen_max, wavelen_step = self.getWavelenLimits(wavelen_min, wavelen_max, wavelen_step)
        # Is method acting on self.wavelen/sb or passed in wavelen/sb? Sort it out.
        update_self = self.checkUseSelf(wavelen, sb)
        if update_self:
            wavelen = self.wavelen
            sb = self.sb
        # Now, on with the resampling.
        if (wavelen.min() > wavelen_max) or (wavelen.max() < wavelen_min):
            raise Exception("No overlap between known wavelength range and desired wavelength range.")
        # Set up gridded wavelength.
        wavelen_grid = numpy.arange(wavelen_min, wavelen_max+wavelen_step/2.0, wavelen_step, dtype='float')
        # Do the interpolation of wavelen/sb onto the grid. (note wavelen/sb type failures will die here).
        f = interpolate.interp1d(wavelen, sb, fill_value=0, bounds_error=False)
        sb_grid = f(wavelen_grid)
        # Update self values if necessary.
        if update_self:
            self.phi = None
            self.wavelen = wavelen_grid
            self.sb = sb_grid
            self.setWavelenLimits(wavelen_min, wavelen_max, wavelen_step)
            return
        return wavelen_grid, sb_grid

    def sbTophi(self):
        """
        Calculate and set phi - the normalized system response.

        This function only pdates self.phi.
        """
        # The definition of phi = (Sb/wavelength)/\int(Sb/wavelength)dlambda.
        # Due to definition of class, self.sb and self.wavelen are guaranteed equal-gridded.
        dlambda = self.wavelen[1]-self.wavelen[0]
        self.phi = self.sb/self.wavelen
        # Normalize phi so that the integral of phi is 1.
        phisum = self.phi.sum()
        if phisum < 1e-300:
            raise Exception("Phi is poorly defined (nearly 0) over bandpass range.")
        norm = phisum * dlambda
        self.phi = self.phi / norm
        return

    def multiplyThroughputs(self, wavelen_other, sb_other):
        """
        Multiply self.sb by another wavelen/sb pair, return wavelen/sb arrays.

        The returned arrays will be gridded like this bandpass.
        This method does not affect self.
        """
        # Resample wavelen_other/sb_other to match this bandpass.
        if self.needResample(wavelen=wavelen_other):
            wavelen_other, sb_other = self.resampleBandpass(wavelen=wavelen_other, sb=sb_other)
        # Make new memory copy of wavelen.
        wavelen_new = numpy.copy(self.wavelen)
        # Calculate new transmission - this is also new memory.
        sb_new = self.sb * sb_other
        return wavelen_new, sb_new
