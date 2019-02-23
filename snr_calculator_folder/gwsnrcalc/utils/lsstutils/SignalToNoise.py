from __future__ import print_function
from __future__ import absolute_import
import numpy
from .Sed import Sed
from .Bandpass import Bandpass

__all__ = ["FWHMeff2FWHMgeom", "FWHMgeom2FWHMeff",
           "calcNeff", "calcInstrNoiseSq", "calcTotalNonSourceNoiseSq", "calcSNR_sed"]

def FWHMeff2FWHMgeom(FWHMeff):
    """
    Convert FWHMeff to FWHMgeom.
    This conversion was calculated by Bo Xin and Zeljko Ivezic (and will be in an update on the LSE-40 and overview papers).

    @param [in] FWHMeff (the single-gaussian equivalent FWHM value, appropriate for calcNeff) in arcseconds

    @param [out] FWHMgeom (the geometric FWHM value, as measured from a typical PSF profile) in arcseconds
    """
    FWHMgeom = 0.822*FWHMeff + 0.052
    return FWHMgeom

def FWHMgeom2FWHMeff(FWHMgeom):
    """
    Convert FWHMgeom to FWHMeff.
    This conversion was calculated by Bo Xin and Zeljko Ivezic (and will be in an update on the LSE-40 and overview papers).

    @param [in] FWHMgeom (the geometric FWHM value, as measured from a typical PSF profile) in arcseconds

    @param [out] FWHMeff (the single-gaussian equivalent FWHM value, appropriate for calcNeff) in arcseconds
    """
    FWHMeff = (FWHMgeom - 0.052)/0.822
    return FWHMeff

def calcNeff(FWHMeff, platescale):
    """
    Calculate the effective number of pixels in a single gaussian PSF.
    This equation comes from LSE-40, equation 27.
    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-40


    @param [in] FWHMeff in arcseconds
       (the width of a single-gaussian that produces correct Neff for typical PSF profile)

    @param [in] platescale in arcseconds per pixel

    @param [out] the effective number of pixels contained in the PSF

    The FWHMeff is a way to represent the equivalent seeing value, if the
    atmosphere could be simply represented as a single gaussian (instead of a more
    complicated von Karman profile for the atmosphere, convolved properly with the
    telescope hardware additional blurring of 0.4").
    A translation from the geometric FWHM to the FWHMeff is provided in FWHMgeom2FWHMeff.
    """
    return 2.266*(FWHMeff/platescale)**2


def calcInstrNoiseSq(photParams):
    """
    Combine all of the noise due to intrumentation into one value

    @param [in] photParams is an instantiation of the
    PhotometricParameters class that carries details about the
    photometric response of the telescope.

    @param [out] The noise due to all of these sources added in quadrature
    in ADU counts
    """

    # instrumental squared noise in electrons
    instNoiseSq = photParams.nexp*photParams.readnoise**2 + \
                  photParams.darkcurrent*photParams.exptime*photParams.nexp + \
                  photParams.nexp*photParams.othernoise**2

    # convert to ADU counts
    instNoiseSq = instNoiseSq/(photParams.gain*photParams.gain)

    return instNoiseSq


def calcTotalNonSourceNoiseSq(skySed, hardwarebandpass, photParams, FWHMeff):
    """
    Calculate the noise due to things that are not the source being observed
    (i.e. intrumentation and sky background)

    @param [in] skySed -- an instantiation of the Sed class representing the sky
    (normalized so that skySed.calcMag() gives the sky brightness in magnitudes
    per square arcsecond)

    @param [in] hardwarebandpass -- an instantiation of the Bandpass class representing
    just the instrumentation throughputs

    @param [in] photParams is an instantiation of the
    PhotometricParameters class that carries details about the
    photometric response of the telescope.

    @param [in] FWHMeff in arcseconds

    @param [out] total non-source noise squared (in ADU counts)
    (this is simga^2_tot * neff in equation 41 of the SNR document
    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-40 )
    """

    # Calculate the effective number of pixels for double-Gaussian PSF
    neff = calcNeff(FWHMeff, photParams.platescale)

    # Calculate the counts from the sky.
    # We multiply by two factors of the platescale because we expect the
    # skySed to be normalized such that calcADU gives counts per
    # square arc second, and we need to convert to counts per pixel.

    skycounts = skySed.calcADU(hardwarebandpass, photParams=photParams) \
                * photParams.platescale * photParams.platescale

    # Calculate the square of the noise due to instrumental effects.
    # Include the readout noise as many times as there are exposures

    noise_instr_sq = calcInstrNoiseSq(photParams=photParams)

    # Calculate the square of the noise due to sky background poisson noise
    noise_sky_sq = skycounts/photParams.gain

    # Discount error in sky measurement for now
    noise_skymeasurement_sq = 0

    total_noise_sq = neff*(noise_sky_sq + noise_instr_sq + noise_skymeasurement_sq)

    return total_noise_sq


def calcSNR_sed(sourceSed, totalbandpass, skysed, hardwarebandpass,
                    photParams, FWHMeff, verbose=False):
    """
    Calculate the signal to noise ratio for a source, given the bandpass(es) and sky SED.

    For a given source, sky sed, total bandpass and hardware bandpass, as well as
    FWHMeff / exptime, calculates the SNR with optimal PSF extraction
    assuming a double-gaussian PSF.

    @param [in] sourceSed is an instantiation of the Sed class containing the SED of
    the object whose signal to noise ratio is being calculated

    @param [in] totalbandpass is an instantiation of the Bandpass class
    representing the total throughput (system + atmosphere)

    @param [in] skysed is an instantiation of the Sed class representing
    the sky emission per square arcsecond.

    @param [in] hardwarebandpass is an instantiation of the Bandpass class
    representing just the throughput of the system hardware.

    @param [in] photParams is an instantiation of the
    PhotometricParameters class that carries details about the
    photometric response of the telescope.

    @param [in] FWHMeff in arcseconds

    @param [in] verbose is a boolean

    @param [out] signal to noise ratio
    """

    # Calculate the counts from the source.
    sourcecounts = sourceSed.calcADU(totalbandpass, photParams=photParams)

    # Calculate the (square of the) noise due to signal poisson noise.
    noise_source_sq = sourcecounts/photParams.gain

    non_source_noise_sq = calcTotalNonSourceNoiseSq(skysed, hardwarebandpass, photParams, FWHMeff)

    # Calculate total noise
    noise = numpy.sqrt(noise_source_sq + non_source_noise_sq)
    # Calculate the signal to noise ratio.
    snr = sourcecounts / noise
    import pdb; pdb.set_trace()
    if verbose:
        skycounts = skysed.calcADU(hardwarebandpass, photParams) * (photParams.platescale**2)
        noise_sky_sq = skycounts/photParams.gain
        neff = calcNeff(FWHMeff, photParams.platescale)
        noise_instr_sq = calcInstrNoiseSq(photParams)

        print("For Nexp %.1f of time %.1f: " % (photParams.nexp, photParams.exptime))
        print("Counts from source: %.2f  Counts from sky: %.2f" %(sourcecounts, skycounts))
        print("FWHMeff: %.2f('')  Neff pixels: %.3f(pix)" %(FWHMeff, neff))
        print("Noise from sky: %.2f Noise from instrument: %.2f" \
            %(numpy.sqrt(noise_sky_sq), numpy.sqrt(noise_instr_sq)))
        print("Noise from source: %.2f" %(numpy.sqrt(noise_source_sq)))
        print(" Total Signal: %.2f   Total Noise: %.2f    SNR: %.2f" %(sourcecounts, noise, snr))
        # Return the signal to noise value.
    return snr
