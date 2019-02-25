import numpy as np


class EMTelescope:
    def _get_source_counts(self, m, band):
        # counts per second
        band_ref = self.source_counts_ref[band]
        counts = band_ref['count_ref']*10**(0.4*(band_ref['m_ref'] - m))
        return counts*self.t_exp

    def _get_sky_counts(self, band):
        # counts per pixel
        band_ref = self.source_counts_ref[band]
        m_sky = self.sky_brightness_dict[band]
        m_pix = m_sky - 2.5*np.log10(self.platescale**2)
        # time * counts/time/pix
        return self.t_exp*(band_ref['count_ref']*10**(0.4*(band_ref['m_ref'] - m_pix)))

    def _get_n_eff(self, band):
        # number of pixels
        FWHM_eff = self.FWHM_eff[band]
        n_eff = 1.0*(FWHM_eff/self.platescale)**2  # factor of order unity ignored outfront
        return n_eff

    def _get_instumental_noise(self):
        self.instrumental_noisesq = (self.read_noise**2
                                     + (self.dark_current * self.t_exp))*self.n_exp
        return

    def get_snr(self, m, band):
        source_counts = self._get_source_counts(m, band)
        sky_counts = self._get_sky_counts(band)
        n_eff = self._get_n_eff(band)
        snr_num = source_counts
        snr_denom = np.sqrt(source_counts/self.gain
                            + (sky_counts/self.gain + self.instrumental_noisesq)*n_eff)
        return snr_num/snr_denom


class LSSTDefaults(EMTelescope):
    def __init__(self, **kwargs):

        # instrumental zero-pionts (counts/sec)
        self.source_counts_ref = {'u': {'m_ref': 26.50, 'count_ref': 1},
                                  'g': {'m_ref': 28.30, 'count_ref': 1},
                                  'r': {'m_ref': 28.13, 'count_ref': 1},
                                  'i': {'m_ref': 27.79, 'count_ref': 1},
                                  'z': {'m_ref': 27.40, 'count_ref': 1},
                                  'y': {'m_ref': 26.58, 'count_ref': 1}}

        # arcseconds
        self.FWHM_eff = {'u': 0.92,  # m''
                         'g': 0.87,
                         'r': 0.83,
                         'i': 0.80,
                         'z': 0.78,
                         'y': 0.76}

        # arcseconds per pixel
        self.platescale = 0.2

        # e- per exposure
        self.read_noise = 8.8

        # e- per exposure per time
        self.dark_current = 0.2

        # number of exposures
        self.n_exp = 1

        # exposure time in seconds
        self.t_exp = 30.0

        # gain e- per ADU
        self.gain = 1.0

        self._get_instumental_noise()

        # mag/arcsecond
        self.sky_brightness_dict = {'u': 22.95,  # m''
                                    'g': 22.24,
                                    'r': 21.20,
                                    'i': 20.47,
                                    'z': 19.60,
                                    'y': 19.60}

        for key, item in kwargs.items():
            setattr(self, key, item)


class SDSSDefaults(EMTelescope):
    def __init__(self, **kwargs):

        # instrumental zero-pionts (counts/sec)
        self.source_counts_ref = {'u': {'m_ref': 25.0, 'count_ref': 18},
                                  'g': {'m_ref': 25.0, 'count_ref': 123},
                                  'r': {'m_ref': 25.0, 'count_ref': 123},
                                  'i': {'m_ref': 25.0, 'count_ref': 89},
                                  'z': {'m_ref': 25.0, 'count_ref': 20}}

        # arcseconds
        self.FWHM_eff = {'u': 0.8,  # m''
                         'g': 0.8,
                         'r': 0.8,
                         'i': 0.8,
                         'z': 0.8}

        # mag/arcsecond
        self.sky_brightness_dict = {'u': 22.1,  # m''
                                    'g': 21.8,
                                    'r': 21.2,
                                    'i': 20.3,
                                    'z': 18.6}

        self.Q_dict = {'u': 0.0116,  # m''
                       'g': 0.113,
                       'r': 0.114,
                       'i': 0.0824,
                       'z': 0.0182}

        # counts per pixel
        self.sky_brightness_counts = {'u': 40,
                                      'g': 390,
                                      'r': 670,
                                      'i': 1110,
                                      'z': 1090}

        # arcseconds per pixel
        self.platescale = 0.396

        # e- per exposure
        self.read_noise = 7.0

        # e- per exposure per time
        self.dark_current = 0.5

        # number of exposures
        self.n_exp = 1

        # exposure time in seconds
        self.t_exp = 54.0

        # gain e- per ADU
        self.gain = 1.0

        self._get_instumental_noise()

        for key, item in kwargs.items():
            setattr(self, key, item)

    def _get_source_counts(self, m, band):
        Q_val = self.Q_dict[band]
        return 1.96e11*self.t_exp*Q_val*10**(-0.4*m)

    def _get_sky_counts(self, band):
        sky_counts = self.sky_brightness_counts[band]
        return sky_counts


if __name__ == '__main__':
    m_rel = 25.0
    band = 'u'
    lsst = LSSTDefaults()
    snr_lsst = lsst.get_snr(m_rel, band)

    sdss = SDSSDefaults()
    snr_sdss = sdss.get_snr(m_rel, band)
    import pdb
    pdb.set_trace()
