import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/snr.h":
    void SNR_function(np.float64_t *snr_all, np.float64_t *snr_ins, np.float64_t *snr_mrg, np.float64_t *snr_rd, np.float64_t *freqs, np.float64_t *hc, np.float64_t *hn, np.float64_t *fmrg, np.float64_t *fpeak, int length_of_signal, int num_binaries);

def GetSNR(np.ndarray[ndim=1, dtype=np.float64_t] freqs,
           np.ndarray[ndim=1, dtype=np.float64_t] hc,
           np.ndarray[ndim=1, dtype=np.float64_t] hn,
           np.ndarray[ndim=1, dtype=np.float64_t] fmrg,
           np.ndarray[ndim=1, dtype=np.float64_t] fpeak,
           length_of_signal,
           num_binaries):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] snr_all = np.zeros(num_binaries, dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] snr_ins = np.zeros(num_binaries, dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] snr_mrg = np.zeros(num_binaries, dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] snr_rd = np.zeros(num_binaries, dtype=np.float64)

    SNR_function(&snr_all[0], &snr_ins[0], &snr_mrg[0], &snr_rd[0],
                         &freqs[0], &hc[0], &hn[0], &fmrg[0], &fpeak[0],
                         length_of_signal, num_binaries)

    return (snr_all, snr_ins, snr_mrg, snr_rd)
