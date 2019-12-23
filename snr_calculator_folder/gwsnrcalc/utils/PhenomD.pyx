import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/phenomd.h":
    void Amplitude(np.float64_t *freqs, np.float64_t *amplitude, np.float64_t *fmrg, np.float64_t *fpeak, np.float64_t *m1, np.float64_t*m2, np.float64_t *chi1, np.float64_t *chi2, np.float64_t *dist, np.float64_t *z, np.float64_t *start_time, np.float64_t *end_time, int length_of_arrays, int num_points)

def GetAmplitude(np.ndarray[ndim=1, dtype=np.float64_t] m1,
                 np.ndarray[ndim=1, dtype=np.float64_t] m2,
                 np.ndarray[ndim=1, dtype=np.float64_t] chi1,
                 np.ndarray[ndim=1, dtype=np.float64_t] chi2,
                 np.ndarray[ndim=1, dtype=np.float64_t] dist,
                 np.ndarray[ndim=1, dtype=np.float64_t] z,
                 np.ndarray[ndim=1, dtype=np.float64_t] start_time,
                 np.ndarray[ndim=1, dtype=np.float64_t] end_time,
                 length_of_arrays,
                 num_points):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] freqs = np.zeros((num_points*length_of_arrays), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] amplitude = np.zeros((num_points*length_of_arrays), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] fmrg = np.zeros((length_of_arrays,), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] fpeak = np.zeros((length_of_arrays,), dtype=np.float64)

    Amplitude(&freqs[0], &amplitude[0], &fmrg[0], &fpeak[0],
                      &m1[0], &m2[0], &chi1[0], &chi2[0], &dist[0],
                      &z[0], &start_time[0], &end_time[0],
                      length_of_arrays, num_points)

    if length_of_arrays == 1:
        return (np.squeeze(freqs), np.squeeze(amplitude), np.squeeze(fmrg), np.squeeze(fpeak))

    else:
        return (freqs.reshape(length_of_arrays, num_points),
                amplitude.reshape(length_of_arrays, num_points),
                fmrg,
                fpeak)
