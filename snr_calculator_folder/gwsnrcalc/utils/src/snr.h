#ifndef __SNR_H__
#define __SNR_H__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

void SNR_function(double *snr_all, double *snr_ins, double *snr_mrg, double *snr_rd, double *freqs, double *hc, double *hn, double *fmrg, double *fpeak, int length_of_signal, int num_binaries);

#endif // __SNR_H__
