/* 
  This code was constructed by Michael Katz using LALSimIMRPhenomD.c, LALSimIMRPhenomD.h, LALSimIMRPhenomD_internals.c, and LALSimIMRPhenomD_internals.h from LALsuite as templates. Here, only the amplitude is implemented so far. The phase may be added in the future. The top part of the code (break will be made clear below) was originally authored by Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London. Below is their license for the redistribution of the LAL codes mentioned above. This code was strategically copied to remove any dependencies on other LAL programs. Below the break, the code was authored by Michael Katz, using the original LAL codes as a guide. PhenomD can be found in Husa et al 2016 (arXiv:1508.07250) and Khan et al 2016 (arXiv:1508.07253). 
*/

/*
 * Copyright (C) 2015 Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *  MA  02111-1307  USA
 */

/* 
  Michael Katz affirms the shame redistributive license under the GNU General Public License. It is recommended to use the original LAL documentation and source code rather than this copy. Any questions regarding this code specifically, email Michael Katz at mikekatz04@gmail.com 
 */ 

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include "ringdown_spectrum_fitting.h"


/**
 * calc square of number without floating point 'pow'
 */
static inline double pow_2_of(double number)
{
	return (number*number);
}

/**
 * calc cube of number without floating point 'pow'
 */
static inline double pow_3_of(double number)
{
	return (number*number*number);
}

/**
 * calc fourth power of number without floating point 'pow'
 */
static inline double pow_4_of(double number)
{
	double pow2 = pow_2_of(number);
	return pow2 * pow2;
}

// struct to hold all amplitude coefficiences
typedef struct tagIMRPhenomDAmplitudeCoefficients {
  double eta;         // symmetric mass-ratio
  double chi1, chi2;  // dimensionless aligned spins, convention m1 >= m2.
  double q;           // asymmetric mass-ratio (q>=1)
  double chi;         // PN reduced spin parameter
  double fRD;         // ringdown frequency
  double fDM;         // imaginary part of the ringdown frequency (damping time)

  double fmaxCalc;    // frequency at which the mrerger-ringdown amplitude is maximum

  // Phenomenological inspiral amplitude coefficients
  double rho1;
  double rho2;
  double rho3;

  // Phenomenological intermediate amplitude coefficients
  double delta0;
  double delta1;
  double delta2;
  double delta3;
  double delta4;

  // Phenomenological merger-ringdown amplitude coefficients
  double gamma1;
  double gamma2;
  double gamma3;

  // Coefficients for collocation method. Used in intermediate amplitude model
  double f1, f2, f3;
  double v1, v2, v3;
  double d1, d2;

  // Transition frequencies for amplitude
  // We don't *have* to store them, but it may be clearer.
  double fInsJoin;    // Ins = Inspiral
  double fMRDJoin;    // MRD = Merger-Ringdown
}
IMRPhenomDAmplitudeCoefficients;

/**
 * used to cache the recurring (frequency-independant) prefactors of AmpInsAnsatz. Must be inited with a call to
 * init_amp_ins_prefactors(&prefactors, p);
 */
typedef struct tagAmpInsPrefactors
{
	double two_thirds;
	double one;
	double four_thirds;
	double five_thirds;
	double two;
	double seven_thirds;
	double eight_thirds;
	double three;

	double amp0;
} AmpInsPrefactors;

/**
   * Structure holding all additional coefficients needed for the delta amplitude functions.
   */
typedef struct tagdeltaUtility {
  double f12;
  double f13;
  double f14;
  double f15;
  double f22;
  double f23;
  double f24;
  double f32;
  double f33;
  double f34;
  double f35;
} DeltaUtility;

//////////////////////// Final spin, final mass, fring, fdamp ///////////////////////

static double FinalSpin0815_s(double eta, double s);
static double FinalSpin0815(double eta, double chi1, double chi2);
static double EradRational0815_s(double eta, double s);
static double EradRational0815(double eta, double chi1, double chi2);
static double fring(double eta, double chi1, double chi2, double finspin, gsl_interp_accel *acc_fring, gsl_spline *iFring);
static double fdamp(double eta, double chi1, double chi2, double finspin,  gsl_interp_accel *acc_fdamp, gsl_spline *iFdamp);

/******************************* Amplitude functions *******************************/

static double amp0Func(double eta);

///////////////////////////// Amplitude: Inspiral functions /////////////////////////

static double rho1_fun(double eta, double chiPN);
static double rho2_fun(double eta, double chiPN);
static double rho3_fun(double eta, double chiPN);
static double AmpInsAnsatz(double Mf, AmpInsPrefactors * prefactors);
static double DAmpInsAnsatz(double Mf, IMRPhenomDAmplitudeCoefficients* p);

////////////////////////// Amplitude: Merger-Ringdown functions //////////////////////

static double gamma1_fun(double eta, double chiPN);
static double gamma2_fun(double eta, double chiPN);
static double gamma3_fun(double eta, double chiPN);
static double AmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p);
static double DAmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p);
static double fmaxCalc(IMRPhenomDAmplitudeCoefficients* p);

//////////////////////////// Amplitude: Intermediate functions ///////////////////////

static double AmpIntAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p);
static double AmpIntColFitCoeff(double eta, double chiPN); //this is the v2 value
static double delta0_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d);
static double delta1_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d);
static double delta2_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d);
static double delta3_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d);
static double delta4_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d);
static IMRPhenomDAmplitudeCoefficients* ComputeDeltasFromCollocation(IMRPhenomDAmplitudeCoefficients* p, AmpInsPrefactors *prefactors);


////////////////////////Initial vertical amplitude function
static double Amp0_from_dist_mass (double M, double distance);



/**

THE BREAK FROM LAL OCCURS HERE. The rest is authored by Michael Katz with the LAL source codes as guides. 

**/

/////////////////// Main functions created separatly from LAL
////////////////////
static double eta_func(double m1, double m2);
static double find_frequency_from_time_before_merger (double time_before_merger, double eta, double M_redshifted_time);

////////////////// Functions called through ctypes
int Amplitude(double *freqs, double *amplitude, double *fmrg, double *fpeak, double *m1, double*m2, double *chi1, double *chi2, double *dist, double *z, double *start_time, double *end_time, int length_of_arrays, int num_points);

int SNR_function(double *snr_all, double *snr_ins, double *snr_mrg, double *snr_rd, double *freqs, double *hc, double *hn, double *fmrg, double *fpeak, int length_of_signal, int num_binaries);







