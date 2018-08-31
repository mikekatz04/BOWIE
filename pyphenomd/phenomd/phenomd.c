/* 
  This code was constructed by Michael Katz using LALSimIMRPhenomD.c, LALSimIMRPhenomD.h, LALSimIMRPhenomD_internals.c, and LALSimIMRPhenomD_internals.h from LALsuite as templates. Here, only the amplitude is implemented so far. The phase may be added in the future. The top part of the code (break will be made clear below) was originally authored by Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London. Below is their license for the redistribution of the LAL codes mentioned above. This code was strategically copied to remove any dependencies on other LAL programs. Below the break, the code was authored by Michael Katz, using the original LAL codes as a guide. PhenomD can be found in Husa et al 2016 (arXiv:1508.07250) and Khan et al 2016 (arXiv:1508.07253). 

  This was used in "Evaluating Black Hole Detectability with LISA" (arXiv:1508.07253), as a part of the BOWIE package (https://github.com/mikekatz04/BOWIE).
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
#include <complex.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <math.h>

#include "phenomd.h"
#include "ringdown_spectrum_fitting.h"

double Pi = 3.1415926536;
double PARSEC_TO_METERS = 3.08567782e+16;
double SECONDS_PER_YEAR = 31557600.0;
double SPEED_OF_LIGHT = 299792458.0;
double M_SUN = 1.98855e30;

//MRSUN and MTSUN are taken from LAL
double MRSUN = 1.476625061404649406193430731479084713e3;
double MTSUN = 4.925491025543575903411922162094833998e-6;

/**
 * amplitude scaling factor defined by eq. 17 in 1508.07253
 */
static double amp0Func(double eta) {
  return (sqrt(2.0/3.0)*sqrt(eta))/pow(Pi, 1.0/6.0);
}

/**
 * PN reduced spin parameter
 * See Eq 5.9 in http://arxiv.org/pdf/1107.1267v2.pdf
 */

static double chiPN(double eta, double chi1, double chi2) {
  // Convention m1 >= m2 and chi1 is the spin on m1
  double delta = sqrt(1.0 - 4.0*eta);
  double chi_s = (chi1 + chi2) / 2.0;
  double chi_a = (chi1 - chi2) / 2.0;
  return chi_s * (1.0 - eta*76.0/113.0) + delta*chi_a;
}

/**
 * rho_1 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
static double rho1_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 3931.8979897196696 - 17395.758706812805*eta
  + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2)*xi
  + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi2
  + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi3;
}

/**
 * rho_2 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
static double rho2_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -40105.47653771657 + 112253.0169706701*eta
  + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2)*xi
  + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi2
  + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi3;
}

/**
 * rho_3 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
static double rho3_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 83208.35471266537 - 191237.7264145924*eta +
  (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2)*xi
  + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi2
  + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi3;
}


/**
 * Inspiral amplitude plus rho phenom coefficents. rho coefficients computed
 * in rho1_fun, rho2_fun, rho3_fun functions.
 * Amplitude is a re-expansion. See 1508.07253 and Equation 29, 30 and Appendix B arXiv:1508.07253 for details
 */

static double AmpInsAnsatz(double Mf, AmpInsPrefactors * prefactors) {
  double Mf2 = Mf*Mf;
  double Mf3 = Mf*Mf2;

  return 1 + pow(Mf, 2.0/3.0) * prefactors->two_thirds
      + Mf * prefactors->one + pow(Mf, 4.0/3.0) * prefactors->four_thirds
      + pow(Mf, 5.0/3.0) * prefactors->five_thirds + Mf2 * prefactors->two
      + pow(Mf, 7.0/3.0) * prefactors->seven_thirds + pow(Mf, 8.0/3.0) * prefactors->eight_thirds
      + Mf3 * prefactors->three;
}

/**
 * Take the AmpInsAnsatz expression and compute the first derivative
 * with respect to frequency to get the expression below.
 */
static double DAmpInsAnsatz(double Mf, IMRPhenomDAmplitudeCoefficients* p) {
  double eta = p->eta;
  double chi1 = p->chi1;
  double chi2 = p->chi2;
  double rho1 = p->rho1;
  double rho2 = p->rho2;
  double rho3 = p->rho3;

  double chi12 = chi1*chi1;
  double chi22 = chi2*chi2;
  double eta2 = eta*eta;
  double eta3 = eta*eta2;
  double Mf2 = Mf*Mf;
  double Pi2 = Pi*Pi;
  double Seta = sqrt(1.0 - 4.0*eta);

   return ((-969 + 1804*eta)*pow(Pi,2.0/3.0))/(1008.*pow(Mf,1.0/3.0))
   + ((chi1*(81*(1 + Seta) - 44*eta) + chi2*(81 - 81*Seta - 44*eta))*Pi)/48.
   + ((-27312085 - 10287648*chi22 - 10287648*chi12*(1 + Seta)
   + 10287648*chi22*Seta + 24*(-1975055 + 857304*chi12 - 994896*chi1*chi2 + 857304*chi22)*eta
   + 35371056*eta2)*pow(Mf,1.0/3.0)*pow(Pi,4.0/3.0))/6.096384e6
   + (5*pow(Mf,2.0/3.0)*pow(Pi,5.0/3.0)*(chi2*(-285197*(-1 + Seta)
   + 4*(-91902 + 1579*Seta)*eta - 35632*eta2) + chi1*(285197*(1 + Seta)
   - 4*(91902 + 1579*Seta)*eta - 35632*eta2) + 42840*(-1 + 4*eta)*Pi))/96768.
   - (Mf*Pi2*(-336*(-3248849057.0 + 2943675504*chi12 - 3339284256*chi1*chi2 + 2943675504*chi22)*eta2 - 324322727232*eta3
   - 7*(-177520268561 + 107414046432*chi22 + 107414046432*chi12*(1 + Seta) - 107414046432*chi22*Seta
   + 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*Pi)
   + 12*eta*(-545384828789.0 - 176491177632*chi1*chi2 + 202603761360*chi22 + 77616*chi12*(2610335 + 995766*Seta)
   - 77287373856*chi22*Seta + 5841690624*(chi1 + chi2)*Pi + 21384760320*Pi2)))/3.0042980352e10
   + (7.0/3.0)*pow(Mf,4.0/3.0)*rho1 + (8.0/3.0)*pow(Mf,5.0/3.0)*rho2 + 3*Mf2*rho3;
}



///////////////////////////// Amplitude: Intermediate functions ////////////////////////

// Phenom coefficients delta0, ..., delta4 determined from collocation method
// (constraining 3 values and 2 derivatives)
// AmpIntAnsatzFunc[]

/**
 * Ansatz for the intermediate amplitude. Equation 21 arXiv:1508.07253
 */
static double AmpIntAnsatz(double Mf, IMRPhenomDAmplitudeCoefficients* p) {
  double Mf2 = Mf*Mf;
  double Mf3 = Mf*Mf2;
  double Mf4 = Mf*Mf3;
  return p->delta0 + p->delta1*Mf + p->delta2*Mf2 + p->delta3*Mf3 + p->delta4*Mf4;
}

/**
 * The function name stands for 'Amplitude Intermediate Collocation Fit Coefficient'
 * This is the 'v2' value in Table 5 of arXiv:1508.07253
 */
static double AmpIntColFitCoeff(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 0.8149838730507785 + 2.5747553517454658*eta
  + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2)*xi
  + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi2
  + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi3;
}

  /**
  * The following functions (delta{0,1,2,3,4}_fun) were derived
  * in mathematica according to
  * the constraints detailed in arXiv:1508.07253,
  * section 'Region IIa - intermediate'.
  * These are not given in the paper.
  * Can be rederived by solving Equation 21 for the constraints
  * given in Equations 22-26 in arXiv:1508.07253
  */
static double delta0_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f14 = d->f14;
  double f15 = d->f15;
  double f22 = d->f22;
  double f23 = d->f23;
  double f24 = d->f24;
  double f32 = d->f32;
  double f33 = d->f33;
  double f34 = d->f34;
  double f35 = d->f35;

  return -((d2*f15*f22*f3 - 2*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32
  - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33
  + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33
  - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35
  + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1
  + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2
  + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3
  + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3
  + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3) / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(f3-f2)));
}

static double delta1_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f14 = d->f14;
  double f15 = d->f15;
  double f22 = d->f22;
  double f23 = d->f23;
  double f24 = d->f24;
  double f32 = d->f32;
  double f33 = d->f33;
  double f34 = d->f34;
  double f35 = d->f35;

  return -((-(d2*f15*f22) + 2*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2*d1*f13*f23*f3
  + 2*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32
  - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32
  - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33
  + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35
  - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1
  + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3
  + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3)
  / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)));
}

static double delta2_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f14 = d->f14;
  double f15 = d->f15;
  double f23 = d->f23;
  double f24 = d->f24;
  double f32 = d->f32;
  double f33 = d->f33;
  double f34 = d->f34;
  double f35 = d->f35;

  return -((d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2*d2*f12*f24 - d2*f15*f3
  + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32
  + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32
  - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33
  + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1
  - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2
  - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3
  + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3)
  / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)));
}

static double delta3_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f14 = d->f14;
  double f22 = d->f22;
  double f24 = d->f24;
  double f32 = d->f32;
  double f33 = d->f33;
  double f34 = d->f34;

  return -((-2*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3
  - 2*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3
  + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32
  + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33
  - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1
  - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2
  - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3
  - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3)
  / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)));
}

static double delta4_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f22 = d->f22;
  double f23 = d->f23;
  double f32 = d->f32;
  double f33 = d->f33;

  return -((d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2*d1*f12*f2*f3
  + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32
  - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1
  - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2
  + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3)
  / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)));
}

/**
 * Calculates delta_i's
 * Method described in arXiv:1508.07253 section 'Region IIa - intermediate'
 */
 static IMRPhenomDAmplitudeCoefficients * ComputeDeltasFromCollocation(IMRPhenomDAmplitudeCoefficients* p, AmpInsPrefactors *prefactors) {
  // Three evenly spaced collocation points in the interval [f1,f3].
  double f1 = AMP_fJoin_INS;
  double f3 = p->fmaxCalc;
  double dfx = (f3 - f1)/2.0;
  double f2 = f1 + dfx;


  // v1 is inspiral model evaluated at f1
  // d1 is derivative of inspiral model evaluated at f1
  double v1 = AmpInsAnsatz(f1, prefactors);
  double d1 = DAmpInsAnsatz(f1, p);

  // v3 is merger-ringdown model evaluated at f3
  // d2 is derivative of merger-ringdown model evaluated at f3
  double v3 = AmpMRDAnsatz(f3, p);
  double d2 = DAmpMRDAnsatz(f3, p);

  // v2 is the value of the amplitude evaluated at f2
  // they come from the fit of the collocation points in the intermediate region
  double v2 = AmpIntColFitCoeff(p->eta, p->chi);

  p->f1 = f1;
  p->f2 = f2;
  p->f3 = f3;
  p->v1 = v1;
  p->v2 = v2;
  p->v3 = v3;
  p->d1 = d1;
  p->d2 = d2;

  // Now compute the delta_i's from the collocation coefficients
  // Precompute common quantities here and pass along to delta functions.
  DeltaUtility d;
  d.f12 = f1*f1;
  d.f13 = f1*d.f12;
  d.f14 = f1*d.f13;
  d.f15 = f1*d.f14;
  d.f22 = f2*f2;
  d.f23 = f2*d.f22;
  d.f24 = f2*d.f23;
  d.f32 = f3*f3;
  d.f33 = f3*d.f32;
  d.f34 = f3*d.f33;
  d.f35 = f3*d.f34;
  p->delta0 = delta0_fun(p, &d);
  p->delta1 = delta1_fun(p, &d);
  p->delta2 = delta2_fun(p, &d);
  p->delta3 = delta3_fun(p, &d);
  p->delta4 = delta4_fun(p, &d);
  return p;
}


// Final Spin and Radiated Energy formulas described in 1508.07250

/**
 * Formula to predict the final spin. Equation 3.6 arXiv:1508.07250
 * s defined around Equation 3.6.
 */
static double FinalSpin0815_s(double eta, double s) {
  double eta2 = eta*eta;
  double eta3 = eta2*eta;
  double eta4 = eta3*eta;
  double s2 = s*s;
  double s3 = s2*s;
  double s4 = s3*s;

return 3.4641016151377544*eta - 4.399247300629289*eta2 +
   9.397292189321194*eta3 - 13.180949901606242*eta4 +
   (1 - 0.0850917821418767*eta - 5.837029316602263*eta2)*s +
   (0.1014665242971878*eta - 2.0967746996832157*eta2)*s2 +
   (-1.3546806617824356*eta + 4.108962025369336*eta2)*s3 +
   (-0.8676969352555539*eta + 2.064046835273906*eta2)*s4;
}

/**
 * Wrapper function for FinalSpin0815_s.
 */
static double FinalSpin0815(double eta, double chi1, double chi2) {
  // Convention m1 >= m2
  double Seta = sqrt(1.0 - 4.0*eta);
  double m1 = 0.5 * (1.0 + Seta);
  double m2 = 0.5 * (1.0 - Seta);
  double m1s = m1*m1;
  double m2s = m2*m2;
  // s defined around Equation 3.6 arXiv:1508.07250
  double s = (m1s * chi1 + m2s * chi2);
  return FinalSpin0815_s(eta, s);
}

/**
 * Formula to predict the radiated energy. Equation 3.8 arXiv:1508.07250
 * s defined around Equation 3.8.
 */

static double EradRational0815_s(double eta, double s) {
  double eta2 = eta*eta;
  double eta3 = eta2*eta;
  double eta4 = eta3*eta;

  return ((0.055974469826360077*eta + 0.5809510763115132*eta2 - 0.9606726679372312*eta3 + 3.352411249771192*eta4)*
    (1. + (-0.0030302335878845507 - 2.0066110851351073*eta + 7.7050567802399215*eta2)*s))/(1. + (-0.6714403054720589 - 1.4756929437702908*eta + 7.304676214885011*eta2)*s);
}

/**
 * Wrapper function for EradRational0815_s.
 */
static double EradRational0815(double eta, double chi1, double chi2) {
  // Convention m1 >= m2
  double Seta = sqrt(1.0 - 4.0*eta);
  double m1 = 0.5 * (1.0 + Seta);
  double m2 = 0.5 * (1.0 - Seta);
  double m1s = m1*m1;
  double m2s = m2*m2;
  // arXiv:1508.07250
  double s = (m1s * chi1 + m2s * chi2) / (m1s + m2s);

  return EradRational0815_s(eta, s);
}

/**
 * fring is the real part of the ringdown frequency
 * 1508.07250 figure 9
 */
static double fring(double eta, double chi1, double chi2, double finspin, gsl_interp_accel *acc_fring, gsl_spline *iFring) {
  double return_val;

  return_val = gsl_spline_eval(iFring, finspin, acc_fring)/ (1.0 - EradRational0815(eta, chi1, chi2));

  return return_val;
}

/**
 * fdamp is the complex part of the ringdown frequency
 * 1508.07250 figure 9
 */
static double fdamp(double eta, double chi1, double chi2, double finspin,  gsl_interp_accel *acc_fdamp, gsl_spline *iFdamp) {
  double return_val;

  return_val = gsl_spline_eval(iFdamp, finspin, acc_fdamp)/ (1.0 - EradRational0815(eta, chi1, chi2));

  return return_val;
}

/**
 * Equation 20 arXiv:1508.07253 (called f_peak in paper)
 * analytic location of maximum of AmpMRDAnsatz
 */
static double fmaxCalc(IMRPhenomDAmplitudeCoefficients* p) {
  double fRD = p->fRD;
  double fDM = p->fDM;
  double gamma2 = p->gamma2;
  double gamma3 = p->gamma3;

  // NOTE: There's a problem with this expression from the paper becoming imaginary if gamma2>=1
  // Fix: if gamma2 >= 1 then set the square root term to zero.
  if (gamma2 <= 1)
    return fabs(fRD + (fDM*(-1 + sqrt(1 - pow_2_of(gamma2)))*gamma3)/gamma2);
  else
    return fabs(fRD + (fDM*(-1)*gamma3)/gamma2);
}


/* gamma 1 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
static double gamma1_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 0.006927402739328343 + 0.03020474290328911*eta
  + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2)*xi
  + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi2
  + (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi3;
}

/**
 * gamma 2 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
static double gamma2_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 1.010344404799477 + 0.0008993122007234548*eta
  + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2)*xi
  + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi2
  + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi3;
}

/**
 * gamma 3 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
static double gamma3_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 1.3081615607036106 - 0.005537729694807678*eta
  + (-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2)*xi
  + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi2
  + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi3;
}



/**
 * Ansatz for the merger-ringdown amplitude. Equation 19 arXiv:1508.07253
 */
 static double AmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p) {
  double fRD = p->fRD;
  double fDM = p->fDM;
  double gamma1 = p->gamma1;
  double gamma2 = p->gamma2;
  double gamma3 = p->gamma3;
  double fDMgamma3 = fDM*gamma3;
  double fminfRD = f - fRD;
  return exp( -(fminfRD)*gamma2 / (fDMgamma3) )
    * (fDMgamma3*gamma1) / (pow_2_of(fminfRD) + pow_2_of(fDMgamma3));
}

/**
 * first frequency derivative of AmpMRDAnsatz
 */
 static double DAmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p) {
  double fRD = p->fRD;
  double fDM = p->fDM;
  double gamma1 = p->gamma1;
  double gamma2 = p->gamma2;
  double gamma3 = p->gamma3;

  double fDMgamma3 = fDM * gamma3;
  double pow2_fDMgamma3 = pow_2_of(fDMgamma3);
  double fminfRD = f - fRD;
  double expfactor = exp(((fminfRD)*gamma2)/(fDMgamma3));
  double pow2pluspow2 = pow_2_of(fminfRD) + pow2_fDMgamma3;

   return (-2*fDM*(fminfRD)*gamma3*gamma1) / ( expfactor * pow_2_of(pow2pluspow2)) -
     (gamma2*gamma1) / ( expfactor * (pow2pluspow2)) ;
}

/*find coefficients for the amplitude part of the waveform. Eautions 31 and 32 from arXiv:1508.07253 provide general form of each equation. */

static IMRPhenomDAmplitudeCoefficients* ComputeIMRPhenomDAmplitudeCoefficients(double eta, double chi1, double chi2, double finspin, gsl_interp_accel *acc_fring, gsl_spline *iFring, gsl_interp_accel *acc_fdamp, gsl_spline *iFdamp) {

  IMRPhenomDAmplitudeCoefficients *p = (IMRPhenomDAmplitudeCoefficients *) malloc(sizeof(IMRPhenomDAmplitudeCoefficients));

  p->eta = eta;
  p->chi1 = chi1;
  p->chi2 = chi2;

  p->q = (1.0 + sqrt(1.0 - 4.0*eta) - 2.0*eta) / (2.0*eta);
  p->chi = chiPN(eta, chi1, chi2);

  p->fRD = fring(eta, chi1, chi2, finspin, acc_fring, iFring);
  p->fDM = fdamp(eta, chi1, chi2, finspin, acc_fdamp, iFdamp);

  // Compute gamma_i's, rho_i's
  p->gamma1 = gamma1_fun(eta, p->chi);
  p->gamma2 = gamma2_fun(eta, p->chi);
  p->gamma3 = gamma3_fun(eta, p->chi);

  p->fmaxCalc = fmaxCalc(p);

  
  p->rho1 = rho1_fun(eta, p->chi);
  p->rho2 = rho2_fun(eta, p->chi);
  p->rho3 = rho3_fun(eta, p->chi);

  //In LAL, the delta's are computed here. 
  return p;
}

/*Find prefactors for each term from eq. 30 in arXiv:1508.07253 for the inspiral phase*/

static AmpInsPrefactors * init_amp_ins_prefactors(IMRPhenomDAmplitudeCoefficients* p)
{
  AmpInsPrefactors *prefactors = (AmpInsPrefactors *) malloc(sizeof(AmpInsPrefactors));
  double eta = p->eta;

  prefactors->amp0 = amp0Func(p->eta);

  double chi1 = p->chi1;
  double chi2 = p->chi2;
  double rho1 = p->rho1;
  double rho2 = p->rho2;
  double rho3 = p->rho3;

  double chi12 = chi1*chi1;
  double chi22 = chi2*chi2;
  double eta2 = eta*eta;
  double eta3 = eta*eta2;


  
  double Pi2 = pow(Pi, 2);
  double Seta = sqrt(1.0 - 4.0*eta);

  prefactors->two_thirds = ((-969 + 1804*eta)*pow(Pi, 2.0/3.0))/672.;
  prefactors->one = ((chi1*(81*(1 + Seta) - 44*eta) + chi2*(81 - 81*Seta - 44*eta))*Pi)/48.;
  prefactors->four_thirds = ( (-27312085.0 - 10287648*chi22 - 10287648*chi12*(1 + Seta) + 10287648*chi22*Seta
                 + 24*(-1975055 + 857304*chi12 - 994896*chi1*chi2 + 857304*chi22)*eta
                 + 35371056*eta2
                 )
              * pow(Pi, 4.0/3.0)) / 8.128512e6;
  prefactors->five_thirds = (pow(Pi, 5.0/3.0) * (chi2*(-285197*(-1 + Seta) + 4*(-91902 + 1579*Seta)*eta - 35632*eta2)
                              + chi1*(285197*(1 + Seta) - 4*(91902 + 1579*Seta)*eta - 35632*eta2)
                              + 42840*(-1.0 + 4*eta)*Pi
                              )
                ) / 32256.;
  prefactors->two = - (Pi2*(-336*(-3248849057.0 + 2943675504*chi12 - 3339284256*chi1*chi2 + 2943675504*chi22)*eta2
                - 324322727232*eta3
                - 7*(-177520268561 + 107414046432*chi22 + 107414046432*chi12*(1 + Seta)
                  - 107414046432*chi22*Seta + 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*Pi
                  )
                + 12*eta*(-545384828789 - 176491177632*chi1*chi2 + 202603761360*chi22
                    + 77616*chi12*(2610335 + 995766*Seta) - 77287373856*chi22*Seta
                    + 5841690624*(chi1 + chi2)*Pi + 21384760320*Pi2
                    )
                )
            )/6.0085960704e10;
  prefactors->seven_thirds= rho1;
  prefactors->eight_thirds = rho2;
  prefactors->three = rho3;

  return prefactors;
}


/*this equation is from the LAL code. However, the prefactors is slightly different*/

static double Amp0_from_dist_mass (double M, double distance){
  return sqrt(5. / (64.*Pi)) * M * MRSUN * M * MTSUN / distance;
}



/**

THE BREAK FROM LAL OCCURS HERE. The rest is authored by Michael Katz with the LAL source codes as guides. 

**/



/* find eta */
static double eta_func(double m1, double m2){
  double M = m1+m2;
  return (m1*m2)/(M*M);
}

/*Find frequency at a time before merger */
static double find_frequency_from_time_before_merger (double time_before_merger, double eta, double M_redshifted_time){
  //Time is in years. 1st order of post-newtonian expansion
  // Returns a dimensionless frequency (M*f)
  double T, tau;
  T = time_before_merger * SECONDS_PER_YEAR;
  tau = (eta*T)/(5.0*M_redshifted_time);
  return (1.0/(8.0*Pi*pow(tau, 3.0/8.0))) * (1.0 + ((11.0/32.0)*eta + (743.0/2688.0))/pow(tau,1.0/4.0));
}

/*Find the amplitude for PhenomD */

int Amplitude(double *freqs, double *amplitude, double *fmrg, double *fpeak, double *m1, double*m2, double *chi1, double *chi2, double *dist, double *z, double *start_time, double *end_time, int length_of_arrays, int num_points){

  //dist is the luminosity distance

  //initialize all parameters

  double f, f_min_log10, f_max_log10, df, AmpPreFac, Amp0_dist_mass, PhenomD_amplitude;
  double M, M_redshifted, M_redshifted_time;
  double MF_ISCO = 1.0/(pow(6.0, 3.0/2.0)*Pi);
  
  int i,j;
  double f_max;
  double f_min=1e-4;
  double f_ins_meets_intermediate = 0.014;

  double finspin, eta;
  double m1_in, m2_in, chi1_in, chi2_in;


  //initialize interpolation functions for fring and fdamp
  gsl_interp_accel *acc_fring = gsl_interp_accel_alloc();
  gsl_spline *iFring = gsl_spline_alloc(gsl_interp_cspline, QNMData_length);
  gsl_spline_init(iFring, QNMData_a, QNMData_fring, QNMData_length);

  gsl_interp_accel *acc_fdamp = gsl_interp_accel_alloc();
  gsl_spline *iFdamp = gsl_spline_alloc(gsl_interp_cspline, QNMData_length);
  gsl_spline_init(iFdamp, QNMData_a, QNMData_fdamp, QNMData_length);


  for(j=0; j<length_of_arrays; j+=1){

    //make sure m1 is the larger mass
    if (m1[j]>m2[j]){
      m1_in = m1[j];
      m2_in = m2[j];
      chi1_in = chi1[j];
      chi2_in = chi2[j];
    } else {
      m1_in = m2[j];
      m2_in = m1[j];
      chi1_in = chi2[j];
      chi2_in = chi1[j];
    }


    //make redshifted mass variables, one with units of seconds
    M = m1_in + m2_in;
    M_redshifted =  M * (1.0+z[j]);
    M_redshifted_time = M_redshifted * MTSUN;

    // find eta and final spin of binary
    eta = eta_func(m1_in, m2_in);
    finspin = FinalSpin0815(eta, chi1_in, chi2_in);

    // find the start frequency based at time before merger
    f_min = find_frequency_from_time_before_merger(start_time[j], eta, M_redshifted_time);

    //if end_time is greater than zero, find corresponding frequency
    if (end_time[j] > 0.0){
      f_max = find_frequency_from_time_before_merger(end_time[j], eta, M_redshifted_time);
    } else {
      f_max = f_CUT;
    }
  
    // initialize all the PhenomD amplitude coefficients
    IMRPhenomDAmplitudeCoefficients *p = ComputeIMRPhenomDAmplitudeCoefficients(eta, chi1_in, chi2_in, finspin, acc_fring, iFring, acc_fdamp, iFdamp);

    // find amplitude prefactors
    AmpInsPrefactors *prefactors = init_amp_ins_prefactors(p);

    //Find where INS joins INT
    p->fInsJoin = f_ins_meets_intermediate;

    //Find where INT joins RD
    p->fMRDJoin = p->fmaxCalc;

    // compute delta_i's
    p = ComputeDeltasFromCollocation(p, prefactors);

    // Find vertical signal factor
    Amp0_dist_mass = Amp0_from_dist_mass(M_redshifted, dist[j]*PARSEC_TO_METERS*1e6);

   // The frequencies are log-spaced
   f_min_log10 = log10(f_min);
   f_max_log10 = log10(f_max);
   df = (f_max_log10-f_min_log10)/(num_points-1);  

  for(i=0; i<num_points; i+=1){

    //find next frequency
     f = pow(10.0, f_min_log10 + (i*df));
     
    //numerical prefactor for amplitude at this frequency
     AmpPreFac = Amp0_dist_mass*prefactors->amp0 / pow(f, 7.0/6.0);

     // calculate amplitdue based on phase
     if (f<=p->fInsJoin){
      PhenomD_amplitude = AmpPreFac * AmpInsAnsatz(f, prefactors);
     } else if (f>=p->fMRDJoin){
      PhenomD_amplitude = AmpPreFac * AmpMRDAnsatz(f, p);
     } else{
      PhenomD_amplitude = AmpPreFac *  AmpIntAnsatz(f, p);
     }

      /*
      THIS RETURNS THE CHARACTERISTIC AMPLITUDE. 
      NEEDS TO BE CHANGED WHEN PHASE IS INCLUDED DUE TO IMAGINARY TERMS.
     */

      freqs[j*num_points+i] = f/M_redshifted_time;

      amplitude[j*num_points+i] = 2.0*PhenomD_amplitude*freqs[j*num_points+i];
  
  }

  //Add merger and peak frequency to know where joining points are.
  fmrg[j] = MF_ISCO/M_redshifted_time;
  fpeak[j] = p->fmaxCalc/M_redshifted_time;
} 

  //free spline memory
  gsl_spline_free(iFring);
  gsl_interp_accel_free(acc_fring);

  gsl_spline_free(iFdamp);
  gsl_interp_accel_free(acc_fdamp);

  return 0;
}

/* This function finds SNR of a signal given its characteristic strain and noise amplitude. It returns an SNR for INS, MRG, RD, and the full signal. It uses the trapezoidal rule. It needs the merger frequency and peak frequency. */

 int SNR_function(double *snr_all, double *snr_ins, double *snr_mrg, double *snr_rd, double *freqs, double *hc, double *hn, double *fmrg, double *fpeak, int length_of_signal, int num_binaries){

  // initialize parameters
  int i, j, ind;
  double snr_all_trans;
  double snr_ins_trans;
  double snr_mrg_trans;
  double snr_rd_trans;
  double trap_val;
  double f;
  double func_a, func_b, strain_ratio_a, strain_ratio_b;

  for(i=0; i<num_binaries; i+=1){
   snr_all_trans=0.0;
   snr_ins_trans=0.0;
   snr_mrg_trans=0.0;
   snr_rd_trans=0.0;

   for(j=1; j<length_of_signal; j+=1){

      // hc and freqs are flattened arrays. Find correct 2d position within flattened array
      ind = i*length_of_signal+j;

      // find points on the trapezoid
      strain_ratio_a = hc[ind]/hn[ind];
      func_a = (1.0/freqs[ind]) * strain_ratio_a * strain_ratio_a;

      strain_ratio_b = hc[ind-1]/hn[ind-1];
      func_b = (1.0/freqs[ind-1]) * strain_ratio_b * strain_ratio_b;

      // find trapezoid area
      trap_val = (freqs[ind] - freqs[ind-1]) * (func_a + func_b)/2.0;
      snr_all_trans += trap_val;

      //Take center of the trapezoid as the frequency value
      f = (freqs[ind] + freqs[ind-1])/2.0;

      //add snr based on phase
     if (f<=fmrg[i]){
      snr_ins_trans += trap_val;
     } else if (f>=fpeak[i]){
      snr_rd_trans += trap_val;
     } else{
      snr_mrg_trans += trap_val;
     }
   }
   
   //add snrs to arrays of return snrs
   snr_all[i] = sqrt(snr_all_trans);
   snr_ins[i] = sqrt(snr_ins_trans);
   snr_mrg[i] = sqrt(snr_mrg_trans);
   snr_rd[i] = sqrt(snr_rd_trans);

  }

  return 0;
}









