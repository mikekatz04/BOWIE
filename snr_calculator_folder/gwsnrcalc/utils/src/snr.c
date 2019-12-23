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

#include "snr.h"

double Pi = 3.1415926536;
double PARSEC_TO_METERS = 3.08567782e+16;
double SECONDS_PER_YEAR = 31557600.0;
double SPEED_OF_LIGHT = 299792458.0;
double M_SUN = 1.98855e30;

//MRSUN and MTSUN are taken from LAL
double MRSUN = 1.476625061404649406193430731479084713e3;
double MTSUN = 4.925491025543575903411922162094833998e-6;

/* This function finds SNR of a signal given its characteristic strain and noise amplitude. It returns an SNR for INS, MRG, RD, and the full signal. It uses the trapezoidal rule. It needs the merger frequency and peak frequency. */

 void SNR_function(double *snr_all, double *snr_ins, double *snr_mrg, double *snr_rd, double *freqs, double *hc, double *hn, double *fmrg, double *fpeak, int length_of_signal, int num_binaries){

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

  return;
}
