//De Boor's algorithm @
//https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/de-Boor.html
//For values outside the domain,
//extrapoltaes the left(right) hand side piece of the curve
//Only works for bspline degree upto 3 becuase of definiation of P
//
#include "pointers.h"

#include <vector>

#ifndef UF3_PAIR_BSPLINE_H
#define UF3_PAIR_BSPLINE_H

namespace LAMMPS_NS {

class uf3_pair_bspline {
 private:
  int bspline_degree;
  int knot_vect_size, coeff_vect_size;
  std::vector<double> knot_vect, dnknot_vect;
  std::vector<double> coeff_vect, dncoeff_vect;
  int pos_i, h, knot_mult, max_count, knot_affect_start, knot_affect_end;
  double temp2, temp3, dntemp4, temp_val;
  //Make P's; size of P is max of what will ever be needed
  double P[4][4] = {};
  LAMMPS *lmp;
  //double main_eval_loop(double ivalue_rij,int ibspline_degree,std::vector<double> iknot_vect,
  //			std::vector<double> icoeff_vect, int ipos_i, int iknot_mult);
 public:
  // dummy constructor
  uf3_pair_bspline();
  uf3_pair_bspline(LAMMPS *ulmp, int ubspline_degree, const std::vector<double> &uknot_vect,
                   const std::vector<double> &ucoeff_vect);
  ~uf3_pair_bspline();
  double bsvalue(double value_rij);
  double bsderivative(double value_rij);
  double main_eval_loop(double ivalue_rij, int ibspline_degree,
                        const std::vector<double> &iknot_vect,
                        const std::vector<double> &icoeff_vect);
};
}    // namespace LAMMPS_NS
#endif
