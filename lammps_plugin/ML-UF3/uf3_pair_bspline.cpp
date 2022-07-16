#include "uf3_pair_bspline.h"

#include "utils.h"
#include <vector>

using namespace LAMMPS_NS;

//Call constructor
//Dummy constructor
uf3_pair_bspline::uf3_pair_bspline() {}
//Constructor
//passing vect by reference
uf3_pair_bspline::uf3_pair_bspline(LAMMPS *ulmp, int ubspline_degree,
                                   const std::vector<double> &uknot_vect,
                                   const std::vector<double> &ucoeff_vect)
{
  lmp = ulmp;
  bspline_degree = ubspline_degree;
  knot_vect = uknot_vect;
  coeff_vect = ucoeff_vect;

  knot_vect_size = uknot_vect.size();
  coeff_vect_size = ucoeff_vect.size();
  //initialize dncoeff_vect and dnknot_coeff for derivates
  for (int i = 0; i < coeff_vect_size - 1; ++i) {
    dntemp4 = bspline_degree / (knot_vect[i + bspline_degree + 1] - knot_vect[i + 1]);
    dncoeff_vect.push_back((coeff_vect[i + 1] - coeff_vect[i]) * dntemp4);
  }
  for (int i = 1; i < knot_vect_size - 1; ++i) { dnknot_vect.push_back(knot_vect[i]); }

  // Cache constants
  double c0, c1, c2, c3;
  for (int i = 0; i < knot_vect.size() - 4; i++) {
    std::vector<double> row;

    c0 = coeff_vect[i] *
        (-pow(knot_vect[i + 0], 3) /
         (-pow(knot_vect[i + 0], 3) + pow(knot_vect[i + 0], 2) * knot_vect[i + 1] +
          pow(knot_vect[i + 0], 2) * knot_vect[i + 2] +
          pow(knot_vect[i + 0], 2) * knot_vect[i + 3] -
          knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] -
          knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
          knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] +
          knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3]));
    c1 = coeff_vect[i] *
        (3 * pow(knot_vect[i + 0], 2) /
         (-pow(knot_vect[i + 0], 3) + pow(knot_vect[i + 0], 2) * knot_vect[i + 1] +
          pow(knot_vect[i + 0], 2) * knot_vect[i + 2] +
          pow(knot_vect[i + 0], 2) * knot_vect[i + 3] -
          knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] -
          knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
          knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] +
          knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3]));
    c2 = coeff_vect[i] *
        (-3 * knot_vect[i + 0] /
         (-pow(knot_vect[i + 0], 3) + pow(knot_vect[i + 0], 2) * knot_vect[i + 1] +
          pow(knot_vect[i + 0], 2) * knot_vect[i + 2] +
          pow(knot_vect[i + 0], 2) * knot_vect[i + 3] -
          knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] -
          knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
          knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] +
          knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3]));
    c3 = coeff_vect[i] *
        (1 /
         (-pow(knot_vect[i + 0], 3) + pow(knot_vect[i + 0], 2) * knot_vect[i + 1] +
          pow(knot_vect[i + 0], 2) * knot_vect[i + 2] +
          pow(knot_vect[i + 0], 2) * knot_vect[i + 3] -
          knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] -
          knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
          knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] +
          knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3]));
    row.push_back(c0);
    row.push_back(c1);
    row.push_back(c2);
    row.push_back(c3);
    c0 = coeff_vect[i] *
        (pow(knot_vect[i + 1], 2) * knot_vect[i + 4] /
             (-pow(knot_vect[i + 1], 3) + pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] +
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4]) +
         pow(knot_vect[i + 0], 2) * knot_vect[i + 2] /
             (-pow(knot_vect[i + 0], 2) * knot_vect[i + 1] +
              pow(knot_vect[i + 0], 2) * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * pow(knot_vect[i + 2], 2) -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 2], 2) * knot_vect[i + 3]) +
         knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] /
             (-knot_vect[i + 0] * pow(knot_vect[i + 1], 2) +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2)));
    c1 = coeff_vect[i] *
        (-pow(knot_vect[i + 1], 2) /
             (-pow(knot_vect[i + 1], 3) + pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] +
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4]) -
         2 * knot_vect[i + 1] * knot_vect[i + 4] /
             (-pow(knot_vect[i + 1], 3) + pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] +
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4]) -
         pow(knot_vect[i + 0], 2) /
             (-pow(knot_vect[i + 0], 2) * knot_vect[i + 1] +
              pow(knot_vect[i + 0], 2) * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * pow(knot_vect[i + 2], 2) -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 2], 2) * knot_vect[i + 3]) -
         2 * knot_vect[i + 0] * knot_vect[i + 2] /
             (-pow(knot_vect[i + 0], 2) * knot_vect[i + 1] +
              pow(knot_vect[i + 0], 2) * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * pow(knot_vect[i + 2], 2) -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 2], 2) * knot_vect[i + 3]) -
         knot_vect[i + 0] * knot_vect[i + 1] /
             (-knot_vect[i + 0] * pow(knot_vect[i + 1], 2) +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2)) -
         knot_vect[i + 0] * knot_vect[i + 3] /
             (-knot_vect[i + 0] * pow(knot_vect[i + 1], 2) +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2)) -
         knot_vect[i + 1] * knot_vect[i + 3] /
             (-knot_vect[i + 0] * pow(knot_vect[i + 1], 2) +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2)));
    c2 = coeff_vect[i] *
        (2 * knot_vect[i + 1] /
             (-pow(knot_vect[i + 1], 3) + pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] +
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4]) +
         knot_vect[i + 4] /
             (-pow(knot_vect[i + 1], 3) + pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] +
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4]) +
         2 * knot_vect[i + 0] /
             (-pow(knot_vect[i + 0], 2) * knot_vect[i + 1] +
              pow(knot_vect[i + 0], 2) * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * pow(knot_vect[i + 2], 2) -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 2], 2) * knot_vect[i + 3]) +
         knot_vect[i + 2] /
             (-pow(knot_vect[i + 0], 2) * knot_vect[i + 1] +
              pow(knot_vect[i + 0], 2) * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * pow(knot_vect[i + 2], 2) -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 2], 2) * knot_vect[i + 3]) +
         knot_vect[i + 0] /
             (-knot_vect[i + 0] * pow(knot_vect[i + 1], 2) +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2)) +
         knot_vect[i + 1] /
             (-knot_vect[i + 0] * pow(knot_vect[i + 1], 2) +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2)) +
         knot_vect[i + 3] /
             (-knot_vect[i + 0] * pow(knot_vect[i + 1], 2) +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2)));
    c3 = coeff_vect[i] *
        (-1 /
             (-pow(knot_vect[i + 1], 3) + pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] +
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4]) -
         1 /
             (-pow(knot_vect[i + 0], 2) * knot_vect[i + 1] +
              pow(knot_vect[i + 0], 2) * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * pow(knot_vect[i + 2], 2) -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 2], 2) * knot_vect[i + 3]) -
         1 /
             (-knot_vect[i + 0] * pow(knot_vect[i + 1], 2) +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] -
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] -
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2)));
    row.push_back(c0);
    row.push_back(c1);
    row.push_back(c2);
    row.push_back(c3);
    c0 = coeff_vect[i] *
        (-knot_vect[i + 0] * pow(knot_vect[i + 3], 2) /
             (-knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] +
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 0] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2) + pow(knot_vect[i + 3], 3)) -
         knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] /
             (-pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 3], 2) * knot_vect[i + 4]) -
         knot_vect[i + 2] * pow(knot_vect[i + 4], 2) /
             (-knot_vect[i + 1] * pow(knot_vect[i + 2], 2) +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 2], 2) * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * pow(knot_vect[i + 4], 2) +
              knot_vect[i + 3] * pow(knot_vect[i + 4], 2)));
    c1 = coeff_vect[i] *
        (2 * knot_vect[i + 0] * knot_vect[i + 3] /
             (-knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] +
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 0] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2) + pow(knot_vect[i + 3], 3)) +
         pow(knot_vect[i + 3], 2) /
             (-knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] +
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 0] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2) + pow(knot_vect[i + 3], 3)) +
         knot_vect[i + 1] * knot_vect[i + 3] /
             (-pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 3], 2) * knot_vect[i + 4]) +
         knot_vect[i + 1] * knot_vect[i + 4] /
             (-pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 3], 2) * knot_vect[i + 4]) +
         knot_vect[i + 3] * knot_vect[i + 4] /
             (-pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 3], 2) * knot_vect[i + 4]) +
         2 * knot_vect[i + 2] * knot_vect[i + 4] /
             (-knot_vect[i + 1] * pow(knot_vect[i + 2], 2) +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 2], 2) * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * pow(knot_vect[i + 4], 2) +
              knot_vect[i + 3] * pow(knot_vect[i + 4], 2)) +
         pow(knot_vect[i + 4], 2) /
             (-knot_vect[i + 1] * pow(knot_vect[i + 2], 2) +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 2], 2) * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * pow(knot_vect[i + 4], 2) +
              knot_vect[i + 3] * pow(knot_vect[i + 4], 2)));
    c2 = coeff_vect[i] *
        (-knot_vect[i + 0] /
             (-knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] +
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 0] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2) + pow(knot_vect[i + 3], 3)) -
         2 * knot_vect[i + 3] /
             (-knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] +
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 0] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2) + pow(knot_vect[i + 3], 3)) -
         knot_vect[i + 1] /
             (-pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 3], 2) * knot_vect[i + 4]) -
         knot_vect[i + 3] /
             (-pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 3], 2) * knot_vect[i + 4]) -
         knot_vect[i + 4] /
             (-pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 3], 2) * knot_vect[i + 4]) -
         knot_vect[i + 2] /
             (-knot_vect[i + 1] * pow(knot_vect[i + 2], 2) +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 2], 2) * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * pow(knot_vect[i + 4], 2) +
              knot_vect[i + 3] * pow(knot_vect[i + 4], 2)) -
         2 * knot_vect[i + 4] /
             (-knot_vect[i + 1] * pow(knot_vect[i + 2], 2) +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 2], 2) * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * pow(knot_vect[i + 4], 2) +
              knot_vect[i + 3] * pow(knot_vect[i + 4], 2)));
    c3 = coeff_vect[i] *
        (1 /
             (-knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 2] +
              knot_vect[i + 0] * knot_vect[i + 1] * knot_vect[i + 3] +
              knot_vect[i + 0] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 0] * pow(knot_vect[i + 3], 2) +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 2] * pow(knot_vect[i + 3], 2) + pow(knot_vect[i + 3], 3)) +
         1 /
             (-pow(knot_vect[i + 1], 2) * knot_vect[i + 2] +
              pow(knot_vect[i + 1], 2) * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * pow(knot_vect[i + 3], 2) -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 3], 2) * knot_vect[i + 4]) +
         1 /
             (-knot_vect[i + 1] * pow(knot_vect[i + 2], 2) +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
              knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] -
              knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] +
              pow(knot_vect[i + 2], 2) * knot_vect[i + 4] -
              knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] -
              knot_vect[i + 2] * pow(knot_vect[i + 4], 2) +
              knot_vect[i + 3] * pow(knot_vect[i + 4], 2)));

    row.push_back(c0);
    row.push_back(c1);
    row.push_back(c2);
    row.push_back(c3);
    c0 = coeff_vect[i] *
        (pow(knot_vect[i + 4], 3) /
         (-knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
          knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] +
          knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] -
          knot_vect[i + 1] * pow(knot_vect[i + 4], 2) +
          knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] -
          knot_vect[i + 2] * pow(knot_vect[i + 4], 2) -
          knot_vect[i + 3] * pow(knot_vect[i + 4], 2) + pow(knot_vect[i + 4], 3)));
    c1 = coeff_vect[i] *
        (-3 * pow(knot_vect[i + 4], 2) /
         (-knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
          knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] +
          knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] -
          knot_vect[i + 1] * pow(knot_vect[i + 4], 2) +
          knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] -
          knot_vect[i + 2] * pow(knot_vect[i + 4], 2) -
          knot_vect[i + 3] * pow(knot_vect[i + 4], 2) + pow(knot_vect[i + 4], 3)));
    c2 = coeff_vect[i] *
        (3 * knot_vect[i + 4] /
         (-knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
          knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] +
          knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] -
          knot_vect[i + 1] * pow(knot_vect[i + 4], 2) +
          knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] -
          knot_vect[i + 2] * pow(knot_vect[i + 4], 2) -
          knot_vect[i + 3] * pow(knot_vect[i + 4], 2) + pow(knot_vect[i + 4], 3)));
    c3 = coeff_vect[i] *
        (-1 /
         (-knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 3] +
          knot_vect[i + 1] * knot_vect[i + 2] * knot_vect[i + 4] +
          knot_vect[i + 1] * knot_vect[i + 3] * knot_vect[i + 4] -
          knot_vect[i + 1] * pow(knot_vect[i + 4], 2) +
          knot_vect[i + 2] * knot_vect[i + 3] * knot_vect[i + 4] -
          knot_vect[i + 2] * pow(knot_vect[i + 4], 2) -
          knot_vect[i + 3] * pow(knot_vect[i + 4], 2) + pow(knot_vect[i + 4], 3)));
    row.push_back(c0);
    row.push_back(c1);
    row.push_back(c2);
    row.push_back(c3);
    constants.push_back(row);
  }

  // Cache derivative constants

  for (int i = 0; i < dnknot_vect.size() - 3; i++) {
    std::vector<double> row;
    c0 = dncoeff_vect[i] *
        (pow(dnknot_vect[i + 0], 2) /
         (pow(dnknot_vect[i + 0], 2) - dnknot_vect[i + 0] * dnknot_vect[i + 1] -
          dnknot_vect[i + 0] * dnknot_vect[i + 2] + dnknot_vect[i + 1] * dnknot_vect[i + 2]));
    c1 = dncoeff_vect[i] *
        (-2 * dnknot_vect[i + 0] /
         (pow(dnknot_vect[i + 0], 2) - dnknot_vect[i + 0] * dnknot_vect[i + 1] -
          dnknot_vect[i + 0] * dnknot_vect[i + 2] + dnknot_vect[i + 1] * dnknot_vect[i + 2]));
    c2 = dncoeff_vect[i] *
        (1 /
         (pow(dnknot_vect[i + 0], 2) - dnknot_vect[i + 0] * dnknot_vect[i + 1] -
          dnknot_vect[i + 0] * dnknot_vect[i + 2] + dnknot_vect[i + 1] * dnknot_vect[i + 2]));
    row.push_back(c0);
    row.push_back(c1);
    row.push_back(c2);
    c0 = dncoeff_vect[i] *
        (-dnknot_vect[i + 1] * dnknot_vect[i + 3] /
             (pow(dnknot_vect[i + 1], 2) - dnknot_vect[i + 1] * dnknot_vect[i + 2] -
              dnknot_vect[i + 1] * dnknot_vect[i + 3] + dnknot_vect[i + 2] * dnknot_vect[i + 3]) -
         dnknot_vect[i + 0] * dnknot_vect[i + 2] /
             (dnknot_vect[i + 0] * dnknot_vect[i + 1] - dnknot_vect[i + 0] * dnknot_vect[i + 2] -
              dnknot_vect[i + 1] * dnknot_vect[i + 2] + pow(dnknot_vect[i + 2], 2)));
    c1 = dncoeff_vect[i] *
        (dnknot_vect[i + 1] /
             (pow(dnknot_vect[i + 1], 2) - dnknot_vect[i + 1] * dnknot_vect[i + 2] -
              dnknot_vect[i + 1] * dnknot_vect[i + 3] + dnknot_vect[i + 2] * dnknot_vect[i + 3]) +
         dnknot_vect[i + 3] /
             (pow(dnknot_vect[i + 1], 2) - dnknot_vect[i + 1] * dnknot_vect[i + 2] -
              dnknot_vect[i + 1] * dnknot_vect[i + 3] + dnknot_vect[i + 2] * dnknot_vect[i + 3]) +
         dnknot_vect[i + 0] /
             (dnknot_vect[i + 0] * dnknot_vect[i + 1] - dnknot_vect[i + 0] * dnknot_vect[i + 2] -
              dnknot_vect[i + 1] * dnknot_vect[i + 2] + pow(dnknot_vect[i + 2], 2)) +
         dnknot_vect[i + 2] /
             (dnknot_vect[i + 0] * dnknot_vect[i + 1] - dnknot_vect[i + 0] * dnknot_vect[i + 2] -
              dnknot_vect[i + 1] * dnknot_vect[i + 2] + pow(dnknot_vect[i + 2], 2)));
    c2 = dncoeff_vect[i] *
        (-1 /
             (pow(dnknot_vect[i + 1], 2) - dnknot_vect[i + 1] * dnknot_vect[i + 2] -
              dnknot_vect[i + 1] * dnknot_vect[i + 3] + dnknot_vect[i + 2] * dnknot_vect[i + 3]) -
         1 /
             (dnknot_vect[i + 0] * dnknot_vect[i + 1] - dnknot_vect[i + 0] * dnknot_vect[i + 2] -
              dnknot_vect[i + 1] * dnknot_vect[i + 2] + pow(dnknot_vect[i + 2], 2)));
    row.push_back(c0);
    row.push_back(c1);
    row.push_back(c2);
    c0 = dncoeff_vect[i] *
        (pow(dnknot_vect[i + 3], 2) /
         (dnknot_vect[i + 1] * dnknot_vect[i + 2] - dnknot_vect[i + 1] * dnknot_vect[i + 3] -
          dnknot_vect[i + 2] * dnknot_vect[i + 3] + pow(dnknot_vect[i + 3], 2)));
    c1 = dncoeff_vect[i] *
        (-2 * dnknot_vect[i + 3] /
         (dnknot_vect[i + 1] * dnknot_vect[i + 2] - dnknot_vect[i + 1] * dnknot_vect[i + 3] -
          dnknot_vect[i + 2] * dnknot_vect[i + 3] + pow(dnknot_vect[i + 3], 2)));
    c2 = dncoeff_vect[i] *
        (1 /
         (dnknot_vect[i + 1] * dnknot_vect[i + 2] - dnknot_vect[i + 1] * dnknot_vect[i + 3] -
          dnknot_vect[i + 2] * dnknot_vect[i + 3] + pow(dnknot_vect[i + 3], 2)));
    row.push_back(c0);
    row.push_back(c1);
    row.push_back(c2);
    dconstants.push_back(row);
  }
}

uf3_pair_bspline::~uf3_pair_bspline() {}

//value function definition
//input -->rij; output -->value on the bspline curve
double uf3_pair_bspline::bsvalue(double value_rij)
{
  //utils::logmesg(lmp,"return this value {}",this->uf3_pair_bspline::main_eval_loop(value_rij,bspline_degree,knot_vect,
  //                                                coeff_vect));
  return this->uf3_pair_bspline::main_eval_loop(value_rij, bspline_degree, knot_vect, coeff_vect);
}

double uf3_pair_bspline::bsderivative(double value_rij)
{
  if (bspline_degree < 1) {
    return 0;
  } else {
    return this->uf3_pair_bspline::main_eval_loop(value_rij, bspline_degree - 1, dnknot_vect,
                                                  dncoeff_vect);
  }
}

double uf3_pair_bspline::main_eval_loop(double r, int ibspline_degree,
                                        const std::vector<double> &iknot_vect,
                                        const std::vector<double> &icoeff_vect)
{
  int ipos_i = 0;
  int iknot_mult = 0;
  h = 0;
  max_count = 0;
  knot_affect_start = 0;
  knot_affect_end = 0;
  temp2 = 0;
  temp3 = 0;
  temp_val = 0;
  if (iknot_vect.front() <= r && r < iknot_vect.back()) {
    //Determine the interval for value_rij
    for (int i = 0; i < knot_vect_size - 1; ++i) {
      if (iknot_vect[i] <= r && r < iknot_vect[i + 1]) {
        ipos_i = i;
        break;
      }
    }
  }
  //#---------------------
  //#----main eval loop---
  //#---------------------
  knot_affect_start = ipos_i - ibspline_degree;

  double rsq = r * r;
  double rth = rsq * r;

  if (ibspline_degree == 3) {
    temp_val = rth * constants[knot_affect_start + 3][3] +
        rsq * constants[knot_affect_start + 3][2] + r * constants[knot_affect_start + 3][1] +
        constants[knot_affect_start + 3][0];
    temp_val += rth * constants[knot_affect_start + 2][7] +
        rsq * constants[knot_affect_start + 2][6] + r * constants[knot_affect_start + 2][5] +
        constants[knot_affect_start + 2][4];
    temp_val += rth * constants[knot_affect_start + 1][11] +
        rsq * constants[knot_affect_start + 1][10] + r * constants[knot_affect_start + 1][9] +
        constants[knot_affect_start + 1][8];
    temp_val += rth * constants[knot_affect_start + 0][15] +
        rsq * constants[knot_affect_start + 0][14] + r * constants[knot_affect_start + 0][13] +
        constants[knot_affect_start + 0][12];
    return temp_val;
  }

  else if (ibspline_degree == 2) {
    temp_val = rsq * dconstants[knot_affect_start + 2][2] +
        r * dconstants[knot_affect_start + 2][1] + dconstants[knot_affect_start + 2][0];
    temp_val += rsq * dconstants[knot_affect_start + 1][5] +
        r * dconstants[knot_affect_start + 1][4] + dconstants[knot_affect_start + 1][3];
    temp_val += rsq * dconstants[knot_affect_start + 0][8] +
        r * dconstants[knot_affect_start + 0][7] + dconstants[knot_affect_start + 0][6];
    return temp_val;
  }

  return 0;
}
