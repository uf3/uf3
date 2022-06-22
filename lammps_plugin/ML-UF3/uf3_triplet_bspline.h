//Simple implementation of 3d bspline
//Takes knot, coeff matrix and (rij,rik,rjk) as input and returns V3 and V'3 as the output
//Repetitive calls uf3_pair_bspline 
//
//
//
#include  "pointers.h"

#include <vector>
#include "uf3_pair_bspline.h"

#ifndef UF3_TRIPLET_BSPLINE_H 
#define UF3_TRIPLET_BSPLINE_H

namespace LAMMPS_NS {
class uf3_triplet_bspline
{
private:
	std::vector<std::vector<std::vector<double>>> coeff_matrix;
	std::vector<std::vector<double>> knot_matrix;
	std::vector<std::vector<double>> alpha, alpha_3rd_term;
	std::vector<double> alpha_2nd_term;
	uf3_pair_bspline UF3Pair;
	double d1st_term, d2nd_term, d3rd_term, derivative;
	double bsderivative_return_val[3];
	double return_val[4];
	LAMMPS* lmp;
public:
	//Dummy Constructor
	uf3_triplet_bspline();
	uf3_triplet_bspline(LAMMPS* ulmp, const std::vector<std::vector<double>> &uknot_matrix,
			const std::vector<std::vector<std::vector<double>>> &ucoeff_matrix);
	//uf3_triplet_bspline(double *uknot_matrix_start, int uknot_matrix_shape[3], 
	//			double *ucoeff_matrix_start, int ucoeff_matrix_shape[3]);
	~uf3_triplet_bspline();
	double bsvalue(double value_rij,double value_rik,double value_rjk);
	double* bsvalue_bsderivative(double value_rij,double value_rik,double value_rjk);
	void main_eval(double ivalue_rij,double ivalue_rik,double ivalue_rjk);
	//static double return_val[2];
};
}
#endif
