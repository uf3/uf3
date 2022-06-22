#include "uf3_triplet_bspline.h"
#include <vector>
#include <iostream>

using namespace LAMMPS_NS;

//Dummy constructor
uf3_triplet_bspline::uf3_triplet_bspline(){}

//Call constructor
uf3_triplet_bspline::uf3_triplet_bspline(LAMMPS* ulmp, const std::vector<std::vector<double>> &uknot_matrix,
			const std::vector<std::vector<std::vector<double>>> &ucoeff_matrix){
	lmp = ulmp;
	knot_matrix = uknot_matrix;
	coeff_matrix = ucoeff_matrix;
	alpha.resize(coeff_matrix.size()+1);
	alpha_3rd_term.resize(coeff_matrix.size()+1);
	for(int l=0; l<coeff_matrix.size();l++){
		alpha[l].resize(coeff_matrix[l].size());
		alpha_3rd_term[l].resize(coeff_matrix[l].size());
	}
	alpha[coeff_matrix.size()].resize(coeff_matrix.size());
	alpha_2nd_term.resize(coeff_matrix.size());
	alpha_3rd_term[coeff_matrix.size()].resize(coeff_matrix.size());
}

//Destructor
uf3_triplet_bspline::~uf3_triplet_bspline(){}

double uf3_triplet_bspline::bsvalue(double value_rij,double value_rik,double value_rjk){
	this->uf3_triplet_bspline::main_eval(value_rij, value_rik, value_rjk);
	return return_val[0];
}

double* uf3_triplet_bspline::bsvalue_bsderivative(double value_rij,double value_rik,double value_rjk){
	//this->uf3_triplet_bspline::main_eval(value_rij, value_rik, value_rjk);
	bsderivative_return_val[0] = return_val[1];
	bsderivative_return_val[1] = return_val[2];
	bsderivative_return_val[2] = return_val[3];
	return bsderivative_return_val;

}

void uf3_triplet_bspline::main_eval(double ivalue_rij,double ivalue_rik,double ivalue_rjk){
	//static double return_val[4];
	//alpha.resize(coeff_matrix.size()+1);
	//alpha_3rd_term.resize(coeff_matrix.size()+1);
	for(int l=0; l<coeff_matrix.size();l++){
		//alpha[l].resize(coeff_matrix[l].size());
		//alpha_3rd_term[l].resize(coeff_matrix[l].size());
		for(int m=0; m<coeff_matrix[l].size();m++){
			UF3Pair = uf3_pair_bspline(lmp,3,knot_matrix[0],coeff_matrix[l][m]);
			alpha[l][m] = UF3Pair.bsvalue(ivalue_rjk);
			alpha_3rd_term[l][m] = UF3Pair.bsderivative(ivalue_rjk);
		}
	}
	//alpha[coeff_matrix.size()].resize(coeff_matrix.size());
	//alpha_2nd_term.resize(coeff_matrix.size());
	//alpha_3rd_term[coeff_matrix.size()].resize(coeff_matrix.size());
	for(int l=0; l<coeff_matrix.size();l++){
		UF3Pair = uf3_pair_bspline(lmp,3,knot_matrix[1],alpha[l]);
		alpha[coeff_matrix.size()][l] = UF3Pair.bsvalue(ivalue_rik);
		alpha_2nd_term[l] = UF3Pair.bsderivative(ivalue_rik);
		
		UF3Pair = uf3_pair_bspline(lmp,3,knot_matrix[1],alpha_3rd_term[l]);
		alpha_3rd_term[coeff_matrix.size()][l] = UF3Pair.bsvalue(ivalue_rik);
	}
	UF3Pair = uf3_pair_bspline(lmp,3,knot_matrix[2],alpha[coeff_matrix.size()]);
	return_val[0] = UF3Pair.bsvalue(ivalue_rij);
	
	d1st_term = UF3Pair.bsderivative(ivalue_rij);
	
	UF3Pair = uf3_pair_bspline(lmp,3,knot_matrix[2],alpha_2nd_term);
	d2nd_term = UF3Pair.bsvalue(ivalue_rij);
	
	UF3Pair = uf3_pair_bspline(lmp,3,knot_matrix[2],alpha_3rd_term[coeff_matrix.size()]);
	d3rd_term = UF3Pair.bsvalue(ivalue_rij);

	//derivative = d1st_term + d2nd_term + d3rd_term;
	return_val[1] = d1st_term;
	return_val[2] = d2nd_term;
	return_val[3] = d3rd_term;
	//return return_val;
}


