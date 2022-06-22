#include "uf3_pair_bspline.h"

#include "utils.h"
#include <vector>

using namespace LAMMPS_NS;

//Call constructor
//Dummy constructor
uf3_pair_bspline::uf3_pair_bspline(){}
//Constructor
//passing vect by reference 
uf3_pair_bspline::uf3_pair_bspline(LAMMPS* ulmp, int ubspline_degree,
				const std::vector<double> &uknot_vect, const std::vector<double> &ucoeff_vect){
	lmp = ulmp;
	bspline_degree = ubspline_degree;
	knot_vect = uknot_vect;
	coeff_vect = ucoeff_vect;
	knot_vect_size = uknot_vect.size();
	coeff_vect_size = ucoeff_vect.size();
	//initialize dncoeff_vect and dnknot_coeff for derivates
	for (int i=0;i<coeff_vect_size-1;++i){
		dntemp4 = bspline_degree/(knot_vect[i+bspline_degree+1]-knot_vect[i+1]);
		dncoeff_vect.push_back((coeff_vect[i+1]-coeff_vect[i])*dntemp4);
	}
	for (int i=1;i<knot_vect_size-1;++i){
		dnknot_vect.push_back(knot_vect[i]);
	}
}

uf3_pair_bspline::~uf3_pair_bspline(){}

//value function definition
//input -->rij; output -->value on the bspline curve
double uf3_pair_bspline::bsvalue(double value_rij)
{
	//utils::logmesg(lmp,"return this value {}",this->uf3_pair_bspline::main_eval_loop(value_rij,bspline_degree,knot_vect,
        //                                                coeff_vect));
	return this->uf3_pair_bspline::main_eval_loop(value_rij,bspline_degree,knot_vect,
							coeff_vect);
}

double uf3_pair_bspline::bsderivative(double value_rij)
{
	if (bspline_degree < 1){
		return 0;}
	else{
		return this->uf3_pair_bspline::main_eval_loop(value_rij,bspline_degree-1,dnknot_vect,
								dncoeff_vect);}
}


double uf3_pair_bspline::main_eval_loop(double ivalue_rij, int ibspline_degree,
					const std::vector<double> &iknot_vect,
					const std::vector<double> &icoeff_vect){
	int ipos_i = 0;
	int iknot_mult = 0;
	h = 0; max_count = 0; knot_affect_start = 0; knot_affect_end = 0;
	temp2 = 0; temp3 = 0;
	temp_val = 0;
	if (iknot_vect.front() <= ivalue_rij && ivalue_rij < iknot_vect.back()){
		//Determine the interval for value_rij
		for (int i=0; i<knot_vect_size-1; ++i){
			if (iknot_vect[i] <= ivalue_rij && ivalue_rij < iknot_vect[i+1])
				{ipos_i = i; break;}
		}
		//if value_rij is equal to existing knot then
		//determine multiplicity of existing knot
		if (ivalue_rij==iknot_vect[ipos_i]){
			for (int i=0;i<knot_vect_size;++i){
				if (iknot_vect[i]==ivalue_rij){
					iknot_mult = iknot_mult + 1;
				}
				else if (ivalue_rij<iknot_vect[i]){break;}
				else {}
			}
		}
	}
	//Extrapolating outside the domain?
	//--value_rij on the left hand side
	else if (ivalue_rij < iknot_vect[0]){
		for (int i=0; i < knot_vect_size-1; ++i){
			if (iknot_vect[i] != iknot_vect[i+1])
				{ipos_i = i; break;}
		}
	}
	//--value_rij on the right hand side
	else
	{
		for (int i=knot_vect_size-1;i>0;--i){
			if (iknot_vect[i] != iknot_vect[i-1])
				{ipos_i = i-1; break;}
		}
	}
	//#---------------------
	//#----main eval loop---
	//#---------------------
	h = ibspline_degree - iknot_mult;
	//value_rij = existing knots and the knot_mult > bspline_dgree
	if (h==-1){
		temp_val = icoeff_vect[ipos_i-ibspline_degree];
		return temp_val;
	}
	else{
		knot_affect_start = ipos_i - ibspline_degree;
		knot_affect_end = ipos_i - iknot_mult;
		//set knot affect start(end) = 0 if < 0; can happen
		//if value_rij is close to the left hand side of the domain
		if (knot_affect_start <0)
			{knot_affect_start = 0;}
		if (knot_affect_end < knot_affect_start)
			{knot_affect_end =knot_affect_start;}
		
		max_count = knot_affect_end + 1 - knot_affect_start;
		//P[3][3] = {0};
		//Set the 0th affected Coeffn
		for (int i=0; i< max_count; ++i)
			{P[0][i] = icoeff_vect[i+knot_affect_start];}
		
		for (int r=1; r<h+1;++r){
			for (int i=ipos_i-ibspline_degree+r; i <ipos_i-iknot_mult+1; ++i){
				temp2 = i+ibspline_degree-r+1;
				temp3 = (ivalue_rij-iknot_vect[i])/(iknot_vect[temp2]-iknot_vect[i]);
				P[r][i-knot_affect_start] = ((1-temp3)*P[r-1][i-knot_affect_start-1]) + (temp3*P[r-1][i-knot_affect_start]);
			}
		}
		temp_val = P[ibspline_degree-iknot_mult][ipos_i-iknot_mult-knot_affect_start];
		return temp_val;
	}
}
