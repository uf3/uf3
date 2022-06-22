/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 *    Contributing author: Ajinkya Hire, Richard Hennig (U of Florida)
 *    ------------------------------------------------------------------------- */

#include "pair_uf3.h"
#include "uf3_pair_bspline.h"
#include "uf3_triplet_bspline.h"

#include "atom.h"
#include "memory.h"
#include "error.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "force.h"
#include "comm.h"
#include "text_file_reader.h"

#include <cmath>

using namespace LAMMPS_NS;

PairUF3::PairUF3(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0; // 1 if single() routine exists
  restartinfo = 0;  // 1 if pair style writes restart info
  maxshort = 10;
  neighshort = nullptr;
}

PairUF3::~PairUF3()
{
  if (copymode) return;
  if (allocated)
  {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    //memory->destroy(cutsq_2b);
    if (pot_3b){
    memory->destroy(setflag_3b);
    memory->destroy(cut_3b);
    memory->destroy(neighshort);
    }
  }
}

/* ----------------------------------------------------------------------
 *     global settings
 *     ------------------------------------------------------------------------- */

void PairUF3::settings(int narg, char **arg)
{
  if (narg!=2) error->all(FLERR, "UF3: Invalid number of argument in pair settings\n\
            Are you running 2-bordy or 2 & 3-body UF potential\n\
            Also how many elements?");
  nbody_flag = utils::numeric(FLERR,arg[0],true,lmp);
  num_of_elements = utils::numeric(FLERR,arg[1],true,lmp);//atom->ntypes;
  if (num_of_elements!=atom->ntypes){
    if(comm->me==0) utils::logmesg(lmp,"\nUF3: Number of elements provided in the input file and \
number of elements detected by lammps in the structure are not same\n\
     proceed with caution"); 
  }
  if (nbody_flag == 2){
    pot_3b = false;
    n2body_pot_files = num_of_elements*(num_of_elements+1)/2;
    tot_pot_files = n2body_pot_files;
  }
  else if (nbody_flag == 3){
    pot_3b = true;
    n2body_pot_files = num_of_elements*(num_of_elements+1)/2;
    n3body_pot_files = num_of_elements*(num_of_elements*(num_of_elements+1)/2);
    tot_pot_files = n2body_pot_files + n3body_pot_files;
  }
  else error->all(FLERR, "UF3: UF3 not yet implemented for {}-body",nbody_flag);
}

/* ----------------------------------------------------------------------
 *    set coeffs for one or more type pairs
 *    ------------------------------------------------------------------------- */
void PairUF3::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg!=tot_pot_files+2) error->all(FLERR, "UF3: UF3 invalid number of argument in pair coeff; Number of potential files provided is not correct");
  // open UF3 potential file on proc 0
  for(int i=2; i<narg; i++){
      uf3_read_pot_file(arg[i]);
  }
  //setflag check needed here
  for (int i =1;i<num_of_elements+1;i++){
    for (int j=1;j<num_of_elements+1;j++){
      if (setflag[i][j]!=1) error->all(FLERR, "UF3: Not all 2-body UF potentials are set, missing potential file for {}-{} interaction",i,j);
    }
  }
  if (pot_3b){
    for (int i =1;i<num_of_elements+1;i++){
      for (int j=1;j<num_of_elements+1;j++){
        for (int k=1;k<num_of_elements+1;k++){
          if(setflag_3b[i][j][k]!=1) error->all(FLERR, "UF3: Not all 3-body UF potentials are set, missing potential file for {}-{}-{} interaction",i,j,k);
        }
      }
    }
  }
  for (int i =1;i<num_of_elements+1;i++){
    for (int j=i;j<num_of_elements+1;j++){
      UFBS2b[i][j] = uf3_pair_bspline(lmp,3,n2b_knot[i][j], n2b_coeff[i][j]);
      UFBS2b[j][i] = uf3_pair_bspline(lmp,3,n2b_knot[j][i], n2b_coeff[j][i]);
    }
    if (pot_3b){
      for (int j=1;j<num_of_elements+1;j++){
        for (int k=j;k<num_of_elements+1;k++){
          key = std::to_string(i)+std::to_string(j)+std::to_string(k);
          UFBS3b[i][j][k] = uf3_triplet_bspline(lmp,n3b_knot_matrix[i][j][k],n3b_coeff_matrix[key]);
          key = std::to_string(i)+std::to_string(k)+std::to_string(j); 
          UFBS3b[i][k][j] = uf3_triplet_bspline(lmp,n3b_knot_matrix[i][k][j],n3b_coeff_matrix[key]);
        }
      }
    }
  }
}

void PairUF3::allocate()
{
  allocated = 1;
  //Contains info about wether UF potential were found for type i and j
  memory->create(setflag,num_of_elements+1,num_of_elements+1,"pair:setflag");
  //Contains info about 2-body cutoff distance for type i and j
  memory->create(cutsq,num_of_elements+1,num_of_elements+1,"pair:cutsq");
  //memory->create(cutsq_2b,num_of_elements+1,num_of_elements+1,"pair:cutsq_2b");
  //Contains knot_vect of 2-body potential for type i and j
  n2b_knot.resize(num_of_elements+1);
  n2b_coeff.resize(num_of_elements+1);
  UFBS2b.resize(num_of_elements+1);
  for(int i =1; i<num_of_elements+1; i++){
   n2b_knot[i].resize(num_of_elements+1);
   n2b_coeff[i].resize(num_of_elements+1);
   UFBS2b[i].resize(num_of_elements+1);
  }
  if(pot_3b)
  {
    //Contains info about wether UF potential were found for type i, j and k
    memory->create(setflag_3b,num_of_elements+1,num_of_elements+1,num_of_elements+1,"pair:setflag_3b");
    //Contains info about 3-body cutoff distance for type i, j and k
    memory->create(cut_3b,num_of_elements+1,num_of_elements+1,"pair:cut_3b");
    n3b_knot_matrix.resize(num_of_elements+1);
    UFBS3b.resize(num_of_elements+1);
    for(int i =1; i<num_of_elements+1; i++){
      n3b_knot_matrix[i].resize(num_of_elements+1);
      UFBS3b[i].resize(num_of_elements+1);
      for(int j =1; j<num_of_elements+1; j++){
        n3b_knot_matrix[i][j].resize(num_of_elements+1);
        UFBS3b[i][j].resize(num_of_elements+1);
      }
    }
    memory->create(neighshort,maxshort,"pair:neighshort");
  }
}

void PairUF3::uf3_read_pot_file(char *potf_name)
{
  if(comm->me==0) utils::logmesg(lmp,"\nUF3: Opening {} file\n",potf_name);
  
  FILE * fp;
  fp = utils::open_potential(potf_name,lmp,nullptr);
  //if (fp) error->all(FLERR,"UF3: {} file not found",potf_name);

  TextFileReader txtfilereader(fp,"UF3:POTFP");
  txtfilereader.ignore_comments = false;
  
  std::string temp_line = txtfilereader.next_line(2);
  Tokenizer fp1st_line(temp_line);
  
  if (fp1st_line.contains("#UF3 POT")==0) error->all(FLERR, "UF3: {} file is not UF3 POT type, found type {} {} on the file"
							,potf_name,fp1st_line.next(),fp1st_line.next());
  
  if(comm->me==0) utils::logmesg(lmp,"UF3: {} file is of type {} {}\n",potf_name,fp1st_line.next(),fp1st_line.next());
  
  temp_line = txtfilereader.next_line(1);
  Tokenizer fp2nd_line(temp_line);
  if (fp2nd_line.contains("2B")==1){
    temp_line = txtfilereader.next_line(4);
    ValueTokenizer fp3rd_line(temp_line);
    temp_type1 = fp3rd_line.next_int();
    temp_type2 = fp3rd_line.next_int();
    if(comm->me==0) utils::logmesg(lmp,"UF3: {} file contains 2-body UF3 potential for {} {}\n",potf_name,temp_type1,temp_type2);

    cutsq[temp_type1][temp_type2] = pow(fp3rd_line.next_double(),2);
    //if(comm->me==0) utils::logmesg(lmp,"UF3: Cutoff {}\n",cutsq[temp_type1][temp_type2]);
    cutsq[temp_type2][temp_type1] = cutsq[temp_type1][temp_type2];
    
    temp_line_len = fp3rd_line.next_int();

    temp_line = txtfilereader.next_line(temp_line_len);     
    ValueTokenizer fp4th_line(temp_line);

    n2b_knot[temp_type1][temp_type2].resize(temp_line_len);
    n2b_knot[temp_type2][temp_type1].resize(temp_line_len);
    for(int k=0;k<temp_line_len;k++){
      n2b_knot[temp_type1][temp_type2][k] = fp4th_line.next_double();
      n2b_knot[temp_type2][temp_type1][k] = n2b_knot[temp_type1][temp_type2][k];
    }

    temp_line = txtfilereader.next_line(1);
    ValueTokenizer fp5th_line(temp_line);

    temp_line_len = fp5th_line.next_int();
    
    temp_line = txtfilereader.next_line(temp_line_len);
    //utils::logmesg(lmp,"UF3:11 {}",temp_line);
    ValueTokenizer fp6th_line(temp_line);
    //if(comm->me==0) utils::logmesg(lmp,"UF3: {}\n",temp_line_len);
    n2b_coeff[temp_type1][temp_type2].resize(temp_line_len);
    n2b_coeff[temp_type2][temp_type1].resize(temp_line_len);
    
    for(int k=0;k<temp_line_len;k++){
      n2b_coeff[temp_type1][temp_type2][k] = fp6th_line.next_double();
      n2b_coeff[temp_type2][temp_type1][k] = n2b_coeff[temp_type1][temp_type2][k];
      //if(comm->me==0) utils::logmesg(lmp,"UF3: {}\n",n2b_coeff[temp_type1][temp_type2][k]);
    }
    //for(int i=0;i<n2b_coeff[temp_type1][temp_type2].size();i++) if(comm->me==0) utils::logmesg(lmp,"UF3: {}\n",n2b_coeff[temp_type1][temp_type2][i]);
    if(n2b_knot[temp_type1][temp_type2].size()!=n2b_coeff[temp_type1][temp_type2].size()+4){
      error->all(FLERR, "UF3: {} has incorrect knot and coeff data nknots!=ncoeffs + 3 +1",potf_name);
    }
    setflag[temp_type1][temp_type2] = 1;
    setflag[temp_type2][temp_type1] = 1;
  }
  else if (fp2nd_line.contains("3B")==1){
    temp_line = txtfilereader.next_line(9);
    ValueTokenizer fp3rd_line(temp_line);
    temp_type1 = fp3rd_line.next_int();
    temp_type2 = fp3rd_line.next_int();
    temp_type3 = fp3rd_line.next_int();
    if(comm->me==0) utils::logmesg(lmp,"UF3: {} file contains 3-body UF3 potential for {} {} {}\n",
					potf_name,temp_type1,temp_type2,temp_type3);
    
    cut3b_rjk = fp3rd_line.next_double();
    cut_3b[temp_type1][temp_type2] = std::max(fp3rd_line.next_double(),
					cut_3b[temp_type1][temp_type2]);
    cut3b_rik = fp3rd_line.next_double();
    if (cut_3b[temp_type1][temp_type2]!=cut3b_rik){
      error->all(FLERR, "UF3: rij!=rik, Current implementation only works for rij=rik");
    }
    if (2*cut3b_rik!=cut3b_rjk){
      error->all(FLERR, "UF3: 2rij=2rik!=rik, Current implementation only works for 2rij=2rik!=rik");
    }
    cut_3b[temp_type1][temp_type3] = std::max(cut_3b[temp_type1][temp_type3],cut3b_rik);
    
    temp_line_len = fp3rd_line.next_int();
    temp_line = txtfilereader.next_line(temp_line_len);
    ValueTokenizer fp4th_line(temp_line);
    
    n3b_knot_matrix[temp_type1][temp_type2][temp_type3].resize(3);
    n3b_knot_matrix[temp_type1][temp_type3][temp_type2].resize(3);

    n3b_knot_matrix[temp_type1][temp_type2][temp_type3][0].resize(temp_line_len);
    n3b_knot_matrix[temp_type1][temp_type3][temp_type2][0].resize(temp_line_len);
    for(int i=0;i<temp_line_len;i++){
      n3b_knot_matrix[temp_type1][temp_type2][temp_type3][0][i] = fp4th_line.next_double();
      n3b_knot_matrix[temp_type1][temp_type3][temp_type2][0][i] = n3b_knot_matrix[temp_type1][temp_type2][temp_type3][0][i];
    }
    
    temp_line_len = fp3rd_line.next_int();
    temp_line = txtfilereader.next_line(temp_line_len);
    ValueTokenizer fp5th_line(temp_line);
    n3b_knot_matrix[temp_type1][temp_type2][temp_type3][1].resize(temp_line_len);
    n3b_knot_matrix[temp_type1][temp_type3][temp_type2][1].resize(temp_line_len);
    for(int i=0;i<temp_line_len;i++){
      n3b_knot_matrix[temp_type1][temp_type2][temp_type3][1][i] = fp5th_line.next_double();
      n3b_knot_matrix[temp_type1][temp_type3][temp_type2][1][i] = n3b_knot_matrix[temp_type1][temp_type2][temp_type3][1][i];
    }
    

    temp_line_len = fp3rd_line.next_int();
    temp_line = txtfilereader.next_line(temp_line_len);
    ValueTokenizer fp6th_line(temp_line);
    n3b_knot_matrix[temp_type1][temp_type2][temp_type3][2].resize(temp_line_len);
    n3b_knot_matrix[temp_type1][temp_type3][temp_type2][2].resize(temp_line_len);
    for(int i=0;i<temp_line_len;i++){
      n3b_knot_matrix[temp_type1][temp_type2][temp_type3][2][i] = fp6th_line.next_double();
      n3b_knot_matrix[temp_type1][temp_type3][temp_type2][2][i] = n3b_knot_matrix[temp_type1][temp_type2][temp_type3][2][i];
    }
    
    /*if(comm->me==0){
    utils::logmesg(lmp,"UF3: knot_matrix\n");
    for(int i=0;i<3;i++){
      for(int j=0;j<n3b_knot_matrix[temp_type1][temp_type2][temp_type3][i].size();j++){
        utils::logmesg(lmp,"{} ",n3b_knot_matrix[temp_type1][temp_type2][temp_type3][i][j]);
      }
      utils::logmesg(lmp,"\n");
    }}*/
 
    temp_line = txtfilereader.next_line(3);
    ValueTokenizer fp7th_line(temp_line);
    
    coeff_matrix_dim1 = fp7th_line.next_int();
    coeff_matrix_dim2 = fp7th_line.next_int();
    coeff_matrix_dim3 = fp7th_line.next_int();
    if (n3b_knot_matrix[temp_type1][temp_type2][temp_type3][0].size()!=coeff_matrix_dim3+3+1){
      error->all(FLERR, "UF3: {} has incorrect knot and coeff data nknots!=ncoeffs + 3 +1",potf_name);
    }
    if (n3b_knot_matrix[temp_type1][temp_type2][temp_type3][1].size()!=coeff_matrix_dim2+3+1){
      error->all(FLERR, "UF3: {} has incorrect knot and coeff data nknots!=ncoeffs + 3 +1",potf_name);
    }
    if (n3b_knot_matrix[temp_type1][temp_type2][temp_type3][2].size()!=coeff_matrix_dim1+3+1){
      error->all(FLERR, "UF3: {} has incorrect knot and coeff data nknots!=ncoeffs + 3 +1",potf_name);
    }

    coeff_matrix_elements_len = coeff_matrix_dim3;
    
    key = std::to_string(temp_type1)+std::to_string(temp_type2)+std::to_string(temp_type3); 
    n3b_coeff_matrix[key].resize(coeff_matrix_dim1);
    for(int i=0;i<coeff_matrix_dim1;i++){
      n3b_coeff_matrix[key][i].resize(coeff_matrix_dim2);
      for(int j=0;j<coeff_matrix_dim2;j++){
        temp_line = txtfilereader.next_line(coeff_matrix_elements_len);
        ValueTokenizer coeff_line(temp_line);
        n3b_coeff_matrix[key][i][j].resize(coeff_matrix_dim3);
        for(int k=0;k<coeff_matrix_dim3;k++) {n3b_coeff_matrix[key][i][j][k] = coeff_line.next_double();}
      }
    }
    //if(comm->me==0) utils::logmesg(lmp,"UF3: Finished reading coeff matrix\n");
    
    //key = std::to_string(temp_type1)+std::to_string(temp_type3)+std::to_string(temp_type2);
    //n3b_coeff_matrix[key] = temp_3d_matrix;
    /*if(comm->me==0){
    utils::logmesg(lmp,"UF3: coeff_matrix\n");
    for(int i=0;i<coeff_matrix_dim1;i++){
      for(int j=0;j<coeff_matrix_dim2;j++){
        for(int k=0;k<coeff_matrix_dim3;k++) utils::logmesg(lmp,"{} ",n3b_coeff_matrix[key][i][j][k]);
        //utils::logmesg(lmp,"\n");
      }
      //utils::logmesg(lmp,"--------------{}\n",i);
    }}*/
    
    key = std::to_string(temp_type1)+std::to_string(temp_type3)+std::to_string(temp_type2);
    n3b_coeff_matrix[key] =  n3b_coeff_matrix[std::to_string(temp_type1)+std::to_string(temp_type2)+std::to_string(temp_type3)];
    setflag_3b[temp_type1][temp_type2][temp_type3] = 1;
    setflag_3b[temp_type1][temp_type3][temp_type2] = 1;
  }
  else error->all(FLERR, "UF3: {} file does not contain right words indicating whether it is 2 or 3 body potential",potf_name);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */
void PairUF3::init_style()
{
  if (force->newton_pair == 0)
  error->all(FLERR,"UF3 Pair style requires newton pair on");
  // request a default neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init list sets the pointer to full neighbour list requested in previous function
------------------------------------------------------------------------- */

void PairUF3::init_list(int /*id*/, class NeighList *ptr)
{
  list = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
double PairUF3::init_one(int i/*i*/, int /*j*/j)
{
  return sqrt(cutsq[i][j]); 
}


void PairUF3::compute(int eflag, int vflag)
{
  int i, j, k, ii, jj, kk, inum, jnum, knum, itype, jtype, ktype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair, fx, fy, fz;
  double del_rji[3], del_rki[3], del_rkj[3];
  double fij[3], fik[3], fjk[3];
  double fji[3], fki[3], fkj[3];
  double Fj[3], Fk[3];
  double rsq, rij, rik, rjk;
  int *ilist, *jlist, *klist, *numneigh, **firstneigh;
  
  ev_init(eflag, vflag);
  
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;  
  
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    evdwl = 0;
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    int numshort = 0;
    for (jj = 0; jj < jnum; jj++) {
      fx = 0;
      fy = 0;
      fz = 0;
      j = jlist[jj];
      j &= NEIGHMASK;
      
      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      if (rsq < cutsq[itype][jtype]) {
        rij = sqrt(rsq);

        if (pot_3b){
          if (rij<=cut_3b[itype][jtype]){
            neighshort[numshort] = j;
            if (numshort >= maxshort-1) {
              maxshort += maxshort/2;
              memory->grow(neighshort,maxshort,"pair:neighshort");
            }
            numshort = numshort + 1;
          }
        }
        
        fpair = UFBS2b[itype][jtype].bsderivative(rij)/rij;
        
        fx = delx * fpair;
        fy = dely * fpair;
        fz = delz * fpair;

        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;
        if (newton_pair || j < nlocal) {
          f[j][0] -= fx;
          f[j][1] -= fy;
          f[j][2] -= fz;
        }
        if (eflag) evdwl = UFBS2b[itype][jtype].bsvalue(rij);

        if (evflag) ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0.0, fx, fy, fz, delx, dely, delz);
      }
    }
    
    //3-body interaction
    //jth atom
    jnum = numshort - 1; 
    for (jj = 0; jj < jnum; jj++) {
      fij[0] = fji[0] = 0;
      fij[1] = fji[1] = 0;
      fij[2] = fji[2] = 0;
      j = neighshort[jj];
      jtype = type[j];
      del_rji[0] = x[j][0] - xtmp;
      del_rji[1] = x[j][1] - ytmp;
      del_rji[2] = x[j][2] - ztmp;
      rij = sqrt(((del_rji[0]*del_rji[0])+(del_rji[1]*del_rji[1])+(del_rji[2]*del_rji[2])));

      //kth atom
      for(kk = jj+1; kk<numshort; kk++){
        fik[0] = fki[0] = 0;
        fik[1] = fki[1] = 0;
        fik[2] = fki[2] = 0;
        
        fjk[0] = fkj[0] = 0;
        fjk[1] = fkj[1] = 0;
        fjk[2] = fkj[2] = 0;

        k = neighshort[kk];
        ktype = type[k];
        del_rki[0] = x[k][0] - xtmp;
        del_rki[1] = x[k][1] - ytmp;
        del_rki[2] = x[k][2] - ztmp;
        rik = sqrt(((del_rki[0]*del_rki[0])+(del_rki[1]*del_rki[1])+(del_rki[2]*del_rki[2])));
        
        del_rkj[0] = x[k][0] - x[j][0];
        del_rkj[1] = x[k][1] - x[j][1];
        del_rkj[2] = x[k][2] - x[j][2];
        rjk = sqrt(((del_rkj[0]*del_rkj[0])+(del_rkj[1]*del_rkj[1])+(del_rkj[2]*del_rkj[2])));
        
        ret_val = UFBS3b[itype][jtype][ktype].bsvalue(rij,rik,rjk);
        ret_val_deri = UFBS3b[itype][jtype][ktype].bsvalue_bsderivative(rij,rik,rjk);

        fij[0] = *(ret_val_deri+0)*(del_rji[0]/rij);
        fji[0] = -fij[0];
        fik[0] = *(ret_val_deri+1)*(del_rki[0]/rik);
        fki[0] = -fik[0];
        fjk[0] = *(ret_val_deri+2)*(del_rkj[0]/rjk);
        fkj[0] = -fjk[0];

        fij[1] = *(ret_val_deri+0)*(del_rji[1]/rij);
        fji[1] = -fij[1];
        fik[1] = *(ret_val_deri+1)*(del_rki[1]/rik);
        fki[1] = -fik[1];
        fjk[1] = *(ret_val_deri+2)*(del_rkj[1]/rjk);
        fkj[1] = -fjk[1];

        fij[2] = *(ret_val_deri+0)*(del_rji[2]/rij);
        fji[2] = -fij[2];
        fik[2] = *(ret_val_deri+1)*(del_rki[2]/rik);
        fki[2] = -fik[2];
        fjk[2] = *(ret_val_deri+2)*(del_rkj[2]/rjk);
        fkj[2] = -fjk[2];
        
        f[i][0] += fij[0] + fik[0];
        f[i][1] += fij[1] + fik[1];
        f[i][2] += fij[2] + fik[2];

        f[j][0] += fji[0] + fjk[0];
        Fj[0]    = fji[0] + fjk[0];
        f[j][1] += fji[1] + fjk[1];
        Fj[1]    = fji[1] + fjk[1];
        f[j][2] += fji[2] + fjk[2];
        Fj[2]    = fji[2] + fjk[2];

        f[k][0] += fki[0] + fkj[0];
        Fk[0]    = fki[0] + fkj[0];
        f[k][1] += fki[1] + fkj[1];
        Fk[1]    = fki[1] + fkj[1];
        f[k][2] += fki[2] + fkj[2];
        Fk[2]    = fki[2] + fkj[2];

        if (eflag) evdwl = ret_val; // 2/3 because of ev_tally_xyz

        if (evflag) {
          ev_tally3(i,j,k, evdwl, 0, Fj, Fk, del_rji, del_rki);
        }

      }
    } 
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

