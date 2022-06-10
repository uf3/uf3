/* ----------------------------------------------------------------------
 *    LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 *       https://www.lammps.org/, Sandia National Laboratories
 *          Steve Plimpton, sjplimp@sandia.gov
 *
 *             Copyright (2003) Sandia Corporation.  Under the terms of Contract
 *                DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 *                   certain rights in this software.  This software is distributed under
 *                      the GNU General Public License.
 *
 *                         See the README file in the top-level LAMMPS directory.
 *                         ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 *    Contributing author: Ajinkya Hire, Richard Hennig (U of Florida)
 *    ------------------------------------------------------------------------- */

#include "pair_uf3.h"
#include "uf3_pair_bspline.h"

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
    memory->destroy(cutsq_3b);
    }
  }
}

/* ----------------------------------------------------------------------
 *     global settings
 *     ------------------------------------------------------------------------- */

void PairUF3::settings(int narg, char **arg)
{
  if (narg!=1) error->all(FLERR, "UF3: UF3 invalid number of argument in pair settings; Are you running 2-bordy or 2 & 3-body UF potential");
  nbody_flag = utils::numeric(FLERR,arg[0],true,lmp);
  num_of_elements = atom->ntypes;
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
  UFBS2b.resize(num_of_elements+1);
  for (int i =1;i<num_of_elements+1;i++){
    UFBS2b[i].resize(num_of_elements+1);
    for (int j=i;j<num_of_elements+1;j++){
      UFBS2b[i][j] = uf3_pair_bspline(lmp,3,n2b_knot[i][j], n2b_coeff[i][j]);
      UFBS2b[j][i] = uf3_pair_bspline(lmp,3,n2b_knot[j][i], n2b_coeff[j][i]);
    }
  }
  if(pot_3b) error->all(FLERR, "UF3: Implementation of 3-body not yet completed");
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
  n3b_knot_matrix.resize(num_of_elements+1);
  //n3b_coeff_matrix.resize(num_of_elements+1);
  for(int i =1; i<num_of_elements+1; i++){
   n2b_knot[i].resize(num_of_elements+1);
   n2b_coeff[i].resize(num_of_elements+1);
   n3b_knot_matrix[i].resize(num_of_elements+1);
   //n3b_coeff_matrix[i].resize(num_of_elements+1);
   for(int j =1; j<num_of_elements+1; j++){
     n3b_knot_matrix[i][j].resize(num_of_elements+1);
     //n3b_coeff_matrix[i][j].resize(num_of_elements+1);
   }
  }
  if(pot_3b)
  {
    //Contains info about wether UF potential were found for type i, j and k
    memory->create(setflag_3b,num_of_elements+1,num_of_elements+1,num_of_elements+1,"pair:setflag_3b");
    //Contains info about 3-body cutoff distance for type i, j and k
    memory->create(cutsq_3b,num_of_elements+1,num_of_elements+1,num_of_elements+1,3,"pair:setflag_3b");
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
    cutsq_3b[temp_type1][temp_type2][temp_type3][0] = fp3rd_line.next_int();
    cutsq_3b[temp_type1][temp_type2][temp_type3][1] = fp3rd_line.next_int();
    cutsq_3b[temp_type1][temp_type2][temp_type3][2] = fp3rd_line.next_int();
    
    cutsq_3b[temp_type1][temp_type3][temp_type2][0] = cutsq_3b[temp_type1][temp_type2][temp_type3][0];
    cutsq_3b[temp_type1][temp_type3][temp_type2][1] = cutsq_3b[temp_type1][temp_type2][temp_type3][1];
    cutsq_3b[temp_type1][temp_type3][temp_type2][2] = cutsq_3b[temp_type1][temp_type2][temp_type3][2];

    temp_line_len = fp3rd_line.next_int();
    temp_line = txtfilereader.next_line(temp_line_len);
    ValueTokenizer fp4th_line(temp_line);
    
    n3b_knot_matrix[temp_type1][temp_type2][temp_type3].resize(3);
    n3b_knot_matrix[temp_type1][temp_type3][temp_type2].resize(3);

    n3b_knot_matrix[temp_type1][temp_type2][temp_type3][0].resize(temp_line_len);
    for(int i=0;i<temp_line_len;i++){
      n3b_knot_matrix[temp_type1][temp_type2][temp_type3][0][i] = fp4th_line.next_double();
      n3b_knot_matrix[temp_type1][temp_type3][temp_type2][0][i] = n3b_knot_matrix[temp_type1][temp_type2][temp_type3][0][i];
    }
    
    temp_line_len = fp3rd_line.next_int();
    temp_line = txtfilereader.next_line(temp_line_len);
    ValueTokenizer fp5th_line(temp_line);
    n3b_knot_matrix[temp_type1][temp_type2][temp_type3][1].resize(temp_line_len);
    for(int i=0;i<temp_line_len;i++){
      n3b_knot_matrix[temp_type1][temp_type2][temp_type3][1][i] = fp5th_line.next_double();
      n3b_knot_matrix[temp_type1][temp_type3][temp_type2][1][i] = n3b_knot_matrix[temp_type1][temp_type2][temp_type3][1][i];
    }
    

    temp_line_len = fp3rd_line.next_int();
    temp_line = txtfilereader.next_line(temp_line_len);
    ValueTokenizer fp6th_line(temp_line);
    n3b_knot_matrix[temp_type1][temp_type2][temp_type3][2].resize(temp_line_len);
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
  neighbor->add_request(this, NeighConst::REQ_DEFAULT);
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
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair, fx, fy, fz;
  double rsq, rij;
  int *ilist, *jlist, *numneigh, **firstneigh;
  
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
    for (jj = 0; jj < jnum; jj++) {
      fx = 0;
      fy = 0;
      fz = 0;
      j = jlist[jj];
      j &= NEIGHMASK;
      
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      //utils::logmesg(lmp,"rsq = {}, cutsq = {}",rsq,cutsq[itype][jtype]);
      if (rsq < cutsq[itype][jtype]) {
        rij = sqrt(rsq);
        fpair = -2*UFBS2b[itype][jtype].bsderivative(rij)/rij;
        
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
        if (eflag) evdwl = 2*UFBS2b[itype][jtype].bsvalue(rij);

        if (evflag) ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0.0, fx, fy, fz, delx, dely, delz);
      }
    }
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

double PairUF3::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                           double /*factor_coul*/, double /*factor_lj*/,
                           double &fforce)
{
  double rij, philj;
  rij = sqrt(rsq);
  fforce = -2*UFBS2b[itype][jtype].bsderivative(rij)/rij;
  philj = 2*UFBS2b[itype][jtype].bsvalue(rij);
  return philj;
}

