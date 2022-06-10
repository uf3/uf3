#include "uf3_read_pot_file.h"

#include "pair_uf3.h"

#include "error.h"
#include "comm.h"
#include "text_file_reader.h"

using namespace LAMMPS_NS;

//Dummy Constructor
uf3_read_pot_file::uf3_read_pot_file(){};

uf3_read_pot_file::~uf3_read_pot_file();
uf3_read_pot_file::uf3_read_pot_file(char *upotf_name)
{
  potf_name = upotf_name;
}

void uf3_read_pot_file::read_file()
{
  if(comm->me==0) utils::logmesg(lmp,"\nUF3: Opening {} file\n",potf_name);
  
  FILE * fp;
  fp = utils::open_potential(potf_name,lmp,nullptr);


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

    cutsq[temp_type1][temp_type2] = fp3rd_line.next_double();
    if(comm->me==0) utils::logmesg(lmp,"UF3: Cutoff {}\n",cutsq[temp_type1][temp_type2]);
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

    ValueTokenizer fp6th_line(temp_line);

    n2b_coeff[temp_type1][temp_type2].resize(temp_line_len);
    n2b_coeff[temp_type2][temp_type1].resize(temp_line_len);
    
    for(int k=0;k<temp_line_len;k++){
      n2b_coeff[temp_type1][temp_type2][k] = fp6th_line.next_double();
      n2b_coeff[temp_type2][temp_type1][k] = n2b_coeff[temp_type1][temp_type2][k];

    }

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
    if(comm->me==0) utils::logmesg(lmp,"UF3: {} file contains 2-body UF3 potential for {} {} {}\n",
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
    
    if(comm->me==0){
    utils::logmesg(lmp,"UF3: knot_matrix\n");
    for(int i=0;i<3;i++){
      for(int j=0;j<n3b_knot_matrix[temp_type1][temp_type2][temp_type3][i].size();j++){
        utils::logmesg(lmp,"{} ",n3b_knot_matrix[temp_type1][temp_type2][temp_type3][i][j]);
      }
      utils::logmesg(lmp,"\n");
    }}
 
    temp_line = txtfilereader.next_line(9);
    ValueTokenizer fp7th_line(temp_line);
    
    coeff_matrix_dim1 = fp7th_line.next_int();
    coeff_matrix_dim2 = fp7th_line.next_int();
    coeff_matrix_dim3 = fp7th_line.next_int();
    coeff_matrix_elements_len = coeff_matrix_dim3;
    
    utils::logmesg(lmp,"UF3: now reading coeff line\n");
    utils::logmesg(lmp,"UF3: length of coeff_line {}\n",coeff_matrix_elements_len);
    utils::logmesg(lmp,"UF3: {} {}\n",coeff_matrix_dim1,coeff_matrix_dim2);
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
    utils::logmesg(lmp,"UF3: Finished reading coeff matrix\n");
    

    if(comm->me==0){
    utils::logmesg(lmp,"UF3: coeff_matrix\n");
    for(int i=0;i<coeff_matrix_dim1;i++){
      for(int j=0;j<coeff_matrix_dim2;j++){
        for(int k=0;k<coeff_matrix_dim3;k++) utils::logmesg(lmp,"{} ",n3b_coeff_matrix[key][i][j][k]);
        utils::logmesg(lmp,"\n");
      }
      utils::logmesg(lmp,"--------------\n");
    }}
    
    //n3b_coeff_matrix[temp_type1][temp_type2][temp_type3].resize();
  }
  else error->all(FLERR, "UF3: {} file does not contain right words indicating whether it is 2 or 3 body potential",potf_name);
}
