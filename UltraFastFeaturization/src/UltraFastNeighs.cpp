/* ----------------------------------------------------------------------
 *    Contributing authors: Ajinkya Hire (U of Florida), 
 * ---------------------------------------------------------------------- */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "UltraFastNeighs.h"

namespace py = pybind11;

UltraFastNeighs::UltraFastNeighs(py::detail::unchecked_reference<double, 2>& _atoms_array_un,
                                 py::detail::unchecked_reference<int, 1>& _crystal_index_un,
                                 py::detail::unchecked_reference<double, 3>& _cell_array_un,
                                 py::detail::unchecked_reference<int, 1>& _geom_posn_un,
                                 py::detail::unchecked_reference<int, 2>& _supercell_factors_un,
                                 std::vector<int>& _num_of_interxns,
                                 std::vector<int>& _n2b_types,
                                 double* _rmin_max_2b_sq)
                : atoms_array_un(_atoms_array_un),
                  crystal_index_un(_crystal_index_un),
                  cell_array_un(_cell_array_un),
                  geom_posn_un(_geom_posn_un),
                  supercell_factors_un(_supercell_factors_un),
                  num_of_interxns(_num_of_interxns),
                  n2b_types(_n2b_types),
                  rmin_max_2b_sq(_rmin_max_2b_sq)
{

}

UltraFastNeighs::~UltraFastNeighs()
{}

void UltraFastNeighs::set_Neighs(int batch_start, int batch_end,
                            std::vector<double>& Neighs,
                            std::vector<double>& Neighs_del,
                            std::vector<int>& Tot_num_Neighs,
                            int rows, int cols)
{
  //Loop over central atoms in this batch
  for (int atom1=batch_start; atom1<batch_end; atom1++) {
    //get the atomic number; x,y,z cart-cordinates; crystal_index; lattice_vects
    //aka cells; geom_posn and supercell_factors
    int Z1 = atoms_array_un(atom1,0); 
    double x1 = atoms_array_un(atom1,1);
    double y1 = atoms_array_un(atom1,2);
    double z1 = atoms_array_un(atom1,3);
    int CI1 = crystal_index_un(atom1);
    double (*cell)[3] = new double[3][3] {
          {cell_array_un(CI1,0,0), cell_array_un(CI1,0,1), cell_array_un(CI1,0,2)},
          {cell_array_un(CI1,1,0), cell_array_un(CI1,1,1), cell_array_un(CI1,1,2)},
          {cell_array_un(CI1,2,0), cell_array_un(CI1,2,1), cell_array_un(CI1,2,2)}
    };

    double latx = sqrt((cell[0][0]*cell[0][0]) + (cell[1][0]*cell[1][0]) + 
              (cell[2][0]*cell[2][0])); //length of x lattice vector
    double laty = sqrt((cell[0][1]*cell[0][1]) + (cell[1][1]*cell[1][1]) + 
              (cell[2][1]*cell[2][1])); //length of y lattice vector
    double latz = sqrt((cell[0][2]*cell[0][2]) + (cell[1][2]*cell[1][2]) + 
              (cell[2][2]*cell[2][2])); //length of z lattice vector

    //Start and end index of atoms contained in the crystal to which this atom 
    //belongs to. This index is the index in atoms_array
    int start_posn = geom_posn_un(CI1);
    int end_posn = geom_posn_un(CI1+1);
    int num_atoms = end_posn-start_posn;
    
    int scf1 = supercell_factors_un(CI1, 0); //supercell size along x
    int scf2 = supercell_factors_un(CI1, 1); //y
    int scf3 = supercell_factors_un(CI1, 2); //z

    int n2b_interactions = num_of_interxns[0]; //total 2-body interactions
    std::vector<int> posn_to_neigh(n2b_interactions,0); //holds index of Neigh 
                                                          //for each interaction
                                                          //at which the neigh is inserted
                                                          //Initialize to 0
    std::vector<int> posn_to_neigh_del(n2b_interactions,0);

    int d = atom1-batch_start; //d=0 is the first atom in this batch

    //Loop over all atoms of the crystal to which this atom belongs to
    for (int atom2=start_posn; atom2<end_posn; atom2++) {
      //if (atom2 == atom1) continue;
      int Z2 = atoms_array_un(atom2,0); 
      double x2 = atoms_array_un(atom2,1);
      double y2 = atoms_array_un(atom2,2);
      double z2 = atoms_array_un(atom2,3);
      if (crystal_index_un(atom2)!=CI1)
        throw std::domain_error("atom1 and atom2 belong to different crystals");
      
      //Determine the interaction type
      int n2b_type = -1;
      for (int i=0; i<n2b_interactions; i++) {
        if ((Z1==n2b_types[2*i]) && (Z2==n2b_types[2*i+1])) n2b_type = i;
        if ((Z2==n2b_types[2*i]) && (Z1==n2b_types[2*i+1])) n2b_type = i;      
      }
      double rmin_sq = rmin_max_2b_sq[2*n2b_type];
      double rmax_sq = rmin_max_2b_sq[2*n2b_type+1];
      //Loop over all periodic images of atom2

      //int d = atom1-batch_start; //d=0 is the first atom in this batch
      int r = n2b_type;
      int c = 0;// = posn_to_neigh[n2b_type];

      // atom in (0,0,0) supercell aka the unit cell

      for (int ix=-1*scf1; ix<=scf1; ix++) {
        double offset_x[3];
        offset_x[0] = ix*cell[0][0];
        offset_x[1] = ix*cell[0][1];
        offset_x[2] = ix*cell[0][2];

        for (int iy=-1*scf2; iy<=scf2; iy++) {
          double offset_y[3];
          offset_y[0] = iy*cell[1][0];
          offset_y[1] = iy*cell[1][1];
          offset_y[2] = iy*cell[1][2];

          for (int iz=-1*scf3; iz<=scf3; iz++) {
            double offset_z[3];
            offset_z[0] = iz*cell[2][0];
            offset_z[1] = iz*cell[2][1];
            offset_z[2] = iz*cell[2][2];
              
            //x2_pi, y2_pi, z2_pi -> cordinate of periodic image of atom2  
            double x2_pi = offset_x[0] + offset_y[0] + offset_z[0] + x2;
            //double x2_pi_neg = (-2*ix*latx) + x2_pi;
            
            double y2_pi = offset_x[1] + offset_y[1] + offset_z[1] + y2;
            //double y2_pi_neg = (-2*iy*laty) + y2_pi;
              
            double z2_pi = offset_x[2] + offset_y[2] + offset_z[2] + z2;
            //double z2_pi_neg = (-2*iz*latz) + z2_pi;
              
            //+x,+y,+z
            double rsq = pow((x2_pi-x1),2) + pow((y2_pi-y1),2) + pow((z2_pi-z1),2);
            if ((rmin_sq <= rsq) && (rsq < rmax_sq)){
              double rij = sqrt(rsq);
              int temp_index = d*(rows*cols)+(r*cols)+posn_to_neigh[n2b_type];
              Neighs[temp_index] = rij;
                
              int temp_index2 = d*(rows*cols*3)+(r*cols*3)+posn_to_neigh_del[n2b_type];
              Neighs_del[temp_index2] = (x2_pi-x1)/rij;
              Neighs_del[temp_index2+1] = (y2_pi-y1)/rij;
              Neighs_del[temp_index2+2] = (z2_pi-z1)/rij;

              posn_to_neigh[n2b_type]++;
              posn_to_neigh_del[n2b_type] = posn_to_neigh_del[n2b_type]+3;
            }
          }
        }
      }
    } //End of atom2 loop
    for (int i=0; i<n2b_interactions; i++) {
      Tot_num_Neighs[d*rows+i] = posn_to_neigh[i];
    }
    delete[] cell;
  } //End of atom1 loop
  ////----------------ALL Neighs have been found---////
}
