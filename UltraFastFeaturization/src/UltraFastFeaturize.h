#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <H5Cpp.h>
//#include <Eigen/Dense>

#include "bspline_config_ff.h"

namespace py = pybind11;

#ifndef UltraFastFeaturize_H // Check if UltraFastFeaturize_H is not defined
#define UltraFastFeaturize_H // Define UltraFastFeaturize_H

class UltraFastFeaturize{
  private:
    //These members are only used to initialize bspline_config_ff
    //These are decleared first as BsplineConfig depends on them, as
    //Members are initialized in the order they are declared in the class, 
    //not in the order listed in the initializer list.
    const int degree, nelements;
    const py::tuple interactions_map;
    const py::array_t<double, py::array::c_style> n2b_knots_map;
    const py::array_t<int, py::array::c_style> n2b_num_knots;
    const py::array_t<double, py::array::c_style> n3b_knots_map;
    const py::array_t<int, py::array::c_style> n3b_num_knots;
    const py::array_t<int, py::array::c_style> n3b_symm_array;
    const py::array_t<int, py::array::c_style> n3b_feature_sizes;
    const int leading_trim;
    const int trailing_trim;

    py::array_t<double, py::array::c_style> atoms_array, supercell_array, cell_array;
    py::array_t<double, py::array::c_style> energy_array, forces_array;
    py::array_t<int, py::array::c_style> crystal_index, geom_posn, supercell_factors;
    std::vector<std::string> structure_names, column_names;

    double *rmin_max_2b_sq, *rmin_max_3b;
    double rcut_max_sq;

    int atom_count=0;
    int prev_CI = 0;
    int tot_crystals, tot_atoms, prev_data_len;
    bool incomplete = false;

    struct N3bInterxnData{
      int bl;
      int bm;
      int bn;
      const std::vector<double>& knots_ij;
      const std::vector<double>& knots_ik;
      const std::vector<double>& knots_jk;
      const std::vector<std::vector<double>>& constants_ij;
      const std::vector<std::vector<double>>& constants_ik;
      const std::vector<std::vector<double>>& constants_jk;
      const std::vector<std::vector<double>>& constants_ij_deri;
      const std::vector<std::vector<double>>& constants_ik_deri;
      const std::vector<std::vector<double>>& constants_jk_deri;
      std::vector<double>& atomic_3b_Reprn_matrix_fx;
      std::vector<double>& atomic_3b_Reprn_matrix_fy;
      std::vector<double>& atomic_3b_Reprn_matrix_fz;

      N3bInterxnData(int _bl,
                     int _bm,
                     int _bn,
                     const std::vector<double>& _knots_ij,
                     const std::vector<double>& _knots_ik,
                     const std::vector<double>& _knots_jk,
                     const std::vector<std::vector<double>>& _constants_ij,
                     const std::vector<std::vector<double>>& _constants_ik,
                     const std::vector<std::vector<double>>& _constants_jk,
                     const std::vector<std::vector<double>>& _constants_ij_deri,
                     const std::vector<std::vector<double>>& _constants_ik_deri,
                     const std::vector<std::vector<double>>& _constants_jk_deri,
                     std::vector<double>& _atomic_3b_Reprn_matrix_fx,
                     std::vector<double>& _atomic_3b_Reprn_matrix_fy,
                     std::vector<double>& _atomic_3b_Reprn_matrix_fz) :
          bl(_bl), bm(_bm), bn(_bn),
          knots_ij(_knots_ij), knots_ik(_knots_ik), knots_jk(_knots_jk),
          constants_ij(_constants_ij), constants_ik(_constants_ik), constants_jk(_constants_jk),
          constants_ij_deri(_constants_ij_deri),
          constants_ik_deri(_constants_ik_deri),
          constants_jk_deri(_constants_jk_deri),
          atomic_3b_Reprn_matrix_fx(_atomic_3b_Reprn_matrix_fx),
          atomic_3b_Reprn_matrix_fy(_atomic_3b_Reprn_matrix_fy),
          atomic_3b_Reprn_matrix_fz(_atomic_3b_Reprn_matrix_fz){}
    };

  public:
    bspline_config_ff BsplineConfig;

    UltraFastFeaturize(int _degree, int _nelements, py::tuple _interactions_map,
                        py::array_t<double, py::array::c_style> _n2b_knots_map,
                        py::array_t<int, py::array::c_style> _n2b_num_knots,
                        py::array_t<double, py::array::c_style> _n3b_knots_map,
                        py::array_t<int, py::array::c_style> _n3b_num_knots,
                        py::array_t<int, py::array::c_style> _n3b_symm_array,
                        py::array_t<int, py::array::c_style> _n3b_feature_sizes,
                        const int _leading_trim,
                        const int _trailing_trim);

    std::vector<int> num_of_interxns, n2b_types, n3b_types, n2b_num_knots_array, elements;
    std::vector<std::vector<double>> n2b_knots_array;
    std::vector<std::vector<int>> n3b_num_knots_array;
    std::vector<std::vector<std::vector<double>>> n3b_knots_array;


    ~UltraFastFeaturize();

    int batch_size = 0;
    int num_batches = 0;
    int max_num_neigh = 0;
    double neigh_in_sphere = 0;
    std::vector<double> Neighs, Neighs_del;
    std::vector<int> Tot_num_Neighs;
    
    void set_geom_data(py::array_t<double, py::array::c_style> _atoms_array,
                       py::array_t<double, py::array::c_style> _energy_array,
                       py::array_t<double, py::array::c_style> _forces_array,
                       py::array_t<double, py::array::c_style> _cell_array,
                       py::array_t<int, py::array::c_style> _crystal_index,
                       py::array_t<int, py::array::c_style> _supercell_factors,
                       py::array_t<int, py::array::c_style> _geom_posn,
                       py::list _structure_names, py::list _column_names);

    py::array featurize(int _batch_size, bool return_Neigh, 
                        std::string& _filename, bool featurize_3b);

    std::vector<std::vector<std::vector<double>>> constants_2b;//, constants_2b_deri;
    std::vector<std::vector<std::vector<double>>> constants_2b_deri1, constants_2b_deri2;
    std::vector<std::vector<std::vector<std::vector<double>>>> constants_3b, constants_3b_deri;
    
    int reprn_length=0, tot_complete_crystals;
    int tot_2b_features_size = 0;
    std::vector<double> atomic_Reprn, crystal_Reprn, incomplete_crystal_Reprn;
    // Representation = [[y, n_El1, n_El2, ..., El1El1_0, El1El1_1, ...],...]

    py::array get_elements();

    std::string filename;
    std::string get_filename();
    int write_hdf5_counter=0;
    void write_hdf5(const hsize_t num_rows, const hsize_t num_cols,
                    const int batch_num, const H5::H5File &file_fp,
                    const std::vector<double> &Data, int first_crystal_CI,
                    int tot_complete_crystals, std::vector<std::string> &column_names,
                    py::array_t<int, py::array::c_style> &geom_posn);

    /*void n3b_compress(std::vector<Eigen::MatrixXd> &atomic_3b_Reprn_matrix,
                      std::vector<double> &atomic_3b_Reprn_flatten,
                      int n3b_symm);*/
    
    void n3b_compress(std::vector<double> &atomic_3b_Reprn_matrix,
                      std::vector<double> &atomic_3b_Reprn_flatten,
                      int n3b_symm, int template_mask_start, int template_mask_end,
                      int bl, int bm, int bn);

    py::array get_symmetry_weights(int interxn, int lead, int trail);

    std::vector<double> template_array_flatten_test, tempftest;
    std::vector<double> flat_weights;
    std::vector<int> template_mask;

    py::array get_flat_weights();
    py::array get_template_mask();
    
    void compute_3b_energy_feature(const double r_ij, const double r_ik, const double r_jk,
                       const double rsq_ij, const double rsq_ik, const double rsq_jk,
                       const double rth_ij, const double rth_ik, const double rth_jk,
                       int knot_posn_ij, int knot_posn_ik, int knot_posn_jk,
                       const int bl, const int bm, const int bn,
                       const std::vector<double>& knots_ij,
                       const std::vector<double>& knots_ik,
                       const std::vector<double>& knots_jk,
                       const std::vector<std::vector<double>>& constants_ij,
                       const std::vector<std::vector<double>>& constants_ik,
                       const std::vector<std::vector<double>>& constants_jk,
                       std::vector<double>& atomic_3b_Reprn_matrix_energy);
    
    std::array<double, 4>  get_basis_set(const double r_ij,
                                                         const double rsq_ij,
                                                         const double rth_ij,
                                                         const std::vector<std::vector<double>>& constants_ij,
                                                         const int knot_posn_ij);
    
    std::array<double, 3>  get_basis_deri_set(const double r_ij,
                                              const double rsq_ij,
                                              const std::vector<std::vector<double>>& constants_ij_deri,
                                              const int knot_posn_ij);
    std::array<double, 12> get_rij_rik_rjk(double r1, double delx1, double dely1, double delz1,
                                          double r2, double delx2, double dely2, double delz2,
                                          int Z1_index_in_3b_interxn);

    void calculate_force_features_for_ij_swap(int atom_of_focus,
            std::array<double, 12>& Rs,
            int Interxn,
            N3bInterxnData& n3b_interxn_data);

    void calculate_force_features_for_ik_swap(int atom_of_focus,
            std::array<double, 12>& Rs,
            int Interxn,
            N3bInterxnData& n3b_interxn_data);
    
    void calculate_force_features_for_jk_swap(int atom_of_focus,
            std::array<double, 12>& Rs,
            int Interxn,
            N3bInterxnData& n3b_interxn_data);
    
    void calculate_force_features_for_ij_swap(int atom_of_focus,
        std::array<double, 12>& Rs,
        int Interxn,
        int bl, int bm, int bn,
        const std::vector<double>& knots_ij,
        const std::vector<double>& knots_ik,
        const std::vector<double>& knots_jk,
        const std::vector<std::vector<double>>& constants_ij,
        const std::vector<std::vector<double>>& constants_ik,
        const std::vector<std::vector<double>>& constants_jk,
        const std::vector<std::vector<double>>& constants_ij_deri,
        const std::vector<std::vector<double>>& constants_ik_deri,
        const std::vector<std::vector<double>>& constants_jk_deri,
        std::vector<double>& atomic_3b_Reprn_matrix_fx,
        std::vector<double>& atomic_3b_Reprn_matrix_fy,
        std::vector<double>& atomic_3b_Reprn_matrix_fz);
    /*std::vector<double> evaluate_basis3(const double r,
                                        const std::vector<double> &knots,
                                        const int num_knots,
                                        const std::vector<std::vector<double>> &constants);*/

};

#endif // End for UltraFastFeaturize_H
