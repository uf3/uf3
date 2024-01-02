#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <H5Cpp.h>

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

    py::array_t<double, py::array::c_style> atoms_array, supercell_array, cell_array;
    py::array_t<int, py::array::c_style> crystal_index, geom_posn, supercell_factors;

    double *rmin_max_2b_sq;

  public:
    bspline_config_ff BsplineConfig;

    UltraFastFeaturize(int _degree, int _nelements, py::tuple _interactions_map,
                        py::array_t<double, py::array::c_style> _n2b_knots_map,
                        py::array_t<int, py::array::c_style> _n2b_num_knots);
                        //py::array_t<double, py::array::c_style> _data);

    std::vector<int> num_of_interxns, n2b_types, n2b_num_knots_array, elements;
    std::vector<std::vector<double>> n2b_knots_array;

    ~UltraFastFeaturize();

    int batch_size = 0;
    int num_batches = 0;
    int max_num_neigh = 0;
    std::vector<double> Neighs;
    std::vector<double> Tot_num_Neighs;
    
    void set_geom_data(py::array_t<double, py::array::c_style> _atoms_array,
                       py::array_t<double, py::array::c_style> _cell_array,
                       py::array_t<int, py::array::c_style> _crystal_index,
                       py::array_t<int, py::array::c_style> _supercell_factors,
                       py::array_t<int, py::array::c_style> _geom_posn);

    py::array featurize(int _batch_size, bool return_Neigh, 
                        std::string& _filename);

    std::vector<std::vector<std::vector<double>>> constants_2b;
    
    int reprn_length=0;
    std::vector<double> atomic_Reprn, crystal_Reprn, incomplete_crystal_Reprn;
    // Representation = [[y, n_El1, n_El2, ..., El1El1_0, El1El1_1, ...],...]

    py::array get_elements();

    std::string filename;
    std::string get_filename();
    int write_hdf5_counter=0;
    void write_hdf5(const hsize_t num_rows, const hsize_t num_cols,
                    const int batch_num, const H5::H5File &file_fp,
                    const std::vector<double> &Data);
    
    /*std::vector<double> evaluate_basis3(const double r,
                                        const std::vector<double> &knots,
                                        const int num_knots,
                                        const std::vector<std::vector<double>> &constants);*/

};

#endif // End for UltraFastFeaturize_H
