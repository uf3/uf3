#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#ifndef BSPLINE_CONFIG_FF_H // Check if BSPLINE_CONFIG_H is not defined
#define BSPLINE_CONFIG_FF_H // Define BSPLINE_CONFIG_H

class bspline_config_ff {
  public:
    const int degree, nelements;
    const py::tuple interactions_map;
    const py::array_t<double, py::array::c_style> n2b_knots_map;
    const py::array_t<int, py::array::c_style> n2b_num_knots;
    const py::array_t<double, py::array::c_style> n3b_knots_map;
    const py::array_t<int, py::array::c_style> n3b_num_knots;
    const py::array_t<int, py::array::c_style> n3b_symm_array;
    const py::array_t<int, py::array::c_style> n3b_feature_sizes;
     
    bspline_config_ff();

    bspline_config_ff(int _degree, int _nelements, py::tuple _interactions_map,
                        py::array_t<double, py::array::c_style> _n2b_knots_map,
                        py::array_t<int, py::array::c_style> _n2b_num_knots,
                        py::array_t<double, py::array::c_style> _n3b_knots_map,
                        py::array_t<int, py::array::c_style> _n3b_num_knots,
                        py::array_t<int, py::array::c_style> _n3b_symm_array,
                        py::array_t<int, py::array::c_style> _n3b_feature_sizes);
    ~bspline_config_ff();

    int n2b_interactions=0;
    int n3b_interactions=0;

    std::vector<int> n2b_types, n3b_types;

    double *rmin_max_2b_sq, *rmin_max_3b;

    py::array get_rmin_max_2b_sq(), get_rmin_max_3b();


};

#endif // End for BSPLINE_CONFIG_H
