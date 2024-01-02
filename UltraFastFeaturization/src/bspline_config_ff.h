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
     
    bspline_config_ff();

    bspline_config_ff(int _degree, int _nelements, py::tuple _interactions_map,
                        py::array_t<double, py::array::c_style> _n2b_knots_map,
                        py::array_t<int, py::array::c_style> _n2b_num_knots);
    ~bspline_config_ff();

    int n2b_interactions;

    std::vector<int> n2b_types;

    double *rmin_max_2b_sq;

    py::array get_rmin_max_2b_sq();

  /*private:
    int n2b_interaction;*/

};

#endif // End for BSPLINE_CONFIG_H
