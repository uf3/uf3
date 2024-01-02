#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "bspline_config_ff.h"
#include "UltraFastFeaturize.h"

namespace py = pybind11;

std::string tupletoString(const py::tuple &t) {
  std::string tuple_str = "(";
  for (int i=0; i < t.size(); i++) {
    if (i!=py::len(t)-1)
      tuple_str += std::string(py::str(t[i])) + ",";
    else
      tuple_str += std::string(py::str(t[i])) + ")";
  }
  return tuple_str;
}


PYBIND11_MODULE(ultra_fast_featurize, m) {

    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif

    py::class_<bspline_config_ff>(m, "bspline_config_ff")
        .def(py::init<int, int, py::tuple, 
                      py::array_t<double, py::array::c_style>,
                      py::array_t<int, py::array::c_style>>(),
             py::arg("degree"),
             py::arg("nelements"),
             py::arg("interactions_map"),
             py::arg("n2b_knots_map"),
             py::arg("n2b_num_knots"))
        .def_readonly("degree", &bspline_config_ff::degree)
        .def_readonly("nelements", &bspline_config_ff::nelements)
        .def_readonly("interactions_map", &bspline_config_ff::interactions_map)
        .def_readonly("n2b_interactions", &bspline_config_ff::n2b_interactions)
        .def_readonly("n2b_knots_map", &bspline_config_ff::n2b_knots_map)
        .def_readonly("n2b_num_knots", &bspline_config_ff::n2b_num_knots)
        .def_readonly("n2b_types", &bspline_config_ff::n2b_types)
        .def("get_rmin_max_2b",
             &bspline_config_ff::get_rmin_max_2b_sq,
             "get_rmin_max_2b")
        .def("__repr__",
            [](const bspline_config_ff &a){
                /*std::string interactions_map_str = "";
                for (int i; i <= a.interactions_map.size(); i++) {
                  interactions_map_str += std::to_string(a.interactions_map[1]);
                }*/
                return ("bspline_config_ff: degree           = " + 
                        std::to_string(a.degree) + "\n" +
                        "                   nelements        = " + 
                        std::to_string(a.nelements) + "\n" +
                        "                   interactions_map = " + 
                        tupletoString(a.interactions_map));
            }
        );
    py::class_<UltraFastFeaturize>(m, "UltraFastFeaturize")
        .def(py::init<int, int, py::tuple, 
                      py::array_t<double, py::array::c_style>,
                      py::array_t<int, py::array::c_style>>(),
                      //py::array_t<double, py::array::c_style>>(),
             py::arg("degree"),
             py::arg("nelements"),
             py::arg("interactions_map"),
             py::arg("n2b_knots_map"),
             py::arg("n2b_num_knots"))
        .def_readonly("BsplineConfig", &UltraFastFeaturize::BsplineConfig)
        .def_readonly("max_num_neigh", &UltraFastFeaturize::max_num_neigh)
        .def_readonly("num_batches", &UltraFastFeaturize::num_batches)
        .def_readonly("reprn_length", &UltraFastFeaturize::reprn_length)
        .def("get_elements", &UltraFastFeaturize::get_elements)
        .def("get_filename", &UltraFastFeaturize::get_filename)

        .def("set_geom_data",
             &UltraFastFeaturize::set_geom_data,
             "Sets the data for geometry and supercell",
             py::arg("atoms_array"),
             py::arg("cell_array"),
             py::arg("crystal_index"),
             py::arg("supercell_factors"),
             py::arg("geom_posn"))
        
        .def("featurize",
             &UltraFastFeaturize::featurize,
             "Featurize the set data",
             py::arg("batch_size"),
             py::arg("bool return_Neigh"),
             py::arg("filename"))
        .def_readonly("constants_2b",
                       &UltraFastFeaturize::constants_2b)
        .def_readonly("write_hdf5_counter",
                &UltraFastFeaturize::write_hdf5_counter);
        /*.def("set_geom_posn",
             &UltraFastFeaturize::set_geom_posn,
             "Sets the posn data for geometry and supercell",
             py::arg("geom_posn"),
             py::arg("supercell_posn"));
             py::arg("data"));*/

}

/*
std::vector<std::vector<float>> create_2d_array() {
    // Example 2D array
    std::vector<std::vector<float>> arr = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    return arr;
}

PYBIND11_MODULE(example, m) {
    m.def("get_2d_array", []() {
        auto arr = create_2d_array();
        size_t row = arr.size();
        size_t col = arr[0].size();

        // Ensure that all inner vectors are of the same size
        for (auto& inner : arr) {
            if (inner.size() != col) {
                throw std::runtime_error("Inner vectors must be of the same size");
            }
        }

        // Create a buffer view on the existing data
        return py::array(
            py::buffer_info(
                arr[0].data(),                           // Pointer to buffer 
                sizeof(float),                           // Size of one scalar
                py::format_descriptor<float>::format(),  // Python struct-style format descriptor 
                2,                                       // Number of dimensions 
                { row, col },                            // Buffer dimensions 
                { sizeof(float) * col, sizeof(float) }   // Strides (in bytes) for each index
            )
        );
    });
}*/
