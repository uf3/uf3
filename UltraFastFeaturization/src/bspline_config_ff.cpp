/* ----------------------------------------------------------------------
 *    Contributing authors: Ajinkya Hire (U of Florida), 
 * ---------------------------------------------------------------------- */

#include <stdexcept>

#include "bspline_config_ff.h"

bspline_config_ff::bspline_config_ff()
        : degree(0), nelements(0),
          interactions_map(py::tuple()),
          n2b_knots_map(py::array_t<double, py::array::c_style>()),
          n2b_num_knots(py::array_t<int, py::array::c_style>())
{}

bspline_config_ff::bspline_config_ff(int _degree,
        int _nelements,
        py::tuple _interactions_map,
        py::array_t<double, py::array::c_style> _n2b_knots_map,
        py::array_t<int, py::array::c_style> _n2b_num_knots)
        : degree(_degree), nelements(_nelements), 
          interactions_map(_interactions_map),
          n2b_knots_map(_n2b_knots_map),
          n2b_num_knots(_n2b_num_knots)
{
  //Curent implementation only for 2-body
  if (degree!=2) {
    throw std::domain_error("Current implementation only works for 2-body UF3 potential");
    }

  /*if (interactions_map.size() <2) {
    std::string error_message = "interactions_map tuple is not of the \n\
              right length";
    throw std::length_error(error_message);
    }*/

  //Compute 2body interactions
  n2b_interactions = nelements*(nelements + 1)/2;

  //Check to size of interactions_map
  //Should be equal to 2body interactions + 1body interactions
  if (degree==2)
    if (interactions_map.size() != n2b_interactions+nelements) {
      std::string error_message = "interactions_map tuple is not of the \n\
                  right length. Expected length " + 
                  std::to_string(n2b_interactions+nelements);
      throw std::length_error(error_message);
    }

  //Check size of n2b_knots_map
  if (n2b_knots_map.shape(0)!=n2b_interactions) {
    std::string error_message = "Incorrect size of dimension 0 of n2b_knots_map\n\
              Expected size is " + std::to_string(n2b_interactions);

    throw std::length_error(error_message);
  }

  //len(n2b_knots_map) == len(n2b_num_knots)
  if (n2b_knots_map.shape(0)!=n2b_num_knots.shape(0))
    throw std::length_error("n2b_knots_map and n2b_num_knots is inconsistent");

  n2b_types = std::vector<int>(n2b_interactions*2);
  for (int i=0; i<n2b_interactions; i++) {
    py::tuple temp = interactions_map[nelements+i].cast<py::tuple>();
    n2b_types[i*2] = temp[0].cast<int>();
    n2b_types[(i*2)+1] = temp[1].cast<int>();
  }
 
  auto n2b_knots_map_un = n2b_knots_map.unchecked<2>();
  auto n2b_num_knots_un = n2b_num_knots.unchecked<1>();
  rmin_max_2b_sq = new double[n2b_interactions*2];
  for (int i=0; i<n2b_interactions; i++) {
    rmin_max_2b_sq[2*i] = pow(n2b_knots_map_un(i,0),2);
    rmin_max_2b_sq[2*i+1] = pow(n2b_knots_map_un(i,n2b_num_knots_un(i)-1),2);
    //py::print(2*i, n2b_knots_map_un(i,0), 2*i+1, n2b_knots_map_un(i,n2b_num_knots_un(i)-1));
    //py::print(2*i, rmin_max_2b[2*i], 2*i+1, rmin_max_2b[2*i+1]);
  }

}


bspline_config_ff::~bspline_config_ff(){
  /*for (int i=0; i<n2b_interactions; i++) {
    delete[] rmin_max_2b[i];
  }*/
  delete[] rmin_max_2b_sq; 

}


py::array bspline_config_ff::get_rmin_max_2b_sq() {
  py::buffer_info rmin_max_2b_buff_info(
    rmin_max_2b_sq,      /* Pointer to buffer */
    sizeof(double),     /* Size of one scalar */
    py::format_descriptor<double>::format(),    /* Python struct-style format descriptor */
    2,                  /* Number of dimensions */
    { n2b_interactions, 2},  /* Buffer dimensions */
    { sizeof(double) * 2,        /* Strides (in bytes) for each index */
    sizeof(double) } 
  );

  return py::array(rmin_max_2b_buff_info);
 };
