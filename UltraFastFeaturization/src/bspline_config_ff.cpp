/* ----------------------------------------------------------------------
 *    Contributing authors: Ajinkya Hire (U of Florida), 
 * ---------------------------------------------------------------------- */

#include <stdexcept>

#include "bspline_config_ff.h"

bspline_config_ff::bspline_config_ff()
        : degree(0), nelements(0),
          interactions_map(py::tuple()),
          n2b_knots_map(py::array_t<double, py::array::c_style>()),
          n2b_num_knots(py::array_t<int, py::array::c_style>()),
          n3b_knots_map(py::array_t<double, py::array::c_style>()),
          n3b_num_knots(py::array_t<int, py::array::c_style>()),
          n3b_symm_array(py::array_t<int, py::array::c_style>()),
          n3b_feature_sizes(py::array_t<int, py::array::c_style>())
{}

/*bspline_config_ff::bspline_config_ff(int _degree,
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
  if (degree!=2) {n3b_feature_sizes
    throw std::domain_error("Wrong function call");
    }

  //Compute 2body interactions
  n2b_interactions = nelements*(nelements + 1)/2;

  //Check the size of interactions_map
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
  }

}*/


bspline_config_ff::bspline_config_ff(int _degree,
        int _nelements,
        py::tuple _interactions_map,
        py::array_t<double, py::array::c_style> _n2b_knots_map,
        py::array_t<int, py::array::c_style> _n2b_num_knots,
        py::array_t<double, py::array::c_style> _n3b_knots_map,
        py::array_t<int, py::array::c_style> _n3b_num_knots,
        py::array_t<int, py::array::c_style> _n3b_symm_array,
        py::array_t<int, py::array::c_style> _n3b_feature_sizes)
        : degree(_degree), nelements(_nelements), 
          interactions_map(_interactions_map),
          n2b_knots_map(_n2b_knots_map),
          n2b_num_knots(_n2b_num_knots),
          n3b_knots_map(_n3b_knots_map),
          n3b_num_knots(_n3b_num_knots),
          n3b_symm_array(_n3b_symm_array),
          n3b_feature_sizes(_n3b_feature_sizes)
{
  //Compute 2body interactions
  n2b_interactions = nelements*(nelements + 1)/2;
  if (degree==3)
    n3b_interactions = nelements*nelements*(nelements + 1)/2;

  //Check the size of interactions_map
  if (degree==2){
    if (interactions_map.size() != n2b_interactions+nelements) {
      std::string error_message = "interactions_map tuple is not of the \n\
                  right length. Expected length " + 
                  std::to_string(n2b_interactions+nelements);
      throw std::length_error(error_message);
    }
  }
  if (degree ==3){
    //Should be equal to 3body interactions + 2body interactions + 1body interactions
    if (interactions_map.size() != n3b_interactions+n2b_interactions+nelements) {
      std::string error_message = "interactions_map tuple is not of the \n\
                                   right length. Expected length " + 
                                   std::to_string(n3b_interactions + 
                                         n2b_interactions+nelements);
      throw std::length_error(error_message);
    }
  }

  //Check size of n2b_knots_map
  if (n2b_knots_map.shape(0)!=n2b_interactions) {
    std::string error_message = "Incorrect size of dimension 0 of n2b_knots_map\n\
              Expected size is " + std::to_string(n2b_interactions);

    throw std::length_error(error_message);
  }
  
  //Check size of n3b_knots_map
  if (degree ==3)
    if (n3b_knots_map.shape(0)!=n3b_interactions) {
        std::string error_message = "Incorrect size of dimension 0 of n3b_knots_map\n\
                                     Expected size is " + std::to_string(n3b_interactions);

      throw std::length_error(error_message);
    }

  if (n2b_knots_map.shape(0)!=n2b_num_knots.shape(0))
    throw std::length_error("n2b_knots_map and n2b_num_knots is inconsistent");

  if (degree ==3)
    if (n3b_knots_map.shape(0)!=n3b_num_knots.shape(0))
      throw std::length_error("n3b_knots_map and n3b_num_knots is inconsistent");

  
  n2b_types = std::vector<int>(n2b_interactions*2);
  for (int i=0; i<n2b_interactions; i++) {
    py::tuple temp = interactions_map[nelements+i].cast<py::tuple>();
    n2b_types[i*2] = temp[0].cast<int>();
    n2b_types[(i*2)+1] = temp[1].cast<int>();
  }
  
  if (degree ==3) {
    n3b_types = std::vector<int>(n3b_interactions*3);
    for (int i=0; i<n3b_interactions; i++) {
      py::tuple temp = interactions_map[nelements+n2b_interactions+i].cast<py::tuple>();
      n3b_types[i*3] = temp[0].cast<int>();
      n3b_types[(i*3)+1] = temp[1].cast<int>();
      n3b_types[(i*3)+2] = temp[2].cast<int>();
    }
  }
 
  auto n2b_knots_map_un = n2b_knots_map.unchecked<2>();
  auto n2b_num_knots_un = n2b_num_knots.unchecked<1>();
  rmin_max_2b_sq = new double[n2b_interactions*2];
  for (int i=0; i<n2b_interactions; i++) {
    rmin_max_2b_sq[2*i] = pow(n2b_knots_map_un(i,0),2);
    rmin_max_2b_sq[2*i+1] = pow(n2b_knots_map_un(i,n2b_num_knots_un(i)-1),2);
  }

  if (degree ==3) {
    auto n3b_knots_map_un = n3b_knots_map.unchecked<3>();
    auto n3b_num_knots_un = n3b_num_knots.unchecked<2>();
    rmin_max_3b = new double[n3b_interactions*2*3];
    for (int i=0; i<n3b_interactions; i++) {
      rmin_max_3b[6*i] = n3b_knots_map_un(i,0,0);
      rmin_max_3b[6*i+1] = n3b_knots_map_un(i,1,0);
      rmin_max_3b[6*i+2] = n3b_knots_map_un(i,2,0);

      rmin_max_3b[6*i+3] = n3b_knots_map_un(i,0,n3b_num_knots_un(i,0)-1);
      rmin_max_3b[6*i+4] = n3b_knots_map_un(i,1,n3b_num_knots_un(i,1)-1);
      rmin_max_3b[6*i+5] = n3b_knots_map_un(i,2,n3b_num_knots_un(i,2)-1);
    }

    if (n3b_symm_array.shape(0) != n3b_interactions){
      std::string error_message = "Incorrect size of n3b_symm_arry. Should be\n\
                                   equal to "+ std::to_string(n3b_interactions);

      throw std::length_error(error_message);
    }
    if (n3b_feature_sizes.shape(0) != n3b_interactions){
      std::string error_message = "Incorrect size of n3b_feature_sizes. Should be\n\
                                   equal to "+ std::to_string(n3b_interactions);

      throw std::length_error(error_message);
    }

  }
  else
    rmin_max_3b = new double[1];
}

bspline_config_ff::~bspline_config_ff(){
  delete[] rmin_max_2b_sq;
  delete[] rmin_max_3b; 
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

py::array bspline_config_ff::get_rmin_max_3b() {
  py::buffer_info rmin_max_3b_buff_info(
    rmin_max_3b,      /* Pointer to buffer */
    sizeof(double),     /* Size of one scalar */
    py::format_descriptor<double>::format(),    /* Python struct-style format descriptor */
    2,                  /* Number of dimensions */
    { n3b_interactions, 6},  /* Buffer dimensions */
    { sizeof(double) * 6,        /* Strides (in bytes) for each index */
    sizeof(double) } 
  );

  return py::array(rmin_max_3b_buff_info);
 };
