#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifndef UltraFastNeighs_H // Check if UltraFastNeighs_H is not defined
#define UltraFastNeighs_H // Define UltraFastNeighs_H

namespace py = pybind11;

class UltraFastNeighs{
  private:
    const py::detail::unchecked_reference<double, 2>& atoms_array_un;
    const py::detail::unchecked_reference<int, 1>& crystal_index_un;
    const py::detail::unchecked_reference<double, 3>& cell_array_un;
    const py::detail::unchecked_reference<int, 1>& geom_posn_un;
    const py::detail::unchecked_reference<int, 2>& supercell_factors_un;
    const std::vector<int>& num_of_interxns;
    const std::vector<int>& n2b_types;
    const double* rmin_max_2b_sq;
    const double rcut_max_sq;
  
  public:
    UltraFastNeighs(py::detail::unchecked_reference<double, 2>& _atoms_array_un,
                    py::detail::unchecked_reference<int, 1>& _crystal_index_un,
                    py::detail::unchecked_reference<double, 3>& _cell_array_un,
                    py::detail::unchecked_reference<int, 1>& _geom_posn_un,
                    py::detail::unchecked_reference<int, 2>& _supercell_factors_un,
                    std::vector<int>& _num_of_interxns,
                    std::vector<int>& _n2b_types,
                    double* _rmin_max_2b_sq,
                    double _rcut_max_sq);
    
    void set_Neighs(int batch_start, int batch_end,
                    std::vector<double>& Neighs,
                    std::vector<double>& Neighs_del,
                    std::vector<int>& Tot_num_Neighs,
                    int rows, int cols);
    
    ~UltraFastNeighs();
};

#endif // End for UltraFastNeighs_H
