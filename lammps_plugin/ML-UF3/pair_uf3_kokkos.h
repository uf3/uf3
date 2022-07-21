/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 *    Contributing authors: Ajinkya Hire(U of Florida), 
 *                          Hendrik Kraß (U of Constance),
 *                          Richard Hennig (U of Florida)
 * ---------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(uf3/kk,PairUF3Kokkos<LMPDeviceType>)
PairStyle(uf3/kk/device,PairUF3Kokkos<LMPDeviceType>)
// clang-format on
#else

#ifndef LMP_PAIR_UF3_KOKKOS_H
#define LMP_PAIR_UF3_KOKKOS_H

#include "uf3_pair_bspline.h"
#include "uf3_triplet_bspline.h"

#include "pair_kokkos.h"

#include <unordered_map>

template <int NEIGHFLAG, int EVFLAG> struct TagPairUF3ComputeFullA {
};
struct TagPairUF3ComputeShortNeigh {};

namespace LAMMPS_NS {

template <class DeviceType> class PairUF3Kokkos : public Pair {
 public:
  PairUF3Kokkos(class LAMMPS *);
  ~PairUF3Kokkos() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  void init_list(int, class NeighList *) override;    // needed for ptr to full neigh list
  double init_one(int, int) override;                 // needed for cutoff radius for neighbour list
  double single(int, int, int, int, double, double, double, double &) override;

 protected:
  void uf3_read_pot_file(char *potf_name);
  int num_of_elements, nbody_flag, n2body_pot_files, n3body_pot_files, tot_pot_files;
  int coeff_matrix_dim1, coeff_matrix_dim2, coeff_matrix_dim3, coeff_matrix_elements_len;
  bool pot_3b;
  int ***setflag_3b;
  double ***cut_3b, **cut_3b_list;
  virtual void allocate();
  std::vector<std::vector<std::vector<double>>> n2b_knot, n2b_coeff;
  std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> n3b_knot_matrix;
  std::unordered_map<std::string, std::vector<std::vector<std::vector<double>>>> n3b_coeff_matrix;
  std::vector<std::vector<uf3_pair_bspline>> UFBS2b;
  std::vector<std::vector<std::vector<uf3_triplet_bspline>>> UFBS3b;
  int *neighshort, maxshort;    // short neighbor list array for 3body interaction

  enum { EnabledNeighFlags = FULL };
  enum { COUL_FLAG = 0 };
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  template <typename T, typename V> void copy_1d(V &d, T *h, int n);

  template <typename T, typename V> void copy_2d(V &d, T **h, int m, int n);

  template <typename T, typename V> void copy_3d(V &d, T ***h, int m, int n, int o);

  template <int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION void operator()(TagPairUF3ComputeFullA<NEIGHFLAG, EVFLAG>, const int &,
                                         EV_FLOAT &) const;

  template <int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION void operator()(TagPairUF3ComputeFullA<NEIGHFLAG, EVFLAG>,
                                         const int &) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairUF3ComputeShortNeigh, const int &) const;

  template <int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION void
  ev_tally(EV_FLOAT &ev, const int &i, const int &j, const F_FLOAT &epair, const F_FLOAT &fpair,
           const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const;

  template <int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION void ev_tally3(EV_FLOAT &ev, const int &i, const int &j, int &k,
                                        const F_FLOAT &evdwl, const F_FLOAT &ecoul, F_FLOAT *fj,
                                        F_FLOAT *fk, F_FLOAT *drji, F_FLOAT *drki) const;

  KOKKOS_INLINE_FUNCTION
  void ev_tally3_atom(EV_FLOAT &ev, const int &i, const F_FLOAT &evdwl, const F_FLOAT &ecoul,
                      F_FLOAT *fj, F_FLOAT *fk, F_FLOAT *drji, F_FLOAT *drki) const;

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_tagint_1d tag;
  typename AT::t_int_1d_randomread type;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  DAT::tdual_virial_array k_cvatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;
  typename AT::t_virial_array d_cvatom;

  int need_dup;
  Kokkos::Experimental::ScatterView<F_FLOAT *[3], typename DAT::t_f_array::array_layout, DeviceType,
                                    Kokkos::Experimental::ScatterSum,
                                    Kokkos::Experimental::ScatterDuplicated>
      dup_f;
  Kokkos::Experimental::ScatterView<E_FLOAT *, typename DAT::t_efloat_1d::array_layout, DeviceType,
                                    Kokkos::Experimental::ScatterSum,
                                    Kokkos::Experimental::ScatterDuplicated>
      dup_eatom;
  Kokkos::Experimental::ScatterView<F_FLOAT *[6], typename DAT::t_virial_array::array_layout,
                                    DeviceType, Kokkos::Experimental::ScatterSum,
                                    Kokkos::Experimental::ScatterDuplicated>
      dup_vatom;
  Kokkos::Experimental::ScatterView<F_FLOAT *[9], typename DAT::t_virial_array::array_layout,
                                    DeviceType, Kokkos::Experimental::ScatterSum,
                                    Kokkos::Experimental::ScatterDuplicated>
      dup_cvatom;
  Kokkos::Experimental::ScatterView<F_FLOAT *[3], typename DAT::t_f_array::array_layout, DeviceType,
                                    Kokkos::Experimental::ScatterSum,
                                    Kokkos::Experimental::ScatterNonDuplicated>
      ndup_f;
  Kokkos::Experimental::ScatterView<E_FLOAT *, typename DAT::t_efloat_1d::array_layout, DeviceType,
                                    Kokkos::Experimental::ScatterSum,
                                    Kokkos::Experimental::ScatterNonDuplicated>
      ndup_eatom;
  Kokkos::Experimental::ScatterView<F_FLOAT *[6], typename DAT::t_virial_array::array_layout,
                                    DeviceType, Kokkos::Experimental::ScatterSum,
                                    Kokkos::Experimental::ScatterNonDuplicated>
      ndup_vatom;
  Kokkos::Experimental::ScatterView<F_FLOAT *[9], typename DAT::t_virial_array::array_layout,
                                    DeviceType, Kokkos::Experimental::ScatterSum,
                                    Kokkos::Experimental::ScatterNonDuplicated>
      ndup_cvatom;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  int neighflag, newton_pair;
  int nlocal, nall, eflag, vflag;

  int inum;
  Kokkos::View<int **, DeviceType> d_neighbors_short;
  Kokkos::View<int *, DeviceType> d_numneigh_short;

  friend void pair_virial_fdotr_compute<PairUF3Kokkos>(PairUF3Kokkos *);
};

}    // namespace LAMMPS_NS

#endif
#endif

    /* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

*/
