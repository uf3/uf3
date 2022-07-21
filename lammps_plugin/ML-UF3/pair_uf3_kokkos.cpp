/* ----------------------------------------------------------------------
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

#include "pair_uf3_kokkos.h"
#include "pair_uf3.h"
#include "uf3_pair_bspline.h"
#include "uf3_triplet_bspline.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair_kokkos.h"
#include "text_file_reader.h"
#include <algorithm>
#include <cmath>
#include <utility>

#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;

template <class DeviceType> PairUF3Kokkos<DeviceType>::PairUF3Kokkos(LAMMPS *lmp) : PairUF3(lmp)
{
  respa_enable = 0;

  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

template <class DeviceType> PairUF3Kokkos<DeviceType>::~PairUF3Kokkos()
{
  if (!copymode) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->destroy_kokkos(k_cvatom, cvatom);
    eatom = NULL;
    vatom = NULL;
    cvatom = NULL;
  }
}

template <class DeviceType> void PairUF3Kokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  no_virial_fdotr_compute = 1;

  ev_init(eflag, vflag, 0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, 6, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }
  if (cvflag_atom) {
    memoryKK->destroy_kokkos(k_cvatom, cvatom);
    memoryKK->create_kokkos(k_cvatom, cvatom, maxcvatom, 9, "pair:vatom");
    d_cvatom = k_cvatom.view<DeviceType>();
  }

  atomKK->sync(execution_space, datamask_read);
  if (eflag || vflag)
    atomKK->modified(execution_space, datamask_modify);
  else
    atomKK->modified(execution_space, F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  newton_pair = force->newton_pair;
  nall = atom->nlocal + atom->nghost;

  const int inum = list->inum;
  const int ignum = inum + list->gnum;
  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_ilist = k_list->d_ilist;
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup) {
    dup_f = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum,
                                                      Kokkos::Experimental::ScatterDuplicated>(f);
    dup_eatom =
        Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum,
                                                  Kokkos::Experimental::ScatterDuplicated>(d_eatom);
    dup_vatom =
        Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum,
                                                  Kokkos::Experimental::ScatterDuplicated>(d_vatom);
    dup_cvatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum,
                                                           Kokkos::Experimental::ScatterDuplicated>(
        d_cvatom);
  } else {
    ndup_f =
        Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum,
                                                  Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_eatom = Kokkos::Experimental::create_scatter_view<
        Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_eatom);
    ndup_vatom = Kokkos::Experimental::create_scatter_view<
        Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
    ndup_cvatom = Kokkos::Experimental::create_scatter_view<
        Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_cvatom);
  }

  copymode = 1;

  EV_FLOAT ev;
  EV_FLOAT ev_all;

  // build short neighbor list

  int max_neighs = d_neighbors.extent(1);

  if ((d_neighbors_short.extent(1) != max_neighs) || (d_neighbors_short.extent(0) != ignum)) {
    d_neighbors_short = Kokkos::View<int **, DeviceType>("UF3::neighbors_short", ignum, max_neighs);
  }
  if (d_numneigh_short.extent(0) != ignum)
    d_numneigh_short = Kokkos::View<int *, DeviceType>("UF3::numneighs_short", ignum);
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairUF3ComputeShortNeigh>(0, ignum),
                       *this);

  // loop over neighbor list of my atoms

  if (evflag)
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<DeviceType, TagPairUF3ComputeFullA<FULL, 1>>(0, inum), *this, ev);
  else
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairUF3ComputeFullA<FULL, 0>>(0, inum),
                         *this);
  ev_all += ev;

  if (need_dup) Kokkos::Experimental::contribute(f, dup_f);

  if (eflag_global) eng_vdwl += ev_all.evdwl;
  if (vflag_global) {
    virial[0] += ev_all.v[0];
    virial[1] += ev_all.v[1];
    virial[2] += ev_all.v[2];
    virial[3] += ev_all.v[3];
    virial[4] += ev_all.v[4];
    virial[5] += ev_all.v[5];
  }

  if (eflag_atom) {
    if (need_dup) Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    if (need_dup) Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  if (cvflag_atom) {
    if (need_dup) Kokkos::Experimental::contribute(d_cvatom, dup_cvatom);
    k_cvatom.template modify<DeviceType>();
    k_cvatom.template sync<LMPHostType>();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;

  // free duplicated memory
  if (need_dup) {
    dup_f = decltype(dup_f)();
    dup_eatom = decltype(dup_eatom)();
    dup_vatom = decltype(dup_vatom)();
    dup_cvatom = decltype(dup_cvatom)();
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairUF3Kokkos<DeviceType>::operator()(TagPairUF3ComputeShortNeigh,
                                                                  const int &ii) const
{
  const int i = d_ilist[ii];
  const X_FLOAT xtmp = x(i, 0);
  const X_FLOAT ytmp = x(i, 1);
  const X_FLOAT ztmp = x(i, 2);

  const int jnum = d_numneigh[i];
  int inside = 0;
  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(i, jj);
    j &= NEIGHMASK;

    const X_FLOAT delx = xtmp - x(j, 0);
    const X_FLOAT dely = ytmp - x(j, 1);
    const X_FLOAT delz = ztmp - x(j, 2);
    const F_FLOAT rsq = delx * delx + dely * dely + delz * delz;

    const int itype = type[i];
    const int jtype = type[j];

    if (rsq < cutsq[itype][jtype]) {
      F_FLOAT rij = sqrt(rsq);

      if (rij <= cut_3b_list[itype][jtype]) {
        d_neighbors_short(i, inside) = j;
        inside++;
      }
    }
  }
  d_numneigh_short(i) = inside;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
template <int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION void
PairUF3Kokkos<DeviceType>::operator()(TagPairUF3ComputeFullA<NEIGHFLAG, EVFLAG>, const int &ii,
                                      EV_FLOAT &ev) const
{

  F_FLOAT del_rji[3], del_rki[3], del_rkj[3];
  F_FLOAT fij[3], fik[3], fjk[3];
  F_FLOAT fji[3], fki[3], fkj[3];
  F_FLOAT Fi[3], Fj[3], Fk[3];
  F_FLOAT evdwl = 0.0;
  F_FLOAT fpair = 0.0;

  const int i = d_ilist[ii];

  const tagint itag = tag[i];
  const int itype = type[i];
  const X_FLOAT xtmp = x(i, 0);
  const X_FLOAT ytmp = x(i, 1);
  const X_FLOAT ztmp = x(i, 2);

  // two-body interactions

  const int jnum = d_numneigh_short[i];

  F_FLOAT fxtmpi = 0.0;
  F_FLOAT fytmpi = 0.0;
  F_FLOAT fztmpi = 0.0;

  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors_short(i, jj);
    j &= NEIGHMASK;
    const tagint jtag = tag[j];

    const int jtype = type[j];

    const X_FLOAT delx = xtmp - x(j, 0);
    const X_FLOAT dely = ytmp - x(j, 1);
    const X_FLOAT delz = ztmp - x(j, 2);
    const F_FLOAT rsq = delx * delx + dely * dely + delz * delz;

    if (rsq >= cutsq[itype][jtype]) continue;

    const F_FLOAT rij = sqrt(rsq);
    F_FLOAT *pair_eval = UFBS2b[itype][jtype].eval(rij);

    fpair = -1 * pair_eval[1] / rij;

    fxtmpi += delx * fpair;
    fytmpi += dely * fpair;
    fztmpi += delz * fpair;

    if (EVFLAG) {
      if (eflag) ev.evdwl += evdwl;
      if (vflag_either || eflag_atom)
        this->template ev_tally<NEIGHFLAG>(ev, i, j, evdwl, fpair, delx, dely, delz);
    }
  }

  // 3-body interaction
  // jth atom
  const int jnumm1 = jnum - 1;
  for (int jj = 0; jj < jnumm1; jj++) {
    int j = d_neighbors_short(i, jj);
    j &= NEIGHMASK;
    const int jtype = type[j];
    del_rji[0] = x(j, 0) - xtmp;
    del_rji[1] = x(j, 1) - ytmp;
    del_rji[2] = x(j, 2) - ztmp;
    const F_FLOAT rij =
        sqrt(del_rji[0] * del_rji[0] + del_rji[1] * del_rji[1] + del_rji[2] * del_rji[2]);

    for (int kk = jj + 1; kk < jnum; kk++) {
      int k = d_neighbors_short(i, kk);
      k &= NEIGHMASK;
      const int ktype = type[k];

      if (rij >= cut_3b[itype][jtype][ktype]) continue;

      del_rki[0] = x(k, 0) - xtmp;
      del_rki[1] = x(k, 1) - ytmp;
      del_rki[2] = x(k, 2) - ztmp;
      const F_FLOAT rik =
          sqrt(del_rki[0] * del_rki[0] + del_rki[1] * del_rki[1] + del_rki[2] * del_rki[2]);

      if (rik >= cut_3b[itype][ktype][jtype]) continue;

      del_rkj[0] = x(k, 0) - x(j, 0);
      del_rkj[1] = x(k, 1) - x(j, 1);
      del_rkj[2] = x(k, 2) - x(j, 2);
      const F_FLOAT rjk =
          sqrt(del_rkj[0] * del_rkj[0] + del_rkj[1] * del_rkj[1] + del_rkj[2] * del_rkj[2]);

      F_FLOAT *triangle_eval = UFBS3b[itype][jtype][ktype].eval(rij, rik, rjk);
      F_FLOAT evdwl3 = *triangle_eval;

      F_FLOAT f_d1, f_d2;

      fij[0] = *(triangle_eval + 1) * (del_rji[0] / rij);
      fji[0] = -fij[0];
      fik[0] = *(triangle_eval + 2) * (del_rki[0] / rik);
      fki[0] = -fik[0];
      fjk[0] = *(triangle_eval + 3) * (del_rkj[0] / rjk);
      fkj[0] = -fjk[0];

      fij[1] = *(triangle_eval + 1) * (del_rji[1] / rij);
      fji[1] = -fij[1];
      fik[1] = *(triangle_eval + 2) * (del_rki[1] / rik);
      fki[1] = -fik[1];
      fjk[1] = *(triangle_eval + 3) * (del_rkj[1] / rjk);
      fkj[1] = -fjk[1];

      fij[2] = *(triangle_eval + 1) * (del_rji[2] / rij);
      fji[2] = -fij[2];
      fik[2] = *(triangle_eval + 2) * (del_rki[2] / rik);
      fki[2] = -fik[2];
      fjk[2] = *(triangle_eval + 3) * (del_rkj[2] / rjk);
      fkj[2] = -fjk[2];

      Fj[0] = fji[0] + fjk[0];
      Fj[1] = fji[1] + fjk[1];
      Fj[2] = fji[2] + fjk[2];

      Fk[0] = fki[0] + fkj[0];
      Fk[1] = fki[1] + fkj[1];
      Fk[2] = fki[2] + fkj[2];

      fxtmpi += 2 * (fij[0] + fik[0]);
      fytmpi += 2 * (fij[1] + fik[1]);
      fztmpi += 2 * (fij[2] + fik[2]);

      if (EVFLAG) {
        if (eflag) { ev.evdwl += evdwl3; }
        if (vflag_either || eflag_atom) {
          this->template ev_tally3<NEIGHFLAG>(ev, i, j, k, evdwl3, 0.0, Fj, Fk, del_rji, del_rki);
        }
      }
    }
  }

  f(i, 0) += fxtmpi;
  f(i, 1) += fytmpi;
  f(i, 2) += fztmpi;
}

template <class DeviceType>
template <int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION void
PairUF3Kokkos<DeviceType>::operator()(TagPairUF3ComputeFullA<NEIGHFLAG, EVFLAG>,
                                      const int &ii) const
{
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG, EVFLAG>(TagPairUF3ComputeFullA<NEIGHFLAG, EVFLAG>(), ii, ev);
}

template <class DeviceType>
template <typename T, typename V>
void PairUF3Kokkos<DeviceType>::copy_1d(V &d, T *h, int n)
{
  Kokkos::DualView<T *, DeviceType> tmp("pair::tmp", n);
  auto h_view = tmp.h_view;

  for (int i = 0; i < n; i++) { h_view(i) = h[i]; }

  tmp.template modify<LMPHostType>();
  tmp.template sync<DeviceType>();

  d = tmp.template view<DeviceType>();
}

template <class DeviceType>
template <typename T, typename V>
void PairUF3Kokkos<DeviceType>::copy_2d(V &d, T **h, int m, int n)
{
  Kokkos::View<T **, Kokkos::LayoutRight> tmp("pair::tmp", m, n);
  auto h_view = Kokkos::create_mirror(tmp);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) { h_view(i, j) = h[i][j]; }
  }

  Kokkos::deep_copy(tmp, h_view);

  d = tmp;
}

template <class DeviceType>
template <typename T, typename V>
void PairUF3Kokkos<DeviceType>::copy_3d(V &d, T ***h, int m, int n, int o)
{
  Kokkos::View<T ***> tmp("pair::tmp", m, n, o);
  auto h_view = Kokkos::create_mirror(tmp);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < o; k++) { h_view(i, j, k) = h[i][j][k]; }
    }
  }

  Kokkos::deep_copy(tmp, h_view);

  d = tmp;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
template <int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION void
PairUF3Kokkos<DeviceType>::ev_tally(EV_FLOAT &ev, const int &i, const int &j, const F_FLOAT &epair,
                                    const F_FLOAT &fpair, const F_FLOAT &delx, const F_FLOAT &dely,
                                    const F_FLOAT &delz) const
{
  const int VFLAG = vflag_either;

  // The eatom and vatom arrays are duplicated for OpenMP, atomic for CUDA,
  // and neither for Serial

  auto v_eatom = ScatterViewHelper<NeedDup<NEIGHFLAG, DeviceType>::value, decltype(dup_eatom),
                                   decltype(ndup_eatom)>::get(dup_eatom, ndup_eatom);
  auto a_eatom = v_eatom.template access<AtomicDup<NEIGHFLAG, DeviceType>::value>();

  auto v_vatom = ScatterViewHelper<NeedDup<NEIGHFLAG, DeviceType>::value, decltype(dup_vatom),
                                   decltype(ndup_vatom)>::get(dup_vatom, ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup<NEIGHFLAG, DeviceType>::value>();

  auto v_cvatom = ScatterViewHelper<NeedDup<NEIGHFLAG, DeviceType>::value, decltype(dup_cvatom),
                                    decltype(ndup_cvatom)>::get(dup_cvatom, ndup_cvatom);
  auto a_cvatom = v_cvatom.template access<AtomicDup<NEIGHFLAG, DeviceType>::value>();

  if (eflag_atom) {
    const E_FLOAT epairhalf = 0.5 * epair;
    a_eatom[i] += epairhalf;
    if (NEIGHFLAG != FULL) a_eatom[j] += epairhalf;
  }

  if (VFLAG) {
    const E_FLOAT v0 = delx * delx * fpair;
    const E_FLOAT v1 = dely * dely * fpair;
    const E_FLOAT v2 = delz * delz * fpair;
    const E_FLOAT v3 = delx * dely * fpair;
    const E_FLOAT v4 = delx * delz * fpair;
    const E_FLOAT v5 = dely * delz * fpair;

    if (vflag_global) {
      ev.v[0] += 0.5 * v0;
      ev.v[1] += 0.5 * v1;
      ev.v[2] += 0.5 * v2;
      ev.v[3] += 0.5 * v3;
      ev.v[4] += 0.5 * v4;
      ev.v[5] += 0.5 * v5;
    }

    if (vflag_atom) {
      a_vatom(i, 0) += 0.5 * v0;
      a_vatom(i, 1) += 0.5 * v1;
      a_vatom(i, 2) += 0.5 * v2;
      a_vatom(i, 3) += 0.5 * v3;
      a_vatom(i, 4) += 0.5 * v4;
      a_vatom(i, 5) += 0.5 * v5;
    }

    if (cvflag_atom) {
      a_cvatom(i, 0) += 0.5 * v0;
      a_cvatom(i, 1) += 0.5 * v1;
      a_cvatom(i, 2) += 0.5 * v2;
      a_cvatom(i, 3) += 0.5 * v3;
      a_cvatom(i, 4) += 0.5 * v4;
      a_cvatom(i, 5) += 0.5 * v5;
      a_cvatom(i, 6) += 0.5 * v3;
      a_cvatom(i, 7) += 0.5 * v4;
      a_cvatom(i, 8) += 0.5 * v5;
    }
  }
}

/* ----------------------------------------------------------------------
   tally eng_vdwl and virial into global and per-atom accumulators
   called by SW and hbond potentials, newton_pair is always on
   virial = riFi + rjFj + rkFk = (rj-ri) Fj + (rk-ri) Fk = drji*fj + drki*fk
 ------------------------------------------------------------------------- */

template <class DeviceType>
template <int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION void
PairUF3Kokkos<DeviceType>::ev_tally3(EV_FLOAT &ev, const int &i, const int &j, int &k,
                                     const F_FLOAT &evdwl, const F_FLOAT &ecoul, F_FLOAT *fj,
                                     F_FLOAT *fk, F_FLOAT *drji, F_FLOAT *drki) const
{
  F_FLOAT epairthird, v[6];

  const int VFLAG = vflag_either;

  // The eatom and vatom arrays are duplicated for OpenMP, atomic for CUDA,
  // and neither for Serial

  auto v_eatom = ScatterViewHelper<NeedDup<NEIGHFLAG, DeviceType>::value, decltype(dup_eatom),
                                   decltype(ndup_eatom)>::get(dup_eatom, ndup_eatom);
  auto a_eatom = v_eatom.template access<AtomicDup<NEIGHFLAG, DeviceType>::value>();

  auto v_vatom = ScatterViewHelper<NeedDup<NEIGHFLAG, DeviceType>::value, decltype(dup_vatom),
                                   decltype(ndup_vatom)>::get(dup_vatom, ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup<NEIGHFLAG, DeviceType>::value>();

  auto v_cvatom = ScatterViewHelper<NeedDup<NEIGHFLAG, DeviceType>::value, decltype(dup_cvatom),
                                    decltype(ndup_cvatom)>::get(dup_cvatom, ndup_cvatom);
  auto a_cvatom = v_cvatom.template access<AtomicDup<NEIGHFLAG, DeviceType>::value>();

  if (eflag_atom) {
    epairthird = THIRD * (evdwl + ecoul);
    a_eatom[i] += epairthird;
    if (NEIGHFLAG != FULL) {
      a_eatom[j] += epairthird;
      a_eatom[k] += epairthird;
    }
  }

  if (VFLAG) {
    v[0] = drji[0] * fj[0] + drki[0] * fk[0];
    v[1] = drji[1] * fj[1] + drki[1] * fk[1];
    v[2] = drji[2] * fj[2] + drki[2] * fk[2];
    v[3] = drji[0] * fj[1] + drki[0] * fk[1];
    v[4] = drji[0] * fj[2] + drki[0] * fk[2];
    v[5] = drji[1] * fj[2] + drki[1] * fk[2];

    if (vflag_global) {
      ev.v[0] += v[0];
      ev.v[1] += v[1];
      ev.v[2] += v[2];
      ev.v[3] += v[3];
      ev.v[4] += v[4];
      ev.v[5] += v[5];
    }

    if (vflag_atom) {
      a_vatom(i, 0) += THIRD * v[0];
      a_vatom(i, 1) += THIRD * v[1];
      a_vatom(i, 2) += THIRD * v[2];
      a_vatom(i, 3) += THIRD * v[3];
      a_vatom(i, 4) += THIRD * v[4];
      a_vatom(i, 5) += THIRD * v[5];
    }

    if (cvflag_atom) {
      F_FLOAT ric[3];
      ric[0] = THIRD * (-drji[0] - drki[0]);
      ric[1] = THIRD * (-drji[0] - drki[0]);
      ric[2] = THIRD * (-drji[0] - drki[0]);
      a_cvatom(i, 0) += ric[0] * (-fj[0] - fk[0]);
      a_cvatom(i, 1) += ric[1] * (-fj[0] - fk[1]);
      a_cvatom(i, 2) += ric[2] * (-fj[0] - fk[2]);
      a_cvatom(i, 3) += ric[0] * (-fj[0] - fk[1]);
      a_cvatom(i, 4) += ric[0] * (-fj[0] - fk[2]);
      a_cvatom(i, 5) += ric[1] * (-fj[0] - fk[2]);
      a_cvatom(i, 6) += ric[1] * (-fj[0] - fk[0]);
      a_cvatom(i, 7) += ric[2] * (-fj[0] - fk[0]);
      a_cvatom(i, 8) += ric[2] * (-fj[0] - fk[1]);
    }
  }
}

/* ----------------------------------------------------------------------
   tally eng_vdwl and virial into global and per-atom accumulators
   called by SW and hbond potentials, newton_pair is always on
   virial = riFi + rjFj + rkFk = (rj-ri) Fj + (rk-ri) Fk = drji*fj + drki*fk
 ------------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void
PairUF3Kokkos<DeviceType>::ev_tally3_atom(EV_FLOAT &ev, const int &i, const F_FLOAT &evdwl,
                                          const F_FLOAT &ecoul, F_FLOAT *fj, F_FLOAT *fk,
                                          F_FLOAT *drji, F_FLOAT *drki) const
{
  F_FLOAT epairthird, v[6];

  const int VFLAG = vflag_either;

  if (eflag_atom) {
    epairthird = THIRD * (evdwl + ecoul);
    d_eatom[i] += epairthird;
  }

  if (VFLAG) {
    v[0] = drji[0] * fj[0] + drki[0] * fk[0];
    v[1] = drji[1] * fj[1] + drki[1] * fk[1];
    v[2] = drji[2] * fj[2] + drki[2] * fk[2];
    v[3] = drji[0] * fj[1] + drki[0] * fk[1];
    v[4] = drji[0] * fj[2] + drki[0] * fk[2];
    v[5] = drji[1] * fj[2] + drki[1] * fk[2];

    if (vflag_atom) {
      d_vatom(i, 0) += THIRD * v[0];
      d_vatom(i, 1) += THIRD * v[1];
      d_vatom(i, 2) += THIRD * v[2];
      d_vatom(i, 3) += THIRD * v[3];
      d_vatom(i, 4) += THIRD * v[4];
      d_vatom(i, 5) += THIRD * v[5];
    }
  }
}