#pragma once
// Grid ↔ QUDA host buffer conversions for clover-Wilson HMC integration.
//
// The QUDA C API takes raw double* host buffers in well-defined layouts.
// We pack/unpack to two flavors:
//
//   Fermion (24 doubles/site, color-inside-spin):
//     buf[24*site + 6*spin + 2*color + ri]
//
//   Gauge (18 doubles/site/dir, row-col color):
//     buf_per_dir[mu][18*site + (row*Nc + col)*2 + ri]
//
// Site index is local to each MPI rank and lex-ordered with x fastest, t
// slowest — Grid's local lex order, which is also QUDA's "t-z-y-x with
// rightmost varying fastest" convention.  The lex_to_eo / eo_to_lex helpers
// reorder per-site blocks for QUDA_QDP_GAUGE_ORDER and QUDA_DIRAC_ORDER
// (which both expect QUDA_EVEN_ODD_SITE_ORDER on the host).
//
// All conversions are CPU-only — the GPU build's autoView triggers the
// device→host copy automatically.  This is fine for HMC: we re-copy at
// most once per smearing-update boundary, which is rare.
//
// Phase 2 deliverable: round-trip identity (Grid → buf → Grid' has
// norm2(diff) < 1e-15).  Phase 3 wires these into QUDA solver classes.

#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid);
namespace Quda {

inline int local_volume(GridBase *grid) {
  Coordinate lc = grid->LocalDimensions();
  int V = 1;
  for (int d = 0; d < (int)lc.size(); ++d) V *= lc[d];
  return V;
}

// site → coords: x fastest, t slowest, matching Grid's local lex.
inline void lex_index_to_coor(int site, const Coordinate &lc, Coordinate &coor) {
  coor.resize(lc.size());
  int s = site;
  for (int d = 0; d < (int)lc.size(); ++d) {
    coor[d] = s % lc[d];
    s /= lc[d];
  }
}

// 4-D parity from Cartesian coords: (x+y+z+t) mod 2.
inline int site_parity(const Coordinate &coor) {
  int p = 0;
  for (int d = 0; d < (int)coor.size(); ++d) p += coor[d];
  return p & 1;
}

// ----------------------------------------------------------------------------
// Fermion: LatticeFermion ↔ flat host buffer (lex sites).
//
// Fast path: Grid's unvectorize/vectorize helpers do a single thread-parallel
// pass over outer SIMD blocks, calling extract/merge once per outer index.
// scalar_object's storage layout (iScalar<iVector<iVector<ComplexD,Nc>,Ns>>)
// is exactly the QUDA_DIRAC_ORDER per-site layout (color-inside-spin, 24
// doubles per site), so the unvectorized array and the QUDA buffer are
// memcpy-compatible.
// ----------------------------------------------------------------------------

template <class FermionField>
inline void fermion_to_lex_buffer(const FermionField &grid_field, double *buf) {
  GridBase *grid = grid_field.Grid();
  int V = local_volume(grid);
  using SiteSpinor = typename FermionField::scalar_object;
  static_assert(sizeof(SiteSpinor) == 24 * sizeof(double),
                "expected Ns·Nc·2 = 24 doubles per fermion site");

  std::vector<SiteSpinor> scalars;
  unvectorizeToLexOrdArray(scalars, grid_field);
  std::memcpy(buf, scalars.data(), V * 24 * sizeof(double));
}

template <class FermionField>
inline void lex_buffer_to_fermion(const double *buf, FermionField &grid_field) {
  GridBase *grid = grid_field.Grid();
  int V = local_volume(grid);
  using SiteSpinor = typename FermionField::scalar_object;
  static_assert(sizeof(SiteSpinor) == 24 * sizeof(double),
                "expected Ns·Nc·2 = 24 doubles per fermion site");

  std::vector<SiteSpinor> scalars(V);
  std::memcpy(scalars.data(), buf, V * 24 * sizeof(double));
  vectorizeFromLexOrdArray(scalars, grid_field);
}

// ----------------------------------------------------------------------------
// Fused fermion → EO buffer (and inverse).
//
// Combines unvectorize, memcpy, and lex→EO permute into a single pass over
// the SIMD outer sites: for each outer index we compute the EO destination
// per SIMD lane and use Grid's `extract` to write the per-lane scalar
// objects directly into the final QUDA EO buffer slots.  No intermediate
// std::vector<SiteSpinor>, no separate permute pass.
//
// Requires Lx even (true for all production lattices) so that lex→EO maps
// cleanly via cb_site = lex_site >> 1 within each parity.
// ----------------------------------------------------------------------------

template <class FermionField>
inline void fermion_to_eo_buffer(const FermionField &grid_field, double *buf_eo) {
  using SiteSpinor = typename FermionField::scalar_object;
  using vobj = typename FermionField::vector_object;
  static_assert(sizeof(SiteSpinor) == 24 * sizeof(double),
                "expected Ns·Nc·2 = 24 doubles per fermion site");

  GridBase *grid = grid_field.Grid();
  Coordinate lc = grid->LocalDimensions();
  assert((lc[0] & 1) == 0);
  int V = local_volume(grid);
  int V_eo = V / 2;
  const int Nsimd = vobj::vector_type::Nsimd();
  const int ndim = grid->Nd();

  // Per-lane Cartesian-shift table (constant across all outer sites).
  std::vector<Coordinate> icoor_table(Nsimd);
  for (int lane = 0; lane < Nsimd; ++lane) {
    icoor_table[lane].resize(ndim);
    grid->iCoorFromIindex(icoor_table[lane], lane);
  }

  autoView(in_v, grid_field, CpuRead);
  thread_for(oidx, grid->oSites(), {
    ExtractPointerArray<SiteSpinor> ptrs(Nsimd);
    Coordinate ocoor(ndim);
    grid->oCoorFromOindex(ocoor, oidx);
    for (int lane = 0; lane < Nsimd; ++lane) {
      int x = ocoor[0] + grid->_rdimensions[0] * icoor_table[lane][0];
      int y = ocoor[1] + grid->_rdimensions[1] * icoor_table[lane][1];
      int z = ocoor[2] + grid->_rdimensions[2] * icoor_table[lane][2];
      int t = ocoor[3] + grid->_rdimensions[3] * icoor_table[lane][3];
      int lex = x + lc[0] * (y + lc[1] * (z + lc[2] * t));
      int parity = (x + y + z + t) & 1;
      int eo_idx = parity * V_eo + (lex >> 1);
      ptrs[lane] = reinterpret_cast<SiteSpinor *>(&buf_eo[eo_idx * 24]);
    }
    extract(in_v[oidx], ptrs, 0);
  });
}

template <class FermionField>
inline void eo_buffer_to_fermion(const double *buf_eo, FermionField &grid_field) {
  using SiteSpinor = typename FermionField::scalar_object;
  using vobj = typename FermionField::vector_object;

  GridBase *grid = grid_field.Grid();
  Coordinate lc = grid->LocalDimensions();
  assert((lc[0] & 1) == 0);
  int V = local_volume(grid);
  int V_eo = V / 2;
  const int Nsimd = vobj::vector_type::Nsimd();
  const int ndim = grid->Nd();

  std::vector<Coordinate> icoor_table(Nsimd);
  for (int lane = 0; lane < Nsimd; ++lane) {
    icoor_table[lane].resize(ndim);
    grid->iCoorFromIindex(icoor_table[lane], lane);
  }

  autoView(out_v, grid_field, CpuWrite);
  thread_for(oidx, grid->oSites(), {
    ExtractPointerArray<SiteSpinor> ptrs(Nsimd);
    Coordinate ocoor(ndim);
    grid->oCoorFromOindex(ocoor, oidx);
    for (int lane = 0; lane < Nsimd; ++lane) {
      int x = ocoor[0] + grid->_rdimensions[0] * icoor_table[lane][0];
      int y = ocoor[1] + grid->_rdimensions[1] * icoor_table[lane][1];
      int z = ocoor[2] + grid->_rdimensions[2] * icoor_table[lane][2];
      int t = ocoor[3] + grid->_rdimensions[3] * icoor_table[lane][3];
      int lex = x + lc[0] * (y + lc[1] * (z + lc[2] * t));
      int parity = (x + y + z + t) & 1;
      int eo_idx = parity * V_eo + (lex >> 1);
      ptrs[lane] = const_cast<SiteSpinor *>(
          reinterpret_cast<const SiteSpinor *>(&buf_eo[eo_idx * 24]));
    }
    vobj vobj_out;
    merge(vobj_out, ptrs, 0);
    out_v[oidx] = vobj_out;
  });
}

// ----------------------------------------------------------------------------
// Gauge: LatticeGaugeField ↔ 4 per-direction host buffers (lex sites).
//
// scalar_object is iVector<iScalar<iMatrix<C,Nc>>,Nd> — 4 dirs × 18 doubles
// stored contiguously per site, dir-slowest.  QUDA wants 4 separate
// per-direction buffers (QUDA_QDP_GAUGE_ORDER), so we de-interleave with a
// thread-parallel loop over sites.
// ----------------------------------------------------------------------------

inline void gauge_to_lex_buffers(const LatticeGaugeField &U_grid,
                                 double *buf_per_dir[4]) {
  int V = local_volume(U_grid.Grid());
  using SiteGauge = LatticeGaugeField::vector_object::scalar_object;
  static_assert(sizeof(SiteGauge) == 4 * 18 * sizeof(double),
                "expected Nd·Nc·Nc·2 = 72 doubles per gauge site");

  std::vector<SiteGauge> scalars;
  unvectorizeToLexOrdArray(scalars, U_grid);
  const double *src = reinterpret_cast<const double *>(scalars.data());
  thread_for(site, V, {
    for (int mu = 0; mu < Nd; ++mu) {
      std::memcpy(&buf_per_dir[mu][18 * site],
                  &src[72 * site + 18 * mu],
                  18 * sizeof(double));
    }
  });
}

inline void lex_buffers_to_gauge(double *const buf_per_dir[4],
                                 LatticeGaugeField &U_grid) {
  int V = local_volume(U_grid.Grid());
  using SiteGauge = LatticeGaugeField::vector_object::scalar_object;

  std::vector<SiteGauge> scalars(V);
  double *dst = reinterpret_cast<double *>(scalars.data());
  thread_for(site, V, {
    for (int mu = 0; mu < Nd; ++mu) {
      std::memcpy(&dst[72 * site + 18 * mu],
                  &buf_per_dir[mu][18 * site],
                  18 * sizeof(double));
    }
  });
  vectorizeFromLexOrdArray(scalars, U_grid);
}

// ----------------------------------------------------------------------------
// RB-half ↔ EO-buffer-half (for HMC: PhiOdd lives on Grid's odd-parity RB
// grid, half-volume).  Within a parity, Grid's RB cb-site order matches
// QUDA's cb_site = full_lex >> 1, so unvectorize → memcpy is a direct copy
// into the parity-half slab of the full-volume QUDA EO buffer.
//
// Caller must zero the *other* parity half of buf_full before calling this
// (or be sure QUDA doesn't read it).
// ----------------------------------------------------------------------------

template <class FermionField>
inline void fermion_rb_to_eo_buffer_half(const FermionField &phi_rb,
                                         int parity, double *buf_full) {
  using SiteSpinor = typename FermionField::scalar_object;
  int V_eo = local_volume(phi_rb.Grid());  // RB grid local vol = V/2
  std::vector<SiteSpinor> scalars;
  unvectorizeToLexOrdArray(scalars, phi_rb);
  std::memcpy(&buf_full[parity * V_eo * 24], scalars.data(),
              V_eo * 24 * sizeof(double));
}

template <class FermionField>
inline void eo_buffer_half_to_fermion_rb(const double *buf_full, int parity,
                                         FermionField &phi_rb) {
  using SiteSpinor = typename FermionField::scalar_object;
  int V_eo = local_volume(phi_rb.Grid());
  std::vector<SiteSpinor> scalars(V_eo);
  std::memcpy(scalars.data(), &buf_full[parity * V_eo * 24],
              V_eo * 24 * sizeof(double));
  vectorizeFromLexOrdArray(scalars, phi_rb);
}

// ----------------------------------------------------------------------------
// Lex ↔ even-odd site permutation.
//
// QUDA's host buffers (DIRAC_ORDER spinor, QDP_GAUGE_ORDER gauge) expect
// QUDA_EVEN_ODD_SITE_ORDER: first all V/2 even sites in lex-by-parity order,
// then all V/2 odd sites.  Within a parity, the index is
//   cb_site = (lex_site >> 1)
// because Grid's local lex with x fastest packs alternating parities into
// adjacent x-pairs, so dropping the LSB of lex_site indexes within parity.
//
// This requires Lx to be even — true for all production lattices.
// ----------------------------------------------------------------------------

inline void lex_to_eo_permute(const double *src, double *dst,
                              int V, int per_site_doubles,
                              const Coordinate &lc) {
  assert((lc[0] & 1) == 0 && "x dimension must be even for EO ordering");
  const int V_eo = V / 2;
  const int Lx = lc[0], Ly = lc[1], Lz = lc[2];
  thread_for(site, V, {
    int s = site;
    int x = s % Lx; s /= Lx;
    int y = s % Ly; s /= Ly;
    int z = s % Lz; int t = s / Lz;
    int parity = (x + y + z + t) & 1;
    int dst_site = parity * V_eo + (site >> 1);
    std::memcpy(&dst[dst_site * per_site_doubles],
                &src[site * per_site_doubles],
                per_site_doubles * sizeof(double));
  });
}

inline void eo_to_lex_permute(const double *src, double *dst,
                              int V, int per_site_doubles,
                              const Coordinate &lc) {
  assert((lc[0] & 1) == 0 && "x dimension must be even for EO ordering");
  const int V_eo = V / 2;
  const int Lx = lc[0], Ly = lc[1], Lz = lc[2];
  thread_for(site, V, {
    int s = site;
    int x = s % Lx; s /= Lx;
    int y = s % Ly; s /= Ly;
    int z = s % Lz; int t = s / Lz;
    int parity = (x + y + z + t) & 1;
    int src_site = parity * V_eo + (site >> 1);
    std::memcpy(&dst[site * per_site_doubles],
                &src[src_site * per_site_doubles],
                per_site_doubles * sizeof(double));
  });
}

}  // namespace Quda
NAMESPACE_END(Grid);
