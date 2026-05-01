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
// ----------------------------------------------------------------------------

template <class FermionField>
inline void fermion_to_lex_buffer(const FermionField &grid_field, double *buf) {
  GridBase *grid = grid_field.Grid();
  Coordinate lc = grid->LocalDimensions();
  int V = local_volume(grid);
  using SiteSpinor = typename FermionField::scalar_object;
  static_assert(sizeof(SiteSpinor) == 24 * sizeof(double),
                "expected Ns·Nc·2 = 24 doubles per fermion site");

  for (int site = 0; site < V; ++site) {
    Coordinate coor;
    lex_index_to_coor(site, lc, coor);
    SiteSpinor s_obj;
    peekLocalSite(s_obj, grid_field, coor);
    for (int spin = 0; spin < Ns; ++spin) {
      for (int color = 0; color < Nc; ++color) {
        ComplexD val = s_obj()(spin)(color);
        buf[24 * site + 6 * spin + 2 * color + 0] = val.real();
        buf[24 * site + 6 * spin + 2 * color + 1] = val.imag();
      }
    }
  }
}

template <class FermionField>
inline void lex_buffer_to_fermion(const double *buf, FermionField &grid_field) {
  GridBase *grid = grid_field.Grid();
  Coordinate lc = grid->LocalDimensions();
  int V = local_volume(grid);
  using SiteSpinor = typename FermionField::scalar_object;

  for (int site = 0; site < V; ++site) {
    Coordinate coor;
    lex_index_to_coor(site, lc, coor);
    SiteSpinor s_obj;
    for (int spin = 0; spin < Ns; ++spin) {
      for (int color = 0; color < Nc; ++color) {
        s_obj()(spin)(color) = ComplexD(
            buf[24 * site + 6 * spin + 2 * color + 0],
            buf[24 * site + 6 * spin + 2 * color + 1]);
      }
    }
    pokeLocalSite(s_obj, grid_field, coor);
  }
}

// ----------------------------------------------------------------------------
// Gauge: LatticeGaugeField ↔ 4 per-direction host buffers (lex sites).
// ----------------------------------------------------------------------------

inline void gauge_to_lex_buffers(const LatticeGaugeField &U_grid,
                                 double *buf_per_dir[4]) {
  GridBase *grid = U_grid.Grid();
  Coordinate lc = grid->LocalDimensions();
  int V = local_volume(grid);

  for (int site = 0; site < V; ++site) {
    Coordinate coor;
    lex_index_to_coor(site, lc, coor);
    LatticeGaugeField::vector_object::scalar_object u;  // iVector<iScalar<iMatrix<C,Nc>>,Nd>
    peekLocalSite(u, U_grid, coor);
    for (int mu = 0; mu < Nd; ++mu) {
      for (int row = 0; row < Nc; ++row) {
        for (int col = 0; col < Nc; ++col) {
          ComplexD val = u(mu)()(row, col);
          buf_per_dir[mu][18 * site + (row * Nc + col) * 2 + 0] = val.real();
          buf_per_dir[mu][18 * site + (row * Nc + col) * 2 + 1] = val.imag();
        }
      }
    }
  }
}

inline void lex_buffers_to_gauge(double *const buf_per_dir[4],
                                 LatticeGaugeField &U_grid) {
  GridBase *grid = U_grid.Grid();
  Coordinate lc = grid->LocalDimensions();
  int V = local_volume(grid);

  for (int site = 0; site < V; ++site) {
    Coordinate coor;
    lex_index_to_coor(site, lc, coor);
    LatticeGaugeField::vector_object::scalar_object u;
    for (int mu = 0; mu < Nd; ++mu) {
      for (int row = 0; row < Nc; ++row) {
        for (int col = 0; col < Nc; ++col) {
          u(mu)()(row, col) = ComplexD(
              buf_per_dir[mu][18 * site + (row * Nc + col) * 2 + 0],
              buf_per_dir[mu][18 * site + (row * Nc + col) * 2 + 1]);
        }
      }
    }
    pokeLocalSite(u, U_grid, coor);
  }
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
  int V_eo = V / 2;
  for (int site = 0; site < V; ++site) {
    Coordinate coor;
    lex_index_to_coor(site, lc, coor);
    int parity = site_parity(coor);
    int cb_site = site >> 1;
    int dst_site = parity * V_eo + cb_site;
    std::memcpy(&dst[dst_site * per_site_doubles],
                &src[site * per_site_doubles],
                per_site_doubles * sizeof(double));
  }
}

inline void eo_to_lex_permute(const double *src, double *dst,
                              int V, int per_site_doubles,
                              const Coordinate &lc) {
  assert((lc[0] & 1) == 0 && "x dimension must be even for EO ordering");
  int V_eo = V / 2;
  for (int site = 0; site < V; ++site) {
    Coordinate coor;
    lex_index_to_coor(site, lc, coor);
    int parity = site_parity(coor);
    int cb_site = site >> 1;
    int src_site = parity * V_eo + cb_site;
    std::memcpy(&dst[site * per_site_doubles],
                &src[src_site * per_site_doubles],
                per_site_doubles * sizeof(double));
  }
}

}  // namespace Quda
NAMESPACE_END(Grid);
