#pragma once
// GPU-side pack for QUDA force-input buffers.
//
// Replaces the per-RHS host-side
//   LatticeFermion scaled = scale * fld;
//   unvectorizeToLexOrdArray(sv, scaled);
//   memcpy(buf, sv.data(), V_eo * 24 * sizeof(double));
// pattern (~1.5 s/eval at 16³×48 with 40 RHS × 4 buffers) with a single
// accelerator_for per RHS that writes lex-ordered doubles directly into a
// device-resident scratch buffer.  Caller does a SINGLE acceleratorCopyFromDevice
// at the end, replacing 160 individual D2H syncs.
//
// The lex-index table maps (oSite, lane) → lex index within the RB
// half-volume.  Built once per RB grid (cached by caller).

#include <Grid/GridCore.h>

NAMESPACE_BEGIN(Grid);
namespace Quda {

// Build a lex-index table for an RB (or full) grid: lex_table[s*Nsimd+lane]
// = lex index of the global site corresponding to (oSite=s, simd-lane).
// Mirrors the pattern in TXQCDWilsonCloverFermionEO.h::ApplyMooeeInvCublas
// step (c) but exposed as a free function.
inline void BuildLexTable(GridBase *grid, deviceVector<int> &lex_table_dev) {
  using vobj_default = vSpinColourVector;  // any SIMD vobj — we just need Nsimd
  constexpr int Nsimd = vobj_default::Nsimd();
  uint64_t oSites = grid->oSites();
  const int ndim = grid->Nd();

  std::vector<Coordinate> icoor(Nsimd);
  for (int lane = 0; lane < Nsimd; ++lane) {
    icoor[lane].resize(ndim);
    grid->iCoorFromIindex(icoor[lane], lane);
  }

  std::vector<int> lex_host(oSites * Nsimd);
  thread_for(oidx, oSites, {
    Coordinate ocoor(ndim), lcoor(ndim);
    grid->oCoorFromOindex(ocoor, oidx);
    for (int lane = 0; lane < Nsimd; ++lane) {
      for (int mu = 0; mu < ndim; ++mu)
        lcoor[mu] = ocoor[mu] + grid->_rdimensions[mu] * icoor[lane][mu];
      int lex;
      Lexicographic::IndexFromCoor(lcoor, lex, grid->_ldimensions);
      lex_host[oidx * Nsimd + lane] = lex;
    }
  });

  lex_table_dev.resize(oSites * Nsimd);
  acceleratorCopyToDevice(&lex_host[0], &lex_table_dev[0],
                          oSites * Nsimd * sizeof(int));
}

// GPU pack: writes one LatticeFermion's scalar-object lex layout into a
// device-resident double buffer, applying `scale` on the fly.
//
//   dst_dev[lex * 24 + (alpha*Nc + c)*2 + 0] = scale * Re v(alpha,c)
//   dst_dev[lex * 24 + (alpha*Nc + c)*2 + 1] = scale * Im v(alpha,c)
//
// Layout matches std::complex<double>'s (re, im) packing inside
// iSpinColourVector<ComplexD>, which is what QUDA's CPU-CSF reads after
// its memcpy from the host buffer in computeCloverWilsonForceWithSchurFields
// / computeCloverSigmaForceWithSchurFields.
template <class vobj>
inline void GpuPackFermionRbLex(const Lattice<vobj> &fld,
                                RealD scale,
                                double *dst_dev,
                                const int *lex_table_dev) {
  GridBase *grid = fld.Grid();
  uint64_t oSites = grid->oSites();
  constexpr int Nsimd = vobj::Nsimd();
  constexpr int Ns_ = Ns;
  constexpr int Nc_ = Nc;

  autoView(in_v, fld, AcceleratorRead);
  accelerator_for(s, oSites, Nsimd, {
    int simt_lane = static_cast<int>(lane);
    int lex = lex_table_dev[s * Nsimd + simt_lane];
    double *dst = &dst_dev[lex * 24];
    auto v = in_v[s];
    for (int alpha = 0; alpha < Ns_; ++alpha) {
      for (int c = 0; c < Nc_; ++c) {
        // getlane returns scalar ComplexD from the SIMD-laned register.
        auto z = getlane(v()(alpha)(c), simt_lane);
        dst[(alpha * Nc_ + c) * 2 + 0] = scale * z.real();
        dst[(alpha * Nc_ + c) * 2 + 1] = scale * z.imag();
      }
    }
  });
}

// Build per-(oSite,lane) table of EO source-site indices for a FULL grid.
// For lex coord (x,y,z,t), eo_site = parity * V_eo + (lex>>1) where
// parity = (x+y+z+t)&1.  This matches eo_to_lex_permute's mapping.
inline void BuildEoTable(GridBase *full_grid,
                         const Coordinate &lc,   // local dims
                         deviceVector<int> &eo_table_dev) {
  using vobj_default = vSpinColourVector;
  constexpr int Nsimd = vobj_default::Nsimd();
  uint64_t oSites = full_grid->oSites();
  const int ndim = full_grid->Nd();
  const int Lx = lc[0], Ly = lc[1], Lz = lc[2];
  const int V = lc[0] * lc[1] * lc[2] * lc[3];
  const int V_eo = V / 2;

  std::vector<Coordinate> icoor(Nsimd);
  for (int lane = 0; lane < Nsimd; ++lane) {
    icoor[lane].resize(ndim);
    full_grid->iCoorFromIindex(icoor[lane], lane);
  }

  std::vector<int> eo_host(oSites * Nsimd);
  thread_for(oidx, oSites, {
    Coordinate ocoor(ndim), lcoor(ndim);
    full_grid->oCoorFromOindex(ocoor, oidx);
    for (int lane = 0; lane < Nsimd; ++lane) {
      for (int mu = 0; mu < ndim; ++mu)
        lcoor[mu] = ocoor[mu] + full_grid->_rdimensions[mu] * icoor[lane][mu];
      int lex;
      Lexicographic::IndexFromCoor(lcoor, lex, full_grid->_ldimensions);
      int parity = (lcoor[0] + lcoor[1] + lcoor[2] + lcoor[3]) & 1;
      int eo_site = parity * V_eo + (lex >> 1);
      eo_host[oidx * Nsimd + lane] = eo_site;
    }
  });

  eo_table_dev.resize(oSites * Nsimd);
  acceleratorCopyToDevice(&eo_host[0], &eo_table_dev[0],
                          oSites * Nsimd * sizeof(int));
}

// GPU unpack: read MILC RECONSTRUCT_10 momentum buffer (device-resident)
// and WRITE (overwriting) into a Grid LatticeGaugeField (SIMD layout) with
// scale.  The kernel writes pure outputs — caller does
// `dst = dst + this_field` via Grid expression to accumulate.
//
// Layout of mom_buf_dev (per (eo_site, μ) block of 10 doubles):
//   [r01, i01, r02, i02, r12, i12, a0, a1, a2, _]
// expanded to anti-Hermitian 3×3:
//   [   i*a0,   r01+i*i01,   r02+i*i02 ]
//   [-r01+i*i01,   i*a1,     r12+i*i12 ]
//   [-r02+i*i02, -r12+i*i12,   i*a2    ]
inline void GpuUnpackMomToGauge(const double *mom_buf_dev,
                                const int *eo_table_dev,
                                LatticeGaugeField &out,
                                RealD scale) {
  GridBase *grid = out.Grid();
  uint64_t oSites = grid->oSites();
  using vobj = typename LatticeGaugeField::vector_object;
  constexpr int Nsimd = vobj::Nsimd();
  constexpr int Nd_ = Nd;
  constexpr int Nc_ = Nc;

  autoView(out_v, out, AcceleratorWrite);
  accelerator_for(s, oSites, Nsimd, {
    int simt_lane = static_cast<int>(lane);
    int eo_site = eo_table_dev[s * Nsimd + simt_lane];
    for (int mu = 0; mu < Nd_; ++mu) {
      const double *m = &mom_buf_dev[(eo_site * Nd_ + mu) * 10];
      double r01 = m[0], i01 = m[1];
      double r02 = m[2], i02 = m[3];
      double r12 = m[4], i12 = m[5];
      double a0  = m[6], a1  = m[7], a2 = m[8];
      ComplexD M[3][3] = {
        {ComplexD(0.0,    a0), ComplexD( r01, i01), ComplexD( r02, i02)},
        {ComplexD(-r01,  i01), ComplexD(0.0,  a1),  ComplexD( r12, i12)},
        {ComplexD(-r02,  i02), ComplexD(-r12, i12), ComplexD(0.0,  a2)}
      };
      // Direct putlane onto the SIMD register — no local-copy race.
      for (int r = 0; r < Nc_; ++r) {
        for (int c = 0; c < Nc_; ++c) {
          putlane(out_v[s](mu)()(r, c),
                  ComplexD(scale, 0.0) * M[r][c], simt_lane);
        }
      }
    }
  });
}

// Accumulate (scaled) RB-grid field into full-grid field at the right CB
// parity, in a SINGLE GPU kernel.  Replaces the
//   tmp = Zero();  setCheckerboard(tmp, rb);  full = full + scale*tmp;
// chain (3 kernels — and the standard setCheckerboard is CPU/host-routed,
// causing D2H+H2D every call).  Only writes the cb sites of `full`; sites
// of the other parity are unchanged.
template <class vobj>
inline void AccumulateRbScaledToFull(Lattice<vobj> &full,
                                     RealD scale,
                                     const Lattice<vobj> &rb,
                                     int checker_dim_half = 0) {
  GridBase *full_grid = full.Grid();
  GridBase *rb_grid   = rb.Grid();
  int cb = rb.Checkerboard();
  uint64_t oSites = full_grid->oSites();
  constexpr int Nsimd = vobj::Nsimd();

  Coordinate rdim_full   = full_grid->_rdimensions;
  Coordinate rdim_half   = rb_grid->_rdimensions;
  Coordinate ostride_half = rb_grid->_ostride;
  Coordinate cdim_mask   = rb_grid->_checker_dim_mask;
  const int ndim = rb_grid->_ndimension;

  autoView(rb_v,   rb,   AcceleratorRead);
  autoView(full_v, full, AcceleratorWrite);

  accelerator_for(ss, oSites, Nsimd, {
    Coordinate coor;
    Lexicographic::CoorFromIndex(coor, ss, rdim_full);

    // Determine the parity at this full-grid oSite.
    int linear = 0;
    for (int d = 0; d < ndim; ++d) {
      if (cdim_mask[d]) linear += coor[d];
    }
    int cbos = linear & 0x1;

    if (cbos == cb) {
      // Map full oSite → corresponding RB oSite (mirrors acceleratorSetCheckerboard).
      int ssh = 0;
      for (int d = 0; d < ndim; ++d) {
        if (d == checker_dim_half) ssh += ostride_half[d] * ((coor[d] / 2) % rdim_half[d]);
        else                         ssh += ostride_half[d] * (coor[d] % rdim_half[d]);
      }
      auto cur = full_v(ss);
      auto add = rb_v(ssh);
      cur = cur + scale * add;
      coalescedWrite(full_v[ss], cur);
    }
  });
}

// Fused t-tensor aux-force kernel for TXQCD: collapses the 6-iteration
// (μ < ν) loop in AccumulateAuxForce into ONE accelerator_for.
//
// For each (μ, ν) with μ<ν, the per-site contribution is the antisymmetric
// (μ,ν)/(ν,μ) accumulation of  ak * F_t^{μν}_{ij}  where
//
//   G_t^{μν}_{ij} = sum_{a,α,β}  conj(Y_a(α,i)) · (iσ_{μν})_{αβ} · X_a(β,j)
//   F_t^{μν}_{ij} = -(G_t^{μν,T} + G_t^{μν,*})_{ij}
//
// Passes a flat array of 6 spin matrices (one per (μ,ν) pair, ordered
// (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)).  Caller flattens via Op(α,β).
//
// Walks full-grid oSites; reads RB Y, X via the standard Grid SIMD coord
// math (mirrors AccumulateRbScaledToFull).  Writes only at cb-parity sites.
inline void FusedTAccumulate(LatticeTField &dst_t_full,
                             RealD ak,
                             const TXQCDFermionNf &Y_rb,
                             const TXQCDFermionNf &X_rb,
                             const std::array<ComplexD, 6 * Ns * Ns> &isig_flat,
                             int checker_dim_half = 0) {
  GridBase *full_grid = dst_t_full.Grid();
  GridBase *rb_grid   = Y_rb.Grid();
  int cb = Y_rb.f[0].Checkerboard();
  uint64_t oSites_full = full_grid->oSites();
  constexpr int Nsimd_T = LatticeTField::vector_object::Nsimd();

  Coordinate rdim_full   = full_grid->_rdimensions;
  Coordinate rdim_half   = rb_grid->_rdimensions;
  Coordinate ostride_half = rb_grid->_ostride;
  Coordinate cdim_mask   = rb_grid->_checker_dim_mask;
  const int ndim = rb_grid->_ndimension;

  autoView(Y0v, Y_rb.f[0], AcceleratorRead);
  autoView(Y1v, Y_rb.f[1], AcceleratorRead);
  autoView(X0v, X_rb.f[0], AcceleratorRead);
  autoView(X1v, X_rb.f[1], AcceleratorRead);
  autoView(dst, dst_t_full, AcceleratorWrite);

  accelerator_for(ss, oSites_full, Nsimd_T, {
    Coordinate coor;
    Lexicographic::CoorFromIndex(coor, ss, rdim_full);
    int linear = 0;
    for (int d = 0; d < ndim; ++d) {
      if (cdim_mask[d]) linear += coor[d];
    }
    int cbos = linear & 0x1;
    if (cbos != cb) return;

    int ssh = 0;
    for (int d = 0; d < ndim; ++d) {
      if (d == checker_dim_half) ssh += ostride_half[d] * ((coor[d] / 2) % rdim_half[d]);
      else                         ssh += ostride_half[d] * (coor[d] % rdim_half[d]);
    }

    auto Y0 = Y0v(ssh);
    auto Y1 = Y1v(ssh);
    auto X0 = X0v(ssh);
    auto X1 = X1v(ssh);
    auto d_lane = dst(ss);

    // For each of 6 (μ,ν) pairs, compute per-site G_t^{μν}_{ij} bilinear,
    // project to Hermitian-anti F_t^{μν}, and accumulate antisymmetrically.
    // Inline (μ,ν) tables for device code.
    int mu_arr[6] = {0, 0, 0, 1, 1, 2};
    int nu_arr[6] = {1, 2, 3, 2, 3, 3};
    for (int p = 0; p < 6; ++p) {
      int mu_l = mu_arr[p];
      int nu_l = nu_arr[p];
      // Gather this pair's 4×4 spin matrix from the flat array (offset p*16).
      const ComplexD *opp = &isig_flat[p * Ns * Ns];

      // Compute 3×3 G_{ij}: for each (i,j), sum over (a, α, β).
      decltype(conjugate(Y0()(0)(0)) * X0()(0)(0)) G_per_lane[Nc][Nc];
      for (int i = 0; i < Nc; ++i)
        for (int j = 0; j < Nc; ++j) zeroit(G_per_lane[i][j]);

      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          for (int a = 0; a < TxqcdNf; ++a) {
            auto &Y_a = (a == 0) ? Y0 : Y1;
            auto &X_a = (a == 0) ? X0 : X1;
            for (int alpha = 0; alpha < Ns; ++alpha) {
              for (int beta = 0; beta < Ns; ++beta) {
                ComplexD op_ab = opp[alpha * Ns + beta];
                G_per_lane[i][j] = G_per_lane[i][j]
                  + conjugate(Y_a()(alpha)(i)) * op_ab * X_a()(beta)(j);
              }
            }
          }
        }
      }

      // F_{ij} = -(G^T + G^*)_{ij}, scaled by ak; accumulate antisym.
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          auto F_ij = -(G_per_lane[j][i] + conjugate(G_per_lane[i][j]));
          auto v = ak * F_ij;
          d_lane()(mu_l, nu_l)(i, j) = d_lane()(mu_l, nu_l)(i, j) + v;
          d_lane()(nu_l, mu_l)(i, j) = d_lane()(nu_l, mu_l)(i, j) - v;
        }
      }
    }
    coalescedWrite(dst[ss], d_lane);
  });
}

}  // namespace Quda
NAMESPACE_END(Grid);
