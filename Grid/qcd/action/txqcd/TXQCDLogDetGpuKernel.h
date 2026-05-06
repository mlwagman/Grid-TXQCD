#pragma once
// Phase J: GPU kernel that replaces TXQCDLogDetCloverEOAction's per-site CPU
// thread_for + Eigen 24×24 inverse + 5 trace loops + vectorize +
// CPU-routed setCheckerboard chain.
//
// Reads a precomputed M^{-1} buffer (column-major 24×24 ComplexD per even
// site, populated by TXQCDWilsonCloverFermionEO::ImportFields) and extracts
// the five aux-force traces into RB-Even Lattice<.> outputs.  Caller pushes
// to full-grid dSdU.* via Quda::AccumulateRbScaledToFull.
//
// For csw≠0, the clover_sigma RB ColourMatrix outputs are derived from F_t_e
// in a thin second pass (clover_sigma[k] = (i·csw/2) * F_t_e[μ_k, ν_k]
// effectively).  Keeping that as a separate Grid expression avoids capturing
// 6 LatticeViews in the main accelerator_for body.

#include <Grid/qcd/action/txqcd/AuxFieldTypes.h>
#include <Grid/qcd/action/txqcd/TXQCDSiteMatrix.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>

NAMESPACE_BEGIN(Grid);
namespace TxqcdLogDet {

// Variant that takes raw device pointers (M_inv_dev, lex_table_dev) directly
// — no TXQCDWilsonCloverFermionEO required.  Used by TXQCDLogDetCloverEOAction
// after a self-managed CPU inversion + acceleratorCopyToDevice upload.
inline void ExtractTracesFromBuffers(const ComplexD *M_inv_dev_ptr,
                                     const int *lex_dev_ptr,
                                     LatticeSigmaField &F_sig_e,
                                     LatticePiField   &F_pi_e,
                                     LatticeSFieldC   &F_s_e,
                                     LatticePFieldC   &F_p_e,
                                     LatticeTField    &F_t_e);

inline void ExtractTraces(TXQCDWilsonCloverFermionEO &eop,
                          LatticeSigmaField &F_sig_e,
                          LatticePiField   &F_pi_e,
                          LatticeSFieldC   &F_s_e,
                          LatticePFieldC   &F_p_e,
                          LatticeTField    &F_t_e) {
  ExtractTracesFromBuffers(&eop.MdevForCb(Even)[0],
                           &eop.LexTableForCb(Even)[0],
                           F_sig_e, F_pi_e, F_s_e, F_p_e, F_t_e);
}

inline void ExtractTracesFromBuffers(const ComplexD *M_inv_dev,
                                     const int *lex_dev,
                                     LatticeSigmaField &F_sig_e,
                                     LatticePiField   &F_pi_e,
                                     LatticeSFieldC   &F_s_e,
                                     LatticePFieldC   &F_p_e,
                                     LatticeTField    &F_t_e) {
  using SMU = TXQCDSiteMatrixUtil;
  constexpr int N = SMU::kDim;       // 24
  constexpr int Nsf = TxqcdNf;
  constexpr int Nsp = Ns;            // 4
  constexpr int Ncc = Nc;            // 3

  F_sig_e.Checkerboard() = Even;
  F_pi_e.Checkerboard()  = Even;
  F_s_e.Checkerboard()   = Even;
  F_p_e.Checkerboard()   = Even;
  F_t_e.Checkerboard()   = Even;

  // Capture spin tables (γ5 + 6 iσ_{μν} for μ<ν) as flat std::array passed
  // by value into the lambda (small POD, fits in CUDA constant memory window).
  SMU::SpinMatrices sm;
  std::array<ComplexD, Nsp * Nsp> g5_flat{};
  for (int a = 0; a < Nsp; ++a)
    for (int b = 0; b < Nsp; ++b)
      g5_flat[a * Nsp + b] = sm.gamma5(a, b);

  std::array<ComplexD, 6 * Nsp * Nsp> isig_flat{};
  static const int MU_PAIR[6] = {0, 0, 0, 1, 1, 2};
  static const int NU_PAIR[6] = {1, 2, 3, 2, 3, 3};
  for (int p = 0; p < 6; ++p) {
    for (int a = 0; a < Nsp; ++a)
      for (int b = 0; b < Nsp; ++b)
        isig_flat[p * Nsp * Nsp + a * Nsp + b] =
            sm.isigma[MU_PAIR[p]][NU_PAIR[p]](a, b);
  }

  GridBase *rbgrid = F_sig_e.Grid();
  uint64_t oSites = rbgrid->oSites();
  using vobj_T = typename LatticeTField::vector_object;
  constexpr int Nsimd = vobj_T::Nsimd();

  autoView(sigv, F_sig_e, AcceleratorWrite);
  autoView(piv,  F_pi_e,  AcceleratorWrite);
  autoView(sv,   F_s_e,   AcceleratorWrite);
  autoView(pv,   F_p_e,   AcceleratorWrite);
  autoView(tv,   F_t_e,   AcceleratorWrite);

  accelerator_for(s, oSites, Nsimd, {
    int simt_lane = static_cast<int>(lane);
    int lex = lex_dev[s * Nsimd + simt_lane];
    const ComplexD *Inv = &M_inv_dev[lex * 576];  // column-major 24×24

    // Inv element (row=ra, col=rb) = Inv[rb*24 + ra]
    auto Mij = [&](int ra, int rb) -> ComplexD { return Inv[rb * N + ra]; };

    // ---- sigma force: F_{ab} = -Σ_{α,i} M^{-1}_{(a,α,i),(b,α,i)} ----
    for (int a = 0; a < Nsf; ++a) {
      for (int b = 0; b < Nsf; ++b) {
        ComplexD val = ComplexD(0.0, 0.0);
        for (int alpha = 0; alpha < Nsp; ++alpha) {
          for (int i = 0; i < Ncc; ++i) {
            int ra = a * Nsp * Ncc + alpha * Ncc + i;
            int rb = b * Nsp * Ncc + alpha * Ncc + i;
            val = val + Mij(ra, rb);
          }
        }
        putlane(sigv[s]()()(a, b), -val, simt_lane);
      }
    }

    // ---- pi force: F_{ab} = -Σ_{α,β,i} γ5(β,α) · M^{-1}_{(a,α,i),(b,β,i)} ----
    for (int a = 0; a < Nsf; ++a) {
      for (int b = 0; b < Nsf; ++b) {
        ComplexD val = ComplexD(0.0, 0.0);
        for (int alpha = 0; alpha < Nsp; ++alpha) {
          for (int beta = 0; beta < Nsp; ++beta) {
            ComplexD g5 = g5_flat[beta * Nsp + alpha];
            for (int i = 0; i < Ncc; ++i) {
              int ra = a * Nsp * Ncc + alpha * Ncc + i;
              int rb = b * Nsp * Ncc + beta  * Ncc + i;
              val = val + g5 * Mij(ra, rb);
            }
          }
        }
        putlane(piv[s]()()(a, b), -val, simt_lane);
      }
    }

    // ---- s force (color, ⨉ inv_sqrt2) ----
    const RealD inv_sqrt2 = 0.7071067811865475;
    for (int i = 0; i < Ncc; ++i) {
      for (int j = 0; j < Ncc; ++j) {
        ComplexD val = ComplexD(0.0, 0.0);
        for (int a = 0; a < Nsf; ++a) {
          for (int alpha = 0; alpha < Nsp; ++alpha) {
            int ri = a * Nsp * Ncc + alpha * Ncc + i;
            int rj = a * Nsp * Ncc + alpha * Ncc + j;
            val = val + Mij(ri, rj);
          }
        }
        putlane(sv[s]()()(i, j), ComplexD(-inv_sqrt2 * val.real(),
                                          -inv_sqrt2 * val.imag()),
                simt_lane);
      }
    }

    // ---- p force (color, γ5 spin, ⨉ inv_sqrt2) ----
    for (int i = 0; i < Ncc; ++i) {
      for (int j = 0; j < Ncc; ++j) {
        ComplexD val = ComplexD(0.0, 0.0);
        for (int a = 0; a < Nsf; ++a) {
          for (int alpha = 0; alpha < Nsp; ++alpha) {
            for (int beta = 0; beta < Nsp; ++beta) {
              ComplexD g5 = g5_flat[beta * Nsp + alpha];
              int ri = a * Nsp * Ncc + alpha * Ncc + i;
              int rj = a * Nsp * Ncc + beta  * Ncc + j;
              val = val + g5 * Mij(ri, rj);
            }
          }
        }
        putlane(pv[s]()()(i, j), ComplexD(-inv_sqrt2 * val.real(),
                                          -inv_sqrt2 * val.imag()),
                simt_lane);
      }
    }

    // ---- t force (color × Lorentz tensor, antisym μν, ⨉ iσ_{μν}) ----
    // Initialize all 16 (μ,ν) entries to zero (μ=ν stays zero by antisymmetry).
    for (int mu_l = 0; mu_l < Nd; ++mu_l)
      for (int nu_l = 0; nu_l < Nd; ++nu_l)
        for (int i = 0; i < Ncc; ++i)
          for (int j = 0; j < Ncc; ++j)
            putlane(tv[s]()(mu_l, nu_l)(i, j), ComplexD(0.0, 0.0), simt_lane);

    int mu_arr[6] = {0, 0, 0, 1, 1, 2};
    int nu_arr[6] = {1, 2, 3, 2, 3, 3};

    for (int p = 0; p < 6; ++p) {
      int mu_l = mu_arr[p];
      int nu_l = nu_arr[p];
      const ComplexD *isig = &isig_flat[p * Nsp * Nsp];
      for (int i = 0; i < Ncc; ++i) {
        for (int j = 0; j < Ncc; ++j) {
          ComplexD val = ComplexD(0.0, 0.0);
          for (int a = 0; a < Nsf; ++a) {
            for (int alpha = 0; alpha < Nsp; ++alpha) {
              for (int beta = 0; beta < Nsp; ++beta) {
                ComplexD isigv = isig[beta * Nsp + alpha];
                int ri = a * Nsp * Ncc + alpha * Ncc + i;
                int rj = a * Nsp * Ncc + beta  * Ncc + j;
                val = val + isigv * Mij(ri, rj);
              }
            }
          }
          ComplexD vneg(-val.real(), -val.imag());
          ComplexD vpos( val.real(),  val.imag());
          putlane(tv[s]()(mu_l, nu_l)(i, j), vneg, simt_lane);
          putlane(tv[s]()(nu_l, mu_l)(i, j), vpos, simt_lane);
        }
      }
    }
  });
}

// Derive the 6 clover_sigma RB ColourMatrix outputs from F_t_e (already
// written above) and the clover coefficient.  Per the original CPU code:
//   clover_sigma[k][x](i,j) = -i*(csw/2) · val   where val = -t_force[μ,ν](i,j)
//                           = +i*(csw/2) · t_force[FmnIndex(μ,ν)] entry.
// The dependence is purely site-local so a small kernel per (μ,ν) pair fits.
inline void DeriveCloverSigma(const LatticeTField &F_t_e,
                              RealD csw,
                              std::array<LatticeColourMatrix, 6> &clover_sigma_e) {
  using SMU = TXQCDSiteMatrixUtil;
  static const int MU_PAIR[6] = {0, 0, 0, 1, 1, 2};
  static const int NU_PAIR[6] = {1, 2, 3, 2, 3, 3};

  GridBase *rbgrid = F_t_e.Grid();
  uint64_t oSites = rbgrid->oSites();
  constexpr int Nsimd = LatticeColourMatrix::vector_object::Nsimd();
  constexpr int Ncc = Nc;
  const ComplexD scale(0.0, 0.5 * csw);  // i*(csw/2)

  autoView(tv, F_t_e, AcceleratorRead);

  for (int p = 0; p < 6; ++p) {
    const int mu_l = MU_PAIR[p];
    const int nu_l = NU_PAIR[p];
    const int k_idx = SMU::FmnIndex(mu_l, nu_l);
    clover_sigma_e[k_idx].Checkerboard() = Even;
    autoView(cv, clover_sigma_e[k_idx], AcceleratorWrite);
    accelerator_for(s, oSites, Nsimd, {
      int simt_lane = static_cast<int>(lane);
      for (int i = 0; i < Ncc; ++i) {
        for (int j = 0; j < Ncc; ++j) {
          ComplexD t_val = getlane(tv[s]()(mu_l, nu_l)(i, j), simt_lane);
          // F_t_e at (μ<ν, μ<ν entry) = -val (per kernel above);
          //   so val = -t_val, and clover_sigma = -i*(csw/2)*val = +i*(csw/2)*t_val.
          ComplexD cv_val = scale * t_val;
          putlane(cv[s]()()(i, j), cv_val, simt_lane);
        }
      }
    });
  }
}

}  // namespace TxqcdLogDet
NAMESPACE_END(Grid);
