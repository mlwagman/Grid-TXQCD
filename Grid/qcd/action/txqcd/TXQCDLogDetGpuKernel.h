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
#include <Grid/qcd/action/txqcd/TxqcdTMode.h>
#include <Grid/qcd/action/txqcd/TXQCDSiteMatrix.h>

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

  // Mode-dependent factors mirroring TXQCDSiteMatrix::BuildSiteMatrix:
  //   mode A: sigma, pi -> 1            ; s, p -> 1/sqrt(2)
  //   mode B: sigma, pi -> 1/sqrt(2)    ; s, p -> 1
  constexpr RealD sig_pi_factor = TxqcdTIsFlavor ? 0.7071067811865475 : 1.0;
  constexpr RealD s_p_factor    = TxqcdTIsFlavor ? 1.0 : 0.7071067811865475;
  constexpr int   TDim          = TxqcdTDim;  // Nc (mode A) or Nf (mode B)

  accelerator_for(s, oSites, Nsimd, {
    int simt_lane = static_cast<int>(lane);
    int lex = lex_dev[s * Nsimd + simt_lane];
    // Per-site stride is N*N (= kDim²) — was hardcoded 576 (= 24²), silently
    // wrong at Nf=3 where kDim=36 → 1296.  Fixed 2026-06-02.
    const ComplexD *Inv = &M_inv_dev[lex * uint64_t(N) * N];  // column-major N×N

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
        putlane(sigv[s]()()(a, b),
                ComplexD(-sig_pi_factor * val.real(),
                         -sig_pi_factor * val.imag()),
                simt_lane);
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
        putlane(piv[s]()()(a, b),
                ComplexD(-sig_pi_factor * val.real(),
                         -sig_pi_factor * val.imag()),
                simt_lane);
      }
    }

    // ---- s force (color, scaled by s_p_factor) ----
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
        putlane(sv[s]()()(i, j), ComplexD(-s_p_factor * val.real(),
                                          -s_p_factor * val.imag()),
                simt_lane);
      }
    }

    // ---- p force (color, γ5 spin, scaled by s_p_factor) ----
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
        putlane(pv[s]()()(i, j), ComplexD(-s_p_factor * val.real(),
                                          -s_p_factor * val.imag()),
                simt_lane);
      }
    }

    // ---- t force (Lorentz antisym, mode-dependent index structure) ----
    // mode A (color-t): writes tv[s](mu,nu)(i,j) — color matrix per (μ,ν).
    // mode B (flavor-t): writes tv[s](mu,nu)(a,b) within upper Nf x Nf block;
    //   inactive color slots (i,j outside Nf x Nf) are zeroed.
    // Initialize all 16 (μ,ν) entries × Nc x Nc slots to zero.
    for (int mu_l = 0; mu_l < Nd; ++mu_l)
      for (int nu_l = 0; nu_l < Nd; ++nu_l)
        for (int i = 0; i < Ncc; ++i)
          for (int j = 0; j < Ncc; ++j)
            putlane(tv[s]()(mu_l, nu_l)(i, j), ComplexD(0.0, 0.0), simt_lane);

    int mu_arr[6] = {0, 0, 0, 1, 1, 2};
    int nu_arr[6] = {1, 2, 3, 2, 3, 3};

    if (TxqcdTIsFlavor) {  // not constexpr-if: nvcc lambda first-capture restriction
      // Mode B: t indexed by (a,b) flavor, color-diagonal.
      // F_t[μν][a,b] = -Σ_{α,β,i} isig(β,α) · M^{-1}_{(a,α,i),(b,β,i)}
      for (int p_idx = 0; p_idx < 6; ++p_idx) {
        int mu_l = mu_arr[p_idx];
        int nu_l = nu_arr[p_idx];
        const ComplexD *isig = &isig_flat[p_idx * Nsp * Nsp];
        for (int a = 0; a < TDim; ++a) {
          for (int b = 0; b < TDim; ++b) {
            ComplexD val = ComplexD(0.0, 0.0);
            for (int alpha = 0; alpha < Nsp; ++alpha) {
              for (int beta = 0; beta < Nsp; ++beta) {
                ComplexD isigv = isig[beta * Nsp + alpha];
                for (int i = 0; i < Ncc; ++i) {
                  int ri = a * Nsp * Ncc + alpha * Ncc + i;
                  int rj = b * Nsp * Ncc + beta  * Ncc + i;
                  val = val + isigv * Mij(ri, rj);
                }
              }
            }
            ComplexD vneg(-val.real(), -val.imag());
            ComplexD vpos( val.real(),  val.imag());
            putlane(tv[s]()(mu_l, nu_l)(a, b), vneg, simt_lane);
            putlane(tv[s]()(nu_l, mu_l)(a, b), vpos, simt_lane);
          }
        }
      }
    } else {
      // Mode A: t indexed by (i,j) color, flavor-diagonal (sum over flavor).
      for (int p_idx = 0; p_idx < 6; ++p_idx) {
        int mu_l = mu_arr[p_idx];
        int nu_l = nu_arr[p_idx];
        const ComplexD *isig = &isig_flat[p_idx * Nsp * Nsp];
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

// Mode-B helper: extract the color sigma-bilinear matrix
//   Sigma_color[mu,nu](i,j) = -Sum_{a,alpha,beta} isigma_{mu,nu}(beta,alpha)
//                                    * M^{-1}_{(a,alpha,i),(a,beta,j)}
// scaled by clover_coeff = (i*csw/2), and write 6 such color matrices into
// clover_sigma_e[k=FmnIndex(mu,nu)].
//
// In mode A this is equal (after the csw scaling) to DeriveCloverSigma(F_t_e),
// because F_t in mode A IS the color sigma-bilinear.  In mode B, F_t becomes a
// flavor matrix (orthogonal channel), so we need to compute the color
// sigma-bilinear directly from M^{-1}.  See memory/project_t_condensate_low_lambda.md
// for why this matters: the clover gauge force depends on a color trace that
// is decoupled from the flavor-t channel in mode B.
inline void ExtractCloverSigmaColorFromBuffers(
    const ComplexD *M_inv_dev,
    const int *lex_dev,
    RealD csw,
    std::array<LatticeColourMatrix, 6> &clover_sigma_e) {
  using SMU = TXQCDSiteMatrixUtil;
  constexpr int N    = SMU::kDim;
  constexpr int Nsf  = TxqcdNf;
  constexpr int Nsp  = Ns;
  constexpr int Ncc  = Nc;

  SMU::SpinMatrices sm;
  std::array<ComplexD, 6 * Nsp * Nsp> isig_flat{};
  static const int MU_PAIR[6] = {0, 0, 0, 1, 1, 2};
  static const int NU_PAIR[6] = {1, 2, 3, 2, 3, 3};
  for (int p = 0; p < 6; ++p)
    for (int a = 0; a < Nsp; ++a)
      for (int b = 0; b < Nsp; ++b)
        isig_flat[p * Nsp * Nsp + a * Nsp + b] =
            sm.isigma[MU_PAIR[p]][NU_PAIR[p]](a, b);

  // The +i·(csw/2) factor (matches DeriveCloverSigma in mode A: clover_sigma
  // = +i·(csw/2)·F_t, where F_t already carries a sign convention).  Encoded
  // as +i·(csw/2) here together with the (-) sign of the trace formula.
  const ComplexD scale(0.0, 0.5 * csw);

  GridBase *rbgrid = clover_sigma_e[0].Grid();
  uint64_t oSites  = rbgrid->oSites();
  constexpr int Nsimd = LatticeColourMatrix::vector_object::Nsimd();

  // Open six write views (one per Fmn slot)
  autoView(c0v, clover_sigma_e[0], AcceleratorWrite);
  autoView(c1v, clover_sigma_e[1], AcceleratorWrite);
  autoView(c2v, clover_sigma_e[2], AcceleratorWrite);
  autoView(c3v, clover_sigma_e[3], AcceleratorWrite);
  autoView(c4v, clover_sigma_e[4], AcceleratorWrite);
  autoView(c5v, clover_sigma_e[5], AcceleratorWrite);
  for (int k = 0; k < 6; ++k) clover_sigma_e[k].Checkerboard() = Even;

  accelerator_for(s, oSites, Nsimd, {
    int simt_lane = static_cast<int>(lane);
    int lex = lex_dev[s * Nsimd + simt_lane];
    const ComplexD *Inv = &M_inv_dev[lex * uint64_t(N) * N];
    auto Mij = [&](int ra, int rb) -> ComplexD { return Inv[rb * N + ra]; };

    for (int p_idx = 0; p_idx < 6; ++p_idx) {
      const ComplexD *isig = &isig_flat[p_idx * Nsp * Nsp];
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
          // Sigma_color[μν](i,j) = -val (matches F_t sign convention in mode A)
          // clover_sigma = scale * Sigma_color = +(i*csw/2) * (-val)
          ComplexD cs = ComplexD(-scale.real(), -scale.imag());
          // (-(scale.re), -(scale.im)) * val
          ComplexD out(cs.real() * val.real() - cs.imag() * val.imag(),
                       cs.real() * val.imag() + cs.imag() * val.real());
          if (p_idx == 0) putlane(c0v[s]()()(i, j), out, simt_lane);
          if (p_idx == 1) putlane(c1v[s]()()(i, j), out, simt_lane);
          if (p_idx == 2) putlane(c2v[s]()()(i, j), out, simt_lane);
          if (p_idx == 3) putlane(c3v[s]()()(i, j), out, simt_lane);
          if (p_idx == 4) putlane(c4v[s]()()(i, j), out, simt_lane);
          if (p_idx == 5) putlane(c5v[s]()()(i, j), out, simt_lane);
        }
      }
    }
  });
}

// Phase J.3 retry: GPU BuildSiteMatrix.
//
// Reads RB-Even aux Lattice fields (5) + optional 6 Fmn ColourMatrix Lattices
// directly via autoView, builds the 24×24 forward M per site, writes to the
// raw M_fwd_dev_ buffer (column-major, 576 ComplexD per site) indexed by
// lex_dev.  Replaces the host-side thread_for + Eigen BuildSiteMatrix block
// that costs ~143 ms/call (cpu_build) AND eliminates the pickCheckerboard
// + UnvectorizeAux host roundtrip (~234 ms/call) since aux are read directly
// from device-resident Lattice views.
//
// View count: 5 aux (Read) + 6 Fmn (Read when csw≠0) = 11.  Output is a raw
// device pointer (M_fwd_dev), not a Lattice view, so no extra view there.
//
// Convention notes:
//   - Lattice<...> "outer scalar" peels via `()` — sigma_site()()(a,b) etc.
//   - Per-lane access via getlane(scalar_object_element, lane).
//   - Same arithmetic as SMU::BuildSiteMatrix on host; bit-exact within IEEE
//     reassociation when compiled without -ffast-math.
inline void BuildSiteMatrixFromLattice(
    ComplexD *M_fwd_dev,
    const int *lex_dev,
    const std::array<RealD, TxqcdNf> &diag_mass,
    RealD csw,
    const LatticeSigmaField &sigma_e,
    const LatticePiField    &pi_e,
    const LatticeSFieldC    &s_e,
    const LatticePFieldC    &p_e,
    const LatticeTField     &t_e,
    const std::array<LatticeColourMatrix, 6> *fmn_e   // nullptr if csw==0
) {
  using SMU = TXQCDSiteMatrixUtil;
  constexpr int N    = SMU::kDim;       // 24
  constexpr int N2   = N * N;           // 576
  constexpr int Nsf  = TxqcdNf;
  constexpr int Nsp  = Ns;              // 4
  constexpr int Ncc  = Nc;              // 3
  const RealD inv_sqrt2 = 1.0 / std::sqrt(2.0);

  // Flatten γ5 and 6 iσ_{μν} into POD arrays passed by value into the lambda.
  SMU::SpinMatrices sm;
  std::array<ComplexD, Nsp * Nsp> g5_flat{};
  for (int a = 0; a < Nsp; ++a)
    for (int b = 0; b < Nsp; ++b)
      g5_flat[a * Nsp + b] = sm.gamma5(a, b);

  std::array<ComplexD, 6 * Nsp * Nsp> isig_flat{};
  static const int MU_PAIR[6] = {0, 0, 0, 1, 1, 2};
  static const int NU_PAIR[6] = {1, 2, 3, 2, 3, 3};
  for (int p = 0; p < 6; ++p)
    for (int a = 0; a < Nsp; ++a)
      for (int b = 0; b < Nsp; ++b)
        isig_flat[p * Nsp * Nsp + a * Nsp + b] =
            sm.isigma[MU_PAIR[p]][NU_PAIR[p]](a, b);

  std::array<RealD, Nsf> diag_mass_flat = diag_mass;
  const RealD csw_local = csw;
  const bool have_csw = (csw != 0.0) && (fmn_e != nullptr);

  GridBase *rbgrid = sigma_e.Grid();
  uint64_t oSites  = rbgrid->oSites();
  using vobj_T = typename LatticeSigmaField::vector_object;
  constexpr int Nsimd = vobj_T::Nsimd();

  autoView(sigv, sigma_e, AcceleratorRead);
  autoView(piv,  pi_e,    AcceleratorRead);
  autoView(sv,   s_e,     AcceleratorRead);
  autoView(pv,   p_e,     AcceleratorRead);
  autoView(tv,   t_e,     AcceleratorRead);

  // Fmn views — opened only when csw != 0 inside the dispatch below.

  // Mode-dependent factors mirroring TXQCDSiteMatrix::BuildSiteMatrix:
  //   mode A: sigma, pi -> 1            ; s, p -> 1/sqrt(2)
  //   mode B: sigma, pi -> 1/sqrt(2)    ; s, p -> 1
  const RealD sig_pi_factor = TxqcdTIsFlavor ? inv_sqrt2 : 1.0;
  const RealD s_p_factor    = TxqcdTIsFlavor ? 1.0       : inv_sqrt2;
  const int   TDim          = TxqcdTDim;

  if (!have_csw) {
    accelerator_for(s, oSites, Nsimd, {
      int simt_lane = static_cast<int>(lane);
      int lex = lex_dev[s * Nsimd + simt_lane];
      ComplexD *M = &M_fwd_dev[(uint64_t)lex * N2];

      // M = 0
      for (int k = 0; k < N2; ++k) M[k] = ComplexD(0.0, 0.0);

      // Diagonal mass
      for (int a = 0; a < Nsf; ++a)
        for (int alpha = 0; alpha < Nsp; ++alpha)
          for (int i = 0; i < Ncc; ++i) {
            int r = a * Nsp * Ncc + alpha * Ncc + i;
            M[r + r * N] = ComplexD(diag_mass_flat[a], 0.0);
          }

      // sigma + pi block (flavor, color-diagonal, sig_pi_factor)
      for (int a = 0; a < Nsf; ++a) {
        for (int b = 0; b < Nsf; ++b) {
          ComplexD sig_ab = getlane(sigv[s]()()(a, b), simt_lane);
          ComplexD pi_ab  = getlane(piv [s]()()(a, b), simt_lane);
          for (int alpha = 0; alpha < Nsp; ++alpha) {
            for (int beta = 0; beta < Nsp; ++beta) {
              ComplexD g5 = g5_flat[alpha * Nsp + beta];
              for (int i = 0; i < Ncc; ++i) {
                int r = a * Nsp * Ncc + alpha * Ncc + i;
                int c = b * Nsp * Ncc + beta  * Ncc + i;
                if (alpha == beta) M[r + c * N] = M[r + c * N] + sig_pi_factor * sig_ab;
                M[r + c * N] = M[r + c * N] + sig_pi_factor * pi_ab * g5;
              }
            }
          }
        }
      }

      // s, p block (color, flavor-diagonal, s_p_factor)
      for (int a = 0; a < Nsf; ++a) {
        for (int i = 0; i < Ncc; ++i) {
          for (int j = 0; j < Ncc; ++j) {
            ComplexD s_ij = getlane(sv[s]()()(i, j), simt_lane);
            ComplexD p_ij = getlane(pv[s]()()(i, j), simt_lane);
            for (int alpha = 0; alpha < Nsp; ++alpha) {
              for (int beta = 0; beta < Nsp; ++beta) {
                int r = a * Nsp * Ncc + alpha * Ncc + i;
                int c = a * Nsp * Ncc + beta  * Ncc + j;
                if (alpha == beta) M[r + c * N] = M[r + c * N] + s_p_factor * s_ij;
                M[r + c * N] = M[r + c * N] + s_p_factor * p_ij *
                               g5_flat[alpha * Nsp + beta];
              }
            }
          }
        }
      }

      // t block (mode-dependent)
      const int MU_LOC[6] = {0, 0, 0, 1, 1, 2};
      const int NU_LOC[6] = {1, 2, 3, 2, 3, 3};
      if (TxqcdTIsFlavor) {  // not constexpr-if: nvcc lambda first-capture restriction
        // Mode B: t indexed by (a,b) flavor (upper Nf x Nf), color-diagonal.
        for (int a = 0; a < TDim; ++a) {
          for (int b = 0; b < TDim; ++b) {
            for (int alpha = 0; alpha < Nsp; ++alpha) {
              for (int beta = 0; beta < Nsp; ++beta) {
                for (int i = 0; i < Ncc; ++i) {
                  int r = a * Nsp * Ncc + alpha * Ncc + i;
                  int c = b * Nsp * Ncc + beta  * Ncc + i;
                  for (int p_idx = 0; p_idx < 6; ++p_idx) {
                    int mu_l = MU_LOC[p_idx];
                    int nu_l = NU_LOC[p_idx];
                    ComplexD t_ab = getlane(tv[s]()(mu_l, nu_l)(a, b), simt_lane);
                    ComplexD isig = isig_flat[p_idx * Nsp * Nsp + alpha * Nsp + beta];
                    M[r + c * N] = M[r + c * N] + t_ab * isig;
                  }
                }
              }
            }
          }
        }
      } else {
        // Mode A: t indexed by (i,j) color, flavor-diagonal.
        for (int a = 0; a < Nsf; ++a) {
          for (int i = 0; i < Ncc; ++i) {
            for (int j = 0; j < Ncc; ++j) {
              for (int alpha = 0; alpha < Nsp; ++alpha) {
                for (int beta = 0; beta < Nsp; ++beta) {
                  int r = a * Nsp * Ncc + alpha * Ncc + i;
                  int c = a * Nsp * Ncc + beta  * Ncc + j;
                  for (int p_idx = 0; p_idx < 6; ++p_idx) {
                    int mu_l = MU_LOC[p_idx];
                    int nu_l = NU_LOC[p_idx];
                    ComplexD t_ij = getlane(tv[s]()(mu_l, nu_l)(i, j), simt_lane);
                    ComplexD isig = isig_flat[p_idx * Nsp * Nsp + alpha * Nsp + beta];
                    M[r + c * N] = M[r + c * N] + t_ij * isig;
                  }
                }
              }
            }
          }
        }
      }
    });
  } else {
    // csw != 0 — open 6 Fmn views too.
    autoView(f0v, (*fmn_e)[0], AcceleratorRead);
    autoView(f1v, (*fmn_e)[1], AcceleratorRead);
    autoView(f2v, (*fmn_e)[2], AcceleratorRead);
    autoView(f3v, (*fmn_e)[3], AcceleratorRead);
    autoView(f4v, (*fmn_e)[4], AcceleratorRead);
    autoView(f5v, (*fmn_e)[5], AcceleratorRead);
    const ComplexD clover_coeff_re(0.0, 0.5 * csw_local);  // i·(csw/2)

    accelerator_for(s, oSites, Nsimd, {
      int simt_lane = static_cast<int>(lane);
      int lex = lex_dev[s * Nsimd + simt_lane];
      ComplexD *M = &M_fwd_dev[(uint64_t)lex * N2];

      for (int k = 0; k < N2; ++k) M[k] = ComplexD(0.0, 0.0);

      // Diagonal mass
      for (int a = 0; a < Nsf; ++a)
        for (int alpha = 0; alpha < Nsp; ++alpha)
          for (int i = 0; i < Ncc; ++i) {
            int r = a * Nsp * Ncc + alpha * Ncc + i;
            M[r + r * N] = ComplexD(diag_mass_flat[a], 0.0);
          }

      // sigma + pi block (flavor, color-diagonal, sig_pi_factor)
      for (int a = 0; a < Nsf; ++a) {
        for (int b = 0; b < Nsf; ++b) {
          ComplexD sig_ab = getlane(sigv[s]()()(a, b), simt_lane);
          ComplexD pi_ab  = getlane(piv [s]()()(a, b), simt_lane);
          for (int alpha = 0; alpha < Nsp; ++alpha) {
            for (int beta = 0; beta < Nsp; ++beta) {
              ComplexD g5 = g5_flat[alpha * Nsp + beta];
              for (int i = 0; i < Ncc; ++i) {
                int r = a * Nsp * Ncc + alpha * Ncc + i;
                int c = b * Nsp * Ncc + beta  * Ncc + i;
                if (alpha == beta) M[r + c * N] = M[r + c * N] + sig_pi_factor * sig_ab;
                M[r + c * N] = M[r + c * N] + sig_pi_factor * pi_ab * g5;
              }
            }
          }
        }
      }

      // s, p, Fmn block (color, flavor-diagonal; s,p use s_p_factor; Fmn unchanged)
      for (int a = 0; a < Nsf; ++a) {
        for (int i = 0; i < Ncc; ++i) {
          for (int j = 0; j < Ncc; ++j) {
            ComplexD s_ij = getlane(sv[s]()()(i, j), simt_lane);
            ComplexD p_ij = getlane(pv[s]()()(i, j), simt_lane);
            // Read all 6 Fmn entries up front (per-lane scalars).
            ComplexD f_ij[6];
            f_ij[0] = getlane(f0v[s]()()(i, j), simt_lane);
            f_ij[1] = getlane(f1v[s]()()(i, j), simt_lane);
            f_ij[2] = getlane(f2v[s]()()(i, j), simt_lane);
            f_ij[3] = getlane(f3v[s]()()(i, j), simt_lane);
            f_ij[4] = getlane(f4v[s]()()(i, j), simt_lane);
            f_ij[5] = getlane(f5v[s]()()(i, j), simt_lane);
            for (int alpha = 0; alpha < Nsp; ++alpha) {
              for (int beta = 0; beta < Nsp; ++beta) {
                int r = a * Nsp * Ncc + alpha * Ncc + i;
                int c = a * Nsp * Ncc + beta  * Ncc + j;
                if (alpha == beta) M[r + c * N] = M[r + c * N] + s_p_factor * s_ij;
                M[r + c * N] = M[r + c * N] + s_p_factor * p_ij *
                               g5_flat[alpha * Nsp + beta];
                const int MU_LOC[6] = {0, 0, 0, 1, 1, 2};
                const int NU_LOC[6] = {1, 2, 3, 2, 3, 3};
                for (int p_idx = 0; p_idx < 6; ++p_idx) {
                  ComplexD isig = isig_flat[p_idx * Nsp * Nsp + alpha * Nsp + beta];
                  M[r + c * N] = M[r + c * N] + clover_coeff_re * f_ij[p_idx] * isig;
                }
              }
            }
          }
        }
      }

      // t block (mode-dependent index structure, decoupled from clover Fmn)
      const int MU_LOC[6] = {0, 0, 0, 1, 1, 2};
      const int NU_LOC[6] = {1, 2, 3, 2, 3, 3};
      if (TxqcdTIsFlavor) {  // not constexpr-if: nvcc lambda first-capture restriction
        // Mode B: t indexed by (a,b) flavor (upper Nf x Nf), color-diagonal.
        for (int a = 0; a < TDim; ++a) {
          for (int b = 0; b < TDim; ++b) {
            for (int alpha = 0; alpha < Nsp; ++alpha) {
              for (int beta = 0; beta < Nsp; ++beta) {
                for (int i = 0; i < Ncc; ++i) {
                  int r = a * Nsp * Ncc + alpha * Ncc + i;
                  int c = b * Nsp * Ncc + beta  * Ncc + i;
                  for (int p_idx = 0; p_idx < 6; ++p_idx) {
                    int mu_l = MU_LOC[p_idx];
                    int nu_l = NU_LOC[p_idx];
                    ComplexD t_ab = getlane(tv[s]()(mu_l, nu_l)(a, b), simt_lane);
                    ComplexD isig = isig_flat[p_idx * Nsp * Nsp + alpha * Nsp + beta];
                    M[r + c * N] = M[r + c * N] + t_ab * isig;
                  }
                }
              }
            }
          }
        }
      } else {
        // Mode A: t indexed by (i,j) color, flavor-diagonal.
        for (int a = 0; a < Nsf; ++a) {
          for (int i = 0; i < Ncc; ++i) {
            for (int j = 0; j < Ncc; ++j) {
              for (int alpha = 0; alpha < Nsp; ++alpha) {
                for (int beta = 0; beta < Nsp; ++beta) {
                  int r = a * Nsp * Ncc + alpha * Ncc + i;
                  int c = a * Nsp * Ncc + beta  * Ncc + j;
                  for (int p_idx = 0; p_idx < 6; ++p_idx) {
                    int mu_l = MU_LOC[p_idx];
                    int nu_l = NU_LOC[p_idx];
                    ComplexD t_ij = getlane(tv[s]()(mu_l, nu_l)(i, j), simt_lane);
                    ComplexD isig = isig_flat[p_idx * Nsp * Nsp + alpha * Nsp + beta];
                    M[r + c * N] = M[r + c * N] + t_ij * isig;
                  }
                }
              }
            }
          }
        }
      }
    });
  }
}

}  // namespace TxqcdLogDet
NAMESPACE_END(Grid);
