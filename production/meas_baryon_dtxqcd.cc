// meas_baryon_dtxqcd: baryon propagator via the diquark-quark construction
// of DTXQCD v2 (dtxqcd_v2.tex Eq 29-36).
//
// Constructs the baryon field B(x) = ε^{ijk} d_iso^{ij}(x) q^k(x) and the
// baryon propagator S^{(B)}_{fsf's'}(t; p=0) by inverting the plain Wilson
// Dirac operator against a custom wall source built from the isosinglet
// diquark aux field d_iso(x',0) at t=0:
//
//   η^{k'}_{β=s}(x', 0) = d_iso^{i'j'}(x', 0) · ε^{i'j'k'}
//   ψ = D^{-1} η                                                (one CG)
//   S^{(B)}(t; p=0) = (1/V_3) Σ_{x} ε^{ijk} d_iso^{ij}(x,t) · ψ^k_s(x,t)
//
// Mass-degenerate Nf=2 gives a flavor factor 2.  Positive-parity spin sum
// {s=1,2} needs 2 CG inversions per cfg (per source momentum p=0).  Output
// is per-t baryon propagator + spin-averaged correlator.
//
// d_iso = (1/√2) ε^{ab} d^{ij}_{ab} (flavor antisymmetric — the standard
// nucleon "good diquark").
//
// Usage:
//   ./meas_baryon_dtxqcd <traj>
//
// Env (in addition to LATT, LAMBDA_DTXQCD, MASS_LIGHT_DTXQCD from params.h):
//   CFG_DIR, DATA_DIR : override I/O directories
//   CG_TOL (1e-10), CG_MAX (10000)

#include "params.h"
#include "dtxqcd_meas_helper.h"
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/qcd/utils/CovariantSmearing.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TXQCDProduction;
using namespace Grid;

// Build the colour-only "isosinglet diquark" lattice field
//   d_iso^{ij}(x) = (1/√2) Σ_{a,b} ε^{ab} d^{ij}_{ab}(x)
// from the 6x6 CF Hermitian d.  Output is a Lattice<iScalar<iScalar<iMatrix<vComplex,Nc>>>>
// (a colour matrix per site).  ε^{ab} for Nf=2 = +1 (a=0,b=1) and -1 (a=1,b=0),
// zero diagonal.
inline LatticeColourMatrix
DtxqcdIsoSingletDiquark(const LatticeDtxqcdD &dF) {
  LatticeColourMatrix out(dF.Grid());
  out = Zero();
  static_assert(DtxqcdNf == 2, "isosinglet diquark code path assumes Nf=2");
  const ComplexD inv_sqrt2 = ComplexD(1.0 / std::sqrt(2.0), 0.0);
  autoView(dv,  dF,  CpuRead);
  autoView(ov,  out, CpuWrite);
  thread_for(ss, dF.Grid()->oSites(), {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        // ε^{01}=+1 → +d_{01}; ε^{10}=−1 → −d_{10}
        auto v = dv[ss]()(0, 1)(i, j) - dv[ss]()(1, 0)(i, j);
        ov[ss]()()(i, j) = inv_sqrt2 * v;
      }
    }
  });
  return out;
}

// Build the custom baryon wall source at t=0 for spin index s∈{0,1}.
// At each spatial site x', the source has:
//   - spin component s nonzero
//   - colour vector η^{k'} = Σ_{i',j'} d_iso^{i'j'}(x',0) · ε^{i'j'k'}
// where ε^{i'j'k'} is the standard 3x3 Levi-Civita symbol.

// Helper: build ε^{ijk} (d_iso^{ij})*(x) colour vector and restrict to t_src.
// For the BARYON SOURCE side we use the complex conjugate of the diquark:
// in the doubled DTXQCD M48, the anti-quark coupling sits in the LL block as
// c·conj(d), so B̄ = ε^{ijk} conj(d)^{ij} q̄^k brings down conj(d_iso) at
// the source.  d_iso is color-antisymmetric (color-antisym in both Hermitian
// and complex-symm conventions); adj(d_iso) = −conj(d_iso) on antisymmetric
// matrices, so previously using adj here flipped the overall correlator
// sign.  Switched to conjugate() so the central value matches the physical
// convention (the extracted baryon mass is unchanged by this sign flip).
//
// `conj_field` controls the conjugation:
//   false: ε^{ijk} d_iso^{ij}(x)    -- use at sink (Eq 36)
//   true : ε^{ijk} (d_iso^{ij})*(x) -- use at source (corresponds to conj(d) in B̄)
inline LatticeColourVector
DiquarkEpsilonWall(const LatticeColourMatrix &d_iso, int t_src,
                   bool conj_field = false) {
  GridBase *g = d_iso.Grid();
  LatticeColourMatrix d_use = conj_field
                                  ? LatticeColourMatrix(conjugate(d_iso))
                                  : d_iso;
  LatticeColourVector wall(g);
  wall = Zero();
  {
    // Scope the views so they CLOSE before the where()/LatticeCoordinate
    // collectives below (which open their own view on `wall`).  Holding an
    // autoView across a same-lattice collective trips GRID_ASSERT
    // AccCache.cpuLock==0 on the non-unified GPU memory manager (this assert
    // is CPU-lenient, which is why the bug stayed latent until the first GPU
    // run of the baryon measurement).
    autoView(dv, d_use, CpuRead);
    autoView(wv, wall,  CpuWrite);
    thread_for(ss, g->oSites(), {
      auto d = dv[ss]()();
      wv[ss]()()(0) = d(1, 2) - d(2, 1);
      wv[ss]()()(1) = d(2, 0) - d(0, 2);
      wv[ss]()()(2) = d(0, 1) - d(1, 0);
    });
  }
  if (t_src >= 0) {
    LatticeInteger t_coord(g);
    LatticeCoordinate(t_coord, Nd - 1);
    LatticeColourVector zero_cv(g);
    zero_cv = Zero();
    wall = where(t_coord == Integer(t_src), wall, zero_cv);
  }
  return wall;
}

// Apply the positive-parity spin-(up|down) projector
//   P₊(s) ≡ (1+γ_4)/2 · (1 ± i γ_3 γ_5)/2
// to a LatticeFermion.  Basis-independent, rank-1 ([γ_4, γ_3γ_5] = 0).
//
// P₋ HANDLING (the uniform-fill trap, found by the NEGPAR gauge gate
// 2026-07-11): the wall source fills ALL 4 spinor components equally, and a
// uniform spinor is a PURE P₊ eigenvector — P₋ annihilates it identically
// (||η||²=0, and the analogous sink sum u†P₋ψ ≡ 0).  So P₋ objects are NOT
// built by passing (1−γ_4)/2 here.  Instead use the γ₅ dressing
//   γ₅ P₊(−s) = P₋(s) γ₅  :
//   source  η₋(s) = γ₅ · [P₊(−s) η_raw]     (γ₅ AFTER the projection)
//   sink    ψ₋(s) = P₊(−s) · (γ₅ ψ)         (γ₅ BEFORE the projection)
// — exact P₋(s) rank-1 pairing, amplitude-matched to the P₊ convention
// (γ₅ unitary), basis-independent.  See the par<0 branches at the call
// sites; this helper itself is P₊-only.
inline LatticeFermion
ApplyPosParitySpinProj(const LatticeFermion &in, int sign) {
  Gamma g4(Gamma::Algebra::GammaT);
  Gamma g3(Gamma::Algebra::GammaZ);
  Gamma g5(Gamma::Algebra::Gamma5);
  // Positive parity (1+γ_4)/2
  LatticeFermion pp = 0.5 * (in + g4 * in);
  // Spin (1 ± iγ_3γ_5)/2 applied to the positive-parity part
  LatticeFermion g3g5_pp = g3 * (g5 * pp);
  ComplexD i_sign = ComplexD(0.0, (RealD)sign);
  return 0.5 * (pp + i_sign * g3g5_pp);
}

// Build a baryon wall source at time t_src and project onto the
// positive-parity spin-up (sign=+1) or spin-down (sign=-1) state.  Populate
// ALL 4 spinor components with the wall colour-vector at t=t_src, then
// apply the rank-1 P_+ P_{↑/↓} projector.  This is basis-independent: the
// projector picks out the right 1-dim subspace regardless of whether |β⟩
// for any single β happens to have overlap with it.
//
// `conj_wall` picks the block:
//   true  (UPPER, default): B̄_U = ε (d_iso)* q̄  → wall from conj(d_iso).
//     Gauge: ε eats two Ω*'s and emits Ω ⇒ η fundamental, matching ψ_u → Ωη.
//   false (LOWER): B̄_L = ε d_iso ψ̄_l → wall from d_iso UNconjugated.
//     Gauge: [εd] is antifundamental (Ω*), matching the charge-conjugate
//     block ψ_l → Ω*ψ_l.  The exact mirror of the upper case.
inline void BuildBaryonWallSourceProj(LatticeFermion &eta,
                                       const LatticeColourMatrix &d_iso,
                                       int spin_sign, int t_src,
                                       bool conj_wall = true, int par = +1) {
  GridBase *g = d_iso.Grid();
  LatticeColourVector wall = DiquarkEpsilonWall(d_iso, t_src,
                                                 /*conj_field=*/conj_wall);
  LatticeFermion eta_raw(g);
  eta_raw = Zero();
  for (int beta = 0; beta < Ns; ++beta) pokeSpin(eta_raw, wall, beta);
  if (par > 0) {
    eta = ApplyPosParitySpinProj(eta_raw, spin_sign);
  } else {
    // P₋(s) source via the γ₅ dressing: η₋(s) = γ₅ [P₊(−s) η_raw].
    // (A direct (1−γ₄)/2 on the uniform fill is IDENTICALLY ZERO — see
    // ApplyPosParitySpinProj's comment.)
    Gamma g5(Gamma::Algebra::Gamma5);
    LatticeFermion tmp = ApplyPosParitySpinProj(eta_raw, -spin_sign);
    eta = g5 * tmp;
  }
}

// Sink contraction with positive-parity spin-(up|down) projection.  Uses
//   S^{(B)}(t) = Σ_x ε^{ijk} d_iso^{ij}(x, t) · Σ_β [P_+ ψ]^k_β(x, t)
// per Eq 36 — the sink weight is ε d_iso UNCONJUGATED (the source side
// carries the d† factor matching B̄ at t_src).
//
// GAUGE-INVARIANCE FIX (2026-07-11): Grid's localInnerProduct CONJUGATES its
// first argument, so passing the unconjugated wall computed ε (d_iso)* ψ —
// d* at BOTH ends.  Rep theory: [ε d] is colour-ANTIfundamental (ε eats two
// Ω's, emits Ω*), so Σ (εd)·ψ is the singlet; conj(εd)·ψ is fund⊗fund =
// Ω^T Ω ≠ 1 ⇒ gauge-VARIANT ⇒ ensemble-ZERO by Elitzur.  Confirmed by the
// GAUGE_TWIST=1 test: the old uu correlator moved O(1) under a random Ω(x)
// (the λ=4 42-cfg ensemble null).  Fix: build the wall from conj(d_iso) so
// localInnerProduct's conjugation restores ε d_iso ψ = the invariant Eq 36.
//
// `conj_wall` picks the block (mirror of the source):
//   true  (UPPER): sink weight ε d_iso (antifund) · ψ_u (fund)      = singlet
//   false (LOWER): sink weight ε (d_iso)* (fund·Ω) · ψ_l (Ω*)      = singlet
// In both cases the wall passed here is the CONJUGATE of the physical sink
// weight, because localInnerProduct conjugates it back.
inline std::vector<ComplexD>
BaryonSinkContractProj(const LatticeColourMatrix &d_iso,
                       const LatticeFermion &psi, int spin_sign,
                       bool conj_wall = true, int par = +1) {
  GridBase *g = d_iso.Grid();
  LatticeColourVector wall = DiquarkEpsilonWall(d_iso, /*t_src=*/-1,
                                                 /*conj_field=*/conj_wall);
  // P₋ sink via the γ₅ dressing: contract u†P₊(−s)(γ₅ψ) = (γ₅P₊(−s)u)†ψ =
  // the exact conjugate of the P₋ source vector (see the source comment) —
  // a plain P₋ projection here would pair against the P₊-pure uniform sum
  // and vanish identically.
  LatticeFermion psi_in(g);
  if (par > 0) {
    psi_in = psi;
  } else {
    Gamma g5(Gamma::Algebra::Gamma5);
    psi_in = g5 * psi;
  }
  LatticeFermion psi_proj =
      ApplyPosParitySpinProj(psi_in, par > 0 ? spin_sign : -spin_sign);
  // Contract: at each site, sum over the 4 spinor components of
  //   conj(wall^k) · ψ_proj^k_β  → ε^{ijk} d_iso^{ij}(x) · Σ_β ψ_proj^k_β(x)
  LatticeComplex Cn(g);
  Cn = Zero();
  for (int beta = 0; beta < Ns; ++beta) {
    LatticeColourVector psi_b = peekSpin(psi_proj, beta);
    Cn = Cn + localInnerProduct(wall, psi_b);
  }
  // sliceSum over t-direction
  std::vector<TComplex> sl;
  sliceSum(Cn, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

// FREE_FIELD validation helper: replace a lattice field by its t=0 slice
// broadcast across ALL timeslices (mask t=0, then sum the T time-shifts) ->
// time-translation-invariant.  With U=I this makes SB_2d[t_src][t_sink] depend
// only on (t_sink - t_src), so the correlator is clean+real and fwd vs bwd
// must agree (the sharp APBC-sign check).
template <class Field>
static void BroadcastTimeZero(Field &f, int T) {
  Grid::GridBase *g = f.Grid();
  Grid::LatticeInteger tc(g);
  Grid::LatticeCoordinate(tc, Grid::Nd - 1);
  Field zero(g); zero = Grid::Zero();
  Field masked = Grid::where(tc == Grid::Integer(0), f, zero);
  Field acc(g); acc = Grid::Zero();
  for (int t = 0; t < T; ++t)
    acc = acc + Grid::Cshift(masked, Grid::Nd - 1, -t);
  f = acc;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: meas_baryon_dtxqcd <traj>" << std::endl;
    return 1;
  }
  int traj = std::atoi(argv[1]);

  Coordinate latt = lattice_size();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  std::string cfg_dir = dtxqcd_cfg_dir();
  if (const char *d = std::getenv("CFG_DIR"); d && *d) cfg_dir = std::string(d);
  std::string data_dir = dtxqcd_data_dir();
  if (const char *d = std::getenv("DATA_DIR"); d && *d) data_dir = std::string(d);
  mkdir_p(data_dir);

  const RealD lam   = lambda_dtxqcd;
  const RealD m     = mass_light_dtxqcd;
  // 1e-6 default (2026-07-06): plenty for correlator-level precision; the
  // old 1e-10 spent most of the CG in the convergence tail.  Override CG_TOL.
  const RealD tol   = TXQCDProduction::detail::env_real("CG_TOL", 1e-6);
  const int   cgmax = TXQCDProduction::detail::env_int("CG_MAX", 100000);
  int T = latt[Nd - 1];
  RealD V3 = 1.0;
  for (int mu = 0; mu < Nd - 1; ++mu) V3 *= (RealD)latt[mu];

  std::cout << GridLogMessage << "======== meas_baryon_dtxqcd ========" << std::endl;
  std::cout << GridLogMessage << "  traj    = " << traj << std::endl;
  std::cout << GridLogMessage << "  lambda  = " << lam << std::endl;
  std::cout << GridLogMessage << "  mass    = " << m << std::endl;
  std::cout << GridLogMessage << "  lattice = " << latt[0] << "x" << latt[1]
            << "x" << latt[2] << "x" << latt[3] << "  T=" << T
            << "  V_3=" << V3 << std::endl;
  std::cout << GridLogMessage << "  t_src   = ALL (translation averaged)" << std::endl;
  std::cout << GridLogMessage << "  cfg dir = " << cfg_dir << std::endl;
  std::cout << GridLogMessage << "  out dir = " << data_dir << std::endl;
  std::cout << GridLogMessage << "====================================" << std::endl;

  DTXQCDField U(&Grid);
  // FREE_FIELD=1: SELF-CONTAINED sign/convention validation (no config load,
  // so it runs at ANY lattice -> use a small LATT + CPU for a fast check).
  // Cold U=I + aux DRAWN in-code, then made time-translation invariant
  // (broadcast the t=0 slice across all time).  Then SB_2d[t_src][t_sink]
  // depends only on (t_sink - t_src): the correlator is CLEAN + REAL and fwd
  // vs bwd (and uu vs ll) MUST agree -> the sharp check of the APBC index maps
  // + the ll charge-conj dressing.
  if (TXQCDProduction::detail::env_int("FREE_FIELD", 0) != 0) {
    U = Zero();
    LatticeColourMatrix id(&Grid); id = 1.0;
    for (int mu = 0; mu < Nd; ++mu) PokeIndex<LorentzIndex>(U.U, id, mu);
    DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lam);
    BroadcastTimeZero(U.sigma, T); BroadcastTimeZero(U.pi, T);
    BroadcastTimeZero(U.d, T);     BroadcastTimeZero(U.n, T);
    BroadcastTimeZero(U.s, T);     BroadcastTimeZero(U.p, T);
    std::cout << GridLogMessage << "[FREE_FIELD] cold U=I + drawn time-constant "
              << "aux (expect clean real correlator, fwd==bwd, uu~ll)"
              << std::endl;
  } else {
    DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                    cfg_dir + "/ckpoint_lat",
                                    cfg_dir + "/ckpoint_rng", traj);
  }

  // Optional random gauge twist (GAUGE_TWIST=1): the gauge-invariance gate.
  // plaq, ||η||², and every correctly-joined correlator must be identical
  // (to CG tol) with and without the twist.  See dtxqcd_meas_helper.h.
  DtxqcdMaybeGaugeTwist(U, pRNG);

  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);

  // Build the isosinglet diquark colour field once for this cfg.
  LatticeColourMatrix d_iso = DtxqcdIsoSingletDiquark(U.d);

  // Optional Gaussian wavefunction smearing on source + sink fermions.
  // Equivalent to smearing the diquark wall via the gauge-covariant identity
  // ⟨smeared d^{ij} | ψ^k⟩ = ⟨d^{ij} | smeared ψ^k⟩ (for spatial-only smearing).
  // SMEAR_WIDTH=0 (default) disables smearing.
  const RealD smear_width =
      TXQCDProduction::detail::env_real("SMEAR_WIDTH", 0.0);
  const int   smear_niter =
      TXQCDProduction::detail::env_int("SMEAR_NITER", 30);
  const bool do_smear = (smear_width > 0.0 && smear_niter > 0);
  // Extract gauge links for the covariant Laplacian used in GaussianSmear.
  std::vector<LatticeColourMatrix> Umu_links(Nd, U.U.Grid());
  if (do_smear) {
    for (int mu = 0; mu < Nd; ++mu)
      Umu_links[mu] = PeekIndex<LorentzIndex>(U.U, mu);
    std::cout << GridLogMessage << "[baryon smear] width=" << smear_width
              << " niter=" << smear_niter
              << " (spatial Gaussian, dir orthog=" << (Nd - 1) << ")"
              << std::endl;
  } else {
    std::cout << GridLogMessage << "[baryon smear] disabled "
              << "(SMEAR_WIDTH=0 or SMEAR_NITER=0)" << std::endl;
  }

  // Build the FULL DOUBLED DTXQCD Wilson-clover Dirac operator on the
  // STOUT-SMEARED gauge + aux fields, at the PRODUCTION csw — i.e. the exact
  // operator the sea quark saw during HMC.  This includes the flavour-dependent
  // σ^{ij}_{ab} insertion in the diagonal block and the d, n diquark coupling
  // between upper (q) and lower (q^C) doubled blocks.  The u and d quark
  // propagators are GENUINELY DIFFERENT cfg-by-cfg because σ^{ij}_{0,b} ≠
  // σ^{ij}_{1,b} on each cfg even though their ensemble averages are equal by
  // isospin symm.
  //
  // (Previously this used csw=0 on the THIN unsmeared gauge — correct only for
  // the old csw=0 scout ensembles, wrong for the csw=1.2493 stout-smeared
  // production line.)  The solve routes through the same QUDA full-volume M
  // operator the non-EO HMC uses (DTXQCD_MEAS_QUDA=1 default; =0 → Grid-CPU
  // reference).  See dtxqcd_meas_helper.h.
  // CSW override (default = production csw).  Set CSW=0 for the FREE_FIELD
  // sign check: at U=I the large time-constant aux + negative mass makes the
  // clover site-block non-HPD (QUDA CloverInvert NaNs); csw=0 removes the
  // clover term and the sign relations don't depend on it.
  const RealD csw_meas = TXQCDProduction::detail::env_real("CSW", csw);
  DtxqcdMeasSolver solver(Grid, RBGrid, U, m, csw_meas, tol, cgmax);

  // Loop over (t_src ∈ {0,…,T-1}) × (flavor ∈ {u, d}) × (spin_sign ∈ {+1, -1}).
  // For each: build a wall fermion source η at t_src with positive-parity
  // spin-(up|down) projector applied.  Embed η into the upper block of a
  // DTXQCDFermionDoubled at flavour slot f (all other slots zero), solve
  // the doubled Dirac equation M^†M ψ = M^† η, extract upper.f[f] of the
  // solution, and contract at the sink with the spin-projector and the
  // diquark·ε wall.  Output S_B^{(f,σ)}(t_sink; t_src) → translation
  // average over t_src.  Cost: 2×2×T = 32 doubled CG solves per cfg.
  //
  // SB_2d[f][s_idx][t_src][t_sink] indexes by f ∈ {0=u,1=d}, s_idx ∈ {0=↑,1=↓}.
  const int NF = 2, NS = 2;
  const int spin_signs[NS] = {+1, -1};  // ↑, ↓
  // MEAS_BARYON_NTSRC: number of (strided) source times used for the
  // translation average.  Default = T (every time slice).  A handful of
  // well-separated t_src already gives most of the noise gain and cuts the
  // solve count from 4·T to 4·NTSRC — this is also the fast path for the
  // QUDA/CPU parity + timing validation.
  int n_tsrc = TXQCDProduction::detail::env_int("MEAS_BARYON_NTSRC", T);
  if (n_tsrc <= 0 || n_tsrc > T) n_tsrc = T;
  std::vector<int> tsrc_list;
  for (int j = 0; j < n_tsrc; ++j) tsrc_list.push_back((j * T) / n_tsrc);
  // MEAS_BARYON_LL=1: ALSO measure the same-block ll baryon (lower-block
  // source + lower sink) — doubles the solve count.
  //
  // NOTE (2026-07-11): the d-wall conjugation per block is FORCED by gauge
  // invariance (source: upper εd*, lower εd; sink weight: upper εd, lower
  // εd*).  The swapped-conjugation "independent antibaryon" pairing was
  // tested and FALSIFIED by the GAUGE_TWIST gate — gauge-variant ⇒
  // Elitzur-zero (see commit a51e5dc5; beware: its contact terms make the
  // raw magnitudes LOOK signal-scale).  uu+ll exhausts the operator family;
  // the physical antibaryon is the bwd fold of the same SB matrix.
  const bool do_ll =
      TXQCDProduction::detail::env_int("MEAS_BARYON_LL", 0) != 0;
  const int NB = do_ll ? 2 : 1;
  // MEAS_BARYON_NEGPAR=1: ALSO run P₋ (negative-parity) source projections —
  // doubles the solves.  Channel content per (projector, fold direction):
  //   P₊ fwd fold = NUCLEON forward                        → "..._fwd"
  //   P₊ bwd fold = parity-partner (N*) tower              → "..._negpar_fwd"
  //   P₋ bwd fold = T-REVERSED NUCLEON (the genuine anti-  → "..._bwd"
  //                 baryon sample; the connected script's
  //                 nucl_fbavg pairing via nucl_neg)
  //   P₋ fwd fold = direct negative-parity channel         → "..._negpar_bwd"
  // (2026-07-11 renaming: the P₊ backward fold was previously misnamed
  // "bwd"; it is the OPPOSITE-parity spectrum, not the antinucleon.)
  const bool do_pneg =
      TXQCDProduction::detail::env_int("MEAS_BARYON_NEGPAR", 0) != 0;
  const int NP = do_pneg ? 2 : 1;
  const int pars[2] = {+1, -1};
  std::cout << GridLogMessage << "[baryon] n_tsrc=" << n_tsrc << " of T=" << T
            << "  (" << (NF * NS * n_tsrc * NB * NP) << " doubled solves"
            << (NB == 2 ? ", uu+ll" : ", uu only")
            << (do_pneg ? ", P+ and P-" : ", P+ only") << ")" << std::endl;
  // SB raw (t_src, t_sink) matrix per (parity × block) × flavour × spin,
  // channel index ch = parity_index * NB + blk.
  //
  // blk 0 = uu: UPPER-block source (wall ε conj(d)), contract psi.upper with
  //             sink weight ε d.
  // blk 1 = ll: LOWER-block source (wall ε d), contract psi.lower with sink
  //             weight ε conj(d) — the exact charge-conjugate mirror,
  //             SAME-block S_ll (Test_dtxqcd_baryon_cross_block: grading-even,
  //             ||S_ll||=||S_uu||; [CCPAR] C-dressing = parity flip, so the
  //             P₊-projected ll carries the P₋ face of uu = complementary
  //             info; alignment vs uu resolved by the saved fwd/bwd folds).
  //             Costs a SECOND solve per (t_src,flavor,spin) — env-gated.
  //
  // The pre-2026-07-11 code took "ll" from psi.lower of an UPPER source —
  // that is the CROSS propagator S_lu: grading-ODD in (d,n) (gate [G2]) ⇒
  // ensemble-zero.  Never the same-block ll baryon.
  std::vector<std::vector<std::vector<std::vector<std::vector<ComplexD>>>>>
      SB_2d(NP * NB,
        std::vector<std::vector<std::vector<std::vector<ComplexD>>>>(
        NF, std::vector<std::vector<std::vector<ComplexD>>>(
          NS, std::vector<std::vector<ComplexD>>(T, std::vector<ComplexD>(T)))));
  for (int t_src : tsrc_list) {
    for (int fi = 0; fi < NF; ++fi) {
      for (int si = 0; si < NS; ++si) {
        // One solve per active (parity, block): blk 0 sources the UPPER
        // doubled block (uu), blk 1 the LOWER (ll) — each contracted at its
        // OWN block's sink with the mirrored conjugation and the SAME parity
        // projector as its source (see BaryonSinkContractProj).
        for (int pi = 0; pi < NP; ++pi) {
        for (int blk = 0; blk < NB; ++blk) {
          const bool upper = (blk == 0);
          const int  par   = pars[pi];
          const int  ch    = pi * NB + blk;
          // Build raw single-flavour fermion source (plain LatticeFermion):
          LatticeFermion eta_f(&Grid);
          BuildBaryonWallSourceProj(eta_f, d_iso, spin_signs[si], t_src,
                                    /*conj_wall=*/upper, par);
          // Apply Gaussian wavefunction smearing to the source (= smearing
          // the diquark source field; commutes with spatial integration).
          if (do_smear) {
            CovariantSmearing<PeriodicGimplR>::GaussianSmear(
                Umu_links, eta_f, smear_width, smear_niter, Nd - 1);
          }
          RealD nrm = norm2(eta_f);
          std::cout << GridLogMessage << "[baryon] t_src=" << t_src
                    << " flavor=" << (fi == 0 ? "u" : "d")
                    << " spin=" << (si == 0 ? "↑" : "↓")
                    << " blk=" << (upper ? "uu" : "ll")
                    << " par=" << (par > 0 ? "+" : "-")
                    << " ||η_f||² = " << nrm << std::endl;
          if (nrm < 1e-28) {
            std::cerr << "FATAL: zero baryon source (projector/fill "
                      << "mismatch?) — refusing to CG on 0/0" << std::endl;
            abort();
          }
          // Embed into the doubled fermion at the block's flavour slot fi.
          DTXQCDFermionDoubled eta(&Grid), psi(&Grid);
          eta = Zero();
          if (upper) eta.upper.f[fi] = eta_f;
          else       eta.lower.f[fi] = eta_f;
          // Solve doubled M psi = eta on the full volume (helper: M†M CG).
          solver.Solve(eta, psi);
          // Contract the SAME block at the sink (same flavour slot fi):
          // psi.upper from an upper source is S_uu; psi.lower from a lower
          // source is S_ll (both grading-EVEN; the cross blocks are odd).
          LatticeFermion psi_q = upper ? psi.upper.f[fi] : psi.lower.f[fi];
          // Apply Gaussian wavefunction smearing to the sink quark field.
          if (do_smear) {
            CovariantSmearing<PeriodicGimplR>::GaussianSmear(
                Umu_links, psi_q, smear_width, smear_niter, Nd - 1);
          }
          SB_2d[ch][fi][si][t_src] =
              BaryonSinkContractProj(d_iso, psi_q, spin_signs[si],
                                     /*conj_wall=*/upper, par);
        }
        }
      }
    }
  }

  // Translation-averaged correlators per (flavor, spin) channel:
  //   C_B^{(f,σ)}(Δt) = (1/T) Σ_{t_src} S_B^{(f,σ)}(t_src+Δt; t_src)
  // and the spin-averaged + flavor-averaged total per Eq 36.
  // Wall-wall normalization 1/V_3², times the SD/Fierz lambda^4 (two diquark
  // aux fields, each == qq/lambda^2, the diquark analog of the aux pion's
  // lambda^4) so the folded correlators are directly comparable IN AMPLITUDE
  // to the QCD nucleon (verified 2026-07-09: aux*lambda^4 = C_nucleon up to a
  // factor 2 = ref spin-SUM Tr[P+ NNbar] vs our spin-AVERAGE; the factor 2 is
  // NOT applied -- our spin-averaged convention is kept, remember it when
  // comparing to the spin-summed reference).  The raw SB_2d matrices below are
  // left UNSCALED (analyze_baryon_vs_ref.py applies lambda^4 itself).
  const RealD lam4 = lam * lam * lam * lam;
  const RealD norm = lam4 / (V3 * V3 * (RealD)n_tsrc);
  const bool apbc = true;   // baryon operator: antiperiodic time (chroma conv)

  // Forward + backward (time-reversed) translation-averaged correlators per
  // (block, flavor, spin), with the SAME APBC index maps as the connected
  // baryon script (meas_conn_qcd.cc shift_fwd_apbc / shift_tr_chroma):
  //   fwd: t_lat = (dt + t_src) % T,   sign = -1 if (dt + t_src) >= T
  //   bwd: t_lat = (t_src - dt + T)%T, sign = -1 if (dt - t_src)  > 0
  // The bwd is the reflection of the SAME forward SB matrix (no extra solves):
  // it reads the backward-propagating (anti-baryon, APBC-wrapped) piece.
  //
  // APBC ALIGNMENT SIGN (fixed 2026-07-09, synthetic_fold_test): for an
  // antiperiodic baryon g(-dt) = -g(T-dt), so the raw reflection gives
  // bwd = -(aligned forward).  We fold in the extra -1 for backward so the
  // saved bwd component is the PHYSICALLY-ALIGNED backward baryon (same
  // positive-decaying signal as fwd) -- matching the connected script's
  // nucl_fbavg = 1/2 (fwd - nucl_neg[T-t]).  Without this the fwd+bwd average
  // partially CANCELS (was the ~0.45 coherence on the real config).
  auto fold = [&](int ch, int fi, int si, bool backward) {
    std::vector<ComplexD> C(T, 0.0);
    const RealD align = backward ? -1.0 : +1.0;   // APBC fwd/bwd alignment
    for (int dt = 0; dt < T; ++dt) {
      for (int t_src : tsrc_list) {
        int t_lat; RealD sgn;
        if (!backward) {
          t_lat = (dt + t_src) % T;
          sgn = (apbc && (dt + t_src) >= T) ? -1.0 : +1.0;
        } else {
          t_lat = ((t_src - dt) % T + T) % T;
          sgn = (apbc && (dt - t_src) > 0) ? -1.0 : +1.0;
        }
        C[dt] += sgn * SB_2d[ch][fi][si][t_src][t_lat];
      }
      C[dt] *= ComplexD(align * norm, 0.0);
    }
    return C;
  };

  // Folded components per (parity, block, flavor, spin, fold direction).
  // CHANNEL NAMING (see the MEAS_BARYON_NEGPAR comment): the suffix labels
  // physical content + time-orientation sample:
  //   fwd        = P₊ fwd fold  = nucleon, forward sample
  //   bwd        = P₋ bwd fold  = nucleon, T-reversed sample   (NEGPAR only)
  //   negpar_fwd = P₊ bwd fold  = parity-partner (N*) tower
  //   negpar_bwd = P₋ fwd fold  = negative-parity, direct      (NEGPAR only)
  // Averages: baryon_nucleon_avg = mean of the uu NUCLEON components
  // (fwd 4, plus bwd 4 when NEGPAR); baryon_negpar_avg likewise for the
  // parity-partner channel; ll components (gate mode) kept separate.
  const char *blk_nm[2] = {"uu", "ll"};
  const char *fl_nm[NF]  = {"p", "n"};
  const char *sp_nm[NS]  = {"up", "dn"};
  std::vector<std::pair<std::string, std::vector<ComplexD>>> comps;
  std::vector<ComplexD> C_nuc_avg(T, 0.0), C_neg_avg(T, 0.0), C_ll_avg(T, 0.0);
  int n_nuc = 0, n_neg = 0, n_llc = 0;
  for (int pi = 0; pi < NP; ++pi)
    for (int blk = 0; blk < NB; ++blk)
      for (int fi = 0; fi < NF; ++fi)
        for (int si = 0; si < NS; ++si)
          for (int bw = 0; bw < 2; ++bw) {
            const int ch = pi * NB + blk;
            auto C = fold(ch, fi, si, bw != 0);
            // nucleon = P₊ fwd fold OR P₋ bwd fold; the other two are negpar.
            const bool nucleon = (pi == 0) ? (bw == 0) : (bw == 1);
            const char *suffix =
                (pi == 0) ? (bw ? "negpar_fwd" : "fwd")
                          : (bw ? "bwd" : "negpar_bwd");
            std::string nm = std::string("baryon_") + blk_nm[blk] + "_" +
                             fl_nm[fi] + "_" + sp_nm[si] + "_" + suffix;
            if (blk == 0) {
              auto &acc = nucleon ? C_nuc_avg : C_neg_avg;
              int  &cnt = nucleon ? n_nuc : n_neg;
              for (int t = 0; t < T; ++t) acc[t] += C[t];
              ++cnt;
            } else {
              for (int t = 0; t < T; ++t) C_ll_avg[t] += C[t];
              ++n_llc;
            }
            comps.emplace_back(nm, std::move(C));
          }
  for (int t = 0; t < T; ++t) C_nuc_avg[t] *= ComplexD(1.0 / n_nuc, 0.0);
  for (int t = 0; t < T; ++t) C_neg_avg[t] *= ComplexD(1.0 / n_neg, 0.0);
  if (n_llc)
    for (int t = 0; t < T; ++t) C_ll_avg[t] *= ComplexD(1.0 / n_llc, 0.0);

  // Back-compat: baryon_total = forward uu, spin+flavor summed (old Eq-36 form,
  // now with the APBC sign the previous version omitted at the wrap).
  std::vector<ComplexD> C_B_total(T, 0.0);
  for (int fi = 0; fi < NF; ++fi)
    for (int si = 0; si < NS; ++si) {
      auto C = fold(0, fi, si, false);
      for (int t = 0; t < T; ++t) C_B_total[t] += C[t];
    }

  std::string outfile = data_dir + "/baryon_dtxqcd_" + std::to_string(traj) + ".h5";
  if (Grid.IsBoss()) {
    Hdf5Writer wr(outfile);
    write(wr, "baryon_nucleon_avg", C_nuc_avg);  // uu nucleon channel average
    write(wr, "baryon_negpar_avg",  C_neg_avg);  // uu parity-partner average
    if (do_ll) write(wr, "baryon_ll_avg", C_ll_avg);  // ll gate diagnostic
    for (auto &c : comps) write(wr, c.first, c.second);   // all components
    write(wr, "baryon_total", C_B_total);         // back-compat (fwd uu sum)
    // Raw 2D per-(t_src,t_sink) matrices for diagnostics.  P₊ names are
    // unchanged (baryon_SB_<blk>_<fl>_<sp>_2d); P₋ gets a _pneg tag.
    auto flatten_2d = [&](int ch, int fi, int si) {
      std::vector<ComplexD> out(T * T);
      for (int ts = 0; ts < T; ++ts)
        for (int tk = 0; tk < T; ++tk)
          out[ts * T + tk] = SB_2d[ch][fi][si][ts][tk];
      return out;
    };
    for (int pi = 0; pi < NP; ++pi)
      for (int blk = 0; blk < NB; ++blk)
        for (int fi = 0; fi < NF; ++fi)
          for (int si = 0; si < NS; ++si)
            write(wr, std::string("baryon_SB_") + blk_nm[blk] + "_" +
                      fl_nm[fi] + "_" + sp_nm[si] +
                      (pi ? "_pneg" : "") + "_2d",
                  flatten_2d(pi * NB + blk, fi, si));
    write(wr, "traj",    traj);
    write(wr, "lambda",  lam);
    write(wr, "mass",    m);
    write(wr, "plaq",    plaq);
    write(wr, "T",       T);
    write(wr, "V3",      V3);
  }
  std::cout << GridLogMessage << "Written " << outfile << std::endl;
  Grid_finalize();
#ifdef GRID_HAVE_QUDA
  // The output is on disk and Grid_finalize() has run MPI_Finalize.  Skip the
  // C++ static-destructor / __cxa_finalize chain: QUDA's own static
  // ColorSpinorField destructors double-free at process exit (the persistent
  // Style-C device state; cf. commit 6023005f which fixed the analogous
  // DTXQCDMpcOpQUDA destructor crash, but QUDA's internal static teardown is
  // out of our reach).  _Exit is safe here — nothing left to flush.
  std::fflush(nullptr);
  std::_Exit(0);
#endif
  return 0;
}
