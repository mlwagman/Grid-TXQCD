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
//   P_{±} ≡ (1+γ_4)/2 · (1 ± i γ_3 γ_5)/2
// to a LatticeFermion.  Basis-independent (works in Grid's chiral basis
// as well as Dirac).  [γ_4, γ_3 γ_5] = 0 so the half-projectors commute
// and the product is a rank-1 projector.
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
inline void BuildBaryonWallSourceProj(LatticeFermion &eta,
                                       const LatticeColourMatrix &d_iso,
                                       int spin_sign, int t_src) {
  GridBase *g = d_iso.Grid();
  // Source uses d†_iso (conjugation matches the B̄ creation operator at the
  // source time in ⟨B(t_sink) B̄(t_src)⟩).  Sink uses d_iso as-is (Eq 36).
  LatticeColourVector wall = DiquarkEpsilonWall(d_iso, t_src,
                                                 /*conj_field=*/true);
  LatticeFermion eta_raw(g);
  eta_raw = Zero();
  for (int beta = 0; beta < Ns; ++beta) pokeSpin(eta_raw, wall, beta);
  eta = ApplyPosParitySpinProj(eta_raw, spin_sign);
}

// Sink contraction with positive-parity spin-(up|down) projection.  Uses
//   S^{(B)}(t) = Σ_x ε^{ijk} d_iso^{ij}(x, t) · Σ_β [P_+ ψ]^k_β(x, t)
// per Eq 36.  Sink wall is built from d_iso (NO conjugation; the source
// side carries the d† factor matching B̄ at t_src).
inline std::vector<ComplexD>
BaryonSinkContractProj(const LatticeColourMatrix &d_iso,
                       const LatticeFermion &psi, int spin_sign) {
  GridBase *g = d_iso.Grid();
  LatticeColourVector wall = DiquarkEpsilonWall(d_iso, /*t_src=*/-1,
                                                 /*conj_field=*/false);
  LatticeFermion psi_proj = ApplyPosParitySpinProj(psi, spin_sign);
  // Contract: at each site, sum over the 4 spinor components of
  //   wall^k · ψ_proj^k_β  → ε^{ijk} (d^{ji}_{ba}(x))* · Σ_β ψ_proj^k_β(x)
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
  std::cout << GridLogMessage << "[baryon] n_tsrc=" << n_tsrc << " of T=" << T
            << "  (" << (NF * NS * n_tsrc) << " doubled solves)" << std::endl;
  // SB raw (t_src, t_sink) matrix per BLOCK (0=uu from upper.f, 1=ll from
  // lower.f) × flavour × spin.  The lower block is the charge-conjugate (q^C)
  // theory; at +parity it is a legitimate baryon correlator equal (per config,
  // up to a parity flip inside the ε-contraction) to the upper one -- verified
  // by Test_dtxqcd_baryon_cross_block ([CCPAR], ||uu||==||ll||).  Saving both
  // gives the 8 uu + 8 ll physical copies the user requested; the cross
  // (upper-source→lower-sink) baryon terms ENSEMBLE-VANISH (S_ul odd in d,n),
  // so only the same-block uu/ll carry signal.
  const int NB = 2;
  std::vector<std::vector<std::vector<std::vector<std::vector<ComplexD>>>>>
      SB_2d(NB, std::vector<std::vector<std::vector<std::vector<ComplexD>>>>(
        NF, std::vector<std::vector<std::vector<ComplexD>>>(
          NS, std::vector<std::vector<ComplexD>>(T, std::vector<ComplexD>(T)))));
  for (int t_src : tsrc_list) {
    for (int fi = 0; fi < NF; ++fi) {
      for (int si = 0; si < NS; ++si) {
        // Build raw single-flavour fermion source (plain LatticeFermion):
        LatticeFermion eta_f(&Grid);
        BuildBaryonWallSourceProj(eta_f, d_iso, spin_signs[si], t_src);
        // Apply Gaussian wavefunction smearing to the source (= smearing the
        // diquark source field; commutes with spatial integration).
        if (do_smear) {
          CovariantSmearing<PeriodicGimplR>::GaussianSmear(
              Umu_links, eta_f, smear_width, smear_niter, Nd - 1);
        }
        RealD nrm = norm2(eta_f);
        std::cout << GridLogMessage << "[baryon] t_src=" << t_src
                  << " flavor=" << (fi == 0 ? "u" : "d")
                  << " spin=" << (si == 0 ? "↑" : "↓")
                  << " ||η_f||² = " << nrm << std::endl;
        // Embed into doubled fermion: upper.f[fi] = eta_f, all else zero.
        DTXQCDFermionDoubled eta(&Grid), psi(&Grid);
        eta = Zero();
        eta.upper.f[fi] = eta_f;
        // Solve doubled M psi = eta on the full volume (helper: M†M CG, QUDA).
        solver.Solve(eta, psi);
        // Contract BOTH doubled blocks at the sink (same flavour slot fi):
        //   blk 0 = upper.f[fi] (uu, quark),  blk 1 = lower.f[fi] (ll, q^C).
        for (int blk = 0; blk < NB; ++blk) {
          LatticeFermion psi_q =
              (blk == 0) ? psi.upper.f[fi] : psi.lower.f[fi];
          // Apply Gaussian wavefunction smearing to the sink quark field.
          if (do_smear) {
            CovariantSmearing<PeriodicGimplR>::GaussianSmear(
                Umu_links, psi_q, smear_width, smear_niter, Nd - 1);
          }
          SB_2d[blk][fi][si][t_src] =
              BaryonSinkContractProj(d_iso, psi_q, spin_signs[si]);
        }
      }
    }
  }

  // Translation-averaged correlators per (flavor, spin) channel:
  //   C_B^{(f,σ)}(Δt) = (1/T) Σ_{t_src} S_B^{(f,σ)}(t_src+Δt; t_src)
  // and the spin-averaged + flavor-averaged total per Eq 36.
  // Wall-wall normalization 1/V_3² applied here.
  const RealD norm = 1.0 / (V3 * V3 * (RealD)n_tsrc);
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
  auto fold = [&](int blk, int fi, int si, bool backward) {
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
        C[dt] += sgn * SB_2d[blk][fi][si][t_src][t_lat];
      }
      C[dt] *= ComplexD(align * norm, 0.0);
    }
    return C;
  };

  // 16 components: block{uu,ll} × flavor{p,n} × spin{up,dn} × dir{fwd,bwd}.
  const char *blk_nm[NB] = {"uu", "ll"};
  const char *fl_nm[NF]  = {"p", "n"};
  const char *sp_nm[NS]  = {"up", "dn"};
  std::vector<std::pair<std::string, std::vector<ComplexD>>> comps;
  std::vector<ComplexD> C_all16_avg(T, 0.0);
  for (int blk = 0; blk < NB; ++blk)
    for (int fi = 0; fi < NF; ++fi)
      for (int si = 0; si < NS; ++si)
        for (int bw = 0; bw < 2; ++bw) {
          auto C = fold(blk, fi, si, bw != 0);
          std::string nm = std::string("baryon_") + blk_nm[blk] + "_" +
                           fl_nm[fi] + "_" + sp_nm[si] + "_" +
                           (bw ? "bwd" : "fwd");
          for (int t = 0; t < T; ++t) C_all16_avg[t] += C[t];
          comps.emplace_back(nm, std::move(C));
        }
  for (int t = 0; t < T; ++t) C_all16_avg[t] *= ComplexD(1.0 / 16.0, 0.0);

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
    write(wr, "baryon_all16_avg", C_all16_avg);   // the 16-component average
    for (auto &c : comps) write(wr, c.first, c.second);   // the 16 independently
    write(wr, "baryon_total", C_B_total);         // back-compat (fwd uu sum)
    // Raw 2D per-(t_src,t_sink) matrices for diagnostics (both blocks).
    auto flatten_2d = [&](int blk, int fi, int si) {
      std::vector<ComplexD> out(T * T);
      for (int ts = 0; ts < T; ++ts)
        for (int tk = 0; tk < T; ++tk)
          out[ts * T + tk] = SB_2d[blk][fi][si][ts][tk];
      return out;
    };
    for (int blk = 0; blk < NB; ++blk)
      for (int fi = 0; fi < NF; ++fi)
        for (int si = 0; si < NS; ++si)
          write(wr, std::string("baryon_SB_") + blk_nm[blk] + "_" + fl_nm[fi] +
                    "_" + sp_nm[si] + "_2d", flatten_2d(blk, fi, si));
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
