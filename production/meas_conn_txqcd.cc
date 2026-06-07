#include "params.h"
#include "quda_txqcd_helper.h"
#include "quda_helper.h"  // QudaPropSolver for the vanilla-QCD strange quark
#include "meas_helper.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDCloverSchurOp.h>
#include <Grid/qcd/action/txqcd/TXQCDSolvers.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CloverHelpers.h>
#include <Grid/qcd/utils/CovariantSmearing.h>
#include <Grid/qcd/utils/BaryonUtils.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TXQCDProduction;

// Schur-EO single-shift propagator solve.  Same call surface as
// QudaTxqcdPropSolver::solve but uses the EO-preconditioned operator with
// half-volume CG — the same operator HMC uses, where TXQCD's CG iter count
// drops 5× in the bistable window vs full-volume MdagM.  ~5× faster than
// solver.solve() at this lattice/mass.
//
// Method:
//   1) Reduce: rhs_o = b_o - M_oe * M_ee^{-1} * b_e
//   2) CG on M_pc^dag·M_pc · x_o = M_pc^dag · rhs_o   (Schur, half volume)
//   3) Reconstruct: x_e = M_ee^{-1} * (b_e - M_eo * x_o)
static void SchurSolveTxqcd(Grid::TXQCDWilsonCloverFermionEO &Meo,
                             Grid::GridRedBlackCartesian *rb,
                             const Grid::TXQCDFermionNf &b,
                             Grid::TXQCDFermionNf &x,
                             RealD tol, int max_iter) {
  using namespace Grid;

  TXQCDFermionNf b_e(rb), b_o(rb);
  for (int a = 0; a < TxqcdNf; ++a) {
    pickCheckerboard(Even, b_e.f[a], b.f[a]);
    pickCheckerboard(Odd,  b_o.f[a], b.f[a]);
  }

  // Source reduction: rhs_o = b_o - M_oe * M_ee^{-1} * b_e
  TXQCDFermionNf tmp_e(rb), tmp_o(rb), rhs_o(rb);
  for (int a = 0; a < TxqcdNf; ++a) {
    tmp_e.f[a].Checkerboard() = Even;
    tmp_o.f[a].Checkerboard() = Odd;
    rhs_o.f[a].Checkerboard() = Odd;
  }
  Meo.MooeeInv(b_e, tmp_e);     // tmp_e = M_ee^{-1} b_e
  Meo.Meooe(tmp_e, tmp_o);      // tmp_o = M_oe * tmp_e
  for (int a = 0; a < TxqcdNf; ++a) rhs_o.f[a] = b_o.f[a] - tmp_o.f[a];

  // CG on M_pc^dag·M_pc · x_o = M_pc^dag · rhs_o
  TXQCDCloverSchurOp SchurOp(Meo);
  TXQCDFermionNf src_o(rb), x_o(rb);
  for (int a = 0; a < TxqcdNf; ++a) {
    src_o.f[a].Checkerboard() = Odd;
    x_o.f[a].Checkerboard() = Odd;
  }
  SchurOp.MpcDag(rhs_o, src_o);
  TXQCDConjugateGradient CG(tol, max_iter);
  CG(SchurOp, src_o, x_o);
  std::cout << GridLogMessage << "[SchurSolveTxqcd] CG iter="
            << CG.IterationsToComplete << "  resid=" << CG.TrueResidual
            << std::endl;

  // Reconstruct: x_e = M_ee^{-1} * (b_e - M_eo * x_o)
  Meo.Meooe(x_o, tmp_e);                                   // tmp_e = M_eo x_o
  for (int a = 0; a < TxqcdNf; ++a) tmp_e.f[a] = b_e.f[a] - tmp_e.f[a];
  TXQCDFermionNf x_e(rb);
  for (int a = 0; a < TxqcdNf; ++a) x_e.f[a].Checkerboard() = Even;
  Meo.MooeeInv(tmp_e, x_e);

  for (int a = 0; a < TxqcdNf; ++a) {
    setCheckerboard(x.f[a], x_e.f[a]);
    setCheckerboard(x.f[a], x_o.f[a]);
  }
}

// Phase M.4 — Multi-RHS Schur EO solver.  Same Schur reduce/reconstruct as
// SchurSolveTxqcd, but the inner CG runs over N RHS in lockstep so each
// iteration can amortize operator-application across columns.  Naive backend
// (LinOp.HermOp called N times per iter); fused HermOpN is the next step.
//
// Activated via TXQCD_MULTIRHS_CG=1 in meas_conn_txqcd.
static void SchurSolveTxqcdMultiRHS(Grid::TXQCDWilsonCloverFermionEO &Meo,
                                     Grid::GridRedBlackCartesian *rb,
                                     const std::vector<Grid::TXQCDFermionNf> &b,
                                     std::vector<Grid::TXQCDFermionNf> &x,
                                     RealD tol, int max_iter) {
  using namespace Grid;
  int N = (int)b.size();
  GRID_ASSERT(N > 0);
  GRID_ASSERT((int)x.size() == N);

  TXQCDCloverSchurOp SchurOp(Meo);

  // For each RHS, do Schur reduction → odd-parity source on RB grid.
  std::vector<TXQCDFermionNf> src_o(N, TXQCDFermionNf(rb));
  std::vector<TXQCDFermionNf> b_e(N, TXQCDFermionNf(rb));  // saved for reconstruct
  std::vector<TXQCDFermionNf> x_o(N, TXQCDFermionNf(rb));
  for (int j = 0; j < N; ++j) {
    TXQCDFermionNf b_o(rb), tmp_e(rb), tmp_o(rb), rhs_o(rb);
    for (int a = 0; a < TxqcdNf; ++a) {
      pickCheckerboard(Even, b_e[j].f[a], b[j].f[a]);
      pickCheckerboard(Odd,  b_o.f[a],   b[j].f[a]);
      tmp_e.f[a].Checkerboard() = Even;
      tmp_o.f[a].Checkerboard() = Odd;
      rhs_o.f[a].Checkerboard() = Odd;
      x_o[j].f[a].Checkerboard() = Odd;
    }
    Meo.MooeeInv(b_e[j], tmp_e);
    Meo.Meooe(tmp_e, tmp_o);
    for (int a = 0; a < TxqcdNf; ++a) rhs_o.f[a] = b_o.f[a] - tmp_o.f[a];
    SchurOp.MpcDag(rhs_o, src_o[j]);
  }

  // Multi-RHS CG: solves Mpc†Mpc · x_o[j] = src_o[j] for each j.
  TXQCDMultiRHSConjugateGradient CG(tol, max_iter);
  CG(SchurOp, src_o, x_o);
  // Per-RHS iter counts: log min/max + which one was the slowest.
  int max_iter_b = 0, max_idx = 0, min_iter_b = max_iter;
  for (int j = 0; j < N; ++j) {
    if (CG.IterationsToComplete[j] > max_iter_b) {
      max_iter_b = CG.IterationsToComplete[j]; max_idx = j;
    }
    if (CG.IterationsToComplete[j] < min_iter_b)
      min_iter_b = CG.IterationsToComplete[j];
  }
  std::cout << GridLogMessage
            << "[SchurMultiRHS] N=" << N
            << "  CG iter min=" << min_iter_b << " max=" << max_iter_b
            << "  slowest_RHS=" << max_idx << std::endl;

  // Per-RHS reconstruct.
  for (int j = 0; j < N; ++j) {
    TXQCDFermionNf tmp_e(rb), x_e(rb);
    for (int a = 0; a < TxqcdNf; ++a) {
      tmp_e.f[a].Checkerboard() = Even;
      x_e.f[a].Checkerboard() = Even;
    }
    Meo.Meooe(x_o[j], tmp_e);
    for (int a = 0; a < TxqcdNf; ++a) tmp_e.f[a] = b_e[j].f[a] - tmp_e.f[a];
    Meo.MooeeInv(tmp_e, x_e);

    for (int a = 0; a < TxqcdNf; ++a) {
      setCheckerboard(x[j].f[a], x_e.f[a]);
      setCheckerboard(x[j].f[a], x_o[j].f[a]);
    }
  }
}

static std::vector<LatticeColourMatrix>
ExtractLinks(const LatticeGaugeField &U) {
  std::vector<LatticeColourMatrix> Umu(Nd, U.Grid());
  for (int mu = 0; mu < Nd; ++mu)
    Umu[mu] = PeekIndex<LorentzIndex>(U, mu);
  return Umu;
}

static std::vector<RealD> PionCorrelator(const LatticePropagator &S_d,
                                         const LatticePropagator &S_u) {
  LatticeComplex corr(S_d.Grid());
  corr = trace(S_d * adj(S_u));
  std::vector<TComplex> Csl;
  sliceSum(corr, Csl, Nd - 1);
  std::vector<RealD> out(Csl.size());
  for (size_t t = 0; t < Csl.size(); ++t)
    out[t] = TensorRemove(Csl[t]).real();
  return out;
}

// Kaon connected correlator — same γ5-γ5 contraction pattern as the pion
// but with one TXQCD light propagator (S_l) and one vanilla-QCD strange
// propagator (S_s, no aux fields).  S_l = S_u → K_u; S_l = S_d → K_d.
static std::vector<RealD> KaonCorrelator(const LatticePropagator &S_l,
                                         const LatticePropagator &S_s) {
  LatticeComplex corr(S_l.Grid());
  corr = trace(S_s * adj(S_l));
  std::vector<TComplex> Csl;
  sliceSum(corr, Csl, Nd - 1);
  std::vector<RealD> out(Csl.size());
  for (size_t t = 0; t < Csl.size(); ++t)
    out[t] = TensorRemove(Csl[t]).real();
  return out;
}

// Time-reversed (charge-conjugated) quark propagator, following chroma's
// barhqlq_w.cc:191-198 recipe:
//   q_TR = -(γ₅γ₄) · S · (γ₅γ₄)
// where γ₅γ₄ = γ₁γ₂γ₃ = chroma's Gamma(7).  NOTE: NO time-reflection of
// the propagator itself — the time-reflection is applied to the OUTPUT
// time index after contraction (see assembly loop below), with per-t_eff
// APBC signs.  This factorization makes the gamma-matrix transformation
// trivially commute with sliceSum.
static LatticePropagator BuildTimeReversedProp(const LatticePropagator &S) {
  Gamma g5(Gamma::Algebra::Gamma5);
  Gamma g4(Gamma::Algebra::GammaT);
  return -(g5 * g4 * S * g5 * g4);
}

// "aab" nucleon contraction with both parity projectors.  q1=q2=S_aa, q3=S_b.
//   - proton (uud):  S_aa=S_u, S_b=S_d
//   - neutron (ddu): S_aa=S_d, S_b=S_u
// Returns {C_pos(t), C_neg(t)} for FB averaging in analysis:
//   C_avg(t) = (C_pos(t) - C_neg((T-t) mod T)) / 2
//
// In TXQCD the auxiliary σ breaks isospin per-cfg (S_u, S_d are independent
// solves), so averaging proton + neutron gives a real noise gain on top of
// FB averaging.
static std::pair<std::vector<ComplexD>, std::vector<ComplexD>>
BaryonAabCorrelatorPosNeg(const LatticePropagator &S_aa,
                          const LatticePropagator &S_b) {
  Gamma G_A(Gamma::Algebra::Identity);
  Gamma G_B(Gamma::Algebra::SigmaXZ);
  int wick = 0;
  BaryonUtils<WilsonImplR>::WickContractions("uud", "uud", wick);
  auto contract_proj = [&](int parity) -> std::vector<ComplexD> {
    LatticeComplex Cn(S_aa.Grid());
    BaryonUtils<WilsonImplR>::ContractBaryons(
        S_aa, S_aa, S_b, G_A, G_B, G_A, G_B, wick, parity, Cn);
    std::vector<TComplex> sl;
    sliceSum(Cn, sl, Nd - 1);
    std::vector<ComplexD> out(sl.size());
    for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
    return out;
  };
  return {contract_proj(+1), contract_proj(-1)};
}

static std::vector<Coordinate> SourceGrid(const Coordinate &latt, int traj) {
  Coordinate origin = src_grid_origin(traj);
  const int sx = space_src_per_dim_runtime();
  const int st = time_src_per_dim_runtime();
  std::vector<Coordinate> sites;
  for (int ix = 0; ix < sx; ++ix)
    for (int iy = 0; iy < sx; ++iy)
      for (int iz = 0; iz < sx; ++iz)
        for (int it = 0; it < st; ++it) {
          Coordinate s(Nd);
          s[0] = (origin[0] + ix * latt[0] / sx) % latt[0];
          s[1] = (origin[1] + iy * latt[1] / sx) % latt[1];
          s[2] = (origin[2] + iz * latt[2] / sx) % latt[2];
          s[3] = (origin[3] + it * latt[3] / st) % latt[3];
          sites.push_back(s);
        }
  return sites;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: meas_conn_txqcd <traj>" << std::endl;
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

  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  mkdir_p(txqcd_data_dir());

  TXQCDField U(&Grid);
  load_txqcd_field(traj, Grid, RBGrid, U, sRNG, pRNG);

  // Stout smearing for inversions
  Smear_Stout<PeriodicGimplR> StoutInv(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> SmearInv(&Grid, stout_nsmear_inv, StoutInv);
  SmearInv.set_Field(U.U);
  LatticeGaugeField U_inv = SmearInv.get_SmearedU();

  // Stout smearing for source/sink Gaussian smearing
  Smear_Stout<PeriodicGimplR> StoutSrc(stout_rho_src);
  SmearedConfiguration<PeriodicGimplR> SmearSrc(&Grid, stout_nsmear_src, StoutSrc);
  SmearSrc.set_Field(U.U);
  LatticeGaugeField U_src = SmearSrc.get_SmearedU();
  auto U_src_links = ExtractLinks(U_src);

  // Build TXQCD operator on inversion-smeared links
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  TXQCDWilsonCloverOp Mop(U_inv, Grid, RBGrid, mass_light,
                           U.sigma, U.pi, U.s, U.p, U.t, csw, impl_p);

  // EO operator (used when TXQCD_EO_SOLVER=1).  Uses the same Schur EO
  // operator as HMC, where TXQCD's CG iter count is 5× lower than full-volume
  // MdagM.  Set TXQCD_EO_SOLVER=0 (or unset) to fall back to full-volume CG.
  bool use_eo_solver       = std::getenv("TXQCD_EO_SOLVER") != nullptr;
  bool use_multirhs_cg     = std::getenv("TXQCD_MULTIRHS_CG") != nullptr;
  TXQCDWilsonCloverFermionEO Meo(U_inv, Grid, RBGrid, mass_light,
                                  U.sigma, U.pi, U.s, U.p, U.t, csw, impl_p);
  if (use_multirhs_cg)
    std::cout << GridLogMessage
              << "[meas_conn_txqcd] TXQCD_MULTIRHS_CG=1 — Schur EO + multi-RHS CG (24 RHS)"
              << std::endl;
  else if (use_eo_solver)
    std::cout << GridLogMessage
              << "[meas_conn_txqcd] TXQCD_EO_SOLVER=1 — Schur EO single-shift CG"
              << std::endl;

  // Full-volume QudaTxqcdPropSolver fallback.
  std::array<RealD, TxqcdNf> mass_arr;
  mass_arr.fill(mass_light);
  Grid::QudaTxqcdPropSolver solver(Mop, mass_arr, csw, U_inv,
                                    cg_tol_runtime(), cg_max);

  // -- Kaon: vanilla-QCD strange Wilson-Clover operator on the same U_inv
  //    (no aux fields).  Off by default for backwards compat with existing
  //    runs; enable with TXQCD_KAON=1.  Adds 12-spin/col inversions per
  //    source (~25% extra wallclock) and emits K_u, K_d correlators built
  //    from (TXQCD u or d) × (vanilla-QCD strange) propagators.
  bool use_kaon = (std::getenv("TXQCD_KAON") != nullptr) &&
                  (std::string(std::getenv("TXQCD_KAON")) != "0");
  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  std::unique_ptr<WCF> Dw_strange;
  std::unique_ptr<MdagMLinearOperator<WCF, LatticeFermion>> HermOp_strange;
  std::unique_ptr<Grid::QudaPropSolver<WCF>> solver_strange;
  if (use_kaon) {
    Dw_strange.reset(new WCF(U_inv, Grid, RBGrid, mass_strange, csw, csw,
                              WilsonAnisotropyCoefficients(), impl_p));
    HermOp_strange.reset(new MdagMLinearOperator<WCF, LatticeFermion>(*Dw_strange));
    solver_strange.reset(new Grid::QudaPropSolver<WCF>(*Dw_strange, *HermOp_strange,
                                                       U_inv, mass_strange, csw,
                                                       cg_tol_runtime(), cg_max));
    std::cout << GridLogMessage
              << "[meas_conn_txqcd] TXQCD_KAON=1 — adding kaon correlators K_u, K_d"
              << " (vanilla-QCD strange at mass=" << mass_strange << ")" << std::endl;
  }

  int T = latt[Nd - 1];
  auto sources = SourceGrid(latt, traj);
  int nsrc = (int)sources.size();

  std::vector<std::vector<RealD>>    all_pion;
  // Kaon correlators (one TXQCD light × vanilla-QCD strange).  K_u is built
  // from (TXQCD u, QCD strange); K_d from (TXQCD d, QCD strange).  Only
  // populated when TXQCD_KAON=1.
  std::vector<std::vector<RealD>>    all_K_u, all_K_d;
  // Proton (uud) and neutron (ddu) correlators with both parity projectors.
  std::vector<std::vector<ComplexD>> all_p_pos, all_p_neg;
  std::vector<std::vector<ComplexD>> all_n_pos, all_n_neg;
  // Time-reversed (parity-flipped, time-reflected) propagator correlators —
  // backward-propagating positive-parity nucleon, used for FB averaging.
  // Gated by TXQCD_TIME_REVERSED=1.
  bool use_time_reversed = std::getenv("TXQCD_TIME_REVERSED") != nullptr;
  std::vector<std::vector<ComplexD>> all_p_tr, all_n_tr;

  for (int isrc = 0; isrc < nsrc; ++isrc) {
    Coordinate &src = sources[isrc];
    std::cout << GridLogMessage << "[conn TXQCD] traj=" << traj
              << " src=(" << src[0] << "," << src[1] << ","
              << src[2] << "," << src[3] << ")" << std::endl;

    LatticePropagator S_u(&Grid), S_d(&Grid);
    S_u = Zero();
    S_d = Zero();

    if (use_multirhs_cg) {
      // Multi-RHS path: build all Nf×Ns×Nc=24 sources, run one batched CG,
      // then unpack to propagators.
      const int Nrhs = TxqcdNf * Ns * Nc;
      std::vector<TXQCDFermionNf> snfs, xs;
      snfs.reserve(Nrhs); xs.reserve(Nrhs);
      for (int j = 0; j < Nrhs; ++j) {
        snfs.emplace_back(&Grid);
        xs.emplace_back(&Grid);
      }
      int j = 0;
      for (int flavor = 0; flavor < TxqcdNf; ++flavor) {
        for (int spin = 0; spin < Ns; ++spin) {
          for (int col = 0; col < Nc; ++col, ++j) {
            LatticePropagator srcP(&Grid);
            srcP = Zero();
            SpinColourMatrix kron; kron = 1.0;
            pokeSite(kron, srcP, src);
            LatticeFermion sf(&Grid);
            PropToFerm<WilsonImplR>(sf, srcP, spin, col);
            CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, sf,
                                                             gauss_width, gauss_niter, Nd - 1);
            snfs[j].f[0] = Zero();
            snfs[j].f[1] = Zero();
            snfs[j].f[flavor] = sf;
          }
        }
      }
      SchurSolveTxqcdMultiRHS(Meo, &RBGrid, snfs, xs, cg_tol_runtime(), cg_max);
      // Unpack to propagators with sink smearing.
      j = 0;
      for (int flavor = 0; flavor < TxqcdNf; ++flavor) {
        LatticePropagator &Sout = (flavor == 0) ? S_u : S_d;
        for (int spin = 0; spin < Ns; ++spin) {
          for (int col = 0; col < Nc; ++col, ++j) {
            CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, xs[j].f[flavor],
                                                             gauss_width, gauss_niter, Nd - 1);
            FermToProp<WilsonImplR>(Sout, xs[j].f[flavor], spin, col);
          }
        }
      }
    } else {
      for (int flavor = 0; flavor < TxqcdNf; ++flavor) {
        LatticePropagator &Sout = (flavor == 0) ? S_u : S_d;
        for (int spin = 0; spin < Ns; ++spin) {
          for (int col = 0; col < Nc; ++col) {
            // Point source
            LatticePropagator srcP(&Grid);
            srcP = Zero();
            SpinColourMatrix kron;
            kron = 1.0;
            pokeSite(kron, srcP, src);

            LatticeFermion sf(&Grid);
            PropToFerm<WilsonImplR>(sf, srcP, spin, col);

            // Gaussian smear source
            CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, sf,
                                                             gauss_width, gauss_niter, Nd - 1);

            // Solve M·x = src.
            TXQCDFermionNf snf(U.Grid()), x(U.Grid());
            snf.f[0] = Zero();
            snf.f[1] = Zero();
            snf.f[flavor] = sf;
            if (use_eo_solver)
              SchurSolveTxqcd(Meo, &RBGrid, snf, x, cg_tol_runtime(), cg_max);
            else
              solver.solve(snf, x);

            // Gaussian smear sink
            CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, x.f[flavor],
                                                             gauss_width, gauss_niter, Nd - 1);

            FermToProp<WilsonImplR>(Sout, x.f[flavor], spin, col);
          }
        }
      }
    }

    all_pion.push_back(PionCorrelator(S_d, S_u));

    // Kaon: build vanilla-QCD strange propagator on the same source, then
    // contract with the TXQCD u and d propagators.  Uses QudaPropSolver's
    // solve_multi for 12 spin/color sources in one batched invert.
    if (use_kaon) {
      const int Nrhs = Ns * Nc;
      std::vector<LatticeFermion> sfs(Nrhs, LatticeFermion(&Grid));
      std::vector<LatticeFermion> xs(Nrhs, LatticeFermion(&Grid));
      int j = 0;
      for (int spin = 0; spin < Ns; ++spin) {
        for (int col = 0; col < Nc; ++col, ++j) {
          LatticePropagator srcP(&Grid); srcP = Zero();
          SpinColourMatrix kron; kron = 1.0;
          pokeSite(kron, srcP, src);
          PropToFerm<WilsonImplR>(sfs[j], srcP, spin, col);
          CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, sfs[j],
                                                           gauss_width, gauss_niter, Nd - 1);
        }
      }
      solver_strange->solve_multi(sfs, xs);
      LatticePropagator S_s(&Grid); S_s = Zero();
      j = 0;
      for (int spin = 0; spin < Ns; ++spin) {
        for (int col = 0; col < Nc; ++col, ++j) {
          CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, xs[j],
                                                           gauss_width, gauss_niter, Nd - 1);
          FermToProp<WilsonImplR>(S_s, xs[j], spin, col);
        }
      }
      all_K_u.push_back(KaonCorrelator(S_u, S_s));
      all_K_d.push_back(KaonCorrelator(S_d, S_s));
    }

    // Proton (uud): S_aa=S_u, S_b=S_d ; Neutron (ddu): S_aa=S_d, S_b=S_u.
    // In TXQCD S_u ≠ S_d on each cfg (auxiliary σ breaks isospin), so
    // averaging proton+neutron gives a noise gain in addition to FB avg.
    auto p_pn = BaryonAabCorrelatorPosNeg(S_u, S_d);
    auto n_pn = BaryonAabCorrelatorPosNeg(S_d, S_u);
    all_p_pos.push_back(std::move(p_pn.first));
    all_p_neg.push_back(std::move(p_pn.second));
    all_n_pos.push_back(std::move(n_pn.first));
    all_n_neg.push_back(std::move(n_pn.second));

    if (use_time_reversed) {
      // Chroma barhqlq recipe: q_TR = -(γ₅γ₄) S (γ₅γ₄), no time-reflection.
      LatticePropagator S_u_TR = BuildTimeReversedProp(S_u);
      LatticePropagator S_d_TR = BuildTimeReversedProp(S_d);
      auto p_tr = BaryonAabCorrelatorPosNeg(S_u_TR, S_d_TR);
      auto n_tr = BaryonAabCorrelatorPosNeg(S_d_TR, S_u_TR);
      // Save the +parity-projector correlators (raw lattice-time, the t_eff
      // mapping with APBC signs is applied in post-processing or below).
      all_p_tr.push_back(std::move(p_tr.first));
      all_n_tr.push_back(std::move(n_tr.first));
    }
  }

  std::cout << GridLogMessage << "[teardown-diag] per-src loop done" << std::endl;

  // Per-source SHIFT to source-relative time (Δt = t-t_s mod T).  For pion
  // (no APBC sensitivity), simple shift.  For baryon FORWARD correlator,
  // apply APBC sign: when t_eff + t_s >= T the shifted slice corresponds
  // to a fermion that crossed the antiperiodic boundary once → factor −1.
  // After this shift, output[t_eff = 0] is the source-source contact, and
  // output[t_eff > 0] is the source-relative correlator at distance t_eff
  // (no further APBC sign needed by analysis).
  bool apbc = true;  // (we always set boundary_phases[Nd-1] = -1)
  auto shift_real = [&](const std::vector<std::vector<RealD>> &raw)
      -> std::vector<std::vector<RealD>> {
    std::vector<std::vector<RealD>> out(nsrc, std::vector<RealD>(T, 0.0));
    for (int i = 0; i < nsrc; ++i) {
      int ts = sources[i][Nd - 1];
      for (int t_eff = 0; t_eff < T; ++t_eff) {
        int t_lat = (t_eff + ts) % T;
        out[i][t_eff] = raw[i][t_lat];   // pion: no APBC sign
      }
    }
    return out;
  };
  auto shift_fwd_apbc = [&](const std::vector<std::vector<ComplexD>> &raw)
      -> std::vector<std::vector<ComplexD>> {
    std::vector<std::vector<ComplexD>> out(nsrc, std::vector<ComplexD>(T, 0.0));
    for (int i = 0; i < nsrc; ++i) {
      int ts = sources[i][Nd - 1];
      for (int t_eff = 0; t_eff < T; ++t_eff) {
        int t_lat = (t_eff + ts) % T;
        double sign_fwd = (apbc && (t_eff + ts) >= T) ? -1.0 : +1.0;
        out[i][t_eff] = sign_fwd * raw[i][t_lat];
      }
    }
    return out;
  };
  // Chroma-style backward shift for TR correlator: t_eff = (T - t_lat + t_s) mod T,
  // sign_back = -1 if (t_eff - t_s) > 0 && APBC.
  auto shift_tr_chroma = [&](const std::vector<std::vector<ComplexD>> &raw)
      -> std::vector<std::vector<ComplexD>> {
    std::vector<std::vector<ComplexD>> out(nsrc, std::vector<ComplexD>(T, 0.0));
    for (int i = 0; i < nsrc; ++i) {
      int ts = sources[i][Nd - 1];
      for (int t_eff = 0; t_eff < T; ++t_eff) {
        int t_lat = ((T - t_eff + ts) % T + T) % T;
        double sign_back = (apbc && (t_eff - ts) > 0) ? -1.0 : +1.0;
        out[i][t_eff] = sign_back * raw[i][t_lat];
      }
    }
    return out;
  };
  auto pion_s  = shift_real(all_pion);
  std::vector<std::vector<RealD>> K_u_s, K_d_s;
  if (use_kaon) {
    K_u_s = shift_real(all_K_u);
    K_d_s = shift_real(all_K_d);
  }
  auto p_pos_s = shift_fwd_apbc(all_p_pos);
  auto p_neg_s = shift_fwd_apbc(all_p_neg);
  auto n_pos_s = shift_fwd_apbc(all_n_pos);
  auto n_neg_s = shift_fwd_apbc(all_n_neg);

  // Chroma-style FB averaging using time-reversed propagator correlator.
  // C_FB(t_eff) = 0.5 * (C_pos_shifted(t_eff) + C_TR_shifted(t_eff))
  // where C_TR_shifted uses chroma's backward t_eff mapping with APBC sign.
  std::vector<std::vector<ComplexD>> all_p_tr_s, all_n_tr_s;
  std::vector<std::vector<ComplexD>> all_p_fb_chroma, all_n_fb_chroma;
  if (use_time_reversed) {
    all_p_tr_s = shift_tr_chroma(all_p_tr);
    all_n_tr_s = shift_tr_chroma(all_n_tr);
    all_p_fb_chroma.assign(nsrc, std::vector<ComplexD>(T, 0.0));
    all_n_fb_chroma.assign(nsrc, std::vector<ComplexD>(T, 0.0));
    for (int i = 0; i < nsrc; ++i) {
      for (int t = 0; t < T; ++t) {
        all_p_fb_chroma[i][t] = 0.5 * (p_pos_s[i][t] + all_p_tr_s[i][t]);
        all_n_fb_chroma[i][t] = 0.5 * (n_pos_s[i][t] + all_n_tr_s[i][t]);
      }
    }
  }

  // Legacy per-source FB (Grid parity convention): kept for backward compat
  // but known to give worse S/N than chroma's TR-based FB above (Grid's
  // C_neg amplitude is ~30% of C_pos, so this halves signal without halving
  // noise).  Use proton_fb_chroma_per_src instead when TXQCD_TIME_REVERSED=1.
  std::vector<std::vector<ComplexD>> all_p_fb(nsrc, std::vector<ComplexD>(T)),
                                     all_n_fb(nsrc, std::vector<ComplexD>(T));
  for (int i = 0; i < nsrc; ++i) {
    for (int t = 0; t < T; ++t) {
      int trev = (T - t) % T;
      all_p_fb[i][t] = 0.5 * (p_pos_s[i][t] - p_neg_s[i][trev]);
      all_n_fb[i][t] = 0.5 * (n_pos_s[i][t] - n_neg_s[i][trev]);
    }
  }

  // Source-average everything in the source-relative frame.
  std::vector<RealD>    pion_avg(T, 0.0);
  std::vector<RealD>    K_u_avg(T, 0.0), K_d_avg(T, 0.0);
  std::vector<ComplexD> p_pos_avg(T, 0.0), p_neg_avg(T, 0.0);
  std::vector<ComplexD> n_pos_avg(T, 0.0), n_neg_avg(T, 0.0);
  std::vector<ComplexD> p_fbavg(T, 0.0), n_fbavg(T, 0.0);
  std::vector<ComplexD> p_fb_chroma_avg(T, 0.0), n_fb_chroma_avg(T, 0.0);
  std::vector<ComplexD> nucl_iso_fbavg(T, 0.0);          // legacy iso+FB (Grid parity)
  std::vector<ComplexD> nucl_iso_fb_chroma_avg(T, 0.0);  // iso + chroma TR FB
  for (int i = 0; i < nsrc; ++i) {
    for (int t = 0; t < T; ++t) {
      pion_avg[t] += pion_s[i][t] / (double)nsrc;
      if (use_kaon) {
        K_u_avg[t] += K_u_s[i][t] / (double)nsrc;
        K_d_avg[t] += K_d_s[i][t] / (double)nsrc;
      }
      p_pos_avg[t] += p_pos_s[i][t] / (double)nsrc;
      p_neg_avg[t] += p_neg_s[i][t] / (double)nsrc;
      n_pos_avg[t] += n_pos_s[i][t] / (double)nsrc;
      n_neg_avg[t] += n_neg_s[i][t] / (double)nsrc;
      p_fbavg[t]   += all_p_fb[i][t] / (double)nsrc;
      n_fbavg[t]   += all_n_fb[i][t] / (double)nsrc;
      if (use_time_reversed) {
        p_fb_chroma_avg[t] += all_p_fb_chroma[i][t] / (double)nsrc;
        n_fb_chroma_avg[t] += all_n_fb_chroma[i][t] / (double)nsrc;
      }
    }
  }
  for (int t = 0; t < T; ++t) {
    nucl_iso_fbavg[t] = 0.5 * (p_fbavg[t] + n_fbavg[t]);
    if (use_time_reversed)
      nucl_iso_fb_chroma_avg[t] = 0.5 * (p_fb_chroma_avg[t] + n_fb_chroma_avg[t]);
  }

  // Compute plaq on ALL ranks first — avgPlaquette is an MPI collective,
  // it must be called outside the IsBoss() guard or rank 0 deadlocks waiting
  // for ranks 1..N to join its Allreduce.
  RealD plaq_for_h5 = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);

  std::string outfile = txqcd_data_dir() + "/conn_txqcd_" + std::to_string(traj) + ".h5";
  std::cout << GridLogMessage << "[teardown-diag] entering h5 write (rank 0 only)" << std::endl;
  if (Grid.IsBoss()) {
    Hdf5Writer wr(outfile);
    std::cout << GridLogMessage << "[teardown-diag] Hdf5Writer opened" << std::endl;
    // All averaged correlators are in SOURCE-RELATIVE TIME (Δt = t - t_s mod T)
    // with APBC sign already applied for baryons.  index 0 = source contact,
    // index t > 0 = source-sink separation.
    write(wr, "pion_conn",        pion_avg);
    if (use_kaon) {
      write(wr, "kaon_u_conn",  K_u_avg);     // (TXQCD u) × (vanilla QCD strange)
      write(wr, "kaon_d_conn",  K_d_avg);     // (TXQCD d) × (vanilla QCD strange)
    }
    // Legacy "nucleon" = proton +parity, in source-relative frame
    write(wr, "nucleon",          p_pos_avg);
    write(wr, "proton_pos",       p_pos_avg);
    write(wr, "proton_neg",       p_neg_avg);
    write(wr, "proton_fbavg",     p_fbavg);                // legacy (Grid parity)
    write(wr, "neutron_pos",      n_pos_avg);
    write(wr, "neutron_neg",      n_neg_avg);
    write(wr, "neutron_fbavg",    n_fbavg);                // legacy
    write(wr, "nucleon_iso_fbavg", nucl_iso_fbavg);        // legacy (Grid parity)
    if (use_time_reversed) {
      write(wr, "proton_fb_chroma",          p_fb_chroma_avg);
      write(wr, "neutron_fb_chroma",         n_fb_chroma_avg);
      write(wr, "nucleon_iso_fb_chroma",     nucl_iso_fb_chroma_avg);
    }
    // Per-source SHIFTED correlators in source-relative time with APBC sign
    // applied for baryons (index 0 = each source's t_src position; index t > 0
    // = source-sink separation).  Pion just shifted (no APBC sign).
    write(wr, "pion_per_src",         pion_s);
    if (use_kaon) {
      write(wr, "kaon_u_per_src",   K_u_s);
      write(wr, "kaon_d_per_src",   K_d_s);
      write(wr, "kaon_u_per_src_lat", all_K_u);
      write(wr, "kaon_d_per_src_lat", all_K_d);
    }
    write(wr, "proton_pos_per_src",   p_pos_s);
    write(wr, "proton_neg_per_src",   p_neg_s);
    write(wr, "neutron_pos_per_src",  n_pos_s);
    write(wr, "neutron_neg_per_src",  n_neg_s);
    write(wr, "nucleon_per_src",      p_pos_s);     // legacy alias = proton_pos shifted
    if (use_time_reversed) {
      // Chroma-style backward-shifted TR correlator and per-source FB average.
      write(wr, "proton_tr_per_src",          all_p_tr_s);
      write(wr, "neutron_tr_per_src",         all_n_tr_s);
      write(wr, "proton_fb_chroma_per_src",   all_p_fb_chroma);
      write(wr, "neutron_fb_chroma_per_src",  all_n_fb_chroma);
    }
    // Also save the LATTICE-TIME raw per-src for backward compat / debugging.
    write(wr, "proton_pos_per_src_lat",   all_p_pos);
    write(wr, "proton_neg_per_src_lat",   all_p_neg);
    write(wr, "neutron_pos_per_src_lat",  all_n_pos);
    write(wr, "neutron_neg_per_src_lat",  all_n_neg);
    if (use_time_reversed) {
      write(wr, "proton_tr_per_src_lat",  all_p_tr);
      write(wr, "neutron_tr_per_src_lat", all_n_tr);
    }
    write(wr, "traj", traj);
    write(wr, "plaq", plaq_for_h5);
    {
      // Per-cfg source-grid origin shift (zero for cfg<1000, deterministic
      // mt19937 shift for cfg>=1000; see params.h src_grid_origin).
      Coordinate s = src_grid_origin(traj);
      std::vector<int> sv(Nd);
      for (int d = 0; d < Nd; ++d) sv[d] = s[d];
      write(wr, "src_shift", sv);
    }
    std::cout << GridLogMessage << "[teardown-diag] all writes queued; Hdf5Writer dtor next" << std::endl;
  }

  std::cout << GridLogMessage << "[teardown-diag] Hdf5Writer block exited" << std::endl;
  std::cout << GridLogMessage << "Written " << outfile << std::endl;
#ifdef GRID_HAVE_QUDA
  std::cout << GridLogMessage << "[teardown-diag] calling Quda::finalize" << std::endl;
  Grid::Quda::finalize();
  std::cout << GridLogMessage << "[teardown-diag] Quda::finalize returned" << std::endl;
#endif
  Grid_finalize();
  return 0;
}
