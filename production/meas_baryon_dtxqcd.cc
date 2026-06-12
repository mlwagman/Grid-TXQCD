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
// For the BARYON SOURCE side we use the Hermitian conjugate of the diquark
// (i.e. of the operator that appears in B̄ = ε d† q̄), so the source wall is
// built from d_iso* = adj(d_iso).  For anti-Hermitian d_iso (which is what
// the flavor-antisymmetric ε^{ab} d^{ij}_{ab} projection gives) adj(d_iso) =
// −d_iso, so structurally this is a sign flip on the source.  We keep the
// two cases separate for clarity / future-proofing for non-anti-Hermitian d.
//
// `conj_field` controls the conjugation:
//   false: ε^{ijk} d_iso^{ij}(x)    -- use at sink (Eq 36)
//   true : ε^{ijk} (d_iso^{ij})*(x) -- use at source (corresponds to d† in B̄)
inline LatticeColourVector
DiquarkEpsilonWall(const LatticeColourMatrix &d_iso, int t_src,
                   bool conj_field = false) {
  GridBase *g = d_iso.Grid();
  LatticeColourMatrix d_use = conj_field ? LatticeColourMatrix(adj(d_iso))
                                          : d_iso;
  LatticeColourVector wall(g);
  wall = Zero();
  autoView(dv, d_use, CpuRead);
  autoView(wv, wall,  CpuWrite);
  thread_for(ss, g->oSites(), {
    auto d = dv[ss]()();
    wv[ss]()()(0) = d(1, 2) - d(2, 1);
    wv[ss]()()(1) = d(2, 0) - d(0, 2);
    wv[ss]()()(2) = d(0, 1) - d(1, 0);
  });
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
  const RealD tol   = TXQCDProduction::detail::env_real("CG_TOL", 1e-10);
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
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);
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

  // Build the FULL DOUBLED DTXQCD Wilson-clover Dirac operator on the gauge
  // + aux fields.  This includes the flavour-dependent σ^{ij}_{ab} insertion
  // in the diagonal block and the d, n diquark coupling between upper (q) and
  // lower (q^C) doubled blocks.  csw=0 (plain Wilson hopping) matches the
  // ensembles' csw=0 generation.  The u and d quark propagators are now
  // GENUINELY DIFFERENT cfg-by-cfg because σ^{ij}_{0,b} ≠ σ^{ij}_{1,b} on
  // each cfg even though their ensemble averages are equal by isospin symm.
  DTXQCDWilsonCloverFermionEO DwDouble(U.U, Grid, RBGrid, m, /*csw=*/0.0,
                                       U.sigma, U.pi, U.d, U.n, U.s, U.p);
  DTXQCDMOp Mop(DwDouble);

  // Hand-rolled CG on M^dag M (mirror of TxqcdCG in tests/txqcd/Test_txqcd_2pt_conn.cc).
  // Solve M^dag M x = b for given b (caller supplies b = M^dag η).
  auto DtxqcdCG = [&](const DTXQCDFermionDoubled &b,
                      DTXQCDFermionDoubled &x) {
    GridBase *g = b.Grid();
    DTXQCDFermionDoubled r(g), p(g), Mp(g), MdMp(g);
    x = Zero();
    r = b;
    p = r;
    RealD rsq = norm2(r);
    RealD bsq = std::max(norm2(b), 1e-30);
    RealD tol2 = tol * tol * bsq;
    int it;
    for (it = 0; it < cgmax; ++it) {
      Mop.M(p, Mp);
      Mop.Mdag(Mp, MdMp);
      ComplexD pAp = innerProduct(p, MdMp);
      ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
      // x += alpha p; r -= alpha MdMp
      for (int a = 0; a < DtxqcdNf; ++a) {
        x.upper.f[a] = x.upper.f[a] + alpha * p.upper.f[a];
        x.lower.f[a] = x.lower.f[a] + alpha * p.lower.f[a];
        r.upper.f[a] = r.upper.f[a] - alpha * MdMp.upper.f[a];
        r.lower.f[a] = r.lower.f[a] - alpha * MdMp.lower.f[a];
      }
      RealD rsq_new = norm2(r);
      if (rsq_new < tol2) { rsq = rsq_new; break; }
      RealD beta = rsq_new / rsq;
      for (int a = 0; a < DtxqcdNf; ++a) {
        p.upper.f[a] = r.upper.f[a] + beta * p.upper.f[a];
        p.lower.f[a] = r.lower.f[a] + beta * p.lower.f[a];
      }
      rsq = rsq_new;
    }
    std::cout << GridLogMessage << "[baryon CG] iter=" << it
              << " rsq/bsq=" << rsq / bsq << std::endl;
  };

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
  std::vector<std::vector<std::vector<std::vector<ComplexD>>>>
      SB_2d(NF, std::vector<std::vector<std::vector<ComplexD>>>(
                NS, std::vector<std::vector<ComplexD>>(T, std::vector<ComplexD>(T))));
  for (int t_src = 0; t_src < T; ++t_src) {
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
        DTXQCDFermionDoubled eta(&Grid), b(&Grid), psi(&Grid);
        eta = Zero();
        eta.upper.f[fi] = eta_f;
        // Solve doubled M^†M CG
        Mop.Mdag(eta, b);
        DtxqcdCG(b, psi);
        // Extract upper.f[fi] for sink contraction (same flavour at source/sink)
        LatticeFermion psi_q = psi.upper.f[fi];
        // Apply Gaussian wavefunction smearing to the sink quark field.
        if (do_smear) {
          CovariantSmearing<PeriodicGimplR>::GaussianSmear(
              Umu_links, psi_q, smear_width, smear_niter, Nd - 1);
        }
        SB_2d[fi][si][t_src] = BaryonSinkContractProj(d_iso, psi_q,
                                                      spin_signs[si]);
      }
    }
  }

  // Translation-averaged correlators per (flavor, spin) channel:
  //   C_B^{(f,σ)}(Δt) = (1/T) Σ_{t_src} S_B^{(f,σ)}(t_src+Δt; t_src)
  // and the spin-averaged + flavor-averaged total per Eq 36.
  // Wall-wall normalization 1/V_3² applied here.
  const RealD norm = 1.0 / (V3 * V3 * (RealD)T);
  std::vector<std::vector<std::vector<ComplexD>>>
      C_per(NF, std::vector<std::vector<ComplexD>>(NS, std::vector<ComplexD>(T, 0.0)));
  for (int fi = 0; fi < NF; ++fi) {
    for (int si = 0; si < NS; ++si) {
      for (int dt = 0; dt < T; ++dt) {
        for (int t_src = 0; t_src < T; ++t_src) {
          int t_sink = (t_src + dt) % T;
          C_per[fi][si][dt] += SB_2d[fi][si][t_src][t_sink];
        }
        C_per[fi][si][dt] *= ComplexD(norm, 0.0);
      }
    }
  }
  // Eq 36: sum over f, s of S^{(B)}_{fsfs}; "averaged at correlator level":
  std::vector<ComplexD> C_B_total(T, 0.0);
  for (int dt = 0; dt < T; ++dt) {
    for (int fi = 0; fi < NF; ++fi)
      for (int si = 0; si < NS; ++si)
        C_B_total[dt] += C_per[fi][si][dt];
  }

  std::string outfile = data_dir + "/baryon_dtxqcd_" + std::to_string(traj) + ".h5";
  if (Grid.IsBoss()) {
    Hdf5Writer wr(outfile);
    // Eq 36 total: ⟨p p̄⟩ + ⟨n n̄⟩, spin-averaged at correlator level
    write(wr, "baryon_total",      C_B_total);
    // Individual flavor × spin channels (4 total), Δt-averaged
    write(wr, "baryon_p_up",       C_per[0][0]);
    write(wr, "baryon_p_dn",       C_per[0][1]);
    write(wr, "baryon_n_up",       C_per[1][0]);
    write(wr, "baryon_n_dn",       C_per[1][1]);
    // Raw 2D per-(t_src,t_sink) matrices (unnormalized Σ_x sums) for diagnostics
    auto flatten_2d = [&](int fi, int si) {
      std::vector<ComplexD> out(T * T);
      for (int ts = 0; ts < T; ++ts)
        for (int tk = 0; tk < T; ++tk)
          out[ts * T + tk] = SB_2d[fi][si][ts][tk];
      return out;
    };
    write(wr, "baryon_SB_p_up_2d", flatten_2d(0, 0));
    write(wr, "baryon_SB_p_dn_2d", flatten_2d(0, 1));
    write(wr, "baryon_SB_n_up_2d", flatten_2d(1, 0));
    write(wr, "baryon_SB_n_dn_2d", flatten_2d(1, 1));
    write(wr, "traj",    traj);
    write(wr, "lambda",  lam);
    write(wr, "mass",    m);
    write(wr, "plaq",    plaq);
    write(wr, "T",       T);
    write(wr, "V3",      V3);
  }
  std::cout << GridLogMessage << "Written " << outfile << std::endl;
  Grid_finalize();
  return 0;
}
