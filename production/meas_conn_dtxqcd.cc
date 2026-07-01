// meas_conn_dtxqcd: connected pion + nucleon correlators on DTXQCD configs.
//
// DEFAULT (physical) path: the light-quark propagator is the upper block of
// the doubled M48^{-1} — the genuine DTXQCD quark propagator, including the
// aux (σ,π,d,n,s,p) backreaction and the d/n coupling to the charge-conjugate
// block.  The operator is built on the STOUT-SMEARED gauge at production csw
// (the same operator the sea quark saw) and solved on the fast QUDA
// device-resident CG (dtxqcd_meas_helper.h).  Per source: 2 flavor × 12
// spin·colour = 24 doubled solves.  u and d propagators differ per-cfg
// (σ^{ij}_{0b} ≠ σ^{ij}_{1b}), so proton+neutron averaging gives a real noise
// gain on top of FB averaging, exactly as in meas_conn_txqcd.
//
// LEGACY path (DTXQCD_CONN_GAUGE_ONLY=1): plain Wilson Dirac on the DTXQCD
// gauge, aux ignored — the original "is the GAUGE sector shifted by aux
// backreaction?" probe.  Kept for that diagnostic; NOT the physical DTXQCD
// correlator.
//
// Usage:
//   ./meas_conn_dtxqcd <traj> [--grid LxLyLzLt] [--mpi mxmymzmt]
//
// Env knobs:
//   CFG_DIR, DATA_DIR         : override input/output directories
//   MEAS_SPACE_SRC, MEAS_TIME_SRC : source-grid density per dim (params.h)
//   CG_TOL (1e-8), CG_MAX     : solver stopping criteria
//   DTXQCD_MEAS_QUDA (1)      : QUDA device CG (0 = Grid-CPU reference)
//   DTXQCD_CONN_GAUGE_ONLY(0) : legacy plain-Wilson gauge-only probe

#include "params.h"
#include "dtxqcd_meas_helper.h"
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/qcd/utils/BaryonUtils.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TXQCDProduction;
using namespace Grid;

// Connected pseudoscalar: tr(S_d · adj(S_u)).  In DTXQCD S_u ≠ S_d per-cfg.
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

// "aab" nucleon (q1=q2=S_aa, q3=S_b) with both parity projectors.
//   proton (uud):  S_aa=S_u, S_b=S_d ;  neutron (ddu): S_aa=S_d, S_b=S_u.
// Returns {C_pos(t), C_neg(t)} for FB averaging:
//   C_avg(t) = (C_pos(t) - C_neg((T-t) mod T)) / 2
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

// Physical DTXQCD light-quark propagators S_u, S_d from the doubled M48:
// for each (flavor, spin, colour) embed a point source in the UPPER block at
// slot `flavor`, solve M48 ψ = η via the helper, and read the UPPER block at
// the same flavor slot as the quark propagator column.
static void DtxqcdM48Props(DtxqcdMeasSolver &solver, GridCartesian &Grid,
                           const Coordinate &src,
                           LatticePropagator &S_u, LatticePropagator &S_d,
                           bool lower = false) {
  S_u = Zero();
  S_d = Zero();
  for (int flavor = 0; flavor < DtxqcdNf; ++flavor) {
    LatticePropagator &Sout = (flavor == 0) ? S_u : S_d;
    for (int spin = 0; spin < Ns; ++spin) {
      for (int col = 0; col < Nc; ++col) {
        LatticePropagator srcP(&Grid);
        srcP = Zero();
        SpinColourMatrix kron;
        kron = 1.0;
        pokeSite(kron, srcP, src);
        LatticeFermion sf(&Grid);
        PropToFerm<WilsonImplR>(sf, srcP, spin, col);

        DTXQCDFermionDoubled eta(&Grid), psi(&Grid);
        eta = Zero();
        if (lower) eta.lower.f[flavor] = sf; else eta.upper.f[flavor] = sf;
        solver.Solve(eta, psi);
        LatticeFermion out = lower ? psi.lower.f[flavor] : psi.upper.f[flavor];
        FermToProp<WilsonImplR>(Sout, out, spin, col);
      }
    }
  }
}

// Legacy plain-Wilson point propagator on the DTXQCD gauge (aux ignored).
static void WilsonPointProp(LatticePropagator &S, LatticeGaugeField &Umu,
                            RealD m, GridCartesian &Grid,
                            GridRedBlackCartesian &RBGrid,
                            const Coordinate &src, RealD tol, int maxit) {
  WilsonFermionD Dw(Umu, Grid, RBGrid, m);
  MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(tol, maxit);

  LatticePropagator srcP(&Grid);
  srcP = Zero();
  SpinColourMatrix kron;
  kron = 1.0;
  pokeSite(kron, srcP, src);
  S = Zero();
  for (int spin = 0; spin < Ns; ++spin) {
    for (int col = 0; col < Nc; ++col) {
      LatticeFermion sf(&Grid), b(&Grid), x(&Grid);
      PropToFerm<WilsonImplR>(sf, srcP, spin, col);
      Dw.Mdag(sf, b);
      x = Zero();
      CG(HermOp, b, x);
      FermToProp<WilsonImplR>(S, x, spin, col);
    }
  }
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
    std::cerr << "Usage: meas_conn_dtxqcd <traj>" << std::endl;
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

  const RealD lam = lambda_dtxqcd;
  const RealD m   = mass_light_dtxqcd;
  const RealD tol = TXQCDProduction::detail::env_real("CG_TOL", 1e-8);
  const int   cgmax = TXQCDProduction::detail::env_int("CG_MAX", 100000);
  const bool  gauge_only =
      TXQCDProduction::detail::env_int("DTXQCD_CONN_GAUGE_ONLY", 0) != 0;

  int T = latt[Nd - 1];
  std::cout << GridLogMessage << "======== meas_conn_dtxqcd ========" << std::endl;
  std::cout << GridLogMessage << "  traj    = " << traj << std::endl;
  std::cout << GridLogMessage << "  lambda  = " << lam << std::endl;
  std::cout << GridLogMessage << "  mass    = " << m << std::endl;
  std::cout << GridLogMessage << "  csw     = " << csw << std::endl;
  std::cout << GridLogMessage << "  mode    = "
            << (gauge_only ? "GAUGE-ONLY (plain Wilson, aux ignored)"
                           : "PHYSICAL M48 (aux + stout + csw)")
            << std::endl;
  std::cout << GridLogMessage << "  lattice = " << latt[0] << "x" << latt[1]
            << "x" << latt[2] << "x" << latt[3] << "  T=" << T << std::endl;
  std::cout << GridLogMessage << "  cfg dir = " << cfg_dir << std::endl;
  std::cout << GridLogMessage << "  out dir = " << data_dir << std::endl;
  std::cout << GridLogMessage << "===================================" << std::endl;

  DTXQCDField U(&Grid);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);
  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);

  // The physical M48 solver (stout + csw + aux, QUDA device CG).  Only built
  // when needed (the gauge-only probe uses a plain Wilson operator instead).
  std::unique_ptr<DtxqcdMeasSolver> solver;
  if (!gauge_only)
    solver.reset(new DtxqcdMeasSolver(Grid, RBGrid, U, m, csw, tol, cgmax));

  auto sources = SourceGrid(latt, traj);
  int nsrc = (int)sources.size();
  std::cout << GridLogMessage << "  nsrc    = " << nsrc << std::endl;

  // Four connected pseudoscalar traces per source:
  //   du = tr(S_d adj S_u)  [ud̄, standard π⁺]   ud = tr(S_u adj S_d)  [dū]
  //   uu = tr(S_u adj S_u)                       dd = tr(S_d adj S_d)
  // In exact isospin all four coincide; at large λ (small aux) they agree up
  // to aux noise, and uu−dd measures the per-cfg isospin breaking directly.
  std::vector<std::vector<RealD>>    all_pi_du, all_pi_ud, all_pi_uu, all_pi_dd;
  std::vector<std::vector<ComplexD>> all_p_pos, all_p_neg, all_n_pos, all_n_neg;

  // Optional LOWER (charge-conjugate) block: source AND sink in the lower block,
  // RAW contraction (no C insertion).  Tests the conjecture that the source and
  // sink C's cancel for a lower-src/lower-sink correlator (as they do for the
  // meson) — so the raw lower nucleon should give the same MASS as the upper.
  const bool do_lower = (!gauge_only) &&
      TXQCDProduction::detail::env_int("DTXQCD_CONN_LOWER", 0) != 0;
  std::vector<std::vector<RealD>>    all_pi_du_lo;
  std::vector<std::vector<ComplexD>> all_p_pos_lo, all_n_pos_lo;

  for (int isrc = 0; isrc < nsrc; ++isrc) {
    Coordinate &src = sources[isrc];
    std::cout << GridLogMessage << "[conn DTXQCD] traj=" << traj
              << " src=(" << src[0] << "," << src[1] << ","
              << src[2] << "," << src[3] << ")" << std::endl;

    LatticePropagator S_u(&Grid), S_d(&Grid);
    if (gauge_only) {
      WilsonPointProp(S_u, U.U, m, Grid, RBGrid, src, tol, cgmax);
      S_d = S_u;  // isospin-symmetric in the gauge-only probe
    } else {
      DtxqcdM48Props(*solver, Grid, src, S_u, S_d);
    }

    all_pi_du.push_back(PionCorrelator(S_d, S_u));
    all_pi_ud.push_back(PionCorrelator(S_u, S_d));
    all_pi_uu.push_back(PionCorrelator(S_u, S_u));
    all_pi_dd.push_back(PionCorrelator(S_d, S_d));
    auto p_pn = BaryonAabCorrelatorPosNeg(S_u, S_d);  // proton (uud)
    auto n_pn = BaryonAabCorrelatorPosNeg(S_d, S_u);  // neutron (ddu = udd)
    all_p_pos.push_back(std::move(p_pn.first));
    all_p_neg.push_back(std::move(p_pn.second));
    all_n_pos.push_back(std::move(n_pn.first));
    all_n_neg.push_back(std::move(n_pn.second));

    if (do_lower) {
      LatticePropagator S_u_lo(&Grid), S_d_lo(&Grid);
      DtxqcdM48Props(*solver, Grid, src, S_u_lo, S_d_lo, /*lower=*/true);
      all_pi_du_lo.push_back(PionCorrelator(S_d_lo, S_u_lo));
      all_p_pos_lo.push_back(BaryonAabCorrelatorPosNeg(S_u_lo, S_d_lo).first);
      all_n_pos_lo.push_back(BaryonAabCorrelatorPosNeg(S_d_lo, S_u_lo).first);
    }
  }

  // Per-source SHIFT to source-relative time + APBC sign for the baryon.
  bool apbc = true;
  auto shift_real = [&](const std::vector<std::vector<RealD>> &raw)
      -> std::vector<std::vector<RealD>> {
    std::vector<std::vector<RealD>> out(nsrc, std::vector<RealD>(T, 0.0));
    for (int i = 0; i < nsrc; ++i) {
      int ts = sources[i][Nd - 1];
      for (int t_eff = 0; t_eff < T; ++t_eff)
        out[i][t_eff] = raw[i][(t_eff + ts) % T];
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
  auto pi_du_s = shift_real(all_pi_du);
  auto pi_ud_s = shift_real(all_pi_ud);
  auto pi_uu_s = shift_real(all_pi_uu);
  auto pi_dd_s = shift_real(all_pi_dd);
  auto p_pos_s = shift_fwd_apbc(all_p_pos);
  auto p_neg_s = shift_fwd_apbc(all_p_neg);
  auto n_pos_s = shift_fwd_apbc(all_n_pos);
  auto n_neg_s = shift_fwd_apbc(all_n_neg);

  std::vector<RealD>    pi_du_avg(T, 0.0), pi_ud_avg(T, 0.0),
                        pi_uu_avg(T, 0.0), pi_dd_avg(T, 0.0);
  std::vector<ComplexD> p_pos_avg(T, 0.0), p_neg_avg(T, 0.0);
  std::vector<ComplexD> n_pos_avg(T, 0.0), n_neg_avg(T, 0.0);
  for (int i = 0; i < nsrc; ++i)
    for (int t = 0; t < T; ++t) {
      pi_du_avg[t] += pi_du_s[i][t] / (double)nsrc;
      pi_ud_avg[t] += pi_ud_s[i][t] / (double)nsrc;
      pi_uu_avg[t] += pi_uu_s[i][t] / (double)nsrc;
      pi_dd_avg[t] += pi_dd_s[i][t] / (double)nsrc;
      p_pos_avg[t] += p_pos_s[i][t] / (double)nsrc;
      p_neg_avg[t] += p_neg_s[i][t] / (double)nsrc;
      n_pos_avg[t] += n_pos_s[i][t] / (double)nsrc;
      n_neg_avg[t] += n_neg_s[i][t] / (double)nsrc;
    }
  // FB average (Grid parity convention) + isospin average.
  std::vector<ComplexD> p_fbavg(T, 0.0), n_fbavg(T, 0.0), nucl_iso_fbavg(T, 0.0);
  for (int t = 0; t < T; ++t) {
    int trev = (T - t) % T;
    p_fbavg[t] = 0.5 * (p_pos_avg[t] - p_neg_avg[trev]);
    n_fbavg[t] = 0.5 * (n_pos_avg[t] - n_neg_avg[trev]);
    nucl_iso_fbavg[t] = 0.5 * (p_fbavg[t] + n_fbavg[t]);
  }

  // Lower-block (charge-conjugate) raw correlators: shift + average.
  std::vector<RealD>    pi_du_lo_avg(T, 0.0);
  std::vector<ComplexD> p_pos_lo_avg(T, 0.0), n_pos_lo_avg(T, 0.0);
  if (do_lower) {
    auto pi_du_lo_s = shift_real(all_pi_du_lo);
    auto p_pos_lo_s = shift_fwd_apbc(all_p_pos_lo);
    auto n_pos_lo_s = shift_fwd_apbc(all_n_pos_lo);
    for (int i = 0; i < nsrc; ++i)
      for (int t = 0; t < T; ++t) {
        pi_du_lo_avg[t] += pi_du_lo_s[i][t] / (double)nsrc;
        p_pos_lo_avg[t] += p_pos_lo_s[i][t] / (double)nsrc;
        n_pos_lo_avg[t] += n_pos_lo_s[i][t] / (double)nsrc;
      }
  }

  std::string outfile = data_dir + "/conn_dtxqcd_" + std::to_string(traj) + ".h5";
  if (Grid.IsBoss()) {
    Hdf5Writer wr(outfile);
    write(wr, "pion_conn",         pi_du_avg);   // = pion_du (standard π⁺, ud̄)
    write(wr, "pion_du",           pi_du_avg);   // tr(S_d adj S_u)
    write(wr, "pion_ud",           pi_ud_avg);   // tr(S_u adj S_d)
    write(wr, "pion_uu",           pi_uu_avg);   // tr(S_u adj S_u)
    write(wr, "pion_dd",           pi_dd_avg);   // tr(S_d adj S_d)
    write(wr, "proton_pos",        p_pos_avg);
    write(wr, "proton_neg",        p_neg_avg);
    write(wr, "proton_fbavg",      p_fbavg);
    write(wr, "neutron_pos",       n_pos_avg);
    write(wr, "neutron_neg",       n_neg_avg);
    write(wr, "neutron_fbavg",     n_fbavg);
    write(wr, "nucleon",           p_pos_avg);        // legacy alias = proton +parity
    write(wr, "nucleon_iso_fbavg", nucl_iso_fbavg);
    if (do_lower) {
      // LOWER (charge-conjugate) block, RAW contraction — same-mass test.
      write(wr, "pion_du_lower",     pi_du_lo_avg);
      write(wr, "proton_pos_lower",  p_pos_lo_avg);
      write(wr, "neutron_pos_lower", n_pos_lo_avg);
    }
    write(wr, "pion_du_per_src",    pi_du_s);
    write(wr, "pion_ud_per_src",    pi_ud_s);
    write(wr, "pion_uu_per_src",    pi_uu_s);
    write(wr, "pion_dd_per_src",    pi_dd_s);
    write(wr, "proton_pos_per_src", p_pos_s);
    write(wr, "neutron_pos_per_src", n_pos_s);
    write(wr, "traj",   traj);
    write(wr, "lambda", lam);
    write(wr, "mass",   m);
    write(wr, "plaq",   plaq);
    write(wr, "gauge_only", (int)gauge_only);
    Coordinate s = src_grid_origin(traj);
    std::vector<int> sv(Nd);
    for (int d = 0; d < Nd; ++d) sv[d] = s[d];
    write(wr, "src_shift", sv);
  }

  std::cout << GridLogMessage << "Written " << outfile << std::endl;
  Grid_finalize();
#ifdef GRID_HAVE_QUDA
  // Skip the C++ static-destructor / __cxa_finalize chain: QUDA's static
  // ColorSpinorField destructors double-free at process exit.  Output is on
  // disk and MPI is finalized — _Exit is safe (cf. meas_baryon_dtxqcd).
  std::fflush(nullptr);
  std::_Exit(0);
#endif
  return 0;
}
