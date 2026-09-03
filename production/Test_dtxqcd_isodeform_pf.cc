// Test_dtxqcd_isodeform_pf: REFERENCE reweighting-factor estimator for the
// U(1)_{u-d} ISOSPIN contour deformation (the conserved-charge analog of the
// baryon-number contour, applied to the meson sector).
//
// Symmetry: degenerate Nf=2 makes U(1)_{u-d} exact.  The t-graded imaginary
// isospin rotation = an isospin-chemical-potential profile.  By the
// Ward-identity/contour-cost law (see dtxqcd-contour-ward-law memory):
//   - UNIFORM grading is a similarity transform => dlogPf == 0 EXACTLY.
//     WARDCHECK=1 runs with prof == const and demands the trace vanish
//     within noise -- the decisive convention gate.
//   - graded cost localizes at profile kinks, paid in PION currency
//     chi ~ e^{-M_pi l}; criticality wall expected at tent slope
//     eta ~ aM_pi = 0.5943(2) (charge-2 units, this file's convention).
//
// Contour (field frame, all SITE-LOCAL -- no hop weights in the reference):
// the I3 = +-2 aux components are the holomorphic pairs, no fictitious
// fields needed (mwagman's |p|,arg construction is automatic here):
//   sigma^{01} -> c(t) sigma^{01},  sigma^{10} -> c(t)^{-1} sigma^{10}
//   pi^{01}    -> c(t) pi^{01},     pi^{10}    -> c(t)^{-1} pi^{10}
//   d(0,0) [uu, I3=+2] -> c(t) d(0,0),   d(1,1) [dd] -> c(t)^{-1} d(1,1)
//   n likewise;  all I3-neutral components untouched;  c(t) = e^{-eta prof(t)}.
// Gaussians |sigma^{01}|^2 etc. INVARIANT, Jacobian 1: weight = complex Pf
// ratio only (O(eta) = pure phase; the u<->d pairing that makes det(mu_I)
// real in pure QCD is broken per config only by the flavor-off-diagonal aux
// fluctuations -- the measured phase tests the near-factorization reprieve).
//
// The lower (charge-conjugate) block: sigma/pi enter via the TRANSPOSE of
// the same fields (DtxqcdApplyDeltaDiagLower) -- transpose + charge flip
// cancel, so ONE weighted field serves both blocks.  d/n enter LL via
// CONJUGATION, so the LL corrections need the INVERSE-weighted fields
// (the d_wm/d_wp pair pattern of Test_dtxqcd_eta_stoch_pf).
//
// Observables this grades (charge q => rate q*eta/2 per unit prof-slope...
// in this file's convention charge +-2 fields carry e^{-+eta prof}, so the
// aux pion wall correlator <pi^{01}(t) pi^{10}(0)> grades at rate eta
// against its StN decay aM_pi: the same marginal race as the baryon 3eta
// vs 2M_N.  Also sigma^{01} (a0+-) and the uu/dd diquark components.
//
// Env: as Test_dtxqcd_pdeform_pf (LATT, CFG_DIR, TRAJ, ETA, PROFILE tent|cos,
//      PROF_T0, N_NOISE, N_GL, MASS, CSW, STOUT_*, LAMBDA_DTXQCD, CG_TOL,
//      CG_MAX, NOISE_SEED, DERIVCHECK) + WARDCHECK (0; =1 forces prof==1,
//      expects dlogPf = 0 within noise).

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>

using namespace Grid;

static RealD env_real(const char *k, RealD d) {
  const char *v = std::getenv(k); return (v && *v) ? std::atof(v) : d;
}
static int env_int(const char *k, int d) {
  const char *v = std::getenv(k); return (v && *v) ? std::atoi(v) : d;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::vector<int> latt_dims{4, 4, 4, 4};
  if (const char *v = std::getenv("LATT"); v && *v) {
    latt_dims.clear();
    std::stringstream ss(v); std::string tok;
    while (std::getline(ss, tok, '.')) latt_dims.push_back(std::atoi(tok.c_str()));
  }
  Coordinate latt(latt_dims);
  GridCartesian Grid(latt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian RBGrid(&Grid);
  const int T = latt[Nd - 1];

  const int traj      = env_int("TRAJ", 250);
  const double eta    = env_real("ETA", 0.3);
  const int n_noise   = env_int("N_NOISE", 16);
  const int n_gl      = env_int("N_GL", 4);
  const RealD mass    = env_real("MASS", -0.245);
  const RealD csw     = env_real("CSW", 1.24930970916466);
  const int nstout    = env_int("STOUT_NSMEAR", 1);
  const RealD rho     = env_real("STOUT_RHO", 0.125);
  const RealD tol     = env_real("CG_TOL", 1e-8);
  const int cgmax     = env_int("CG_MAX", 20000);
  const int prof_t0   = env_int("PROF_T0", 0);
  const bool wardchk  = env_int("WARDCHECK", 0) != 0;
  const std::string prof_kind = [&] {
    const char *v = std::getenv("PROFILE");
    return std::string(v && *v ? v : "tent");
  }();
  const bool prof_cos = prof_kind == "cos";
  // ISO_DN_ONLY=1 (mwagman probe): grade ONLY the diquark d/n cross-blocks
  // with the isospin pattern, leave the MESON sigma/pi UNDEFORMED.  Tests
  // whether the residual isospin-contour phase came from the off-diagonal
  // meson spurions (sigma_ud, pi_ud) -- if darg collapses toward 0, the
  // d/n-only contour is (near-)real; if it stays ~1.7 eta, the phase lives
  // in the diquark UR/LL doubling structure (the real-vs-cheap tension).
  // NOTE: no longer the isospin SYMMETRY (mesons carry isospin too), so the
  // WARDCHECK uniform gate is NOT expected to give 0 here -- it measures the
  // uniform d/n-only response instead.
  const bool dn_only = env_int("ISO_DN_ONLY", 0) != 0;
  const bool prof_rounded = prof_kind == "tent_rounded" ||
                            prof_kind == "tent_rounded2";
  const bool prof_round_src = prof_kind == "tent_rounded2";
  // rounding width: default 1/aM_pi -- the isospin kink pays broadband
  // susceptibility above the pion; rounding at the pion Compton wavelength
  // is the tent_rounded2 move that won big for the d contour.
  const double prof_w = env_real("PROFILE_W", 1.0 / 0.5943);
  const double prof_w_src = env_real("PROFILE_W_SRC", prof_w);
  std::string cfg_dir = "/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/"
                        "dtxqcd_lam5.0000_4t4_signhop_lam5";
  if (const char *d = std::getenv("CFG_DIR"); d && *d) cfg_dir = d;

  std::cout << GridLogMessage << "==== Test_dtxqcd_isodeform_pf ====" << std::endl;
  std::cout << GridLogMessage << " cfg=" << cfg_dir << " traj=" << traj
            << " eta=" << eta << " profile="
            << (wardchk ? "UNIFORM(wardcheck)" : (prof_cos ? "cos" : "tent"))
            << " t0=" << prof_t0 << " N_NOISE=" << n_noise
            << " N_GL=" << n_gl << std::endl;

  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid);
  DTXQCDField U(&Grid);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);
  {
    const int s = env_int("NOISE_SEED", 77);
    pRNG.SeedFixedIntegers({s, s + 3, s + 7, s + 13, s + 21});
  }
  LatticeGaugeField Usm(U.U);
  if (nstout > 0) {
    Smear_Stout<PeriodicGimplR> stout(rho);
    LatticeGaugeField tmp(&Grid);
    for (int i = 0; i < nstout; ++i) { stout.smear(tmp, Usm); Usm = tmp; }
  }
  DTXQCDWilsonCloverFermionEO Dw(Usm, Grid, RBGrid, mass, csw,
                                 U.sigma, U.pi, U.d, U.n, U.s, U.p);
  DTXQCDMOp Mop(Dw);

  auto prof = [&](int t) -> double {
    if (wardchk) return 1.0;                    // Ward gate: uniform profile
    double d = (double)(((t - prof_t0) % T + T) % T);
    if (prof_cos)
      return 0.25 * (double)T * (1.0 - std::cos(2.0 * M_PI * d / (double)T));
    double v = std::min(d, (double)T - d);      // sharp tent
    if (prof_rounded) {                         // apex rounding, width prof_w
      const double a = std::abs(d - 0.5 * T);
      if (a < prof_w)
        v = 0.5 * T - 0.5 * prof_w - a * a / (2.0 * prof_w);
      if (prof_round_src && prof_w_src > 0) {   // trough rounding
        const double b = std::min(d, (double)T - d);
        if (b < prof_w_src)
          v = 0.5 * prof_w_src + b * b / (2.0 * prof_w_src);
      }
    }
    return v;
  };

  // ---- component-weighted aux fields ------------------------------------
  LatticeInteger tc(&Grid);
  LatticeCoordinate(tc, Nd - 1);
  auto WeightComp = [&](LatticeColourMatrix &out,
                        const LatticeColourMatrix &base, auto fac) {
    out = Zero();
    for (int t = 0; t < T; ++t) {
      auto scaled = base;
      scaled = ComplexD(fac(t), 0.0) * scaled;
      out = where(tc == Integer(t), scaled, out);
    }
  };

  // base charged components
  LatticeColourMatrix sig01(PeekIndex<1>(U.sigma, 0, 1)),
      sig10(PeekIndex<1>(U.sigma, 1, 0)), pi01(PeekIndex<1>(U.pi, 0, 1)),
      pi10(PeekIndex<1>(U.pi, 1, 0)), d00(PeekIndex<1>(U.d, 0, 0)),
      d11(PeekIndex<1>(U.d, 1, 1)), n00(PeekIndex<1>(U.n, 0, 0)),
      n11(PeekIndex<1>(U.n, 1, 1));

  // weighted field sets: _m ("minus" = M corrections), _p (dagger/LL),
  // _dm/_dp (eta-derivative vertex).  Weights (charge-2 factor c=e^{-e prof}):
  //   wm = c-1, wp = 1/c-1, dm = -prof*c, dp = +prof/c.
  LatticeDtxqcdSigma sig_wm(&Grid), sig_wp(&Grid), sig_dm(&Grid);
  LatticeDtxqcdPi pi_wm(&Grid), pi_wp(&Grid), pi_dm(&Grid);
  LatticeDtxqcdD d_wm(&Grid), d_wp(&Grid), d_dm(&Grid), d_dp(&Grid);
  LatticeDtxqcdN n_wm(&Grid), n_wp(&Grid), n_dm(&Grid), n_dp(&Grid);
  LatticeDtxqcdS s0(&Grid); s0 = Zero();
  LatticeDtxqcdP p0(&Grid); p0 = Zero();

  auto BuildWeights = [&](double e) {
    auto wm = [&](int t) { return std::exp(-e * prof(t)) - 1.0; };
    auto wp = [&](int t) { return std::exp(+e * prof(t)) - 1.0; };
    auto dm = [&](int t) { return -prof(t) * std::exp(-e * prof(t)); };
    auto dp = [&](int t) { return +prof(t) * std::exp(+e * prof(t)); };
    LatticeColourMatrix a(&Grid), b(&Grid);
    auto pack2 = [&](auto &out, const LatticeColourMatrix &c01,
                     const LatticeColourMatrix &c10, auto f01, auto f10,
                     int i0, int j0, int i1, int j1) {
      out = Zero();
      WeightComp(a, c01, f01);
      WeightComp(b, c10, f10);
      PokeIndex<1>(out, a, i0, j0);
      PokeIndex<1>(out, b, i1, j1);
    };
    // sigma/pi: (0,1) carries the +2 charge -> c; (1,0) -> 1/c
    pack2(sig_wm, sig01, sig10, wm, wp, 0, 1, 1, 0);
    pack2(sig_wp, sig01, sig10, wp, wm, 0, 1, 1, 0);
    pack2(sig_dm, sig01, sig10, dm, dp, 0, 1, 1, 0);
    pack2(pi_wm, pi01, pi10, wm, wp, 0, 1, 1, 0);
    pack2(pi_wp, pi01, pi10, wp, wm, 0, 1, 1, 0);
    pack2(pi_dm, pi01, pi10, dm, dp, 0, 1, 1, 0);
    // d/n: (0,0)=uu -> c; (1,1)=dd -> 1/c
    pack2(d_wm, d00, d11, wm, wp, 0, 0, 1, 1);
    pack2(d_wp, d00, d11, wp, wm, 0, 0, 1, 1);
    pack2(d_dm, d00, d11, dm, dp, 0, 0, 1, 1);
    pack2(d_dp, d00, d11, dp, dm, 0, 0, 1, 1);
    pack2(n_wm, n00, n11, wm, wp, 0, 0, 1, 1);
    pack2(n_wp, n00, n11, wp, wm, 0, 0, 1, 1);
    pack2(n_dm, n00, n11, dm, dp, 0, 0, 1, 1);
    pack2(n_dp, n00, n11, dp, dm, 0, 0, 1, 1);
  };

  // Vertex fields at e=0 for an ARBITRARY profile (ISO_LINEAR mode: the
  // O(eta) phase coefficient is profile-linear, so one undeformed solve
  // serves every profile's vertex).
  auto BuildVertexE0 = [&](const std::function<double(int)> &pf) {
    auto dm = [&](int t) { return -pf(t); };
    auto dp = [&](int t) { return +pf(t); };
    LatticeColourMatrix a(&Grid), b(&Grid);
    auto pack2 = [&](auto &out, const LatticeColourMatrix &c01,
                     const LatticeColourMatrix &c10, auto f01, auto f10,
                     int i0, int j0, int i1, int j1) {
      out = Zero();
      WeightComp(a, c01, f01);
      WeightComp(b, c10, f10);
      PokeIndex<1>(out, a, i0, j0);
      PokeIndex<1>(out, b, i1, j1);
    };
    pack2(sig_dm, sig01, sig10, dm, dp, 0, 1, 1, 0);
    pack2(pi_dm, pi01, pi10, dm, dp, 0, 1, 1, 0);
    pack2(d_dm, d00, d11, dm, dp, 0, 0, 1, 1);
    pack2(d_dp, d00, d11, dp, dm, 0, 0, 1, 1);
    pack2(n_dm, n00, n11, dm, dp, 0, 0, 1, 1);
    pack2(n_dp, n00, n11, dp, dm, 0, 0, 1, 1);
  };

  // ---- deformed operator + vertex ---------------------------------------
  typedef DTXQCDFermionDoubled F;
  auto AddDiag = [&](const LatticeDtxqcdSigma &sw, const LatticeDtxqcdPi &pw,
                     const F &in, F &out) {
    DTXQCDFermionNf c(&Grid);
    DtxqcdApplyDeltaDiag(sw, pw, s0, p0, in.upper, c);
    for (int a = 0; a < DtxqcdNf; ++a)
      out.upper.f[a] = out.upper.f[a] + c.f[a];
    DtxqcdApplyDeltaDiagLower(sw, pw, s0, p0, in.lower, c);
    for (int a = 0; a < DtxqcdNf; ++a)
      out.lower.f[a] = out.lower.f[a] + c.f[a];
  };
  auto AddCross = [&](const LatticeDtxqcdD &dd, const LatticeDtxqcdN &nn,
                      const F &in, F &out, bool lower_to_upper) {
    DTXQCDFermionNf c(&Grid);
    if (lower_to_upper) {
      DtxqcdApplyDnCross(dd, nn, in.lower, c, /*apply_conj=*/false);
      for (int a = 0; a < DtxqcdNf; ++a)
        out.upper.f[a] = out.upper.f[a] + c.f[a];
    } else {
      DtxqcdApplyDnCross(dd, nn, in.upper, c, /*apply_conj=*/true);
      for (int a = 0; a < DtxqcdNf; ++a)
        out.lower.f[a] = out.lower.f[a] + c.f[a];
    }
  };
  auto Mdef = [&](const F &in, F &out) {
    Mop.M(in, out);
    if (!dn_only) AddDiag(sig_wm, pi_wm, in, out);
    AddCross(d_wm, n_wm, in, out, true);    // UR: uu weighted c, dd 1/c
    AddCross(d_wp, n_wp, in, out, false);   // LL (conj'd): inverse weights
  };
  auto MdefDag = [&](const F &in, F &out) {
    Mop.Mdag(in, out);
    if (!dn_only) AddDiag(sig_wp, pi_wp, in, out);   // Hermitian pair swaps
    AddCross(d_wp, n_wp, in, out, true);
    AddCross(d_wm, n_wm, in, out, false);
  };
  auto dMiso = [&](const F &in, F &out) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = Zero();
      out.lower.f[a] = Zero();
    }
    if (!dn_only) AddDiag(sig_dm, pi_dm, in, out);
    AddCross(d_dm, n_dm, in, out, true);
    AddCross(d_dp, n_dp, in, out, false);
  };

  auto axpyF = [&](F &y, RealD a, const F &x) {
    for (int b = 0; b < DtxqcdNf; ++b) {
      y.upper.f[b] = y.upper.f[b] + a * x.upper.f[b];
      y.lower.f[b] = y.lower.f[b] + a * x.lower.f[b];
    }
  };
  auto norm2F = [&](const F &x) {
    RealD n = 0.0;
    for (int b = 0; b < DtxqcdNf; ++b)
      n += norm2(x.upper.f[b]) + norm2(x.lower.f[b]);
    return n;
  };
  auto innerF = [&](const F &x, const F &y) {
    ComplexD z(0.0, 0.0);
    for (int b = 0; b < DtxqcdNf; ++b)
      z += innerProduct(x.upper.f[b], y.upper.f[b]) +
           innerProduct(x.lower.f[b], y.lower.f[b]);
    return z;
  };

  // ---- DERIVCHECK: dMiso == FD of Mdef in eta ---------------------------
  if (env_int("DERIVCHECK", 1) != 0) {
    const double e0 = 0.5 * eta, de = 1e-5;
    F v(&Grid), ya(&Grid), yp(&Grid), ym(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(pRNG, v.upper.f[a]);
      gaussian(pRNG, v.lower.f[a]);
    }
    BuildWeights(e0);
    dMiso(v, ya);
    BuildWeights(e0 + de);
    Mdef(v, yp);
    BuildWeights(e0 - de);
    Mdef(v, ym);
    axpyF(yp, -1.0, ym);
    RealD fd2 = 0.0, dev2 = 0.0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      LatticeFermion du(&Grid), dl(&Grid);
      du = ComplexD(0.5 / de, 0.0) * yp.upper.f[a] - ya.upper.f[a];
      dl = ComplexD(0.5 / de, 0.0) * yp.lower.f[a] - ya.lower.f[a];
      dev2 += norm2(du) + norm2(dl);
      du = ComplexD(0.5 / de, 0.0) * yp.upper.f[a];
      dl = ComplexD(0.5 / de, 0.0) * yp.lower.f[a];
      fd2 += norm2(du) + norm2(dl);
    }
    const double rel = std::sqrt(dev2 / std::max(fd2, 1e-300));
    std::cout << GridLogMessage << "ISO-DERIVCHECK |dMiso - FD(Mdef)|/|FD| = "
              << rel << (rel < 1e-4 ? "   MATCH" : "   MISMATCH") << std::endl;
    if (rel >= 1e-4) { Grid_finalize(); std::_Exit(2); }
  }

  // ---- ISO_LINEAR: O(eta) phase coefficients for a PROFILE LADDER --------
  // The small-eta phase is eta * c_config with c_config = Im (1/2)Tr[M^-1
  // dM_prof] at e=0, LINEAR in the profile.  One UNDEFORMED solve per noise
  // vector serves every profile's vertex, so the sharp-vs-rounded kink
  // comparison (the sigma_theta rounding lever) costs a single solve set.
  // Output: per-profile mean re/im +- err; the config-to-config spread of
  // the im column across a config scan IS sigma_theta/eta per profile.
  if (env_int("ISO_LINEAR", 0) != 0) {
    struct PV { std::string name; std::function<double(int)> p; };
    auto tent_r = [&](double w, double wsrc) {
      return std::function<double(int)>([&, w, wsrc](int t) {
        double d = (double)(((t - prof_t0) % T + T) % T);
        double v = std::min(d, (double)T - d);
        const double a = std::abs(d - 0.5 * T);
        if (w > 0 && a < w) v = 0.5 * T - 0.5 * w - a * a / (2.0 * w);
        if (wsrc > 0) {
          const double b = std::min(d, (double)T - d);
          if (b < wsrc) v = 0.5 * wsrc + b * b / (2.0 * wsrc);
        }
        return v;
      });
    };
    const double wpi = 1.0 / 0.5943;
    std::vector<PV> pv = {
        {"tent_sharp", tent_r(0.0, 0.0)},
        {"round_1mpi", tent_r(wpi, wpi)},
        {"round_2mpi", tent_r(2.0 * wpi, 2.0 * wpi)},
        {"round_3mpi", tent_r(3.0 * wpi, 3.0 * wpi)},
    };
    const int np = (int)pv.size();
    const RealD noise_var0 = 2.0;            // Grid gaussian per-component
    std::vector<ComplexD> acc(np, ComplexD(0, 0));
    std::vector<RealD> a2r(np, 0), a2i(np, 0);
    BuildWeights(0.0);                       // Mdef == undeformed M
    for (int s = 0; s < n_noise; ++s) {
      F xi(&Grid), x(&Grid), b(&Grid), r(&Grid), pvv(&Grid), t1(&Grid),
          Ap(&Grid);
      for (int a = 0; a < DtxqcdNf; ++a) {
        gaussian(pRNG, xi.upper.f[a]);
        gaussian(pRNG, xi.lower.f[a]);
      }
      MdefDag(xi, b);
      for (int a = 0; a < DtxqcdNf; ++a) {
        x.upper.f[a] = Zero(); x.lower.f[a] = Zero();
      }
      r = b; pvv = b;
      RealD r2 = norm2F(r), b2 = std::max(norm2F(b), 1e-300);
      const RealD tol2 = tol * tol * b2;
      for (int k = 0; k < cgmax; ++k) {
        Mdef(pvv, t1);
        MdefDag(t1, Ap);
        RealD alpha = r2 / innerF(pvv, Ap).real();
        axpyF(x, +alpha, pvv);
        axpyF(r, -alpha, Ap);
        RealD rn = norm2F(r);
        if (rn < tol2) break;
        RealD beta = rn / r2;
        for (int a = 0; a < DtxqcdNf; ++a) {
          pvv.upper.f[a] = r.upper.f[a] + beta * pvv.upper.f[a];
          pvv.lower.f[a] = r.lower.f[a] + beta * pvv.lower.f[a];
        }
        r2 = rn;
      }
      if (std::sqrt(norm2F(r) / b2) > tol) {
        std::cout << GridLogMessage << "[iso-CG] NOT CONVERGED -- abort"
                  << std::endl;
        Grid_finalize();
        std::_Exit(3);
      }
      for (int j = 0; j < np; ++j) {
        BuildVertexE0(pv[j].p);
        dMiso(x, t1);
        const ComplexD est = 0.5 * innerF(xi, t1) / noise_var0;
        acc[j] += est;
        a2r[j] += est.real() * est.real();
        a2i[j] += est.imag() * est.imag();
      }
    }
    for (int j = 0; j < np; ++j) {
      const ComplexD m = acc[j] / (RealD)n_noise;
      auto serr = [&](RealD a2, RealD mm) {
        return std::sqrt(std::max(a2 / n_noise - mm * mm, 0.0) /
                         std::max(n_noise - 1, 1));
      };
      std::cout << GridLogMessage << "ISO-LINEAR " << pv[j].name
                << "  c = " << m.real() << " +- " << serr(a2r[j], m.real())
                << "  im " << m.imag() << " +- " << serr(a2i[j], m.imag())
                << std::endl;
    }
    Grid_finalize();
    std::fflush(nullptr);
    std::_Exit(0);
  }

  // ---- GL quadrature + Hutchinson ---------------------------------------
  static const double gl_x1[1] = {0.5};
  static const double gl_w1[1] = {1.0};
  static const double gl_x2[2] = {0.21132486540518713, 0.78867513459481290};
  static const double gl_w2[2] = {0.5, 0.5};
  static const double gl_x4[4] = {0.06943184420297371, 0.33000947820757187,
                                  0.66999052179242810, 0.93056815579702629};
  static const double gl_w4[4] = {0.17392742256872693, 0.32607257743127307,
                                  0.32607257743127307, 0.17392742256872693};
  static const double gl_x8[8] = {0.01985507, 0.10166676, 0.23723379, 0.40828268,
                                  0.59171732, 0.76276621, 0.89833324, 0.98014493};
  static const double gl_w8[8] = {0.05061427, 0.11119052, 0.15685332, 0.18134189,
                                  0.18134189, 0.15685332, 0.11119052, 0.05061427};
  GRID_ASSERT(n_gl == 1 || n_gl == 2 || n_gl == 4 || n_gl == 8);
  const double *gl_x = (n_gl == 1) ? gl_x1 : (n_gl == 2) ? gl_x2
                       : (n_gl == 4) ? gl_x4 : gl_x8;
  const double *gl_w = (n_gl == 1) ? gl_w1 : (n_gl == 2) ? gl_w2
                       : (n_gl == 4) ? gl_w4 : gl_w8;

  const RealD noise_var = 2.0;   // Grid gaussian: N(0,1) per real component

  ComplexD dlogpf(0.0, 0.0);
  RealD dlogpf_var_re = 0.0, dlogpf_var_im = 0.0;
  for (int q = 0; q < n_gl; ++q) {
    const double e = eta * gl_x[q];
    BuildWeights(e);
    ComplexD acc(0.0, 0.0);
    RealD acc2_re = 0.0, acc2_im = 0.0;
    int last_it = 0;
    for (int s = 0; s < n_noise; ++s) {
      F xi(&Grid), x(&Grid), b(&Grid), r(&Grid), pv(&Grid), t1(&Grid), Ap(&Grid);
      for (int a = 0; a < DtxqcdNf; ++a) {
        gaussian(pRNG, xi.upper.f[a]);
        gaussian(pRNG, xi.lower.f[a]);
      }
      MdefDag(xi, b);
      for (int a = 0; a < DtxqcdNf; ++a) {
        x.upper.f[a] = Zero(); x.lower.f[a] = Zero();
      }
      r = b; pv = b;
      RealD r2 = norm2F(r), b2 = std::max(norm2F(b), 1e-300);
      const RealD tol2 = tol * tol * b2;
      int k = 0;
      for (; k < cgmax; ++k) {
        Mdef(pv, t1);
        MdefDag(t1, Ap);
        RealD pAp = innerF(pv, Ap).real();
        RealD alpha = r2 / pAp;
        axpyF(x, +alpha, pv);
        axpyF(r, -alpha, Ap);
        RealD rn = norm2F(r);
        if (rn < tol2) break;
        RealD beta = rn / r2;
        for (int a = 0; a < DtxqcdNf; ++a) {
          pv.upper.f[a] = r.upper.f[a] + beta * pv.upper.f[a];
          pv.lower.f[a] = r.lower.f[a] + beta * pv.lower.f[a];
        }
        r2 = rn;
      }
      const RealD relres = std::sqrt(norm2F(r) / b2);
      if (relres > tol) {
        std::cout << GridLogMessage << "[iso-CG] NOT CONVERGED at CG_MAX="
                  << cgmax << " (relres " << relres << ") -- aborting"
                  << std::endl;
        Grid_finalize();
        std::_Exit(3);
      }
      last_it = k + 1;
      dMiso(x, t1);
      const ComplexD est = 0.5 * innerF(xi, t1) / noise_var;
      acc += est;
      acc2_re += est.real() * est.real();
      acc2_im += est.imag() * est.imag();
    }
    const ComplexD mean = acc / (RealD)n_noise;
    auto serr = [&](RealD a2, RealD m) {
      return std::sqrt(std::max(a2 / n_noise - m * m, 0.0) /
                       std::max(n_noise - 1, 1));
    };
    const RealD err_re = serr(acc2_re, mean.real());
    const RealD err_im = serr(acc2_im, mean.imag());
    std::cout << GridLogMessage << "ISO-NODE q=" << q << " e=" << e
              << " (1/2)Tr[M^-1 dM/de] = " << mean.real() << " +- " << err_re
              << "  im " << mean.imag() << " +- " << err_im
              << "  iters=" << last_it << std::endl;
    dlogpf += eta * gl_w[q] * mean;
    dlogpf_var_re += (eta * gl_w[q] * err_re) * (eta * gl_w[q] * err_re);
    dlogpf_var_im += (eta * gl_w[q] * err_im) * (eta * gl_w[q] * err_im);
  }
  std::cout << GridLogMessage << "ISO-RESULT eta=" << eta
            << (wardchk ? " [WARDCHECK: expect 0]" : "")
            << " dlogPf re = " << dlogpf.real() << " +- "
            << std::sqrt(dlogpf_var_re) << "  im (phase) = " << dlogpf.imag()
            << " +- " << std::sqrt(dlogpf_var_im) << std::endl;
  if (wardchk) {
    const RealD z = std::abs(dlogpf.real()) /
                    std::max(std::sqrt(dlogpf_var_re), (RealD)1e-30);
    const RealD zi = std::abs(dlogpf.imag()) /
                     std::max(std::sqrt(dlogpf_var_im), (RealD)1e-30);
    std::cout << GridLogMessage << "ISO-WARDCHECK z_re=" << z
              << " z_im=" << zi
              << ((z < 4 && zi < 4) ? "   PASSED" : "   FAILED") << std::endl;
  }

  Grid_finalize();
  std::fflush(nullptr);
  std::_Exit(0);
}
