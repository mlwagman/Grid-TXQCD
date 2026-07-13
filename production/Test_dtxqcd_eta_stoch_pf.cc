// Test_dtxqcd_eta_stoch_pf: STOCHASTIC estimator of the eta-deformation
// reweighting factor  Delta log det M48(eta)  -- the production-scalable
// replacement for the dense LU of Test_dtxqcd_eta_deform_pf.
//
//   Delta log det(eta) = int_0^eta deta' Tr[ M(eta')^{-1} dM(eta')/deta' ]
//
// with the deformation d,n -> e^{-eta p(t)} (UR), d*,n* -> e^{+eta p(t)} (LL),
// p(t) the profile (tent: min(t, T-t); linear: t).  dM/deta is SITE-LOCAL
// (the -/+ p(t)-weighted scaled cross blocks), so the trace is Hutchinson-
// estimable with gaussian noise + CG solves of the DEFORMED operator:
//
//   M_def = M + C_UR[w-] + C_LL[w+],   w-+ = e^{-+eta p(t)} - 1
//   M_def^dag = M^dag + C_UR-slot[w+] + C_LL-slot[w-]     (weights swap;
//     complex symmetry d^T = d gives d^dag = d*, so the adjoint of each
//     cross branch is the OTHER branch with the same real weight)
//
// realized entirely with the exposed DtxqcdApplyDnCross (coefficient +1 in
// AddDiagAndCrossAndClover; the apply_conj branch conjugates internally and
// real weights pass through conj()).
//
// Quadrature: Gauss-Legendre (N_GL nodes) in eta' -- the integrand is
// analytic.  Noise: Grid gaussian (E<v,Av> = 2 Tr A -> divide by 2),
// N_NOISE vectors per node.  Output: per-node complex traces + the
// integrated d log|det| (Re) and d arg (Im) with jackknife-style noise
// errors.  Validate against the dense ETA_SCAN values on 4^4 / 4^3x8;
// then any geometry (8^4, production 16^3x48) is reachable.
//
// Env: LATT (4.4.4.4), CFG_DIR, TRAJ (250), ETA (0.3), PROFILE (tent|linear),
//      N_NOISE (16), N_GL (8), MASS (-0.245), CSW (1.24930970916466),
//      STOUT_NSMEAR (1), STOUT_RHO (0.125), CG_TOL (1e-8), CG_MAX (20000),
//      NOISE_SEED (77).

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

  RealD mass = env_real("MASS", -0.245);
  RealD csw  = env_real("CSW", 1.24930970916466);
  int nstout = env_int("STOUT_NSMEAR", 1);
  RealD rho  = env_real("STOUT_RHO", 0.125);
  RealD eta  = env_real("ETA", 0.3);
  int n_noise = env_int("N_NOISE", 16);
  int n_gl    = env_int("N_GL", 8);
  RealD tol   = env_real("CG_TOL", 1e-8);
  int cgmax   = env_int("CG_MAX", 20000);
  int traj    = env_int("TRAJ", 250);
  const bool tent = [&] {
    const char *v = std::getenv("PROFILE");
    return !v || std::string(v) != "linear";   // tent is the default
  }();
  std::string cfg_dir = "/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/"
                        "dtxqcd_lam4.0000_4t4_signhop_lam4";
  if (const char *d = std::getenv("CFG_DIR"); d && *d) cfg_dir = d;

  std::cout << GridLogMessage << "==== Test_dtxqcd_eta_stoch_pf ====" << std::endl;
  std::cout << GridLogMessage << " cfg=" << cfg_dir << " traj=" << traj
            << " eta=" << eta << " profile=" << (tent ? "tent" : "linear")
            << " N_NOISE=" << n_noise << " N_GL=" << n_gl << std::endl;

  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid);
  DTXQCDField U(&Grid);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);
  // Independent noise seed (decoupled from the checkpoint RNG state).
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

  // t-profile weight applied to a d/n-type aux field: out = f(t) * field.
  LatticeInteger tc(&Grid);
  LatticeCoordinate(tc, Nd - 1);
  auto prof = [&](int t) { return tent ? (double)std::min(t, T - t) : (double)t; };
  auto WeightField = [&](const auto &f, auto &out, auto fac) {
    out = f;
    for (int t = 0; t < T; ++t) {
      auto scaled = f;
      scaled = ComplexD(fac(t), 0.0) * scaled;
      out = where(tc == Integer(t), scaled, out);
    }
  };

  // Scaled copies of d, n for the deformed operator and its eta-derivative,
  // rebuilt per quadrature node.
  LatticeDtxqcdD d_wm(&Grid), d_wp(&Grid), d_dm(&Grid), d_dp(&Grid);
  LatticeDtxqcdN n_wm(&Grid), n_wp(&Grid), n_dm(&Grid), n_dp(&Grid);

  auto BuildWeights = [&](double e) {
    WeightField(U.d, d_wm, [&](int t) { return std::exp(-e * prof(t)) - 1.0; });
    WeightField(U.n, n_wm, [&](int t) { return std::exp(-e * prof(t)) - 1.0; });
    WeightField(U.d, d_wp, [&](int t) { return std::exp(+e * prof(t)) - 1.0; });
    WeightField(U.n, n_wp, [&](int t) { return std::exp(+e * prof(t)) - 1.0; });
    WeightField(U.d, d_dm, [&](int t) { return -prof(t) * std::exp(-e * prof(t)); });
    WeightField(U.n, n_dm, [&](int t) { return -prof(t) * std::exp(-e * prof(t)); });
    WeightField(U.d, d_dp, [&](int t) { return +prof(t) * std::exp(+e * prof(t)); });
    WeightField(U.n, n_dp, [&](int t) { return +prof(t) * std::exp(+e * prof(t)); });
  };

  typedef DTXQCDFermionDoubled F;
  auto AddCross = [&](const LatticeDtxqcdD &dd, const LatticeDtxqcdN &nn,
                      const F &in, F &out, bool lower_to_upper) {
    DTXQCDFermionNf c(&Grid);
    if (lower_to_upper) {          // UR slot: in.lower -> out.upper
      DtxqcdApplyDnCross(dd, nn, in.lower, c, /*apply_conj=*/false);
      for (int a = 0; a < DtxqcdNf; ++a)
        out.upper.f[a] = out.upper.f[a] + c.f[a];
    } else {                       // LL slot: in.upper -> out.lower (conj'd)
      DtxqcdApplyDnCross(dd, nn, in.upper, c, /*apply_conj=*/true);
      for (int a = 0; a < DtxqcdNf; ++a)
        out.lower.f[a] = out.lower.f[a] + c.f[a];
    }
  };
  auto Mdef = [&](const F &in, F &out) {
    Mop.M(in, out);
    AddCross(d_wm, n_wm, in, out, true);    // UR weighted by e^{-} - 1
    AddCross(d_wp, n_wp, in, out, false);   // LL weighted by e^{+} - 1
  };
  auto MdefDag = [&](const F &in, F &out) {
    Mop.Mdag(in, out);
    AddCross(d_wp, n_wp, in, out, true);    // weights SWAP under dagger
    AddCross(d_wm, n_wm, in, out, false);
  };
  auto dM = [&](const F &in, F &out) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = Zero();
      out.lower.f[a] = Zero();
    }
    AddCross(d_dm, n_dm, in, out, true);
    AddCross(d_dp, n_dp, in, out, false);
  };

  // Custom CG on Mdef^dag Mdef (pattern of dtxqcd_diag.h compute_trminv).
  auto SolveDef = [&](const F &src, F &x) {
    F b(&Grid), r(&Grid), p(&Grid), t1(&Grid), Ap(&Grid);
    MdefDag(src, b);
    x = Zero();
    r = b; p = b;
    RealD r2 = norm2(r), b2 = std::max(norm2(b), 1e-300);
    RealD tol2 = tol * tol * b2;
    int k = 0;
    for (; k < cgmax; ++k) {
      Mdef(p, t1);
      MdefDag(t1, Ap);
      RealD pAp = innerProduct(p, Ap).real();
      RealD alpha = r2 / pAp;
      for (int a = 0; a < DtxqcdNf; ++a) {
        x.upper.f[a] = x.upper.f[a] + alpha * p.upper.f[a];
        x.lower.f[a] = x.lower.f[a] + alpha * p.lower.f[a];
        r.upper.f[a] = r.upper.f[a] - alpha * Ap.upper.f[a];
        r.lower.f[a] = r.lower.f[a] - alpha * Ap.lower.f[a];
      }
      RealD rn = norm2(r);
      if (rn < tol2) break;
      RealD beta = rn / r2;
      for (int a = 0; a < DtxqcdNf; ++a) {
        p.upper.f[a] = r.upper.f[a] + beta * p.upper.f[a];
        p.lower.f[a] = r.lower.f[a] + beta * p.lower.f[a];
      }
      r2 = rn;
    }
  };

  // OPCHECK=1 (4^4-class only): machine-precision validation of the lattice
  // deformed operator against the INDEPENDENT dense implementation --
  // assemble dense M48 columnwise from the UNDEFORMED Mop, sector-scale it
  // (the construction already validated by the det/gauge tests), and compare
  // H_def . v against Mdef(v) and H_def^dag . v against MdefDag(v) on random
  // vectors.  Proves the weighted-cross corrections exactly, no statistics.
  if (env_int("OPCHECK", 0) != 0) {
    const int V = (int)Grid.gSites();
    GRID_ASSERT(Grid.gSites() == Grid.lSites() && V <= 4096);
    const int dim = 48 * V;
    auto SiteCoor = [&](int lex, Coordinate &c) {
      const Coordinate ld = Grid.GlobalDimensions();
      c.resize(4);
      c[0] = lex % ld[0]; lex /= ld[0];
      c[1] = lex % ld[1]; lex /= ld[1];
      c[2] = lex % ld[2]; lex /= ld[2];
      c[3] = lex;
    };
    typedef typename LatticeFermion::scalar_object FSobj;
    std::vector<std::complex<double>> H((size_t)dim * dim);
    {
      DTXQCDFermionDoubled e(&Grid), y(&Grid);
      std::vector<FSobj> col(V);
      for (int site = 0; site < V; ++site) {
        Coordinate c; SiteCoor(site, c);
        for (int blk = 0; blk < 2; ++blk)
          for (int a = 0; a < DtxqcdNf; ++a)
            for (int sp = 0; sp < Ns; ++sp)
              for (int co = 0; co < Nc; ++co) {
                const int j = ((site * 2 + blk) * DtxqcdNf + a) * 12 + sp * 3 + co;
                for (int aa = 0; aa < DtxqcdNf; ++aa) {
                  e.upper.f[aa] = Zero(); e.lower.f[aa] = Zero();
                }
                FSobj one; one = Zero();
                one()(sp)(co) = ComplexD(1.0, 0.0);
                if (blk == 0) pokeSite(one, e.upper.f[a], c);
                else          pokeSite(one, e.lower.f[a], c);
                Mop.M(e, y);
                for (int b2 = 0; b2 < 2; ++b2)
                  for (int a2 = 0; a2 < DtxqcdNf; ++a2) {
                    unvectorizeToLexOrdArray(col, b2 == 0 ? y.upper.f[a2]
                                                          : y.lower.f[a2]);
                    for (int s2 = 0; s2 < V; ++s2)
                      for (int sp2 = 0; sp2 < Ns; ++sp2)
                        for (int co2 = 0; co2 < Nc; ++co2) {
                          const int i =
                              ((s2 * 2 + b2) * DtxqcdNf + a2) * 12 + sp2 * 3 + co2;
                          auto z = col[s2]()(sp2)(co2);
                          H[(size_t)j * dim + i] =
                              std::complex<double>(z.real(), z.imag());
                        }
                  }
              }
      }
    }
    // Sector-scale (the tent/linear deformation) exactly as the dense test.
    for (int sj = 0; sj < V; ++sj)
      for (int bj = 0; bj < 2; ++bj)
        for (int k = 0; k < 24; ++k) {
          const int j = (sj * 2 + bj) * 24 + k;
          for (int si = 0; si < V; ++si) {
            Coordinate c; SiteCoor(si, c);
            const double t = prof(c[3]);
            for (int bi = 0; bi < 2; ++bi) {
              double f = 1.0;
              if (bi == 0 && bj == 1) f = std::exp(-eta * t);
              if (bi == 1 && bj == 0) f = std::exp(+eta * t);
              if (f == 1.0) continue;
              for (int m = 0; m < 24; ++m)
                H[(size_t)j * dim + (si * 2 + bi) * 24 + m] *= f;
            }
          }
        }
    BuildWeights(eta);
    // Random vector comparison for Mdef and MdefDag.
    auto ToVec = [&](const F &f, std::vector<std::complex<double>> &v) {
      std::vector<FSobj> col(V);
      for (int b = 0; b < 2; ++b)
        for (int a = 0; a < DtxqcdNf; ++a) {
          unvectorizeToLexOrdArray(col, b == 0 ? f.upper.f[a] : f.lower.f[a]);
          for (int s = 0; s < V; ++s)
            for (int sp = 0; sp < Ns; ++sp)
              for (int co = 0; co < Nc; ++co) {
                auto z = col[s]()(sp)(co);
                v[((s * 2 + b) * DtxqcdNf + a) * 12 + sp * 3 + co] =
                    std::complex<double>(z.real(), z.imag());
              }
        }
    };
    F vf(&Grid), yf(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(pRNG, vf.upper.f[a]);
      gaussian(pRNG, vf.lower.f[a]);
    }
    std::vector<std::complex<double>> vv(dim), yl(dim), yd(dim);
    ToVec(vf, vv);
    for (int pass = 0; pass < 2; ++pass) {
      if (pass == 0) Mdef(vf, yf); else MdefDag(vf, yf);
      ToVec(yf, yl);
      // dense: yd = H.v (pass 0) or H^dag.v (pass 1); column-major H.
      for (int i = 0; i < dim; ++i) yd[i] = 0.0;
      for (int j = 0; j < dim; ++j)
        for (int i = 0; i < dim; ++i) {
          if (pass == 0) yd[i] += H[(size_t)j * dim + i] * vv[j];
          else           yd[j] += std::conj(H[(size_t)j * dim + i]) * vv[i];
        }
      double dmax = 0.0, scale = 0.0;
      for (int i = 0; i < dim; ++i) {
        dmax = std::max(dmax, std::abs(yl[i] - yd[i]));
        scale = std::max(scale, std::abs(yd[i]));
      }
      std::cout << GridLogMessage << "OPCHECK " << (pass ? "Mdag" : "M   ")
                << " eta=" << eta << " profile=" << (tent ? "tent" : "linear")
                << ": max|lattice-dense| = " << dmax << "  (scale " << scale
                << ")  " << (dmax < 1e-8 * scale ? "MATCH" : "MISMATCH")
                << std::endl;
    }
    Grid_finalize();
    std::fflush(nullptr);
    std::_Exit(0);
  }

  // 8-node Gauss-Legendre on [0,1] (rescaled to [0,eta]); truncate to n_gl.
  static const double gl_x[8] = {0.01985507, 0.10166676, 0.23723379, 0.40828268,
                                 0.59171732, 0.76276621, 0.89833324, 0.98014493};
  static const double gl_w[8] = {0.05061427, 0.11119052, 0.15685332, 0.18134189,
                                 0.18134189, 0.15685332, 0.11119052, 0.05061427};
  GRID_ASSERT(n_gl >= 1 && n_gl <= 8);

  ComplexD dlogdet(0.0, 0.0), dlogdet_var(0.0, 0.0);
  for (int q = 0; q < n_gl; ++q) {
    const double ep = eta * gl_x[q];
    BuildWeights(ep);
    // Hutchinson: Tr[Mdef^{-1} dM] = E<v, Mdef^{-1} dM v> / 2 (gaussian).
    ComplexD acc(0.0, 0.0), acc2r(0.0, 0.0), acc2i(0.0, 0.0);
    for (int h = 0; h < n_noise; ++h) {
      // Grading-antithetic pair: dM is block-grading ODD, so the odd part of
      // M^{-1} dM dominates the Hutchinson variance yet contributes ZERO to
      // the trace.  Averaging v with Gv (G = block-parity flip of the lower
      // components) cancels it exactly per pair: 2x solves, ~10-40x variance.
      F v(&Grid), y(&Grid), x(&Grid);
      for (int a = 0; a < DtxqcdNf; ++a) {
        gaussian(pRNG, v.upper.f[a]);
        gaussian(pRNG, v.lower.f[a]);
      }
      dM(v, y);
      SolveDef(y, x);
      ComplexD est1 = innerProduct(v, x) / 2.0;
      for (int a = 0; a < DtxqcdNf; ++a)
        v.lower.f[a] = -v.lower.f[a];                    // v -> Gv
      dM(v, y);
      SolveDef(y, x);
      ComplexD est2 = innerProduct(v, x) / 2.0;
      ComplexD est = 0.5 * (est1 + est2);
      acc += est;
      acc2r += est.real() * est.real();
      acc2i += est.imag() * est.imag();
    }
    ComplexD g = acc / (double)n_noise;
    double er = std::sqrt(std::max(0.0, acc2r.real() / n_noise -
                                        g.real() * g.real()) / n_noise);
    double ei = std::sqrt(std::max(0.0, acc2i.real() / n_noise -
                                        g.imag() * g.imag()) / n_noise);
    std::cout << GridLogMessage << "GLNODE q=" << q << " eta'=" << ep
              << " Tr[Minv dM] = " << g.real() << " +- " << er
              << "  (im " << g.imag() << " +- " << ei << ")" << std::endl;
    dlogdet += eta * gl_w[q] * g;
    dlogdet_var += ComplexD(eta * gl_w[q] * er * eta * gl_w[q] * er,
                            eta * gl_w[q] * ei * eta * gl_w[q] * ei);
  }
  std::cout << GridLogMessage << "STOCHPF traj=" << traj << " eta=" << eta
            << " dlogdet= " << dlogdet.real()
            << " +- " << std::sqrt(dlogdet_var.real())
            << " darg= " << dlogdet.imag()
            << " +- " << std::sqrt(dlogdet_var.imag()) << std::endl;

  Grid_finalize();
  std::fflush(nullptr);
  std::_Exit(0);
}
