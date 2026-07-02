// Test_dtxqcd_mpc_op_quda_eo — Gate 1 of the EO Style C port
// (HANDOFF_EO_STYLE_C_PORT.md): bit-exactness of the device-resident odd-CB
// Schur operator DTXQCDMpcOpQUDA_EO against the Grid reference DTXQCDMpcOp,
// decomposed piecewise so a failure localizes immediately:
//   (a) hop   : Meooe  (odd->even)        device vs Grid
//   (b) siteI : MooeeInv (even)           device vs Grid
//   (c) siteF : Mooee   (odd)             device vs Grid
//   (d) full  : Mpc                       device vs Grid
//   (e) dag   : Mpc^dag                   device vs Grid
// Each comparison prints rel-diff, norm ratio and Re-cosine — the latter two
// auto-diagnose scale/sign convention errors (bisect via DTXQCD_EO_HOP_SCALE).
//
// Run: ./Test_dtxqcd_mpc_op_quda_eo --grid 4.4.4.4 --mpi 1.1.1.1
// Env: LAMBDA (4.0), MASS (-0.245), CSW (1.24930970916466), NRAND (10),
//      GAUGE_KIND (tepid|hot|weak, default tepid)

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA_EO.h>
#include <Grid/qcd/action/dtxqcd/DoubledStateCSF.h>
#include <Grid/util/QudaInit.h>
#include <cstdlib>

using namespace Grid;

namespace {

RealD rel_diff(const DTXQCDFermionDoubled &a, const DTXQCDFermionDoubled &b) {
  RealD num = 0, den = 0;
  for (int f = 0; f < DtxqcdNf; ++f) {
    LatticeFermion du(a.upper.f[f].Grid()), dl(a.upper.f[f].Grid());
    du = a.upper.f[f] - b.upper.f[f];
    dl = a.lower.f[f] - b.lower.f[f];
    num += norm2(du) + norm2(dl);
    den += norm2(b.upper.f[f]) + norm2(b.lower.f[f]);
  }
  return std::sqrt(num / std::max(den, 1e-300));
}

RealD dnorm2(const DTXQCDFermionDoubled &a) {
  RealD n = 0;
  for (int f = 0; f < DtxqcdNf; ++f)
    n += norm2(a.upper.f[f]) + norm2(a.lower.f[f]);
  return n;
}

RealD recos(const DTXQCDFermionDoubled &a, const DTXQCDFermionDoubled &b) {
  ComplexD ip(0, 0);
  for (int f = 0; f < DtxqcdNf; ++f) {
    ip += innerProduct(a.upper.f[f], b.upper.f[f]);
    ip += innerProduct(a.lower.f[f], b.lower.f[f]);
  }
  return ip.real() / std::sqrt(dnorm2(a) * dnorm2(b));
}

void pick_cb(int cb, DTXQCDFermionDoubled &half,
             const DTXQCDFermionDoubled &full) {
  for (int f = 0; f < DtxqcdNf; ++f) {
    pickCheckerboard(cb, half.upper.f[f], full.upper.f[f]);
    pickCheckerboard(cb, half.lower.f[f], full.lower.f[f]);
  }
}

}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Grid::Quda::initialize();

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--grid" && i + 1 < argc) {
      std::vector<int> dv; std::stringstream ss(argv[i + 1]); std::string t;
      while (std::getline(ss, t, '.')) dv.push_back(std::stoi(t));
      if (dv.size() == (size_t)Nd) latt = Coordinate(dv);
    }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid_(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid_);
  GridParallelRNG pRNG(&Grid_); pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  auto env_real = [](const char *n, RealD d) {
    const char *e = std::getenv(n); return (e && *e) ? std::atof(e) : d;
  };
  auto env_int = [](const char *n, int d) {
    const char *e = std::getenv(n); return (e && *e) ? std::atoi(e) : d;
  };
  const RealD lambda = env_real("LAMBDA", 4.0);
  const RealD mass   = env_real("MASS", -0.245);
  const RealD csw    = env_real("CSW", 1.24930970916466);
  const int   nrand  = env_int("NRAND", 10);

  DTXQCDField U(&Grid_);
  std::string gk = std::getenv("GAUGE_KIND") ? std::getenv("GAUGE_KIND") : "tepid";
  if (gk == "hot")       SU<Nc>::HotConfiguration(pRNG, U.U);
  else if (gk == "weak") DTXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, 0.1);
  else                   PeriodicGimplR::TepidConfiguration(pRNG, U.U);
  DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, /*Sigma=*/0.0);

  std::cout << GridLogMessage << "[gate1-eo] gauge=" << gk << " lambda=" << lambda
            << " mass=" << mass << " csw=" << csw << " nrand=" << nrand
            << std::endl;

  DTXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass, csw,
                                 U.sigma, U.pi, U.d, U.n, U.s, U.p);
  DTXQCDMpcOp mpc(Dw);
  DTXQCDMpcOpQUDA_EO Mq(Dw, U.U, Grid_, RBGrid, mass, csw,
                        U.sigma, U.pi, U.d, U.n, U.s, U.p);

  int X_full_dims[4];
  for (int d_ = 0; d_ < 4; ++d_) X_full_dims[d_] = Mq.GaugeParam().X[d_];
  quda::ColorSpinorParam ptmpl = Mq.MakeNativeCsfParam();

  DtxqcdQudaStyleC::DoubledStateCSF in_q, out_q, tmp_q, scr_q;
  in_q.allocate(ptmpl);
  out_q.allocate(ptmpl);
  tmp_q.allocate(ptmpl);
  scr_q.allocate_lower_only(ptmpl);

  auto ptrs = [](DtxqcdQudaStyleC::DoubledStateCSF &s,
                 quda::ColorSpinorField *up[DtxqcdNf],
                 quda::ColorSpinorField *lo[DtxqcdNf]) {
    up[0] = s.csf[0].get(); up[1] = s.csf[1].get();
    lo[0] = s.csf[2].get(); lo[1] = s.csf[3].get();
  };

  int fails = 0;
  auto check = [&](const char *name, const DTXQCDFermionDoubled &dev,
                   const DTXQCDFermionDoubled &ref, RealD tol) {
    RealD rd = rel_diff(dev, ref);
    RealD nr = std::sqrt(dnorm2(dev) / std::max(dnorm2(ref), 1e-300));
    RealD cs = recos(dev, ref);
    bool ok = rd < tol;
    if (!ok) ++fails;
    std::cout << GridLogMessage << (ok ? "[ok]   " : "[FAIL] ") << name
              << "  rel=" << rd << "  norm_ratio=" << nr << "  cos=" << cs
              << "  (tol " << tol << ")" << std::endl;
  };

  for (int r = 0; r < nrand; ++r) {
    // Random odd-CB doubled fermion.
    DTXQCDFermionDoubled full(&Grid_);
    for (int f = 0; f < DtxqcdNf; ++f) {
      gaussian(pRNG, full.upper.f[f]);
      gaussian(pRNG, full.lower.f[f]);
    }
    DTXQCDFermionDoubled psi_o(&RBGrid), tmp_e(&RBGrid), tmp_e2(&RBGrid);
    DTXQCDFermionDoubled ref_o(&RBGrid), dev_o(&RBGrid), dev_e(&RBGrid);
    pick_cb(Odd, psi_o, full);

    // ---- Grid reference pieces ----
    Dw.Meooe(psi_o, tmp_e);        // odd -> even
    Dw.MooeeInv(tmp_e, tmp_e2);    // even
    DTXQCDFermionDoubled mooee_o(&RBGrid);
    Dw.Mooee(psi_o, mooee_o);      // odd
    mpc.M(psi_o, ref_o);           // full Mpc

    // ---- (a) hop device vs Grid ----
    in_q.copy_from_grid(psi_o, Mq.InvertParam(), X_full_dims);
    quda::ColorSpinorField *iu[DtxqcdNf], *il[DtxqcdNf];
    quda::ColorSpinorField *ou[DtxqcdNf], *ol[DtxqcdNf];
    quda::ColorSpinorField *su[DtxqcdNf], *sl[DtxqcdNf];
    ptrs(in_q, iu, il);
    ptrs(out_q, ou, ol);
    su[0] = scr_q.csf[0].get(); sl[0] = su[0];
    su[1] = scr_q.csf[1].get(); sl[1] = su[1];
    quda::ColorSpinorField *in_all[4]  = {iu[0], iu[1], il[0], il[1]};
    quda::ColorSpinorField *out_all[4] = {ou[0], ou[1], ol[0], ol[1]};
    out_q.zero(); scr_q.zero();
    Mq.TestHopDevice(in_all, out_all, QUDA_EVEN_PARITY, su);
    dev_e = tmp_e;   // shape/cb template
    out_q.copy_to_grid(dev_e, Mq.InvertParam(), X_full_dims);
    for (int f = 0; f < DtxqcdNf; ++f) {
      dev_e.upper.f[f].Checkerboard() = Even;
      dev_e.lower.f[f].Checkerboard() = Even;
    }
    if (r == 0) check("hop  Meooe (o->e)", dev_e, tmp_e, 1e-10);

    // ---- (b) site inverse (even) ----
    in_q.copy_from_grid(tmp_e, Mq.InvertParam(), X_full_dims);
    out_q.zero();
    Mq.TestSiteMatDevice(in_all, out_all, Even, /*use_inverse=*/true);
    DTXQCDFermionDoubled dev_e2 = tmp_e2;
    out_q.copy_to_grid(dev_e2, Mq.InvertParam(), X_full_dims);
    for (int f = 0; f < DtxqcdNf; ++f) {
      dev_e2.upper.f[f].Checkerboard() = Even;
      dev_e2.lower.f[f].Checkerboard() = Even;
    }
    if (r == 0) check("siteI MooeeInv (e)", dev_e2, tmp_e2, 1e-10);

    // ---- (c) site forward (odd) ----
    in_q.copy_from_grid(psi_o, Mq.InvertParam(), X_full_dims);
    out_q.zero();
    Mq.TestSiteMatDevice(in_all, out_all, Odd, /*use_inverse=*/false);
    DTXQCDFermionDoubled dev_mo = mooee_o;
    out_q.copy_to_grid(dev_mo, Mq.InvertParam(), X_full_dims);
    for (int f = 0; f < DtxqcdNf; ++f) {
      dev_mo.upper.f[f].Checkerboard() = Odd;
      dev_mo.lower.f[f].Checkerboard() = Odd;
    }
    if (r == 0) check("siteF Mooee (o)", dev_mo, mooee_o, 1e-10);

    // ---- (d) full Mpc ----
    in_q.copy_from_grid(psi_o, Mq.InvertParam(), X_full_dims);
    out_q.zero(); scr_q.zero();
    Mq.M_device_csf(iu, il, ou, ol, su);
    dev_o = ref_o;
    out_q.copy_to_grid(dev_o, Mq.InvertParam(), X_full_dims);
    for (int f = 0; f < DtxqcdNf; ++f) {
      dev_o.upper.f[f].Checkerboard() = Odd;
      dev_o.lower.f[f].Checkerboard() = Odd;
    }
    check((std::string("full Mpc  rand ") + std::to_string(r)).c_str(),
          dev_o, ref_o, 1e-10);

    // ---- (e) Mpc^dag ----
    DTXQCDFermionDoubled refd_o(&RBGrid);
    mpc.Mdag(psi_o, refd_o);
    in_q.copy_from_grid(psi_o, Mq.InvertParam(), X_full_dims);
    out_q.zero(); scr_q.zero();
    Mq.Mdag_device_csf(iu, il, ou, ol, su);
    DTXQCDFermionDoubled devd_o = refd_o;
    out_q.copy_to_grid(devd_o, Mq.InvertParam(), X_full_dims);
    for (int f = 0; f < DtxqcdNf; ++f) {
      devd_o.upper.f[f].Checkerboard() = Odd;
      devd_o.lower.f[f].Checkerboard() = Odd;
    }
    if (r == 0) check("Mpc^dag", devd_o, refd_o, 1e-10);
  }

  std::cout << GridLogMessage << "==== gate1-eo "
            << (fails ? "FAILED" : "PASSED") << " (" << fails << " failures) ===="
            << std::endl;
  Grid_finalize();
  return fails ? 1 : 0;
}
