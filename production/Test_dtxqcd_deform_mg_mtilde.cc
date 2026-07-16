// Test_dtxqcd_deform_mg_mtilde: prototype for MG built on M~ ITSELF
// (Petrov-Galerkin coarsening of the non-Hermitian deformed EO operator),
// per DTXQCD_DEFORM_MG_MTILDE_DESIGN.md.  The normal-op (Mpc~^dag Mpc~)
// MG route is CLOSED (commit 58b9197f): its transfer must represent
// singular vectors of a strongly non-normal operator and measurably
// cannot, at any quality/count/geometry/level.  Coarsening M~ directly
// (i) un-squares kappa (1.6e4 vs 2.7e8 at lambda=5 eta=0.8) and (ii)
// aggregates the EIGENVECTOR near-kernel — the thing blocked transfers
// are good at.
//
// This program implements Steps 0-1 of the design + their gates:
//   S0a  adjointness:  <u, M~ v> == <M~dag u, v>          (device legs)
//   S0b  quasi-g5-Hermiticity:  M~(eta)^dag u == G5 M~(-eta) G5 u
//        (the load-bearing identity: left null spaces come free from the
//        reciprocal-weight right set; at eta=0 this IS g5-Hermiticity)
//   S1   right/left near-kernel generation by inverse iteration through
//        the certified CGNR solver at loose tol (NV_TOL, default 0.1);
//        left vectors = G5 x (near-kernel of M~(-eta)).  Reports |M~ v|
//        per vector — the design's block-locality precondition check.
//
// Envs: DEFORM_ETA(0.6) DEFORM_T0(0) NV_NEV(16) NV_TOL(0.1) NV_MAXIT(20000)
//       LAMBDA MASS CSW CFG_DIR TRAJ GAUGE_KIND (as Test_dtxqcd_grid_mg).
// Run (4^4 mechanics):  env LAMBDA=5 DEFORM_ETA=0.4 NV_NEV=4 ./this --grid 4.4.4.8
// Run (16^3x48):        env CFG_DIR=... TRAJ=300 LAMBDA=5 DEFORM_ETA=0.6 ./this ...

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDPackedField.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/util/QudaInit.h>
#endif

#include "params.h"
#include "dtxqcd_meas_helper.h"
#ifdef GRID_HAVE_QUDA
#include "dtxqcd_deform_quda.h"
#endif

#include <cstdlib>
#include <string>
#if defined(GRID_CUDA) && defined(GRID_HAVE_QUDA)
#include <cusolverDn.h>
#endif

using namespace Grid;

namespace {

double env_real(const char *n, double d) {
  const char *e = std::getenv(n);
  return (e && *e) ? std::atof(e) : d;
}
int env_int(const char *n, int d) {
  const char *e = std::getenv(n);
  return (e && *e) ? std::atoi(e) : d;
}
std::string env_str(const char *n, const char *d) {
  const char *e = std::getenv(n);
  return (e && *e) ? std::string(e) : std::string(d);
}

Coordinate parse_dotted(const std::string &s, Coordinate def) {
  Coordinate out = def;
  size_t p = 0;
  for (int d = 0; d < Nd && p < s.size(); ++d) {
    size_t q = s.find('.', p);
    out[d] = std::atoi(s.substr(p, q - p).c_str());
    if (q == std::string::npos) break;
    p = q + 1;
  }
  return out;
}

#ifdef GRID_HAVE_QUDA

typedef DTXQCDFermionDoubled F;

// G5 on the doubled struct field: gamma5 per fermion component (spin-only;
// identity on the doubling/flavor indices).  Gate S0b at eta=0 CERTIFIES
// this is the convention under which the Schur complement is
// g5-Hermitian; if the doubled block needed a relative sign, S0b would
// fail at eta=0 and this is the first place to look.
void Gamma5All(const F &in, F &out) {
  Gamma g5(Gamma::Algebra::Gamma5);
  for (int a = 0; a < DtxqcdNf; ++a) {
    out.upper.f[a] = g5 * in.upper.f[a];
    out.lower.f[a] = g5 * in.lower.f[a];
  }
}

RealD Fnorm2(const F &x) {
  RealD s = 0;
  for (int a = 0; a < DtxqcdNf; ++a)
    s += norm2(x.upper.f[a]) + norm2(x.lower.f[a]);
  return s;
}
ComplexD Finner(const F &x, const F &y) {
  ComplexD s = 0;
  for (int a = 0; a < DtxqcdNf; ++a) {
    s = s + innerProduct(x.upper.f[a], y.upper.f[a]);
    s = s + innerProduct(x.lower.f[a], y.lower.f[a]);
  }
  return s;
}
void Fzero(F &x, int cb) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    x.upper.f[a] = Zero();
    x.lower.f[a] = Zero();
    x.upper.f[a].Checkerboard() = cb;
    x.lower.f[a].Checkerboard() = cb;
  }
}
void Fgaussian(GridParallelRNG &rng, GridBase *full, GridBase *rb, F &x_o) {
  // RB noise house pattern: full-grid gaussian + pickCheckerboard (a
  // GridParallelRNG constructed ON an RB grid HANGS).
  F tmp(full);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(rng, tmp.upper.f[a]);
    gaussian(rng, tmp.lower.f[a]);
    pickCheckerboard(Odd, x_o.upper.f[a], tmp.upper.f[a]);
    pickCheckerboard(Odd, x_o.lower.f[a], tmp.lower.f[a]);
  }
}

#endif  // GRID_HAVE_QUDA

}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
#ifndef GRID_HAVE_QUDA
  std::cout << GridLogMessage
            << "[mtilde] requires a QUDA build (deform EO stack)"
            << std::endl;
  std::_Exit(1);
#else
  Coordinate latt(std::vector<int>{4, 4, 4, 8});
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--grid" && i + 1 < argc)
      latt = parse_dotted(argv[i + 1], latt);
  GridCartesian Grid_(latt, GridDefaultSimd(Nd, vComplex::Nsimd()),
                      GridDefaultMpi());
  GridRedBlackCartesian RBGrid(&Grid_);
  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid_);
  pRNG.SeedFixedIntegers({51, 52, 53, 54, 55});

  const RealD lambda = env_real("LAMBDA", 5.0);
  const RealD mass = env_real("MASS", -0.245);
  const RealD csw_v = env_real("CSW", 1.24930970916466);
  const double eta = env_real("DEFORM_ETA", 0.6);
  const int t0 = env_int("DEFORM_T0", 0);
  const int T = latt[Nd - 1];

  DTXQCDField U(&Grid_);
  std::string cfg_dir = env_str("CFG_DIR", "");
  if (!cfg_dir.empty()) {
    const int traj = env_int("TRAJ", 0);
    DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG, cfg_dir + "/ckpoint_lat",
                                   cfg_dir + "/ckpoint_rng", traj);
    std::cout << GridLogMessage << "[mtilde] cfg " << cfg_dir << " traj "
              << traj << " plaq "
              << WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U) << std::endl;
  } else {
    std::string gk = env_str("GAUGE_KIND", "tepid");
    if (gk == "hot")
      SU<Nc>::HotConfiguration(pRNG, U.U);
    else
      PeriodicGimplR::TepidConfiguration(pRNG, U.U);
    DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda,
                                       env_real("AUX_SIGMA", 0.0));
    std::cout << GridLogMessage << "[mtilde] synthetic gauge=" << gk
              << " lambda=" << lambda << std::endl;
  }
  LatticeGaugeField U_inv =
      TXQCDProduction::DtxqcdInversionSmear(Grid_, U.U);

  DTXQCDWilsonCloverFermionEO Dw(U_inv, Grid_, RBGrid, mass, csw_v, U.sigma,
                                 U.pi, U.d, U.n, U.s, U.p);
  DTXQCDMOp Mop(Dw);
  DTXQCDMpcOp Mpc_cpu(Dw);
  Grid::Quda::initialize();
  DtxqcdPackedFullHermOp HopFull(Mop, &Grid_, Even);
  DTXQCDMpcOpQUDA_EO Mq_eo(Dw, U_inv, Grid_, RBGrid, mass, csw_v, U.sigma,
                           U.pi, U.d, U.n, U.s, U.p);
  DtxqcdPackedMpcHermOp HopEO(Mpc_cpu, Mq_eo, &RBGrid, Odd);
  DtxqcdDeformQuda Qdef(HopFull, Grid_, U_inv, T);
  Qdef.AttachEO(HopEO, Mq_eo, RBGrid, Dw);
  // PROFILE support (2026-07-16): tent (default) | tent_rounded |
  // tent_rounded2 | cos, with PROFILE_W / PROFILE_W_SRC — same lambda as
  // the meas/probe programs.  The MG question for SMOOTH profiles: their
  // deep modes near ignition are WIDE supercritical-window states (not
  // wall-localized) — exactly what defeated the sharp-tent MG is absent.
  // SetProf(+-eta) replaces all SetTentProfile sites (recip = -eta works
  // for ANY profile: the weights are e^{-+eta dp/2}).
  const std::string prof_kind = env_str("PROFILE", "tent");
  const double prof_w = env_real("PROFILE_W", 12.0);
  const double prof_w_src = env_real("PROFILE_W_SRC", prof_w);
  const bool prof_rounded =
      prof_kind == "tent_rounded" || prof_kind == "tent_rounded2";
  const bool prof_round_src = prof_kind == "tent_rounded2";
  const bool prof_cos = prof_kind == "cos";
  auto prof = [&](int t) {
    const int dt = ((t - t0) % T + T) % T;
    const double d = (double)dt;
    if (prof_cos)
      return 0.25 * (double)T * (1.0 - std::cos(2.0 * M_PI * d / (double)T));
    double v = std::min(d, (double)T - d);
    if (!prof_rounded) return v;
    const double a = std::abs(d - 0.5 * T);
    if (a < prof_w)
      v = 0.5 * (double)T - 0.5 * prof_w - a * a / (2.0 * prof_w);
    if (prof_round_src && prof_w_src > 0) {
      const double b = std::min(d, (double)T - d);
      if (b < prof_w_src)
        v = 0.5 * prof_w_src + b * b / (2.0 * prof_w_src);
    }
    return v;
  };
  auto SetProf = [&](double e) {
    if (prof_kind == "tent")
      Qdef.SetTentProfile(e, t0);
    else
      Qdef.SetProfile(e, prof);
  };
  SetProf(eta);
  std::cout << GridLogMessage << "[mtilde] deformed EO op ready: eta=" << eta
            << " t0=" << t0 << " profile=" << prof_kind << " w=" << prof_w
            << "/" << prof_w_src << std::endl;

  int fails = 0;
  auto check = [&](const char *name, RealD val, RealD tol) {
    bool ok = val < tol;
    if (!ok) ++fails;
    std::cout << GridLogMessage << (ok ? "[ok]   " : "[FAIL] ") << name
              << " = " << val << " (tol " << tol << ")" << std::endl;
  };

  //////////////////////////////////////////////////////////////////////
  // Gate S0a: device adjointness  <u, M~ v> == <M~dag u, v>
  //////////////////////////////////////////////////////////////////////
  F u(&RBGrid), v(&RBGrid), Mv(&RBGrid), Mdu(&RBGrid);
  Fgaussian(pRNG, &Grid_, &RBGrid, u);
  Fgaussian(pRNG, &Grid_, &RBGrid, v);
  Qdef.MpcTil(v, Mv);
  Qdef.MpcTilDag(u, Mdu);
  {
    const ComplexD a = Finner(u, Mv);
    const ComplexD b = Finner(Mdu, v);
    const RealD scale = std::sqrt(Fnorm2(u) * Fnorm2(Mv));
    check("S0a adjointness |<u,Mv>-<Mdag u,v>|/scale",
          DtxqcdAbs(ComplexD(a - b)) / scale, 1e-10);
  }

  //////////////////////////////////////////////////////////////////////
  // Gate S0b: quasi-g5-Hermiticity  M~(eta)^dag u == G5 M~(-eta) G5 u.
  // At eta=0 this is plain g5-Hermiticity of the Schur complement (also
  // certifies the Gamma5All convention on the doubled index).  This
  // identity is what makes Petrov-Galerkin setup HALF price: left null
  // spaces = G5 x right near-kernel of M~(-eta).
  //////////////////////////////////////////////////////////////////////
  {
    F g5u(&RBGrid), t1(&RBGrid), rhs(&RBGrid);
    Gamma5All(u, g5u);
    SetProf(-eta);
    Qdef.MpcTil(g5u, t1);
    SetProf(eta);
    Gamma5All(t1, rhs);
    F diff(&RBGrid);
    RealD n2 = 0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      diff.upper.f[a] = rhs.upper.f[a] - Mdu.upper.f[a];
      diff.lower.f[a] = rhs.lower.f[a] - Mdu.lower.f[a];
    }
    n2 = Fnorm2(diff);
    check("S0b quasi-g5-Herm |Mdag u - G5 M(-eta) G5 u|/|Mdag u|",
          std::sqrt(n2 / Fnorm2(Mdu)), 1e-9);
  }

  //////////////////////////////////////////////////////////////////////
  // Step 1: near-kernel generation by inverse iteration.  Right set:
  // v_i = normalize( M~(eta)^{-1} r_i ) at loose NV_TOL via the certified
  // CGNR solve; left set: u_i = G5 x normalize( M~(-eta)^{-1} r_i ).
  // Report |M~ v_i| / |v_i| — the block-locality precondition metric
  // (design kill criterion: if this does not decay well below the bulk
  // scale, M~'s near-kernel is not there to aggregate).
  //////////////////////////////////////////////////////////////////////
  const int nv = env_int("NV_NEV", 16);
  const RealD nv_tol = env_real("NV_TOL", 0.1);
  const int nv_maxit = env_int("NV_MAXIT", 20000);
  // Compile-time basis size (Aggregation template).  Build variants via
  // -DMTILDE_NB=32/48/128 (Makefile: Test_dtxqcd_deform_mg_mtilde_nb*).
  // NB the dense coarse LU is (sites*NB)^2: pair larger NB with coarser
  // BLOCK (nb32/nb48 @ 4.4.4.4 -> N=24,576/36,864 -> 9.7/21.7 GB on GPU 1).
#ifndef MTILDE_NB
#define MTILDE_NB 16
#endif
  constexpr int NB = MTILDE_NB;
  GRID_ASSERT(nv == NB && "prototype instantiates Aggregation at nbasis=16");
  Coordinate block = parse_dotted(env_str("BLOCK", "4.4.4.2"),
                                  Coordinate(std::vector<int>{4, 4, 4, 2}));
  Coordinate clatt(Nd);
  for (int d = 0; d < Nd; ++d) {
    GRID_ASSERT(latt[d] % block[d] == 0);
    clatt[d] = latt[d] / block[d];
  }
  GridCartesian Coarse4d(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()),
                         GridDefaultMpi());
  typedef GeneralCoarsenedMatrix<vDtxqcdPackedVector, vTComplex, NB>
      LittleDiracOperator;
  typedef typename LittleDiracOperator::CoarseVector CoarseVector;
  typedef Aggregation<vDtxqcdPackedVector, vTComplex, NB> Subspace;
  std::cout << GridLogMessage << "[mtilde] coarse=" << clatt << " nbasis="
            << NB << std::endl;
  // Aggregates constructed BEFORE the generation loop so each vector is
  // packed the moment it exists — keeping the struct basis (2*NB fields)
  // resident until a later pack loop was a 19.3 GB duplication at nb128
  // (OOM at pack start, 2026-07-16); at no point below do two copies of
  // the basis coexist.
  Subspace AggR(&Coarse4d, &RBGrid, Odd), AggL(&Coarse4d, &RBGrid, Odd);
  std::cout << GridLogMessage << "[mtilde] generating " << nv
            << " right + " << nv << " left near-kernel vectors (invit tol "
            << nv_tol << ")" << std::endl;
  // Production-vs-standalone NV discriminators (2026-07-16 stall hunt):
  // the header's Setup differs from this file's validated recipe in exactly
  // two ways — flip each here in isolation to find which kills the solve.
  // NV_RHS_FULL=1: right vectors from a FULL-grid gaussian src through
  // SolveTilEO's genuine Schur prep (the header semantics) instead of the
  // RB-allocated struct quirk this file validated with.
  // NV_LEFT_DAG=1: left vectors by dagger-CGNR on Mpc~ (header semantics)
  // instead of G5 x the recip-profile solve.
  const int nv_rhs_full = env_int("NV_RHS_FULL", 0);
  const int nv_left_dag = env_int("NV_LEFT_DAG", 0);
  for (int i = 0; i < nv; ++i) {
    F r(&RBGrid), x(&RBGrid);
    RealD rel = 0;
    int it = 0;
    if (nv_rhs_full) {
      F rf(&Grid_), xf(&Grid_);
      for (int a = 0; a < DtxqcdNf; ++a) {
        gaussian(pRNG, rf.upper.f[a]);
        gaussian(pRNG, rf.lower.f[a]);
        xf.upper.f[a] = Zero();
        xf.lower.f[a] = Zero();
      }
      it = Qdef.SolveTilEO(rf, xf, nv_tol, nv_maxit, rel);
      for (int a = 0; a < DtxqcdNf; ++a) {
        pickCheckerboard(Odd, x.upper.f[a], xf.upper.f[a]);
        pickCheckerboard(Odd, x.lower.f[a], xf.lower.f[a]);
      }
    } else {
      Fgaussian(pRNG, &Grid_, &RBGrid, r);
      // right: M~(eta)^{-1} r
      Fzero(x, Odd);
      it = Qdef.SolveTilEO(r, x, nv_tol, nv_maxit, rel);
    }
    RealD xn = std::sqrt(Fnorm2(x));
    for (int a = 0; a < DtxqcdNf; ++a) {
      x.upper.f[a] = x.upper.f[a] * (1.0 / xn);
      x.lower.f[a] = x.lower.f[a] * (1.0 / xn);
    }
    F Mx(&RBGrid);
    Qdef.MpcTil(x, Mx);
    std::cout << GridLogMessage << "[mtilde] right nv[" << i
              << "]: invit iters=" << it << "  |M~v|/|v|="
              << std::sqrt(Fnorm2(Mx)) << std::endl;
    // DtxqcdPackDoubled copies the physical Odd cb from the struct; restamp
    // Even after — packed metadata stays Even program-wide (see MtilOpShim).
    DtxqcdPackDoubled(x, AggR.subspace[i]);
    AggR.subspace[i].Checkerboard() = Even;
    F g5x(&RBGrid);
    if (nv_left_dag) {
      // header semantics: z ~ Mpc~^{-dag} r by CG on (Mpc~ Mpc~^dag) z = Mpc~ r
      Fgaussian(pRNG, &Grid_, &RBGrid, r);
      F bb(&RBGrid), res(&RBGrid), p(&RBGrid), t(&RBGrid), Ap(&RBGrid);
      Qdef.MpcTil(r, bb);
      Fzero(g5x, Odd);
      res = bb;
      p = bb;
      RealD r2 = Fnorm2(res);
      const RealD tol2 = nv_tol * nv_tol * std::max(Fnorm2(bb), 1e-300);
      int k = 0;
      for (; k < nv_maxit && r2 > tol2; ++k) {
        Qdef.MpcTilDag(p, t);
        Qdef.MpcTil(t, Ap);
        const RealD pAp = DtxqcdDeformQuda::finner_(p, Ap).real();
        const RealD alpha = r2 / std::max(pAp, 1e-300);
        for (int a = 0; a < DtxqcdNf; ++a) {
          g5x.upper.f[a] = g5x.upper.f[a] + alpha * p.upper.f[a];
          g5x.lower.f[a] = g5x.lower.f[a] + alpha * p.lower.f[a];
          res.upper.f[a] = res.upper.f[a] - alpha * Ap.upper.f[a];
          res.lower.f[a] = res.lower.f[a] - alpha * Ap.lower.f[a];
        }
        const RealD rn = Fnorm2(res);
        const RealD beta = rn / r2;
        for (int a = 0; a < DtxqcdNf; ++a) {
          p.upper.f[a] = res.upper.f[a] + beta * p.upper.f[a];
          p.lower.f[a] = res.lower.f[a] + beta * p.lower.f[a];
        }
        r2 = rn;
      }
      it = k;
      xn = std::sqrt(Fnorm2(g5x));
      for (int a = 0; a < DtxqcdNf; ++a) {
        g5x.upper.f[a] = g5x.upper.f[a] * (1.0 / xn);
        g5x.lower.f[a] = g5x.lower.f[a] * (1.0 / xn);
      }
    } else {
    // left: G5 M~(-eta)^{-1} r'
    Fgaussian(pRNG, &Grid_, &RBGrid, r);
    Fzero(x, Odd);
    SetProf(-eta);
    it = Qdef.SolveTilEO(r, x, nv_tol, nv_maxit, rel);
    SetProf(eta);
    xn = std::sqrt(Fnorm2(x));
    for (int a = 0; a < DtxqcdNf; ++a) {
      x.upper.f[a] = x.upper.f[a] * (1.0 / xn);
      x.lower.f[a] = x.lower.f[a] * (1.0 / xn);
    }
    Gamma5All(x, g5x);
    }
    // gate the identity in anger: |M~(eta)^dag g5x| should equal the
    // right-style residual of the recip solve (small)
    Qdef.MpcTilDag(g5x, Mx);
    std::cout << GridLogMessage << "[mtilde] left  nv[" << i
              << "]: invit iters=" << it << "  |M~dag u|/|u|="
              << std::sqrt(Fnorm2(Mx)) << std::endl;
    DtxqcdPackDoubled(g5x, AggL.subspace[i]);
    AggL.subspace[i].Checkerboard() = Even;
  }
  std::cout << GridLogMessage
            << "[mtilde] Step 0-1 complete; Petrov-Galerkin coarsening next"
            << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Step 2: Petrov-Galerkin coarsening of M~ itself.  Grid's
  // GeneralCoarsenedMatrix::CoarsenOperator(linop, U, V) is NATIVELY
  // Petrov-Galerkin (probes linop.Op against V's phased subspace,
  // projects with U): A_c = U^dag M~ V with U = left basis (G5 x recip
  // near-kernel), V = right near-kernel.  Nearest coarse geometry covers
  // the distance-2 Schur footprint for blocks >= 2 in every dimension.
  //////////////////////////////////////////////////////////////////////
  // Non-Hermitian fine shim: Op = M~, AdjOp = M~^dag, on packed fields.
  struct MtilOpShim : public LinearOperatorBase<LatticeDtxqcdPacked> {
    DtxqcdDeformQuda &q;
    GridBase *rb;
    mutable size_t nops = 0;
    MtilOpShim(DtxqcdDeformQuda &q_, GridBase *rb_) : q(q_), rb(rb_) {}
    void Op(const LatticeDtxqcdPacked &in, LatticeDtxqcdPacked &out) {
      F fi(rb), fo(rb);
      DtxqcdUnpackDoubled(in, fi);
      for (int a = 0; a < DtxqcdNf; ++a) {
        fi.upper.f[a].Checkerboard() = Odd;
        fi.lower.f[a].Checkerboard() = Odd;
      }
      q.MpcTil(fi, fo);
      DtxqcdPackDoubled(fo, out);
      // Packed-field cb METADATA is kept Even throughout this program (the
      // grid_mg RB convention): CoarsenOperator's internal phase fields are
      // default-constructed Even and its ET products assert on mismatch.
      // Physical parity is stamped on the STRUCT side above.
      out.Checkerboard() = Even;
      ++nops;
    }
    void AdjOp(const LatticeDtxqcdPacked &in, LatticeDtxqcdPacked &out) {
      F fi(rb), fo(rb);
      DtxqcdUnpackDoubled(in, fi);
      for (int a = 0; a < DtxqcdNf; ++a) {
        fi.upper.f[a].Checkerboard() = Odd;
        fi.lower.f[a].Checkerboard() = Odd;
      }
      q.MpcTilDag(fi, fo);
      DtxqcdPackDoubled(fo, out);
      out.Checkerboard() = Even;  // see Op(): packed metadata stays Even
      ++nops;
    }
    void HermOp(const LatticeDtxqcdPacked &in, LatticeDtxqcdPacked &out) {
      LatticeDtxqcdPacked t(in.Grid());
      Op(in, t);
      AdjOp(t, out);
    }
    void HermOpAndNorm(const LatticeDtxqcdPacked &in,
                       LatticeDtxqcdPacked &out, RealD &n1, RealD &n2) {
      HermOp(in, out);
      ComplexD dot = innerProduct(in, out);
      n1 = real(dot);
      n2 = norm2(out);
    }
    void OpDiag(const LatticeDtxqcdPacked &, LatticeDtxqcdPacked &) {
      GRID_ASSERT(0);
    }
    void OpDir(const LatticeDtxqcdPacked &, LatticeDtxqcdPacked &, int,
               int) {
      GRID_ASSERT(0);
    }
    void OpDirAll(const LatticeDtxqcdPacked &,
                  std::vector<LatticeDtxqcdPacked> &) {
      GRID_ASSERT(0);
    }
  };
  MtilOpShim shim(Qdef, &RBGrid);

  // Schur footprint is distance-2 in FINE sites: with t-blocks of 2, a
  // (x+-1, t+-1) fine hop crosses a coarse CORNER — nearest geometry
  // truncates it (measured: S2 consistency 0.12).  Default nnn (81-point,
  // the full +-1 hypercube), same as the grid_mg EO runs.
  std::unique_ptr<NonLocalStencilGeometry> geomp;
  std::string gs = env_str("GEOM", "nnn");
  if (gs == "nearest")
    geomp.reset(new NearestStencilGeometry4D(&Coarse4d));
  else if (gs == "nn")
    geomp.reset(new NextToNearestStencilGeometry4D(&Coarse4d));
  else
    geomp.reset(new NextToNextToNextToNearestStencilGeometry4D(&Coarse4d));
  LittleDiracOperator Ac(*geomp, &RBGrid, &Coarse4d);
  double tc0 = usecond();
  Ac.CoarsenOperator(shim, AggL, AggR);
  std::cout << GridLogMessage << "[mtilde] PG coarsen done in "
            << (usecond() - tc0) * 1e-6 << " s" << std::endl;

  // Gate S2: coarse-op consistency.  A_c c must equal
  // ProjL( M~ PromR(c) ) on random coarse vectors (uses the
  // POST-orthogonalisation bases — CoarsenOperator orthogonalises U,V in
  // place, so this is exact up to roundoff if the stencil captured the
  // full footprint).
  {
    GridParallelRNG CRNG(&Coarse4d);
    CRNG.SeedFixedIntegers({61, 62, 63, 64});
    CoarseVector c(&Coarse4d), y1(&Coarse4d), y2(&Coarse4d);
    gaussian(CRNG, c);
    Ac.M(c, y1);
    LatticeDtxqcdPacked f(&RBGrid), Mf(&RBGrid);
    f.Checkerboard() = Even;  // packed metadata convention
    AggR.PromoteFromSubspace(c, f);
    shim.Op(f, Mf);
    AggL.ProjectToSubspace(y2, Mf);
    CoarseVector d(&Coarse4d);
    d = y1 - y2;
    check("S2 PG coarse-op consistency |Ac c - R M~ P c|/|R M~ P c|",
          std::sqrt(norm2(d) / norm2(y2)), 1e-8);
  }

  //////////////////////////////////////////////////////////////////////
  // COARSE_DIRECT (default 1): dense LU of A_c on a SECOND GPU.
  // Motivation (eta=0.8 v1/v2): restarted GMRES does not converge on the
  // inherited-band coarse system even at restart 768 — so make the coarse
  // correction EXACT and remove the inner-solver variable entirely.  At
  // 24,576-dim the dense matrix is 9.7 GB (does NOT fit GPU 0 beside the
  // fine stack -> cudaSetDevice(1), context restored after every call)
  // and cusolverDnZgetrf factors it in seconds.  The assembly gate is
  // self-calibrating: it tests BOTH gather conventions against Ac.M and
  // keeps the one that matches.
  //////////////////////////////////////////////////////////////////////
  const int Ncc = (int)(Coarse4d.gSites()) * NB;
  const int npoint = geomp->npoint;
  Coordinate cdims = Coarse4d.GlobalDimensions();
  auto cflat = [&](const Coordinate &c) {
    int64_t idx = 0;
    for (int mu = Nd - 1; mu >= 0; --mu) idx = idx * cdims[mu] + c[mu];
    return idx;
  };
  typedef typename LittleDiracOperator::CoarseVector CoarseVectorT;
  auto packc = [&](const CoarseVectorT &v, std::vector<ComplexD> &out) {
    out.resize(Ncc);
    Coordinate c(Nd);
    for (c[3] = 0; c[3] < cdims[3]; ++c[3])
      for (c[2] = 0; c[2] < cdims[2]; ++c[2])
        for (c[1] = 0; c[1] < cdims[1]; ++c[1])
          for (c[0] = 0; c[0] < cdims[0]; ++c[0]) {
            auto s = peekSite(v, c);
            const int64_t b = cflat(c) * NB;
            for (int i = 0; i < NB; ++i)
              out[b + i] = TensorRemove(s(i));
          }
  };
  auto pokec = [&](const std::vector<ComplexD> &in, CoarseVectorT &v) {
    Coordinate c(Nd);
    for (c[3] = 0; c[3] < cdims[3]; ++c[3])
      for (c[2] = 0; c[2] < cdims[2]; ++c[2])
        for (c[1] = 0; c[1] < cdims[1]; ++c[1])
          for (c[0] = 0; c[0] < cdims[0]; ++c[0]) {
            decltype(peekSite(v, c)) s;
            const int64_t b = cflat(c) * NB;
            for (int i = 0; i < NB; ++i) s(i) = in[b + i];
            pokeSite(s, v, c);
          }
  };
#if defined(GRID_CUDA) && defined(GRID_HAVE_QUDA)
  struct CoarseDirect {
    int N = 0, dev = 1;
    cusolverDnHandle_t h = nullptr;
    cuDoubleComplex *dA = nullptr, *dWork = nullptr, *dB = nullptr;
    int *dPiv = nullptr, *dInfo = nullptr;
    bool ready = false;
    void Factor(std::vector<std::complex<double>> &Dcol, int n) {
      N = n;
      int dev0 = 0;
      cudaGetDevice(&dev0);
      cudaSetDevice(dev);
      if (ready) {  // bootstrap re-factor: free the previous factorization
        cudaFree(dA);
        cudaFree(dWork);
        cudaFree(dPiv);
        cudaFree(dInfo);
        cudaFree(dB);
        cusolverDnDestroy(h);
        ready = false;
      }
      cudaMalloc(&dA, sizeof(cuDoubleComplex) * (size_t)N * N);
      cudaMemcpy(dA, Dcol.data(), sizeof(cuDoubleComplex) * (size_t)N * N,
                 cudaMemcpyHostToDevice);
      cusolverDnCreate(&h);
      int lwork = 0;
      cusolverDnZgetrf_bufferSize(h, N, N, dA, N, &lwork);
      cudaMalloc(&dWork, sizeof(cuDoubleComplex) * (size_t)lwork);
      cudaMalloc(&dPiv, sizeof(int) * N);
      cudaMalloc(&dInfo, sizeof(int));
      cudaMalloc(&dB, sizeof(cuDoubleComplex) * N);
      cusolverDnZgetrf(h, N, N, dA, N, dWork, dPiv, dInfo);
      int info = -1;
      cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost);
      GRID_ASSERT(info == 0 && "coarse dense LU failed");
      // Singularity meter: the U-diagonal spread bounds cond(A_c) from
      // below.  The eta=0.8 bootstrap collapse (solve residual 6.6e6)
      // was a NEAR-SINGULAR A_c from left/right subspace-angle loss —
      // this print is the early warning.
      {
        std::vector<std::complex<double>> diag(N);
        for (int i = 0; i < N; ++i)
          cudaMemcpy(&diag[i], dA + (size_t)i + (size_t)N * i,
                     sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        double dmin = 1e300, dmax = 0;
        for (int i = 0; i < N; ++i) {
          const double a = std::abs(diag[i]);
          dmin = std::min(dmin, a);
          dmax = std::max(dmax, a);
        }
        std::cout << GridLogMessage << "[mtilde] A_c LU diag |U_ii|: min "
                  << dmin << " max " << dmax << " ratio " << dmax / dmin
                  << std::endl;
      }
      ready = true;
      cudaSetDevice(dev0);
    }
    void Solve(std::vector<ComplexD> &b) {
      int dev0 = 0;
      cudaGetDevice(&dev0);
      cudaSetDevice(dev);
      cudaMemcpy(dB, b.data(), sizeof(cuDoubleComplex) * N,
                 cudaMemcpyHostToDevice);
      cusolverDnZgetrs(h, CUBLAS_OP_N, N, 1, dA, N, dPiv, dB, N, dInfo);
      cudaMemcpy(b.data(), dB, sizeof(cuDoubleComplex) * N,
                 cudaMemcpyDeviceToHost);
      cudaSetDevice(dev0);
    }
    // Solve A_c^dag x = b from the SAME factorization (getrs op=C) — the
    // dagger V-cycle's coarse correction for left-vector bootstrap.
    void SolveDag(std::vector<ComplexD> &b) {
      int dev0 = 0;
      cudaGetDevice(&dev0);
      cudaSetDevice(dev);
      cudaMemcpy(dB, b.data(), sizeof(cuDoubleComplex) * N,
                 cudaMemcpyHostToDevice);
      cusolverDnZgetrs(h, CUBLAS_OP_C, N, 1, dA, N, dPiv, dB, N, dInfo);
      cudaMemcpy(b.data(), dB, sizeof(cuDoubleComplex) * N,
                 cudaMemcpyDeviceToHost);
      cudaSetDevice(dev0);
    }
  };
  CoarseDirect cdirect;
  const bool use_direct = env_int("COARSE_DIRECT", 1) != 0;
  int best_variant = -1;  // calibrated once, reused by bootstrap re-factors
  // Assemble + calibrate + LU-factor the CURRENT Ac.  Convention space:
  // shift sign x link-index transpose x link-site (Grid's coarse _A comes
  // out of an FT reconstruction whose index conventions are not worth
  // reverse-engineering — calibrate against Ac.M directly; measured
  // winner: sign +1, TRANSPOSED indices, link at x).
  auto build_direct = [&]() {
    double td0 = usecond();
    std::vector<std::complex<double>> Dcol;
    CoarseVectorT tv(&Coarse4d), ty(&Coarse4d);
    GridParallelRNG CR2(&Coarse4d);
    CR2.SeedFixedIntegers({71, 72, 73, 74});
    gaussian(CR2, tv);
    Ac.M(tv, ty);
    std::vector<ComplexD> vin, yref;
    packc(tv, vin);
    packc(ty, yref);
    auto assemble = [&](int variant) {
      const int sign = (variant & 1) ? -1 : +1;
      const int trans = (variant >> 1) & 1;
      const int atnbr = (variant >> 2) & 1;
      Dcol.assign((size_t)Ncc * Ncc, std::complex<double>(0, 0));
      Coordinate x(Nd);
      for (x[3] = 0; x[3] < cdims[3]; ++x[3])
        for (x[2] = 0; x[2] < cdims[2]; ++x[2])
          for (x[1] = 0; x[1] < cdims[1]; ++x[1])
            for (x[0] = 0; x[0] < cdims[0]; ++x[0]) {
              const int64_t rb = cflat(x) * NB;
              for (int p = 0; p < npoint; ++p) {
                Coordinate y = x;
                for (int mu = 0; mu < Nd; ++mu)
                  y[mu] = (x[mu] + sign * geomp->shifts[p][mu] +
                           cdims[mu]) % cdims[mu];
                const int64_t cb = cflat(y) * NB;
                auto link = peekSite(Ac._A[p], atnbr ? y : x);
                for (int i = 0; i < NB; ++i)
                  for (int j = 0; j < NB; ++j) {
                    const ComplexD z =
                        TensorRemove(trans ? link(j, i) : link(i, j));
                    // column-major for cuSOLVER
                    Dcol[(size_t)(rb + i) + (size_t)Ncc * (cb + j)] +=
                        std::complex<double>(DtxqcdToStd(z).real(),
                                             DtxqcdToStd(z).imag());
                  }
              }
            }
    };
    auto dense_rel = [&]() -> RealD {
      RealD dn = 0, yn = 0;
      for (int r = 0; r < Ncc; ++r) {
        std::complex<double> acc(0, 0);
        for (int cidx = 0; cidx < Ncc; ++cidx)
          acc += Dcol[(size_t)r + (size_t)Ncc * cidx] *
                 std::complex<double>(DtxqcdToStd(vin[cidx]).real(),
                                      DtxqcdToStd(vin[cidx]).imag());
        const std::complex<double> ry(DtxqcdToStd(yref[r]).real(),
                                      DtxqcdToStd(yref[r]).imag());
        dn += std::norm(acc - ry);
        yn += std::norm(ry);
      }
      return std::sqrt(dn / yn);
    };
    RealD bestrel = 1e30;
    if (best_variant < 0) {
      for (int variant = 0; variant < 8; ++variant) {
        assemble(variant);
        const RealD rel = dense_rel();
        std::cout << GridLogMessage << "[mtilde] dense assembly variant "
                  << variant << " vs Ac.M rel = " << rel << std::endl;
        if (rel < bestrel) {
          bestrel = rel;
          best_variant = variant;
        }
        if (rel < 1e-10) break;  // Dcol holds the winner
      }
    } else {
      assemble(best_variant);
      bestrel = dense_rel();
    }
    check("S2b dense assembly matches Ac.M", bestrel, 1e-10);
    cdirect.Factor(Dcol, Ncc);
    std::cout << GridLogMessage << "[mtilde] coarse dense LU ready in "
              << (usecond() - td0) * 1e-6 << " s (N=" << Ncc
              << ", variant " << best_variant << ", GPU 1)" << std::endl;
  };
  if (use_direct) build_direct();
#else
  struct CoarseDirect {
    bool ready = false;
    void Solve(std::vector<ComplexD> &) {}
  };
  CoarseDirect cdirect;
#endif

  //////////////////////////////////////////////////////////////////////
  // Step 3: two-level solve.  Outer = PGCR (non-Hermitian, flexible) on
  // M~ x = b; preconditioner = V-cycle: MR pre-smooth -> coarse GMRES
  // correction (P Ac^{-1} R) -> MR post-smooth.  MR = minimal-residual
  // iteration (z += alpha r_s, alpha = <Ar,r>/<Ar,Ar>), the cheapest
  // non-Hermitian smoother.  Baseline = the certified CGNR on the same
  // source; cross-gate = |x_mg - x_cgnr| / |x|.
  //////////////////////////////////////////////////////////////////////
  const int run_solve = env_int("RUN_SOLVE", 1);
  if (run_solve) {
    typedef LatticeDtxqcdPacked Field;
    struct VCycle : public LinearFunction<Field> {
      MtilOpShim &op;
      LittleDiracOperator &ac;
      Subspace &aggL, &aggR;
      GridCartesian *cgrid;
      int smooth_it;
      RealD ctol;
      int cmaxit, crestart;
      int use_coarse;  // MG_COARSE=0: smoother-only (attribution control)
      int dag = 0;     // dagger V-cycle: AdjOp smoother, A_c^dag coarse,
                       // aggregates swapped ((U^dag A V)^dag = V^dag A^dag U)
      std::function<void(const CoarseVector &, CoarseVector &)> csolve;
      VCycle(MtilOpShim &o, LittleDiracOperator &a, Subspace &l, Subspace &r,
             GridCartesian *cg, int si, RealD ct, int cm, int cr, int uc)
          : op(o), ac(a), aggL(l), aggR(r), cgrid(cg), smooth_it(si),
            ctol(ct), cmaxit(cm), crestart(cr), use_coarse(uc) {}
      void Apply_(const Field &in, Field &out) {
        if (dag)
          op.AdjOp(in, out);
        else
          op.Op(in, out);
      }
      void MR_(const Field &r, Field &z) {
        Field rs(r.Grid()), Ar(r.Grid());
        z = Zero();
        z.Checkerboard() = r.Checkerboard();
        rs = r;
        for (int k = 0; k < smooth_it; ++k) {
          Apply_(rs, Ar);
          ComplexD num = innerProduct(Ar, rs);
          RealD den = norm2(Ar);
          ComplexD alpha = num * (1.0 / std::max(den, 1e-300));
          z = z + alpha * rs;
          rs = rs - alpha * Ar;
        }
      }
      void operator()(const Field &r, Field &z) {
        Field t(r.Grid()), resid(r.Grid());
        t.Checkerboard() = r.Checkerboard();
        resid.Checkerboard() = r.Checkerboard();
        // pre-smooth
        MR_(r, z);
        if (!use_coarse) return;  // attribution control: smoother only
        Apply_(z, t);
        resid = r - t;
        // coarse correction
        CoarseVector cr(cgrid), cx(cgrid);
        Subspace &restrictor = dag ? aggR : aggL;
        Subspace &prolongator = dag ? aggL : aggR;
        restrictor.ProjectToSubspace(cr, resid);
        cx = Zero();
        if (csolve) {
          csolve(cr, cx);  // exact dense LU (or LU^dag) on GPU 1
        } else {
          GRID_ASSERT(!dag && "GMRES fallback has no dagger coarse path");
          NonHermitianLinearOperator<LittleDiracOperator, CoarseVector>
              cop(ac);
          GeneralisedMinimalResidual<CoarseVector> gmres(ctol, cmaxit,
                                                         crestart, false);
          gmres(cop, cr, cx);
        }
        Field corr(r.Grid());
        corr.Checkerboard() = r.Checkerboard();
        prolongator.PromoteFromSubspace(cx, corr);
        z = z + corr;
        // post-smooth
        Apply_(z, t);
        resid = r - t;
        Field z2(r.Grid());
        z2.Checkerboard() = r.Checkerboard();
        MR_(resid, z2);
        z = z + z2;
      }
    };
    // Dagger-system shim for the left-vector bootstrap: Op = M~^dag.
    struct DagShim : public LinearOperatorBase<Field> {
      MtilOpShim &s;
      DagShim(MtilOpShim &s_) : s(s_) {}
      void Op(const Field &in, Field &out) { s.AdjOp(in, out); }
      void AdjOp(const Field &in, Field &out) { s.Op(in, out); }
      void HermOp(const Field &in, Field &out) { s.HermOp(in, out); }
      void HermOpAndNorm(const Field &in, Field &out, RealD &n1, RealD &n2) {
        s.HermOpAndNorm(in, out, n1, n2);
      }
      void OpDiag(const Field &, Field &) { GRID_ASSERT(0); }
      void OpDir(const Field &, Field &, int, int) { GRID_ASSERT(0); }
      void OpDirAll(const Field &, std::vector<Field> &) { GRID_ASSERT(0); }
    };
    VCycle vcyc(shim, Ac, AggL, AggR, &Coarse4d,
                env_int("SMOOTH_IT", 8), env_real("COARSE_TOL", 1e-2),
                env_int("COARSE_MAXIT", 2000),
                env_int("COARSE_RESTART", 64), env_int("MG_COARSE", 1));
#if defined(GRID_CUDA) && defined(GRID_HAVE_QUDA)
    if (cdirect.ready)
      vcyc.csolve = [&](const CoarseVectorT &cr, CoarseVectorT &cx) {
        std::vector<ComplexD> b;
        packc(cr, b);
        cdirect.Solve(b);
        pokec(b, cx);
      };

    //////////////////////////////////////////////////////////////////
    // NV_BOOTSTRAP (default 2): adaptive setup.  Refine the null
    // vectors THROUGH the current two-level solver (even a
    // plateau-stalled solve carries full 1/lambda weighting on its
    // converged components — ~10x band enrichment per round at V-cycle
    // prices), re-coarsen, re-factor.  Right vectors under M~ (vcyc);
    // left under M~^dag (dagger V-cycle + A_c^dag LU).
    //////////////////////////////////////////////////////////////////
    const int nboot = env_int("NV_BOOTSTRAP", 2);
    if (nboot > 0 && cdirect.ready) {
      DagShim dshim(shim);
      VCycle vcycd(shim, Ac, AggL, AggR, &Coarse4d,
                   env_int("SMOOTH_IT", 8), 0, 0, 0, 1);
      vcycd.dag = 1;
      vcycd.csolve = [&](const CoarseVectorT &cr, CoarseVectorT &cx) {
        std::vector<ComplexD> b;
        packc(cr, b);
        cdirect.SolveDag(b);
        pokec(b, cx);
      };
      const RealD btol = env_real("NV_BOOT_TOL", 0.05);
      const int bmax = env_int("NV_BOOT_MAXIT", 60);
      for (int round = 0; round < nboot; ++round) {
        double tb0 = usecond();
        size_t ops0 = shim.nops;
        for (int i = 0; i < NB; ++i) {
          Field b(&RBGrid), z(&RBGrid);
          b = AggR.subspace[i];
          z = Zero();
          z.Checkerboard() = Even;
          PrecGeneralisedConjugateResidualNonHermitian<Field> BR(
              btol, bmax, shim, vcyc, 16, 16);
          BR(b, z);
          // MIX, don't replace: the band modes are t-LOCALIZED (tent
          // apex), so a raw refined vector has roundoff-junk support in
          // far blocks which blockOrthogonalise then normalizes to O(1)
          // NOISE — the coarse correction injects it volume-wide and the
          // solve blows up (measured 6e6 resid, A_c diag healthy).
          // v + z keeps volume-filling support while adding band weight.
          RealD zn = std::sqrt(norm2(z));
          z = b + (1.0 / std::max(zn, 1e-300)) * z;
          zn = std::sqrt(norm2(z));
          AggR.subspace[i] = (1.0 / std::max(zn, 1e-300)) * z;
          b = AggL.subspace[i];
          z = Zero();
          z.Checkerboard() = Even;
          PrecGeneralisedConjugateResidualNonHermitian<Field> BL(
              btol, bmax, dshim, vcycd, 16, 16);
          BL(b, z);
          zn = std::sqrt(norm2(z));
          z = b + (1.0 / std::max(zn, 1e-300)) * z;
          zn = std::sqrt(norm2(z));
          AggL.subspace[i] = (1.0 / std::max(zn, 1e-300)) * z;
        }
        Ac.CoarsenOperator(shim, AggL, AggR);
        build_direct();
        std::cout << GridLogMessage << "[mtilde] bootstrap round "
                  << round + 1 << "/" << nboot << ": "
                  << (shim.nops - ops0) << " fine Ops, "
                  << (usecond() - tb0) * 1e-6 << " s" << std::endl;
      }
    }
#endif

    // The measurement source: gaussian (the reweighting-estimator shape;
    // couples O(1) to the band — the HARD case for plain CG).
    F src(&RBGrid), x_cg(&RBGrid);
    Fgaussian(pRNG, &Grid_, &RBGrid, src);
    const RealD stol = env_real("CG_TOL", 1e-7);
    const int scgmax = env_int("CG_MAX", 200000);
    // Baseline: CGNR on the SAME raw odd-block Schur system (via the
    // shim's HermOp = M~^dag M~; src' = M~^dag b).  NB SolveTilEO is NOT
    // comparable — it wraps Schur prep + even backsubstitution, a
    // different linear system (measured: solution rel 1.1 vs the raw
    // system — apples to oranges).
    LatticeDtxqcdPacked psrc(&RBGrid), px(&RBGrid), pxc(&RBGrid);
    DtxqcdPackDoubled(src, psrc);
    psrc.Checkerboard() = Even;  // packed metadata convention
    bool have_baseline = env_int("RUN_BASELINE", 1) != 0;
    size_t base_ops = 0;
    if (have_baseline) {
      LatticeDtxqcdPacked bsrc(&RBGrid);
      bsrc.Checkerboard() = Even;
      shim.AdjOp(psrc, bsrc);
      pxc = Zero();
      pxc.Checkerboard() = Even;
      shim.nops = 0;
      double tb = usecond();
      ConjugateGradient<LatticeDtxqcdPacked> CG(stol, scgmax, false);
      CG(shim, bsrc, pxc);
      base_ops = shim.nops;
      std::cout << GridLogMessage << "[mtilde] baseline CGNR (raw Schur): "
                << base_ops << " fine Ops in " << (usecond() - tb) * 1e-6
                << " s" << std::endl;
    }
    // Two-level PGCR
    px = Zero();
    px.Checkerboard() = Even;
    shim.nops = 0;
    PrecGeneralisedConjugateResidualNonHermitian<Field> PGCR(
        stol, env_int("MG_MAXIT", 1000), shim, vcyc,
        env_int("PGCR_MMAX", 16), env_int("PGCR_NSTEP", 16));
    double tm = usecond();
    PGCR(psrc, px);
    const double t_mg = (usecond() - tm) * 1e-6;
    // true residual on M~
    LatticeDtxqcdPacked Mx(&RBGrid);
    shim.Op(px, Mx);
    Mx = Mx - psrc;
    const RealD true_rel = std::sqrt(norm2(Mx) / norm2(psrc));
    std::cout << GridLogMessage << "[mtilde] two-level PGCR: fine Ops="
              << shim.nops << " in " << t_mg << " s; true relres |M~x-b|/|b|="
              << true_rel << std::endl;
    check("S3 two-level true residual", true_rel, 100.0 * stol);
    if (have_baseline) {
      LatticeDtxqcdPacked dd(&RBGrid);
      dd = px - pxc;
      // Bound: two solves converged to ~stol in DIFFERENT norms (direct
      // vs normal-eq) legitimately differ by up to ~kappa(M~)*stol
      // (measured 2.6e-4 at eta=0.6, kappa~1.3e3, stol 1e-7 — right at
      // the bound).  Gate at 100x stol*sqrt(kappa-ish) via env.
      check("S3 MG vs CGNR solution rel",
            std::sqrt(norm2(dd) / norm2(pxc)),
            env_real("SOLCMP_TOL", 1e-3));
      std::cout << GridLogMessage << "[mtilde] fine-Op ratio (CGNR/MG) = "
                << (double)base_ops /
                       std::max<size_t>(shim.nops, 1)
                << std::endl;
    }
    // Attribution control (MG_COARSE=0 leg runs INSTEAD when set):
    // smoother-only preconditioner isolates the kappa-un-squaring win
    // (direct PGCR on M~) from the coarse-space contribution.
  }

  std::cout << GridLogMessage << "==== mtilde "
            << (fails ? "FAILED" : "PASSED") << " (" << fails
            << " failures) ====" << std::endl;
  std::cout.flush();
  std::_Exit(fails ? 1 : 0);
#endif
}
