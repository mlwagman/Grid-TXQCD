// Eigenvalue spectrum diagnostics: lowest few eigenvalues of M†M for QCD
// vs TXQCD at multiple λ on a fixed gauge config (IMPORT_CFG or weak field).
//
// Used to (a) test whether TXQCD has worse-conditioned M than QCD at small λ
// (more CG iterations) even though the connected pion is heavier, and
// (b) check for residual sign-problem: if any |eig(γ5 M)| = √eig(M†M) is
// near zero, the sign of det(M) is fragile under integrator noise.
//
// Implementation: Chebyshev-filter Lanczos with full re-orthogonalization,
// applied to MdagM where MdagM is templated on Field type.  Works for both
// LatticeFermion (QCD) and TXQCDFermionNf (TXQCD).

#include "params.h"
#include <cstdio>
#include <cstring>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDCheckpointer.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <Grid/qcd/smearing/StoutSmearing.h>
#include <Grid/Grid_Eigen_Dense.h>
#include <Grid/Eigen/Eigenvalues>

using namespace TXQCDProduction;

typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

// ---- field-type-generic helpers ---------------------------------------------
// axpy_ip(y, a, x): y += a*x
// scale_ip(y, a):   y *= a
// copy_field(dst, src): dst = src
// random_field(rng, f): gaussian fill

template <class Field>
inline void axpy_ip(Field &y, const ComplexD &a, const Field &x);

template <class Field>
inline void scale_ip(Field &y, const RealD &a);

template <class Field>
inline void copy_field(Field &dst, const Field &src);

template <class Field>
inline void random_field(GridParallelRNG &rng, Field &f);

// axpby_field(out, a, b, x, y): out = a*x + b*y
template <class Field>
inline void axpby_field(Field &out, const RealD &a, const RealD &b,
                        const Field &x, const Field &y);

// LatticeFermion specializations
template <>
inline void axpy_ip(LatticeFermion &y, const ComplexD &a, const LatticeFermion &x) {
  y = y + a * x;
}
template <>
inline void scale_ip(LatticeFermion &y, const RealD &a) { y = a * y; }
template <>
inline void copy_field(LatticeFermion &dst, const LatticeFermion &src) { dst = src; }
template <>
inline void random_field(GridParallelRNG &rng, LatticeFermion &f) { gaussian(rng, f); }
template <>
inline void axpby_field(LatticeFermion &out, const RealD &a, const RealD &b,
                        const LatticeFermion &x, const LatticeFermion &y) {
  out = a * x + b * y;
}

// TXQCDFermionNf specializations
template <>
inline void axpy_ip(TXQCDFermionNf &y, const ComplexD &a, const TXQCDFermionNf &x) {
  for (int aa = 0; aa < TxqcdNf; ++aa) y.f[aa] = y.f[aa] + a * x.f[aa];
}
template <>
inline void scale_ip(TXQCDFermionNf &y, const RealD &a) {
  for (int aa = 0; aa < TxqcdNf; ++aa) y.f[aa] = a * y.f[aa];
}
template <>
inline void copy_field(TXQCDFermionNf &dst, const TXQCDFermionNf &src) {
  for (int aa = 0; aa < TxqcdNf; ++aa) dst.f[aa] = src.f[aa];
}
template <>
inline void random_field(GridParallelRNG &rng, TXQCDFermionNf &f) {
  for (int a = 0; a < TxqcdNf; ++a) gaussian(rng, f.f[a]);
}
template <>
inline void axpby_field(TXQCDFermionNf &out, const RealD &a, const RealD &b,
                        const TXQCDFermionNf &x, const TXQCDFermionNf &y) {
  for (int aa = 0; aa < TxqcdNf; ++aa) out.f[aa] = a * x.f[aa] + b * y.f[aa];
}

// AutoMeasureSigma — same as in meas_conn_weakfield.
static RealD AutoMeasureSigma(GridCartesian &Grid, GridRedBlackCartesian &RBGrid,
                              GridParallelRNG &noisePRNG,
                              const LatticeGaugeField &Usm, int n_noise = 4) {
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  WCF Dw(const_cast<LatticeGaugeField&>(Usm), Grid, RBGrid, mass_light, csw, csw,
         WilsonAnisotropyCoefficients(), impl_p);
  MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
  RealD V = (RealD)Grid.gSites();
  RealD acc = 0.0;
  for (int h = 0; h < n_noise; ++h) {
    LatticeFermion eta(&Grid), b(&Grid), x(&Grid);
    gaussian(noisePRNG, eta);
    Dw.Mdag(eta, b);
    x = Zero();
    CG(HermOp, b, x);
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  return (acc / n_noise) / 2.0;
}

// HermOp wrappers (LinearFunction<Field>) — Field is LatticeFermion or TXQCDFermionNf.

class QcdMdagM : public LinearFunction<LatticeFermion> {
public:
  WCF &Dw_;
  QcdMdagM(WCF &Dw) : Dw_(Dw) {}
  void operator()(const LatticeFermion &in, LatticeFermion &out) override {
    LatticeFermion tmp(in.Grid());
    Dw_.M(in, tmp);
    Dw_.Mdag(tmp, out);
  }
};

class TxqcdMdagM : public LinearFunction<TXQCDFermionNf> {
public:
  TXQCDWilsonCloverOp &Mop_;
  TxqcdMdagM(TXQCDWilsonCloverOp &Mop) : Mop_(Mop) {}
  void operator()(const TXQCDFermionNf &in, TXQCDFermionNf &out) override {
    TXQCDFermionNf tmp(in.Grid());
    Mop_.M(in, tmp);
    Mop_.Mdag(tmp, out);
  }
};

// γ5·M wrappers — Hermitian (since M† = γ5 M γ5  ⇒  (γ5 M)† = M† γ5 = γ5 M).
// Signed eigenvalues: zero crossings ⇒ sign of det(M) = ∏ λ_i changes.
class QcdG5M : public LinearFunction<LatticeFermion> {
public:
  WCF &Dw_;
  QcdG5M(WCF &Dw) : Dw_(Dw) {}
  void operator()(const LatticeFermion &in, LatticeFermion &out) override {
    LatticeFermion tmp(in.Grid());
    Dw_.M(in, tmp);
    Gamma g5(Gamma::Algebra::Gamma5);
    out = g5 * tmp;
  }
};

class TxqcdG5M : public LinearFunction<TXQCDFermionNf> {
public:
  TXQCDWilsonCloverOp &Mop_;
  TxqcdG5M(TXQCDWilsonCloverOp &Mop) : Mop_(Mop) {}
  void operator()(const TXQCDFermionNf &in, TXQCDFermionNf &out) override {
    TXQCDFermionNf tmp(in.Grid());
    Mop_.M(in, tmp);
    Gamma g5(Gamma::Algebra::Gamma5);
    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = g5 * tmp.f[a];
  }
};

// Power iteration for λ_max.
template <class Field>
static RealD PowerIterMaxEval(LinearFunction<Field> &Op, GridParallelRNG &pRNG,
                              GridBase *grid, int n_iter = 80) {
  Field v(grid), w(grid);
  random_field(pRNG, v);
  RealD nv = std::sqrt(norm2(v));
  scale_ip(v, 1.0 / nv);
  RealD ev = 0.0;
  for (int it = 0; it < n_iter; ++it) {
    Op(v, w);
    ev = std::sqrt(norm2(w));
    if (ev == 0.0) break;
    copy_field(v, w);
    scale_ip(v, 1.0 / ev);
  }
  return ev;  // eigenvalue (since v is unit-norm and Op v = ev v)
}

// Apply T_{order-1}(M̃) where M̃ = (2*Op - (hi+lo))/(hi-lo).  T_n is a Chebyshev
// polynomial of the first kind.  This amplifies eigenvalues of Op outside [lo,hi]
// (and most strongly the smallest, when lo > 0) and suppresses those inside.
// Used as a low-pass filter for Lanczos.
template <class Field>
void ApplyChebyT(LinearFunction<Field> &Op, const Field &in, Field &out,
                 RealD lo, RealD hi, int order) {
  GridBase *grid = in.Grid();
  Field T0(grid), T1(grid), T2(grid), y(grid);
  RealD xscale = 2.0 / (hi - lo);
  RealD mscale = -(hi + lo) / (hi - lo);
  copy_field(T0, in);
  Op(T0, y);
  axpby_field(T1, xscale, mscale, y, T0);
  if (order <= 1) { copy_field(out, T0); return; }
  if (order == 2) { copy_field(out, T1); return; }
  Field *Tnm = &T0, *Tn = &T1, *Tnp = &T2;
  for (int n = 2; n < order; ++n) {
    Op(*Tn, y);
    axpby_field(y, xscale, mscale, y, *Tn);  // y = M̃ T_n
    axpby_field(*Tnp, 2.0, -1.0, y, *Tnm);  // T_{n+1} = 2 M̃ T_n − T_{n-1}
    Field *sw = Tnm;
    Tnm = Tn; Tn = Tnp; Tnp = sw;
  }
  copy_field(out, *Tn);
}

template <class Field>
class ChebFilter : public LinearFunction<Field> {
public:
  LinearFunction<Field> &Op_;
  RealD lo_, hi_;
  int order_;
  ChebFilter(LinearFunction<Field> &Op, RealD lo, RealD hi, int order)
      : Op_(Op), lo_(lo), hi_(hi), order_(order) {}
  void operator()(const Field &in, Field &out) override {
    ApplyChebyT<Field>(Op_, in, out, lo_, hi_, order_);
  }
};

// Manual Lanczos with full re-orthogonalization on a Hermitian operator FilterOp.
// Returns Ritz pairs (eigenvalue of FilterOp, vector index in V).  Use the
// vectors to evaluate Rayleigh quotients for the underlying MdagM AND the
// (different, Hermitian) γ5·M operator — the latter gives SIGNED eigenvalues
// whose zero crossings flag sign-of-det(M) changes.
//
// Symmetric tridiagonal T = symtridiag(alpha[0..k-1], beta[0..k-2]).  Ritz
// values are eigenvalues of T; Ritz vectors are V * eigenvec(T).
template <class Field>
static void Lanczos(LinearFunction<Field> &FilterOp,
                    LinearFunction<Field> &MdagM,
                    LinearFunction<Field> &G5M,
                    GridBase *grid, GridParallelRNG &rng, int Nm,
                    std::vector<RealD> &mdagm_evals,
                    std::vector<RealD> &g5m_evals,
                    int Nev) {
  std::vector<Field> V; V.reserve(Nm);
  for (int i = 0; i < Nm; ++i) V.emplace_back(grid);
  std::vector<RealD> alpha(Nm, 0.0), beta(Nm, 0.0);

  Field w(grid), tmp(grid);
  random_field(rng, V[0]);
  RealD n0 = std::sqrt(norm2(V[0]));
  scale_ip(V[0], 1.0 / n0);

  int k_done = 0;
  for (int k = 0; k < Nm; ++k) {
    FilterOp(V[k], w);
    if (k > 0) axpy_ip(w, ComplexD(-beta[k-1], 0), V[k-1]);
    alpha[k] = real(innerProduct(V[k], w));
    axpy_ip(w, ComplexD(-alpha[k], 0), V[k]);
    // Full re-orthogonalization (twice for stability).
    for (int pass = 0; pass < 2; ++pass) {
      for (int j = 0; j <= k; ++j) {
        ComplexD c = innerProduct(V[j], w);
        axpy_ip(w, -c, V[j]);
      }
    }
    beta[k] = std::sqrt(norm2(w));
    k_done = k + 1;
    if (beta[k] < 1e-12) break;
    if (k + 1 < Nm) {
      copy_field(V[k+1], w);
      scale_ip(V[k+1], 1.0 / beta[k]);
    }
  }

  // Diagonalize tridiag T (size k_done × k_done).
  Eigen::MatrixXd T = Eigen::MatrixXd::Zero(k_done, k_done);
  for (int i = 0; i < k_done; ++i) {
    T(i, i) = alpha[i];
    if (i + 1 < k_done) { T(i+1, i) = beta[i]; T(i, i+1) = beta[i]; }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
  // We want the LARGEST Ritz values of FilterOp (= Cheb(MdagM)) — these
  // correspond to SMALLEST eigenvalues of MdagM.
  // es.eigenvectors() columns sorted ascending → use the last Nev columns.
  auto evals = es.eigenvalues();
  auto evecs = es.eigenvectors();

  // For each top Ritz pair (large FilterOp eval → small MdagM eval),
  // build the Ritz vector u = sum_j evecs(j, idx) V_j and compute Rayleigh
  // quotients <u, MdagM u>/<u,u> (positive) and <u, γ5·M u>/<u,u> (signed).
  int top = std::min(Nev, k_done);
  mdagm_evals.clear();
  g5m_evals.clear();
  // Keep (m2, g5) pairs together so we can sort by m2 (smallest-magnitude first)
  // and retain the matching signed γ5·M eigenvalue.
  std::vector<std::pair<RealD, RealD>> pairs;
  pairs.reserve(top);
  for (int rank = 0; rank < top; ++rank) {
    int idx = k_done - 1 - rank;  // largest FilterOp eval first
    Field u(grid); u = Zero();
    for (int j = 0; j < k_done; ++j)
      axpy_ip(u, ComplexD(evecs(j, idx), 0), V[j]);
    RealD nu = norm2(u);
    if (nu < 1e-30) continue;
    MdagM(u, tmp);
    RealD ray_m2 = real(innerProduct(u, tmp)) / nu;
    G5M(u, tmp);
    RealD ray_g5 = real(innerProduct(u, tmp)) / nu;
    pairs.emplace_back(ray_m2, ray_g5);
  }
  // Sort by M†M eigenvalue ascending (smallest magnitude first).
  std::sort(pairs.begin(), pairs.end(),
            [](const std::pair<RealD,RealD> &a, const std::pair<RealD,RealD> &b) {
              return a.first < b.first;
            });
  for (const auto &p : pairs) {
    mdagm_evals.push_back(p.first);
    g5m_evals.push_back(p.second);
  }
}

// Returns the smallest-magnitude eigenvalues of γ5·M (signed), along with the
// matching M†M Rayleigh quotients (positive).  Same ordering for both vectors.
template <class Field>
static void
SmallestEvals(LinearFunction<Field> &MdagM, LinearFunction<Field> &G5M,
              GridBase *grid, GridParallelRNG &pRNG, int Nev, RealD cheby_lo,
              RealD cheby_hi, int cheby_ord, int Nm,
              std::vector<RealD> &m2_evals, std::vector<RealD> &g5_evals) {
  ChebFilter<Field> Filter(MdagM, cheby_lo, cheby_hi, cheby_ord);
  Lanczos(Filter, MdagM, G5M, grid, pRNG, Nm, m2_evals, g5_evals, Nev);
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  if (latt.size() != 4 || latt[0] <= 0) latt = Coordinate({16, 16, 16, 48});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  std::cout << GridLogMessage << "Lattice: " << latt[0] << "."
            << latt[1] << "." << latt[2] << "." << latt[3] << std::endl;
  std::cout << GridLogMessage << "mass=" << mass_light << " csw=" << csw
            << std::endl;

  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({42, 43, 44, 45, 46});
  pRNG.SeedFixedIntegers({142, 143, 144, 145, 146});

  TXQCDField U(&Grid);
  // prod_aux_loaded=true means we loaded BOTH gauge AND TXQCD aux fields from
  // the per-traj production checkpoint (TXQCDCheckpointer sidecar) — i.e. the
  // actual cfg the HMC stream sampled.  In that mode we skip the FillAuxFields
  // call inside the λ loop and force the λ scan to the single LAMBDA env var.
  bool prod_aux_loaded = false;
  if (const char *pt = std::getenv("TXQCD_PROD_TRAJ"); pt && *pt) {
    int traj_load = std::atoi(pt);
    std::cout << GridLogMessage << "TXQCD_PROD_TRAJ=" << traj_load
              << " — loading gauge+aux from production checkpoint "
              << txqcd_cfg_dir() << "/ckpoint_lat." << traj_load << std::endl;
    TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  txqcd_cfg_dir() + "/ckpoint_lat",
                                  txqcd_cfg_dir() + "/ckpoint_rng", traj_load);
    prod_aux_loaded = true;
  } else if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
    std::cout << GridLogMessage << "IMPORT_CFG=" << ic << std::endl;
    FILE *f = std::fopen(ic, "rb"); char magic[16] = {0};
    if (f) { std::fread(magic, 1, sizeof(magic), f); std::fclose(f); }
    FieldMetaData header;
    if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
      typedef GaugeStatistics<PeriodicGimplR> GS;
      NerscIO::readConfiguration<GS>(U.U, header, std::string(ic));
    } else {
      IldgReader IR;
      IR.open(std::string(ic));
      IR.readConfiguration(U.U, header);
      IR.close();
    }
  } else {
    TXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, 0.1);
  }
  std::cout << GridLogMessage << "gauge plaq="
            << WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U) << std::endl;

  Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> Smear(&Grid, stout_nsmear_inv, Stout);
  Smear.set_Field(U.U);
  LatticeGaugeField Usm = Smear.get_SmearedU();

  GridParallelRNG noisePRNG(&Grid);
  noisePRNG.SeedFixedIntegers({1042, 1043, 1044, 1045, 1046});
  RealD Sigma = AutoMeasureSigma(Grid, RBGrid, noisePRNG, Usm, 4);
  std::cout << GridLogMessage << "Σ (auto) = " << Sigma << std::endl;

  // Tunables.
  RealD cheby_lo = 0.05;          // upper edge of "low" band to amplify
  RealD cheby_hi = 80.0;          // upper edge of full M†M spectrum
  int   cheby_ord = 51;           // Chebyshev polynomial order
  int   Nev = 5;                  // number of low Ritz values to return
  int   Nm = 40;                  // Krylov subspace dimension
  if (const char *e = std::getenv("CHEBY_LO");  e && *e) cheby_lo = std::atof(e);
  if (const char *e = std::getenv("CHEBY_HI");  e && *e) cheby_hi = std::atof(e);
  if (const char *e = std::getenv("CHEBY_ORD"); e && *e) cheby_ord = std::atoi(e);
  if (const char *e = std::getenv("NEV");       e && *e) Nev = std::atoi(e);
  if (const char *e = std::getenv("NM");        e && *e) Nm = std::atoi(e);
  std::cout << GridLogMessage << "Cheby(lo=" << cheby_lo << ", hi=" << cheby_hi
            << ", ord=" << cheby_ord << ")  Nev=" << Nev
            << "  Nm=" << Nm << std::endl;

  bool skip_qcd = false;
  if (const char *sq = std::getenv("SKIP_QCD"); sq && *sq && std::atoi(sq))
    skip_qcd = true;

  WilsonImplParams impl_p_outer;  // re-used by both sections; mirror impl_p below
  impl_p_outer.boundary_phases.resize(Nd, 1.0);
  impl_p_outer.boundary_phases[Nd - 1] = -1.0;
  RealD lmax_qcd = 0.0;
  std::vector<RealD> evals_qcd, g5_evals_qcd;
  if (skip_qcd) {
    std::cout << GridLogMessage << "SKIP_QCD=1 — skipping QCD spectrum"
              << std::endl;
  } else {
  // ---- QCD ----
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "==== QCD ====" << std::endl;
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  WCF DwQ(Usm, Grid, RBGrid, mass_light, csw, csw,
          WilsonAnisotropyCoefficients(), impl_p);
  QcdMdagM MdagM_qcd(DwQ);
  QcdG5M   G5M_qcd(DwQ);
  lmax_qcd = PowerIterMaxEval<LatticeFermion>(MdagM_qcd, pRNG, &Grid, 80);
  std::cout << GridLogMessage << "  λ_max(M†M) ≈ " << lmax_qcd << std::endl;
  SmallestEvals<LatticeFermion>(MdagM_qcd, G5M_qcd, &Grid, pRNG,
                                Nev, cheby_lo, cheby_hi,
                                cheby_ord, Nm,
                                evals_qcd, g5_evals_qcd);
  std::cout << GridLogMessage << "  evals(γ5·M) signed, lowest |·|:";
  for (auto e : g5_evals_qcd) std::cout << " " << e;
  std::cout << std::endl;
  std::cout << GridLogMessage << "  evals(M†M) lowest "
            << evals_qcd.size() << ":";
  for (auto e : evals_qcd) std::cout << " " << e;
  std::cout << std::endl;
  if (!evals_qcd.empty())
    std::cout << GridLogMessage << "  κ ≈ " << lmax_qcd / evals_qcd.front()
              << std::endl;
  }  // end !skip_qcd

  // ---- TXQCD λ scan ----
  std::vector<RealD> lambdas;
  if (prod_aux_loaded) {
    // Use the single LAMBDA from env (the one the production stream was run at).
    lambdas = {lambda};
    std::cout << GridLogMessage
              << "Production-aux mode: single λ=" << lambda << std::endl;
  } else if (const char *l = std::getenv("LAMBDAS"); l && *l) {
    std::stringstream ss(l); RealD x; while (ss >> x) lambdas.push_back(x);
  } else {
    lambdas = {4, 6, 12, 18};
  }
  if (const char *sx = std::getenv("SKIP_TXQCD"); sx && *sx && std::atoi(sx))
    lambdas.clear();

  std::map<RealD, std::pair<RealD, std::vector<RealD>>> tx_results;
  for (RealD lam : lambdas) {
    std::cout << GridLogMessage << std::endl;
    std::cout << GridLogMessage << "==== TXQCD λ=" << lam << " ====" << std::endl;
    if (!prod_aux_loaded) {
      GridParallelRNG pRNG_lam(&Grid);
      pRNG_lam.SeedFixedIntegers({500 + (int)(10*lam), 501 + (int)(10*lam),
                                   502 + (int)(10*lam), 503 + (int)(10*lam),
                                   504 + (int)(10*lam)});
      TXQCDCompositeImpl::FillAuxFields(pRNG_lam, U, lam, Sigma);
    } else {
      std::cout << GridLogMessage
                << "  (using production aux fields, NOT re-drawing)" << std::endl;
    }

    std::array<RealD, TxqcdNf> mass_arr;
    for (int a = 0; a < TxqcdNf; ++a) mass_arr[a] = mass_light;
    if (TxqcdNf >= 3) mass_arr[TxqcdNf - 1] = mass_strange;
    TXQCDWilsonCloverOp Mop(Usm, Grid, RBGrid, mass_arr,
                             U.sigma, U.pi, U.s, U.p, U.t, csw, impl_p_outer);
    TxqcdMdagM MdagM_tx(Mop);
    TxqcdG5M   G5M_tx(Mop);
    RealD lmax_tx = PowerIterMaxEval<TXQCDFermionNf>(MdagM_tx, pRNG, &Grid, 80);
    std::cout << GridLogMessage << "  λ_max(M†M) ≈ " << lmax_tx << std::endl;
    std::vector<RealD> evals_tx, g5_evals_tx;
    SmallestEvals<TXQCDFermionNf>(MdagM_tx, G5M_tx, &Grid, pRNG,
                                  Nev, cheby_lo, cheby_hi,
                                  cheby_ord, Nm,
                                  evals_tx, g5_evals_tx);
    std::cout << GridLogMessage << "  evals(γ5·M) signed, lowest |·|:";
    for (auto e : g5_evals_tx) std::cout << " " << e;
    std::cout << std::endl;
    std::cout << GridLogMessage << "  evals(M†M) lowest "
              << evals_tx.size() << ":";
    for (auto e : evals_tx) std::cout << " " << e;
    std::cout << std::endl;
    if (!evals_tx.empty())
      std::cout << GridLogMessage << "  κ ≈ " << lmax_tx / evals_tx.front()
                << std::endl;
    tx_results[lam] = {lmax_tx, evals_tx};
  }

  // Summary table.
  std::cout << std::endl;
  std::cout << "# === eigenvalue spectrum summary ===" << std::endl;
  std::cout << "# action      λ_max     λ_min     |γ5M|_min      κ" << std::endl;
  auto print_row = [](const std::string &label, RealD lmax,
                      const std::vector<RealD> &evals) {
    std::cout << label << "  " << lmax;
    if (evals.empty()) {
      std::cout << "  -  -  -" << std::endl;
    } else {
      RealD lmin = evals.front();
      std::cout << "  " << lmin << "  " << std::sqrt(std::abs(lmin))
                << "  " << lmax / lmin << std::endl;
    }
  };
  print_row("QCD       ", lmax_qcd, evals_qcd);
  for (RealD lam : lambdas) {
    auto &p = tx_results[lam];
    char buf[32]; std::snprintf(buf, sizeof(buf), "TXQCD-λ=%-4g", lam);
    print_row(buf, p.first, p.second);
  }

  Grid_finalize();
  return 0;
}
