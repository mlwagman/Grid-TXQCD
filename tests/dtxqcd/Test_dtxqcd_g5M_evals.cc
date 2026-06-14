// Test_dtxqcd_g5M_evals: lowest |eigenvalues| of γ5·M48 (Hermitian, real evals)
// for the doubled DTXQCD operator.  γ5·M48 has the same det/Pf as M48 up to
// the (fixed) det(γ5) factor, so signs of γ5·M48 eigenvalues directly track
// Pfaffian sign through HMC.
//
// Method: simple Lanczos (no reorthogonalization) for K Krylov dim,
//          extract smallest |λ| Ritz values from the tridiagonal T.
//
// Usage:
//   MASS=0.3 CSW=0.0 N_EV=6 N_KRYLOV=40 ./Test_dtxqcd_g5M_evals <cfg_dir> <traj> --grid ... --mpi ...

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/Eigen/Eigenvalues>

using namespace Grid;

// γ5 applied to doubled fermion (acts on spin only, both blocks).
inline void Gamma5_doubled(DTXQCDFermionDoubled &x) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    x.upper.f[a] = Gamma(Gamma::Algebra::Gamma5) * x.upper.f[a];
    x.lower.f[a] = Gamma(Gamma::Algebra::Gamma5) * x.lower.f[a];
  }
}

// y = γ5 · M48 · x  (Hermitian on doubled space when M48 is γ5-Hermitian).
inline void G5M(DTXQCDWilsonCloverFermionEO &Dw,
                const DTXQCDFermionDoubled &x, DTXQCDFermionDoubled &y) {
  Dw.M(const_cast<DTXQCDFermionDoubled&>(x), y);
  Gamma5_doubled(y);
}

inline void Normalize(DTXQCDFermionDoubled &x) {
  RealD n = std::sqrt(norm2(x));
  if (n < 1e-30) return;
  RealD inv = 1.0 / n;
  for (int a = 0; a < DtxqcdNf; ++a) {
    x.upper.f[a] = inv * x.upper.f[a];
    x.lower.f[a] = inv * x.lower.f[a];
  }
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  if (argc < 3) { std::cerr << "Usage: <cfg_dir> <traj>\n"; return 1; }
  std::string cfg_dir = argv[1];
  int traj = std::atoi(argv[2]);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid_(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid_);
  GridSerialRNG sRNG;  sRNG.SeedFixedIntegers({1});
  GridParallelRNG pRNG(&Grid_);
  pRNG.SeedFixedIntegers({1234, 5678, 91011, 121314, 151617});

  DTXQCDField U(&Grid_);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);

  RealD mass = 0.3, csw = 0.0;
  if (const char *s = std::getenv("MASS"); s && *s) mass = std::atof(s);
  if (const char *s = std::getenv("CSW");  s && *s) csw  = std::atof(s);
  int Nev = 6;
  if (const char *s = std::getenv("N_EV"); s && *s) Nev = std::atoi(s);
  int Nm = 40;
  if (const char *s = std::getenv("N_KRYLOV"); s && *s) Nm = std::atoi(s);

  DTXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass, csw,
                                  U.sigma, U.pi, U.d, U.n, U.s, U.p);

  // ---------- Simple Lanczos ----------
  std::vector<DTXQCDFermionDoubled> V;
  V.reserve(Nm + 1);
  V.emplace_back(&Grid_);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, V[0].upper.f[a]);
    gaussian(pRNG, V[0].lower.f[a]);
  }
  Normalize(V[0]);

  std::vector<RealD> alpha(Nm), beta(Nm + 1);
  beta[0] = 0.0;
  DTXQCDFermionDoubled w(&Grid_);

  int j_final = Nm;
  for (int j = 0; j < Nm; ++j) {
    G5M(Dw, V[j], w);
    // w -= β_{j-1} V[j-1]
    if (j > 0) {
      for (int a = 0; a < DtxqcdNf; ++a) {
        w.upper.f[a] = w.upper.f[a] - beta[j] * V[j-1].upper.f[a];
        w.lower.f[a] = w.lower.f[a] - beta[j] * V[j-1].lower.f[a];
      }
    }
    alpha[j] = innerProduct(V[j], w).real();
    for (int a = 0; a < DtxqcdNf; ++a) {
      w.upper.f[a] = w.upper.f[a] - alpha[j] * V[j].upper.f[a];
      w.lower.f[a] = w.lower.f[a] - alpha[j] * V[j].lower.f[a];
    }
    // Re-orthogonalize against all previous (one pass) to combat loss of
    // orthogonality in finite precision Lanczos.  Cost = O(j · N), worth it.
    for (int i = 0; i < j; ++i) {
      ComplexD c = innerProduct(V[i], w);
      for (int a = 0; a < DtxqcdNf; ++a) {
        w.upper.f[a] = w.upper.f[a] - c * V[i].upper.f[a];
        w.lower.f[a] = w.lower.f[a] - c * V[i].lower.f[a];
      }
    }
    RealD bnext = std::sqrt(norm2(w));
    if (bnext < 1e-12) { j_final = j + 1; break; }
    beta[j + 1] = bnext;
    if (j < Nm - 1) {
      V.emplace_back(&Grid_);
      RealD inv = 1.0 / bnext;
      for (int a = 0; a < DtxqcdNf; ++a) {
        V[j + 1].upper.f[a] = inv * w.upper.f[a];
        V[j + 1].lower.f[a] = inv * w.lower.f[a];
      }
    }
  }

  // ---------- Diagonalize T ----------
  int Nactual = j_final;
  Eigen::MatrixXd T = Eigen::MatrixXd::Zero(Nactual, Nactual);
  for (int i = 0; i < Nactual; ++i) {
    T(i, i) = alpha[i];
    if (i + 1 < Nactual) {
      T(i, i + 1) = beta[i + 1];
      T(i + 1, i) = beta[i + 1];
    }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
  Eigen::VectorXd evals = es.eigenvalues();

  // Sort by absolute value (lowest |λ| first) — these are the ones near zero.
  std::vector<RealD> evals_vec(evals.data(), evals.data() + evals.size());
  std::sort(evals_vec.begin(), evals_vec.end(),
            [](RealD a, RealD b){ return std::abs(a) < std::abs(b); });

  int n_neg = 0;
  for (auto e : evals) if (e < 0) ++n_neg;

  std::cout << GridLogMessage << "Lanczos: " << Nactual << " Krylov dim, "
            << evals.size() << " Ritz values" << std::endl;
  std::cout << GridLogMessage << "RESULT: lowest " << Nev
            << " |λ(γ5M)| (signed):";
  for (int i = 0; i < std::min(Nev, (int)evals_vec.size()); ++i) {
    std::cout << "  " << std::setprecision(6) << std::fixed << evals_vec[i];
  }
  std::cout << std::endl;
  std::cout << GridLogMessage << "RESULT: n_neg(T) = " << n_neg
            << " / " << evals.size() << "  (parity = " << (n_neg & 1)
            << ")" << std::endl;

  Grid_finalize();
  return 0;
}
