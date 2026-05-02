// Diagnostic: connected pion on TXQCD configs with off-diagonal flavor aux
// fields (sigma_12, pi_12) zeroed at inversion time.  Distinguishes the
// connected-pion contamination contributions from off-diagonal aux (cross-
// flavor sigma/pi fluctuations) versus diagonal aux (sigma_aa, pi_aa, s, p, t).
//
// NOTE: this is *not* a true Nf=1+1 head-to-head — the gauge field was
// generated under the unconstrained Nf=2 dynamics and the color aux s,p,t
// remain shared between flavors.  It only isolates how much of the connected-
// pion shift comes specifically from the off-diagonal flavor mixing in
// sigma/pi.
//
// Writes: meas_<lam>/meas_diagonal_conn.h5

#include "Test_txqcd_2pt_wilson8x24_utils.h"
#include <Grid/serialisation/Hdf5IO.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TxqcdTest2ptWilson8x24;

static void PointSource(const Coordinate &site, LatticePropagator &src) {
  src = Zero();
  SpinColourMatrix kron;
  kron = 1.0;
  pokeSite(kron, src, site);
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

// Zero the off-diagonal entries of a flavor-Hermitian aux field (sigma, pi),
// keeping the diagonal entries unchanged.  Uses the lattice site loop pattern
// from the disc/EO action code.
static void ZeroOffDiagonal(LatticeSigmaField &fld) {
  GridBase *g = fld.Grid();
  autoView(v, fld, AcceleratorWrite);
  accelerator_for(ss, g->oSites(), 1, {
    auto sl = v(ss);
    for (int a = 0; a < TxqcdNf; ++a)
      for (int b = 0; b < TxqcdNf; ++b)
        if (a != b) sl()()(a, b) = Zero();
    coalescedWrite(v[ss], sl);
  });
}

// Hand-rolled CG on M^dag M for TXQCDFermionNf (matches conn test).
static void TxqcdCG(TXQCDWilsonOp &Mop, const TXQCDFermionNf &b,
                     TXQCDFermionNf &x, RealD tol, int maxit) {
  GridBase *g = b.Grid();
  TXQCDFermionNf r(g), p(g), Mp(g), MdMp(g);
  x = Zero();
  r = b;
  p = r;
  RealD rsq = norm2(r);
  RealD bsq = std::max(norm2(b), 1e-30);
  RealD tol2 = tol * tol * bsq;
  int it;
  for (it = 0; it < maxit; ++it) {
    Mop.M(p, Mp);
    Mop.Mdag(Mp, MdMp);
    ComplexD pAp = innerProduct(p, MdMp);
    ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
    axpy(x, alpha, p);
    axpy(r, -alpha, MdMp);
    RealD rsq_new = norm2(r);
    if (rsq_new < tol2) { rsq = rsq_new; break; }
    RealD beta_cg = rsq_new / rsq;
    for (int a = 0; a < TxqcdNf; ++a) p.f[a] = r.f[a] + beta_cg * p.f[a];
    rsq = rsq_new;
  }
  std::cout << GridLogMessage << "[diag CG] iter=" << it
            << " rsq/bsq=" << rsq / bsq << std::endl;
}

static void TxqcdPointPropDiagAux(LatticePropagator &S_u, LatticePropagator &S_d,
                                  TXQCDField &U, RealD m,
                                  const Coordinate &src, RealD tol, int maxit) {
  GridBase *g = U.Grid();
  GridCartesian *Ug = dynamic_cast<GridCartesian *>(g);
  GridRedBlackCartesian RB(Ug);

  // Build off-diagonal-zeroed copies of sigma and pi.
  LatticeSigmaField sigma_diag = U.sigma;  ZeroOffDiagonal(sigma_diag);
  LatticePiField    pi_diag    = U.pi;     ZeroOffDiagonal(pi_diag);

  // s, p, t remain shared (flavor-trivial color-Hermitian aux).
  TXQCDWilsonOp Mop(U.U, *Ug, RB, m, sigma_diag, pi_diag, U.s, U.p, U.t);

  LatticePropagator srcP(g);
  PointSource(src, srcP);
  S_u = Zero();
  S_d = Zero();

  for (int flavor = 0; flavor < TxqcdNf; ++flavor) {
    LatticePropagator &Sout = (flavor == 0) ? S_u : S_d;
    for (int spin = 0; spin < Ns; ++spin) {
      for (int col = 0; col < Nc; ++col) {
        LatticeFermion sf(g);
        PropToFerm<WilsonImplR>(sf, srcP, spin, col);
        TXQCDFermionNf snf(g), b(g), x(g);
        for (int a = 0; a < TxqcdNf; ++a) snf.f[a] = Zero();
        snf.f[flavor] = sf;
        Mop.Mdag(snf, b);
        TxqcdCG(Mop, b, x, tol, maxit);
        FermToProp<WilsonImplR>(Sout, x.f[flavor], spin, col);
      }
    }
  }
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  auto trajs = meas_trajs();
  Coordinate src = src_site();
  mkdir_p(meas_dir());

  if (!txqcd_configs_exist()) {
    std::cout << GridLogMessage
              << "[diagonal] TXQCD configs not found; nothing to do."
              << std::endl;
    Grid_finalize();
    return 0;
  }

  std::cout << GridLogMessage << "[diagonal] LAMBDA=" << lambda_runtime()
            << " mass=" << mass_runtime()
            << " (sigma_12, pi_12 zeroed at inversion)" << std::endl;

  std::vector<std::vector<RealD>> pion_diag;
  std::vector<RealD> plaq;

  TXQCDField U(&Grid);
  for (int traj : trajs) {
    std::cout << GridLogMessage << "[diagonal] TXQCD traj=" << traj << std::endl;
    LoadTxqcdConfig(U, sRNG, pRNG, traj);
    plaq.push_back(WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U));
    LatticePropagator Su(&Grid), Sd(&Grid);
    TxqcdPointPropDiagAux(Su, Sd, U, mass_runtime(), src, meas_tol, cg_max);
    pion_diag.push_back(PionCorrelator(Sd, Su));
  }

  Hdf5Writer wr(meas_dir() + "/meas_diagonal_conn.h5");
  write(wr, "pion_conn", pion_diag);
  write(wr, "plaq", plaq);

  std::cout << GridLogMessage << "[diagonal] wrote " << meas_dir()
            << "/meas_diagonal_conn.h5" << std::endl;
  Grid_finalize();
  return 0;
}
