// Step 2: Compute connected pion and nucleon correlators from point-source
// quark propagators on TXQCD and QCD configurations.
//
// Writes: meas_2pt/{pion_conn,nucleon}_{txqcd,qcd}.dat

#include "Test_txqcd_2pt_utils.h"
#include <Grid/qcd/utils/BaryonUtils.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>

using namespace TxqcdTest2pt;

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

static std::vector<ComplexD> NucleonCorrelator(const LatticePropagator &S_u,
                                               const LatticePropagator &S_d) {
  Gamma G_A(Gamma::Algebra::Identity);
  Gamma G_B(Gamma::Algebra::SigmaXZ);
  int wick = 0;
  BaryonUtils<WilsonImplR>::WickContractions("uud", "uud", wick);
  LatticeComplex Cn(S_u.Grid());
  BaryonUtils<WilsonImplR>::ContractBaryons(S_u, S_u, S_d, G_A, G_B, G_A, G_B,
                                            wick, +1, Cn);
  std::vector<TComplex> sl;
  sliceSum(Cn, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

// Hand-rolled CG on M^dag M for TXQCDFermionNf.
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
  std::cout << GridLogMessage << "[conn CG] iter=" << it
            << " rsq/bsq=" << rsq / bsq << std::endl;
}

static void TxqcdPointProp(LatticePropagator &S_u, LatticePropagator &S_d,
                           TXQCDField &U, RealD m,
                           const Coordinate &src, RealD tol, int maxit) {
  GridBase *g = U.Grid();
  GridCartesian *Ug = dynamic_cast<GridCartesian *>(g);
  GridRedBlackCartesian RB(Ug);
  TXQCDWilsonOp Mop(U.U, *Ug, RB, m, U.sigma, U.pi, U.s, U.p, U.t);

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
        snf.f[0] = Zero();
        snf.f[1] = Zero();
        snf.f[flavor] = sf;
        Mop.Mdag(snf, b);
        TxqcdCG(Mop, b, x, tol, maxit);
        FermToProp<WilsonImplR>(Sout, x.f[flavor], spin, col);
      }
    }
  }
}

static void QcdPointProp(LatticePropagator &S, LatticeGaugeField &Umu,
                         RealD m, GridCartesian &Grid,
                         GridRedBlackCartesian &RBGrid,
                         const Coordinate &src, RealD tol, int maxit) {
  WilsonFermionD Dw(Umu, Grid, RBGrid, m);
  MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(tol, maxit);

  LatticePropagator srcP(&Grid);
  PointSource(src, srcP);
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

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  int T = latt[Nd - 1];
  auto trajs = meas_trajs();
  Coordinate src = src_site();
  mkdir_p(meas_dir());

  std::vector<std::vector<RealD>>    pion_tx, pion_qcd;
  std::vector<std::vector<ComplexD>> nucl_tx, nucl_qcd;

  // TXQCD
  {
    TXQCDField U(&Grid);
    for (int traj : trajs) {
      std::cout << GridLogMessage << "[conn] TXQCD traj=" << traj << std::endl;
      LoadTxqcdConfig(U, sRNG, pRNG, traj);
      LatticePropagator Su(&Grid), Sd(&Grid);
      TxqcdPointProp(Su, Sd, U, mass, src, meas_tol, cg_max);
      pion_tx.push_back(PionCorrelator(Sd, Su));
      nucl_tx.push_back(NucleonCorrelator(Su, Sd));
    }
  }

  // QCD
  {
    LatticeGaugeField Umu(&Grid);
    for (int traj : trajs) {
      std::cout << GridLogMessage << "[conn] QCD traj=" << traj << std::endl;
      LoadQcdConfig(Umu, sRNG, pRNG, traj);
      LatticePropagator S(&Grid);
      QcdPointProp(S, Umu, mass, Grid, RBGrid, src, meas_tol, cg_max);
      pion_qcd.push_back(PionCorrelator(S, S));
      nucl_qcd.push_back(NucleonCorrelator(S, S));
    }
  }

  WriteMeasReal(meas_dir() + "/pion_conn_txqcd.dat", pion_tx, T);
  WriteMeasReal(meas_dir() + "/pion_conn_qcd.dat", pion_qcd, T);
  WriteMeasComplex(meas_dir() + "/nucleon_txqcd.dat", nucl_tx, T);
  WriteMeasComplex(meas_dir() + "/nucleon_qcd.dat", nucl_qcd, T);

  std::cout << GridLogMessage << "Connected 2pt measurements written to "
            << meas_dir() << "/" << std::endl;
  Grid_finalize();
  return 0;
}
