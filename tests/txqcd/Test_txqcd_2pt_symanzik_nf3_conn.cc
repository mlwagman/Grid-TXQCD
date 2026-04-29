// Symanzik Nf=3 connected correlators: pion (light-light) and kaon
// (light-strange) on TXQCD-Nf=3 cfgs.  Compares against QCD Nf=2+1 reference.
//
// Compile with -DTXQCD_Nf=3.
//
// Key idea: in Nf=3 TXQCD, the propagator is naturally a (Nf*Ns*Nc)x(Nf*Ns*Nc)
// site matrix.  We invert with the source on each (flavor, spin, color) basis
// vector, giving us per-source-flavor LatticePropagators.  Then:
//   pion = trace[γ5 S_l(0,x) γ5 S_l(x,0)†]   (light = flavor 0)
//   kaon = trace[γ5 S_s(0,x) γ5 S_l(x,0)†]   (strange = flavor 2)
// where S_a(x, 0) means the propagator from a unit source on flavor a sampled
// at the flavor-a sink.

#ifndef TXQCD_Nf
#error "Compile with -DTXQCD_Nf=3"
#endif
static_assert(TXQCD_Nf == 3, "expects TXQCD_Nf=3");

#include "Test_txqcd_2pt_symanzik_nf3_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <dirent.h>
#include <algorithm>

using namespace TxqcdTest2ptSymanzikNf3;

// Discover the trajectories actually available (useful for a quick-test
// gencfgs run where we override N_THERM/N_PROD): list ckpoint_lat.<n> files
// in cfg_dir, parse the suffix, and intersect with the QCD reference set so
// the comparison stays apples-to-apples.
static std::vector<int> available_trajs(const std::string &cfg_dir) {
  std::vector<int> out;
  DIR *dp = opendir(cfg_dir.c_str());
  if (!dp) return out;
  struct dirent *ent;
  const std::string prefix = "ckpoint_lat.";
  while ((ent = readdir(dp)) != nullptr) {
    std::string name = ent->d_name;
    if (name.compare(0, prefix.size(), prefix) != 0) continue;
    std::string num = name.substr(prefix.size());
    if (!num.empty() && std::all_of(num.begin(), num.end(), ::isdigit))
      out.push_back(std::stoi(num));
  }
  closedir(dp);
  std::sort(out.begin(), out.end());
  return out;
}

static void PointSource(const Coordinate &site, LatticePropagator &src) {
  src = Zero();
  SpinColourMatrix kron;
  kron = 1.0;
  pokeSite(kron, src, site);
}

// trace[γ5 S_a(0,x) γ5 S_b(x,0)†]: meson built from flavor-a propagator
// at the source, flavor-b at the sink.  For pion, S_a = S_b = S_light.
static std::vector<RealD> MesonCorrelator(const LatticePropagator &S_a,
                                          const LatticePropagator &S_b) {
  LatticeComplex corr(S_a.Grid());
  corr = trace(S_a * adj(S_b));
  std::vector<TComplex> Csl;
  sliceSum(corr, Csl, Nd - 1);
  std::vector<RealD> out(Csl.size());
  for (size_t t = 0; t < Csl.size(); ++t)
    out[t] = TensorRemove(Csl[t]).real();
  return out;
}

static void TxqcdCG(TXQCDWilsonCloverOp &Mop, const TXQCDFermionNf &b,
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
    for (int aa = 0; aa < TxqcdNf; ++aa) p.f[aa] = r.f[aa] + beta_cg * p.f[aa];
    rsq = rsq_new;
  }
  std::cout << GridLogMessage << "[conn CG] iter=" << it
            << " rsq/bsq=" << rsq / bsq << std::endl;
}

// Compute three propagators S_l (flavor 0), S_l' (flavor 1) [should equal S_l
// up to noise from off-diagonal aux], and S_s (flavor 2) on a Nf=3 TXQCD cfg.
// Source flavor selects which row of TXQCDFermionNf is non-zero; we extract
// the same flavor at the sink (flavor-diagonal block of full propagator).
static void TxqcdPointPropNf3(LatticePropagator &S_l, LatticePropagator &S_s,
                              LatticeGaugeField &Ulinks, TXQCDField &U,
                              const std::array<RealD, 3> &m,
                              const Coordinate &src, RealD tol, int maxit) {
  GridBase *g = U.Grid();
  GridCartesian *Ug = dynamic_cast<GridCartesian *>(g);
  GridRedBlackCartesian RB(Ug);
  TXQCDWilsonCloverOp Mop(Ulinks, *Ug, RB, m, U.sigma, U.pi, U.s, U.p, U.t,
                          csw);

  LatticePropagator srcP(g);
  PointSource(src, srcP);
  S_l = Zero();
  S_s = Zero();

  // For each source flavor a in {0, 2} (light, strange), invert and read off
  // the flavor-a sink component (light/strange propagator respectively).
  for (int flavor : {0, 2}) {
    LatticePropagator &Sout = (flavor == 0) ? S_l : S_s;
    for (int spin = 0; spin < Ns; ++spin) {
      for (int col = 0; col < Nc; ++col) {
        LatticeFermion sf(g);
        PropToFerm<WilsonImplR>(sf, srcP, spin, col);
        TXQCDFermionNf snf(g), b(g), x(g);
        for (int aa = 0; aa < TxqcdNf; ++aa) snf.f[aa] = Zero();
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
  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  WCF Dw(Umu, Grid, RBGrid, m, csw, csw);
  MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
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

  // Intersect TXQCD-Nf=3 cfgs with QCD reference cfgs so both ensembles
  // have the same trajectory list.
  auto tx_avail = available_trajs(txqcd_nf3_cfg_dir());
  auto qcd_avail = available_trajs(qcd_cfg_dir());
  std::vector<int> trajs;
  std::set_intersection(tx_avail.begin(), tx_avail.end(),
                        qcd_avail.begin(), qcd_avail.end(),
                        std::back_inserter(trajs));
  std::cout << GridLogMessage << "[conn-nf3] " << trajs.size()
            << " common cfgs (TXQCD has " << tx_avail.size()
            << ", QCD has " << qcd_avail.size() << ")" << std::endl;
  Coordinate src = src_site();
  mkdir_p(meas_dir());

  std::vector<std::vector<RealD>> pion_tx, kaon_tx, pion_qcd, kaon_qcd;
  std::vector<RealD> plaq_tx, plaq_qcd;

  Smear_Stout<PeriodicGimplR> Stout(stout_rho);
  SmearedConfiguration<PeriodicGimplR> SmearPolicy(&Grid, stout_nsmear, Stout);

  std::array<RealD, 3> mass_diag = nf3_mass();

  // TXQCD Nf=3
  {
    TXQCDField U(&Grid);
    for (int traj : trajs) {
      std::cout << GridLogMessage
                << "[conn-nf3] TXQCD traj=" << traj << std::endl;
      LoadTxqcdConfig(U, sRNG, pRNG, traj);
      plaq_tx.push_back(WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U));
      SmearPolicy.set_Field(U.U);
      LatticeGaugeField Usmeared = SmearPolicy.get_SmearedU();
      LatticePropagator Sl(&Grid), Ss(&Grid);
      TxqcdPointPropNf3(Sl, Ss, Usmeared, U, mass_diag, src, meas_tol, cg_max);
      pion_tx.push_back(MesonCorrelator(Sl, Sl));
      kaon_tx.push_back(MesonCorrelator(Sl, Ss));
    }
  }

  // QCD Nf=2+1 reference
  {
    LatticeGaugeField Umu(&Grid);
    for (int traj : trajs) {
      std::cout << GridLogMessage
                << "[conn-nf3] QCD traj=" << traj << std::endl;
      TxqcdTest2ptSymanzik::LoadQcdConfig(Umu, sRNG, pRNG, traj);
      plaq_qcd.push_back(WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu));
      SmearPolicy.set_Field(Umu);
      LatticeGaugeField Usmeared = SmearPolicy.get_SmearedU();
      LatticePropagator Sl(&Grid), Ss(&Grid);
      QcdPointProp(Sl, Usmeared, mass,   Grid, RBGrid, src, meas_tol, cg_max);
      QcdPointProp(Ss, Usmeared, mass_s, Grid, RBGrid, src, meas_tol, cg_max);
      pion_qcd.push_back(MesonCorrelator(Sl, Sl));
      kaon_qcd.push_back(MesonCorrelator(Sl, Ss));
    }
  }

  {
    Hdf5Writer wr(meas_dir() + "/meas_txqcd_nf3_conn.h5");
    write(wr, "pion", pion_tx);
    write(wr, "kaon", kaon_tx);
    write(wr, "plaq", plaq_tx);
  }
  {
    Hdf5Writer wr(meas_dir() + "/meas_qcd_conn.h5");
    write(wr, "pion", pion_qcd);
    write(wr, "kaon", kaon_qcd);
    write(wr, "plaq", plaq_qcd);
  }

  std::cout << GridLogMessage
            << "Symanzik Nf=3 conn measurements written to "
            << meas_dir() << std::endl;
  Grid_finalize();
  return 0;
}
