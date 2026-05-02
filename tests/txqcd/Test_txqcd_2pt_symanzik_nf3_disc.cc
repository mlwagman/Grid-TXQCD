// Symanzik Nf=3 disconnected loops + VEVs: check that TXQCD-Nf=3 with
// diag mass (m_l, m_l, m_s) reproduces the QCD Nf=2+1 condensates.
//
// Compile with -DTXQCD_Nf=3.
//
// What's measured per cfg:
//   - light  Tr M^-1  (per-flavor noise on flavor 0/1, averaged)
//   - strange Tr M^-1 (per-flavor noise on flavor 2)
//   - aux VEVs (Tr σ, Tr s) — TXQCD only

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
  for (int it = 0; it < maxit; ++it) {
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
}

// Stochastic Tr_{spin,color}(M^-1)_{aa,aa} on flavor `flav`: noise source on
// that flavor only, solve, extract <eta|M^-1|eta> on the same flavor block.
// Returned value is normalized the same way as StochasticTrMinv in the Nf=2
// symanzik test: factor 1/(2V) (the /2 absorbs the pseudofermion doubling
// that appears when Tr M^-1 enters via det(M^dag M)).
static RealD StochasticTrMinv_TXQCDNf3(TXQCDWilsonCloverOp &Mop, GridBase *grid,
                                       GridParallelRNG &pRNG, int flav, int nn,
                                       RealD tol, int maxit) {
  RealD V = (RealD)grid->gSites();
  RealD acc = 0.0;
  for (int h = 0; h < nn; ++h) {
    TXQCDFermionNf eta(grid), b(grid), x(grid);
    for (int a = 0; a < TxqcdNf; ++a) eta.f[a] = Zero();
    gaussian(pRNG, eta.f[flav]);
    Mop.Mdag(eta, b);
    TxqcdCG(Mop, b, x, tol, maxit);
    // Extract only the same-flavor block: <eta_flav|M^-1 eta_flav>.
    acc += innerProduct(eta.f[flav], x.f[flav]).real() / (2.0 * V);
  }
  return acc / nn;
}

// Stochastic estimator for sum_x Tr_{spin,color}[gamma5 (M^-1)_{a_sink,b_src}](x,t)
// per time slice, used as the "disconnected" piece subtraction in the TXQCD
// flavor-non-singlet meson Fierz comparison: the off-diagonal flavor block of
// M^-1 (induced by sigma/pi aux fields mixing flavors) contributes a Wick
// pairing that vanishes in QCD (M^-1 flavor-diagonal there).
//
// Noise on src flavor only; solve M^-1; pull the sink-flavor component;
// localInnerProduct at the source flavor gives the Hutchinson estimator.
// /2.0 normalization matches Nf=2 StochasticLoop_ud (absorbs det(M^dag M)
// pseudofermion doubling).
static std::vector<ComplexD>
StochasticOffDiagFlavorLoop_Nf3(TXQCDWilsonCloverOp &Mop, GridBase *grid,
                                GridParallelRNG &pRNG,
                                int sink_flav, int src_flav,
                                int nn, RealD tol, int maxit) {
  int T = grid->GlobalDimensions()[Nd - 1];
  Gamma g5(Gamma::Algebra::Gamma5);
  std::vector<ComplexD> L(T, 0.0);

  for (int h = 0; h < nn; ++h) {
    TXQCDFermionNf src(grid), b(grid), x(grid);
    for (int a = 0; a < TxqcdNf; ++a) src.f[a] = Zero();
    gaussian(pRNG, src.f[src_flav]);
    Mop.Mdag(src, b);
    TxqcdCG(Mop, b, x, tol, maxit);

    LatticeFermion g5x(grid);
    g5x = g5 * x.f[sink_flav];
    LatticeComplex lf(grid);
    lf = localInnerProduct(src.f[src_flav], g5x);
    std::vector<TComplex> sl;
    sliceSum(lf, sl, Nd - 1);
    for (int t = 0; t < T; ++t)
      L[t] += TensorRemove(sl[t]) / (2.0 * nn);
  }
  return L;
}

typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
static RealD StochasticTrMinv_QCD(WCF &Dw, GridBase *grid,
                                  GridParallelRNG &pRNG, int nn, RealD tol,
                                  int maxit) {
  RealD V = (RealD)grid->gSites();
  MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(tol, maxit);
  RealD acc = 0.0;
  for (int h = 0; h < nn; ++h) {
    LatticeFermion eta(grid), b(grid), x(grid);
    gaussian(pRNG, eta);
    Dw.Mdag(eta, b);
    x = Zero();
    CG(HermOp, b, x);
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  return acc / nn;
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

  auto tx_avail = available_trajs(txqcd_nf3_cfg_dir());
  auto qc_avail = available_trajs(qcd_cfg_dir());
  std::vector<int> trajs;
  std::set_intersection(tx_avail.begin(), tx_avail.end(),
                        qc_avail.begin(), qc_avail.end(),
                        std::back_inserter(trajs));
  std::cout << GridLogMessage << "[disc-nf3] " << trajs.size()
            << " common cfgs" << std::endl;

  mkdir_p(meas_dir());

  std::vector<RealD> vev_sigma_tx, vev_s_tx;
  std::vector<RealD> trminv_l_tx, trminv_s_tx;
  std::vector<RealD> trminv_l_qc, trminv_s_qc;
  // Off-diagonal flavor blocks: pion-disc uses (sink=0, src=1); kaon-disc
  // uses (sink=0, src=2).  TXQCD-only -- vanishes identically in QCD.
  std::vector<std::vector<ComplexD>> loop_pion_tx, loop_kaon_tx;

  Smear_Stout<PeriodicGimplR> Stout(stout_rho);
  SmearedConfiguration<PeriodicGimplR> SmearPolicy(&Grid, stout_nsmear, Stout);

  std::array<RealD, 3> mass_diag = nf3_mass();

  // TXQCD Nf=3
  {
    TXQCDField U(&Grid);
    for (int traj : trajs) {
      std::cout << GridLogMessage << "[disc-nf3] TXQCD traj=" << traj << std::endl;
      sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
      pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
      LoadTxqcdConfig(U, sRNG, pRNG, traj);

      RealD V = (RealD)Grid.gSites();
      vev_sigma_tx.push_back(TensorRemove(sum(trace(U.sigma))).real() / V);
      vev_s_tx.push_back(TensorRemove(sum(trace(U.s))).real() / V);

      SmearPolicy.set_Field(U.U);
      LatticeGaugeField Usmeared = SmearPolicy.get_SmearedU();

      GridCartesian *Ug = dynamic_cast<GridCartesian *>(U.Grid());
      GridRedBlackCartesian RB(Ug);
      TXQCDWilsonCloverOp Mop(Usmeared, *Ug, RB, mass_diag, U.sigma, U.pi,
                              U.s, U.p, U.t, csw);

      // Light = avg over (flavor 0, flavor 1).
      RealD tr_l0 = StochasticTrMinv_TXQCDNf3(Mop, &Grid, pRNG, 0,
                                              n_noise, meas_tol, cg_max);
      RealD tr_l1 = StochasticTrMinv_TXQCDNf3(Mop, &Grid, pRNG, 1,
                                              n_noise, meas_tol, cg_max);
      trminv_l_tx.push_back(0.5 * (tr_l0 + tr_l1));
      trminv_s_tx.push_back(StochasticTrMinv_TXQCDNf3(
          Mop, &Grid, pRNG, 2, n_noise, meas_tol, cg_max));

      // Off-diagonal flavor M^-1 loops for the meson Fierz disc subtraction.
      // pion: between flavors 0 and 1 (both light).
      // kaon: between flavor 0 (light) and flavor 2 (strange).
      loop_pion_tx.push_back(StochasticOffDiagFlavorLoop_Nf3(
          Mop, &Grid, pRNG, /*sink=*/0, /*src=*/1, n_noise, meas_tol, cg_max));
      loop_kaon_tx.push_back(StochasticOffDiagFlavorLoop_Nf3(
          Mop, &Grid, pRNG, /*sink=*/0, /*src=*/2, n_noise, meas_tol, cg_max));
    }
  }

  // QCD Nf=2+1 reference
  {
    LatticeGaugeField Umu(&Grid);
    for (int traj : trajs) {
      std::cout << GridLogMessage << "[disc-nf3] QCD traj=" << traj << std::endl;
      sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
      pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});
      TxqcdTest2ptSymanzik::LoadQcdConfig(Umu, sRNG, pRNG, traj);
      SmearPolicy.set_Field(Umu);
      LatticeGaugeField Usmeared = SmearPolicy.get_SmearedU();

      WCF Dw_l(Usmeared, Grid, RBGrid, mass,   csw, csw);
      WCF Dw_s(Usmeared, Grid, RBGrid, mass_s, csw, csw);
      trminv_l_qc.push_back(
          StochasticTrMinv_QCD(Dw_l, &Grid, pRNG, n_noise, meas_tol, cg_max));
      trminv_s_qc.push_back(
          StochasticTrMinv_QCD(Dw_s, &Grid, pRNG, n_noise, meas_tol, cg_max));
    }
  }

  {
    Hdf5Writer wr(meas_dir() + "/meas_txqcd_nf3_disc.h5");
    write(wr, "vev_sigma",  vev_sigma_tx);
    write(wr, "vev_s",      vev_s_tx);
    write(wr, "trminv_l",   trminv_l_tx);
    write(wr, "trminv_s",   trminv_s_tx);
    write(wr, "loop_pion",  loop_pion_tx);
    write(wr, "loop_kaon",  loop_kaon_tx);
  }
  {
    Hdf5Writer wr(meas_dir() + "/meas_qcd_disc.h5");
    write(wr, "trminv_l", trminv_l_qc);
    write(wr, "trminv_s", trminv_s_qc);
  }

  std::cout << GridLogMessage << "Symanzik Nf=3 disc/VEV measurements written to "
            << meas_dir() << std::endl;
  Grid_finalize();
  return 0;
}
