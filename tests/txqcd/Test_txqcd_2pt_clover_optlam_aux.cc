// Step 4 (Clover): Auxiliary field correlators from clover TXQCD configs.
// Identical to Wilson version — aux fields have the same structure regardless
// of csw. Only the config directory differs.

#include "Test_txqcd_2pt_clover_optlam_utils.h"
#include <Grid/serialisation/Hdf5IO.h>

using namespace TxqcdTest2ptCloverOptlam;

static std::vector<std::vector<ComplexD>>
SliceSumPiAll(const LatticePiField &piF) {
  GridBase *g = piF.Grid();
  int T = g->GlobalDimensions()[Nd - 1];
  const int Nf2 = TxqcdNf * TxqcdNf;
  std::vector<std::vector<ComplexD>> out(Nf2, std::vector<ComplexD>(T));
  for (int a = 0; a < TxqcdNf; ++a) {
    for (int b = 0; b < TxqcdNf; ++b) {
      LatticeComplex piab(g);
      piab = PeekIndex<2>(piF, a, b);
      std::vector<TComplex> sl;
      sliceSum(piab, sl, Nd - 1);
      for (int t = 0; t < T; ++t)
        out[a * TxqcdNf + b][t] = TensorRemove(sl[t]);
    }
  }
  return out;
}

static std::vector<ComplexD> SliceSumTrace(const LatticePiField &piF) {
  GridBase *g = piF.Grid();
  LatticeComplex tr(g);
  tr = trace(piF);
  std::vector<TComplex> sl;
  sliceSum(tr, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

static std::vector<ComplexD> SliceSumColorTrace(const LatticeSFieldC &sF) {
  GridBase *g = sF.Grid();
  LatticeComplex tr(g);
  tr = trace(sF);
  std::vector<TComplex> sl;
  sliceSum(tr, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

static std::vector<ComplexD> SliceSumSigmaTrace(const LatticeSigmaField &sigF) {
  GridBase *g = sigF.Grid();
  LatticeComplex tr(g);
  tr = trace(sigF);
  std::vector<TComplex> sl;
  sliceSum(tr, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  int T = latt[Nd - 1];
  RealD V4 = 1.0;
  for (int mu = 0; mu < Nd; ++mu) V4 *= latt[mu];
  const RealD lam4 = lambda * lambda * lambda * lambda;

  auto trajs = meas_trajs();
  mkdir_p(meas_dir());

  std::vector<std::vector<ComplexD>> aux_pi, aux_sigma, aux_s;
  std::vector<RealD> vev_pi, vev_p;
  std::vector<std::vector<RealD>> vev_t;

  TXQCDField U(&Grid);
  for (int traj : trajs) {
    std::cout << GridLogMessage << "[aux clover optlam] traj=" << traj << std::endl;
    LoadTxqcdConfig(U, sRNG, pRNG, traj);

    RealD V = (RealD)Grid.gSites();
    vev_pi.push_back(TensorRemove(sum(trace(U.pi))).real() / V);
    vev_p.push_back(TensorRemove(sum(trace(U.p))).real() / V);
    {
      std::vector<RealD> t6;
      for (int mu = 0; mu < Nd; ++mu)
        for (int nu = mu + 1; nu < Nd; ++nu) {
          auto t_mn = PeekIndex<1>(U.t, mu, nu);
          t6.push_back(TensorRemove(sum(trace(t_mn))).real() / V);
        }
      vev_t.push_back(t6);
    }

    auto pi_all = SliceSumPiAll(U.pi);
    auto pi_tr  = SliceSumTrace(U.pi);
    auto C_disc = CorrelatorFromSlice(pi_tr, V4);
    std::vector<ComplexD> C_total(T, 0.0);
    for (const auto &ps : pi_all) {
      auto Cab = CorrelatorFromSlice(ps, V4);
      for (int t = 0; t < T; ++t) C_total[t] += Cab[t];
    }
    std::vector<ComplexD> C_pi(T);
    for (int t = 0; t < T; ++t) C_pi[t] = lam4 * (C_total[t] - C_disc[t]);
    aux_pi.push_back(C_pi);

    auto sig_tr = SliceSumSigmaTrace(U.sigma);
    auto C_sig  = CorrelatorFromSlice(sig_tr, V4);
    std::vector<ComplexD> csig(T);
    for (int t = 0; t < T; ++t) csig[t] = lam4 * C_sig[t];
    aux_sigma.push_back(csig);

    auto s_tr = SliceSumColorTrace(U.s);
    auto C_s  = CorrelatorFromSlice(s_tr, V4);
    std::vector<ComplexD> cs(T);
    for (int t = 0; t < T; ++t) cs[t] = lam4 * C_s[t];
    aux_s.push_back(cs);
  }

  {
    Hdf5Writer wr(meas_dir() + "/meas_txqcd_aux.h5");
    write(wr, "aux_pi", aux_pi);
    write(wr, "aux_sigma", aux_sigma);
    write(wr, "aux_s", aux_s);
    write(wr, "vev_pi", vev_pi);
    write(wr, "vev_p", vev_p);
    write(wr, "vev_t", vev_t);
  }

  std::cout << GridLogMessage << "Auxiliary clover correlators written to "
            << meas_dir() << "/*.h5" << std::endl;
  Grid_finalize();
  return 0;
}
