// Step 4: Compute auxiliary field correlators from TXQCD configurations.
//
// C_pi(dt)    = lam^4 [<Tr_f[pi(t) pi^dag(0)]> - <Tr_f[pi(t)] Tr_f[pi^dag(0)]>]
//             = lam^4 (I=1 pion, traceless flavor projection, volume-averaged over sources)
// C_sigma(dt) = lam^4 <Tr_f[sigma(t)] Tr_f[sigma^dag(0)]>
// C_s(dt)     = lam^4 <Tr_c[s(t)] Tr_c[s^dag(0)]>
//
// No fermion inversions needed — correlators are computed directly from aux fields.
//
// Writes: meas_2pt/{aux_pi, aux_sigma, aux_s}_txqcd.dat

#include "Test_txqcd_2pt_utils.h"

using namespace TxqcdTest2pt;

// Per-element slice sums for Nf x Nf flavor matrix field.
// Returns Nf*Nf vectors of length T.
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

// Flavor trace slice sum: Tr_f[pi](t).
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

// Color trace slice sum: Tr_c[s](t).
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

// Flavor trace slice sum for sigma (same structure as pi).
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

  TXQCDField U(&Grid);
  for (int traj : trajs) {
    std::cout << GridLogMessage << "[aux] traj=" << traj << std::endl;
    LoadTxqcdConfig(U, sRNG, pRNG, traj);

    // C_pi = lam^4 [Tr_f(pi pi^dag) - Tr_f(pi) Tr_f(pi^dag)]
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

    // C_sigma = lam^4 <Tr_f[sigma] Tr_f[sigma]*>
    auto sig_tr = SliceSumSigmaTrace(U.sigma);
    auto C_sig  = CorrelatorFromSlice(sig_tr, V4);
    std::vector<ComplexD> csig(T);
    for (int t = 0; t < T; ++t) csig[t] = lam4 * C_sig[t];
    aux_sigma.push_back(csig);

    // C_s = lam^4 <Tr_c[s] Tr_c[s]*>
    auto s_tr = SliceSumColorTrace(U.s);
    auto C_s  = CorrelatorFromSlice(s_tr, V4);
    std::vector<ComplexD> cs(T);
    for (int t = 0; t < T; ++t) cs[t] = lam4 * C_s[t];
    aux_s.push_back(cs);
  }

  WriteMeasComplex(meas_dir() + "/aux_pi_txqcd.dat", aux_pi, T);
  WriteMeasComplex(meas_dir() + "/aux_sigma_txqcd.dat", aux_sigma, T);
  WriteMeasComplex(meas_dir() + "/aux_s_txqcd.dat", aux_s, T);

  std::cout << GridLogMessage << "Auxiliary correlators written to "
            << meas_dir() << "/" << std::endl;
  Grid_finalize();
  return 0;
}
