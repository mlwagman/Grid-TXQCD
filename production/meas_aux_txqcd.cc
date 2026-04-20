#include "params.h"

using namespace TXQCDProduction;

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

static std::vector<ComplexD>
CorrelatorFromSlice(const std::vector<ComplexD> &s, RealD V4) {
  int T = (int)s.size();
  std::vector<ComplexD> C(T, 0.0);
  for (int dt = 0; dt < T; ++dt)
    for (int t0 = 0; t0 < T; ++t0)
      C[dt] += s[(t0 + dt) % T] * std::conj(s[t0]);
  for (auto &c : C) c /= V4;
  return C;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: meas_aux_txqcd <traj>" << std::endl;
    return 1;
  }
  int traj = std::atoi(argv[1]);

  Coordinate latt = lattice_size();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  mkdir_p(data_dir());

  int T = latt[Nd - 1];
  RealD V4 = 1.0;
  for (int mu = 0; mu < Nd; ++mu) V4 *= latt[mu];
  const RealD lam4 = lambda * lambda * lambda * lambda;

  TXQCDField U(&Grid);
  TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                txqcd_cfg_dir() + "/ckpoint_lat",
                                txqcd_cfg_dir() + "/ckpoint_rng", traj);

  std::cout << GridLogMessage << "[aux TXQCD] traj=" << traj << std::endl;

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

  auto sig_tr = SliceSumSigmaTrace(U.sigma);
  auto C_sig  = CorrelatorFromSlice(sig_tr, V4);
  std::vector<ComplexD> aux_sigma(T);
  for (int t = 0; t < T; ++t) aux_sigma[t] = lam4 * C_sig[t];

  auto s_tr = SliceSumColorTrace(U.s);
  auto C_s  = CorrelatorFromSlice(s_tr, V4);
  std::vector<ComplexD> aux_s(T);
  for (int t = 0; t < T; ++t) aux_s[t] = lam4 * C_s[t];

  std::string outfile = data_dir() + "/aux_txqcd_" + std::to_string(traj) + ".h5";
  {
    Hdf5Writer wr(outfile);
    write(wr, "aux_pi", C_pi);
    write(wr, "aux_sigma", aux_sigma);
    write(wr, "aux_s", aux_s);
    write(wr, "traj", traj);
  }

  std::cout << GridLogMessage << "Written " << outfile << std::endl;
  Grid_finalize();
  return 0;
}
