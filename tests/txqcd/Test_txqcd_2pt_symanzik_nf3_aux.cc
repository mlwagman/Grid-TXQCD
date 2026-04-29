// Symanzik Nf=3 auxiliary-field correlators on TXQCD-Nf=3 cfgs.
// Compile with -DTXQCD_Nf=3.
//
// Same observables as the Nf=2 symanzik_aux test:
//   aux_pi    - lambda^4 * (sum_{a,b} <pi_ab pi_ba> - <Tr pi>(t)<Tr pi>(0))
//   aux_sigma - lambda^4 * <Tr sigma(t) Tr sigma(0)>_c
//   aux_s     - lambda^4 * <Tr s(t)     Tr s(0)>_c
//   vev_pi, vev_p, vev_t (TXQCD-only diagnostics)
// All are Nf-generic (the SliceSumPiAll loop already runs over (a,b) pairs).

#ifndef TXQCD_Nf
#error "Compile with -DTXQCD_Nf=3"
#endif
static_assert(TXQCD_Nf == 3, "expects TXQCD_Nf=3");

#include "Test_txqcd_2pt_symanzik_nf3_utils.h"
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

template <class HermField>
static std::vector<ComplexD> SliceSumTrace(const HermField &F) {
  LatticeComplex tr(F.Grid());
  tr = trace(F);
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
  const RealD lam = lambda_runtime();
  const RealD lam4 = lam * lam * lam * lam;

  auto trajs = available_trajs(txqcd_nf3_cfg_dir());
  std::cout << GridLogMessage << "[aux-nf3] " << trajs.size()
            << " cfgs in " << txqcd_nf3_cfg_dir() << std::endl;
  mkdir_p(meas_dir());

  std::vector<std::vector<ComplexD>> aux_pi, aux_sigma, aux_s;
  std::vector<RealD> vev_pi, vev_p;
  std::vector<std::vector<RealD>> vev_t;

  TXQCDField U(&Grid);
  for (int traj : trajs) {
    std::cout << GridLogMessage << "[aux-nf3] traj=" << traj << std::endl;
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

    // Connected pi-pi correlator: sum_{a,b} <pi_ab pi_ba> - <Tr pi(t)><Tr pi(0)>
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

    auto sig_tr = SliceSumTrace(U.sigma);
    auto C_sig  = CorrelatorFromSlice(sig_tr, V4);
    std::vector<ComplexD> csig(T);
    for (int t = 0; t < T; ++t) csig[t] = lam4 * C_sig[t];
    aux_sigma.push_back(csig);

    auto s_tr = SliceSumTrace(U.s);
    auto C_s  = CorrelatorFromSlice(s_tr, V4);
    std::vector<ComplexD> cs(T);
    for (int t = 0; t < T; ++t) cs[t] = lam4 * C_s[t];
    aux_s.push_back(cs);
  }

  {
    Hdf5Writer wr(meas_dir() + "/meas_txqcd_nf3_aux.h5");
    write(wr, "aux_pi", aux_pi);
    write(wr, "aux_sigma", aux_sigma);
    write(wr, "aux_s", aux_s);
    write(wr, "vev_pi", vev_pi);
    write(wr, "vev_p", vev_p);
    write(wr, "vev_t", vev_t);
  }

  std::cout << GridLogMessage << "Aux Nf=3 correlators written to "
            << meas_dir() << "/meas_txqcd_nf3_aux.h5" << std::endl;
  Grid_finalize();
  return 0;
}
