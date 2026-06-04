#include "params.h"

using namespace TXQCDProduction;

// ---- Slice helpers --------------------------------------------------------
// All return one complex per global timeslice (sliceSum is a Grid collective).

// Σ_{x} (Tr_flavor of a flavor-Hermitian aux at site x), per t.
static std::vector<ComplexD>
SliceSumFlavorTrace(const LatticeSigmaField &F) {
  GridBase *g = F.Grid();
  LatticeComplex tr(g);
  tr = trace(F);
  std::vector<TComplex> sl;
  sliceSum(tr, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

// Σ_{x} (Tr_color of a color-Hermitian aux at site x), per t.
static std::vector<ComplexD>
SliceSumColorTrace(const LatticeSFieldC &F) {
  GridBase *g = F.Grid();
  LatticeComplex tr(g);
  tr = trace(F);
  std::vector<TComplex> sl;
  sliceSum(tr, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

// Σ_{x} π_ab(x) per (a,b,t), shape [Nf²][T] flat (row-major a*Nf+b).
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

// Σ_{x} Tr_color t_{μν}(x), one channel per antisymmetric (μ,ν) pair, per t.
// Returns 6 channels for Nd=4 (pairs: 01,02,03,12,13,23) in row-major order.
static std::vector<std::vector<ComplexD>>
SliceSumTAll(const LatticeTField &tF) {
  GridBase *g = tF.Grid();
  int T = g->GlobalDimensions()[Nd - 1];
  const int Npairs = Nd * (Nd - 1) / 2;
  std::vector<std::vector<ComplexD>> out(Npairs, std::vector<ComplexD>(T));
  int idx = 0;
  for (int mu = 0; mu < Nd; ++mu) {
    for (int nu = mu + 1; nu < Nd; ++nu, ++idx) {
      auto t_munu = PeekIndex<1>(tF, mu, nu);
      LatticeComplex tr(g);
      tr = trace(reinterpret_cast<const LatticeColourMatrix &>(t_munu));
      std::vector<TComplex> sl;
      sliceSum(tr, sl, Nd - 1);
      for (int t = 0; t < T; ++t)
        out[idx][t] = TensorRemove(sl[t]);
    }
  }
  return out;
}

// ---- Correlator + vacuum subtraction --------------------------------------
//
// s[t] = Σ_{spatial x} O(x,t), a spatial volume sum.  Note <s[t]> = V3·<O>.
// Cross-correlator:
//   C_AB(τ) ≡ (1/V4)·Σ_{t0} s_A(t0+τ)·s_B(t0)*  -- matches the original code.
// Vacuum (disconnected) piece:
//   C_AB_disc = (1/V4)·T·V3²·<O_A>·<O_B>* = V3·<O_A><O_B>*
//             = (1/V3) · vev_slice_A · conj(vev_slice_B),    V3 = V4/T,
//   where vev_slice = (1/T)·Σ_t s[t] ≈ V3·<O>.
// Subtracted correlator: C_sub = C_raw - C_disc (a constant in τ).

static std::vector<ComplexD>
CorrelatorFromSlice(const std::vector<ComplexD> &sA,
                    const std::vector<ComplexD> &sB,
                    RealD V4) {
  int T = (int)sA.size();
  std::vector<ComplexD> C(T, 0.0);
  for (int dt = 0; dt < T; ++dt)
    for (int t0 = 0; t0 < T; ++t0)
      C[dt] += sA[(t0 + dt) % T] * conjugate(sB[t0]);
  for (auto &c : C) c /= V4;
  return C;
}

static std::vector<ComplexD>
CorrelatorFromSlice(const std::vector<ComplexD> &s, RealD V4) {
  return CorrelatorFromSlice(s, s, V4);
}

static ComplexD SliceMean(const std::vector<ComplexD> &s) {
  ComplexD acc = 0.0;
  for (auto v : s) acc += v;
  return acc / RealD(s.size());
}

static std::vector<ComplexD>
VacSubtractCorrelator(const std::vector<ComplexD> &C_raw,
                      ComplexD vev_slice_A, ComplexD vev_slice_B,
                      RealD V3) {
  std::vector<ComplexD> out = C_raw;
  ComplexD disc = vev_slice_A * conjugate(vev_slice_B) / V3;
  for (auto &c : out) c -= disc;
  return out;
}

static std::vector<ComplexD>
VacSubtractCorrelator(const std::vector<ComplexD> &C_raw,
                      ComplexD vev_slice, RealD V3) {
  return VacSubtractCorrelator(C_raw, vev_slice, vev_slice, V3);
}

static void Scale(std::vector<ComplexD> &v, RealD c) {
  for (auto &x : v) x *= c;
}

// ---- Main -----------------------------------------------------------------

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

  mkdir_p(txqcd_data_dir());

  int T = latt[Nd - 1];
  RealD V4 = 1.0;
  for (int mu = 0; mu < Nd; ++mu) V4 *= latt[mu];
  RealD V3 = V4 / RealD(T);
  const RealD lam4 = lambda * lambda * lambda * lambda;

  // --- Startup banner ---
  std::cout << GridLogMessage << "======== meas_aux_txqcd ========" << std::endl;
  std::cout << GridLogMessage << "  traj    = " << traj << std::endl;
  std::cout << GridLogMessage << "  lambda  = " << lambda
            << "    lambda^4 = " << lam4 << std::endl;
  std::cout << GridLogMessage << "  lattice = " << latt[0] << "x" << latt[1]
            << "x" << latt[2] << "x" << latt[3]
            << "    V4 = " << V4 << "    V3 = " << V3
            << "    T = " << T << std::endl;
  std::cout << GridLogMessage << "  Nf      = " << TxqcdNf
            << "    Nc = " << Nc << "    Nd = " << Nd << std::endl;
  std::cout << GridLogMessage << "  cfg dir = " << txqcd_cfg_dir() << std::endl;
  std::cout << GridLogMessage << "  out dir = " << txqcd_data_dir() << std::endl;
  std::cout << GridLogMessage << "================================" << std::endl;

  TXQCDField U(&Grid);
  TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                txqcd_cfg_dir() + "/ckpoint_lat",
                                txqcd_cfg_dir() + "/ckpoint_rng", traj);

  // =====================================================================
  // Channel computations.
  //
  // For each channel, we save 4 quantities:
  //   <ch>_C_raw     : λ⁴ · C(τ) (raw correlator, includes vacuum disconnected piece)
  //   <ch>_C_sub     : λ⁴ · (C(τ) - (1/V3)·vev_slice·vev_slice*)  (vac-subtracted)
  //   <ch>_vev_slice : (1/T)·Σ_t s[t]  ≡  V3·<O> (single complex scalar)
  //   <ch>_vev_per_site : vev_slice / V3  ≡  <O> (single complex scalar)
  //
  // The λ⁴ prefactor matches the original code's convention.  All time
  // dimensions are length T.
  // =====================================================================

  // ----- sigma channel (Tr_flavor σ) ---------------------------------------
  std::vector<ComplexD> sig_tr  = SliceSumFlavorTrace(U.sigma);
  std::vector<ComplexD> C_sig   = CorrelatorFromSlice(sig_tr, V4);
  ComplexD              vev_sig = SliceMean(sig_tr);
  std::vector<ComplexD> C_sig_sub = VacSubtractCorrelator(C_sig, vev_sig, V3);
  Scale(C_sig, lam4);
  Scale(C_sig_sub, lam4);

  // ----- pi iso-triplet channel (λ⁴ (Σ_ab C_{ab,ab} - C_TrTr)) -----------
  auto                  pi_all  = SliceSumPiAll(U.pi);
  std::vector<ComplexD> pi_tr   = SliceSumFlavorTrace(U.pi);
  std::vector<ComplexD> C_disc  = CorrelatorFromSlice(pi_tr, V4);
  ComplexD              vev_pi_tr = SliceMean(pi_tr);
  std::vector<ComplexD> C_disc_sub =
      VacSubtractCorrelator(C_disc, vev_pi_tr, V3);

  std::vector<ComplexD> C_total_raw(T, 0.0), C_total_sub(T, 0.0);
  std::vector<ComplexD> vev_pi_ab(TxqcdNf * TxqcdNf, 0.0);
  std::vector<std::vector<ComplexD>> C_pi_ab_raw(TxqcdNf * TxqcdNf);
  std::vector<std::vector<ComplexD>> C_pi_ab_sub(TxqcdNf * TxqcdNf);
  for (int ab = 0; ab < TxqcdNf * TxqcdNf; ++ab) {
    auto C_ab    = CorrelatorFromSlice(pi_all[ab], V4);
    vev_pi_ab[ab] = SliceMean(pi_all[ab]);
    auto C_ab_sub = VacSubtractCorrelator(C_ab, vev_pi_ab[ab], V3);
    for (int t = 0; t < T; ++t) {
      C_total_raw[t] += C_ab[t];
      C_total_sub[t] += C_ab_sub[t];
    }
    C_pi_ab_raw[ab] = std::move(C_ab);
    C_pi_ab_sub[ab] = std::move(C_ab_sub);
  }
  std::vector<ComplexD> C_pi_raw(T), C_pi_sub(T);
  for (int t = 0; t < T; ++t) {
    C_pi_raw[t] = lam4 * (C_total_raw[t] - C_disc[t]);
    C_pi_sub[t] = lam4 * (C_total_sub[t] - C_disc_sub[t]);
  }
  // Apply λ⁴ to per-flavor matrix correlators too, for consistency.
  for (int ab = 0; ab < TxqcdNf * TxqcdNf; ++ab) {
    Scale(C_pi_ab_raw[ab], lam4);
    Scale(C_pi_ab_sub[ab], lam4);
  }
  // Flatten per-flavor matrices to a single contiguous [Nf², T] block for h5.
  std::vector<ComplexD> C_pi_ab_raw_flat(TxqcdNf * TxqcdNf * T);
  std::vector<ComplexD> C_pi_ab_sub_flat(TxqcdNf * TxqcdNf * T);
  for (int ab = 0; ab < TxqcdNf * TxqcdNf; ++ab) {
    for (int t = 0; t < T; ++t) {
      C_pi_ab_raw_flat[ab * T + t] = C_pi_ab_raw[ab][t];
      C_pi_ab_sub_flat[ab * T + t] = C_pi_ab_sub[ab][t];
    }
  }

  // ----- s channel (Tr_color s) --------------------------------------------
  std::vector<ComplexD> s_tr   = SliceSumColorTrace(U.s);
  std::vector<ComplexD> C_s    = CorrelatorFromSlice(s_tr, V4);
  ComplexD              vev_s  = SliceMean(s_tr);
  std::vector<ComplexD> C_s_sub = VacSubtractCorrelator(C_s, vev_s, V3);
  Scale(C_s, lam4);
  Scale(C_s_sub, lam4);

  // ----- p channel (Tr_color p) --------------------------------------------
  std::vector<ComplexD> p_tr   = SliceSumColorTrace(U.p);
  std::vector<ComplexD> C_p    = CorrelatorFromSlice(p_tr, V4);
  ComplexD              vev_p  = SliceMean(p_tr);
  std::vector<ComplexD> C_p_sub = VacSubtractCorrelator(C_p, vev_p, V3);
  Scale(C_p, lam4);
  Scale(C_p_sub, lam4);

  // ----- t channels (Tr_color t_{μν}), 6 antisymmetric pairs ---------------
  auto t_all = SliceSumTAll(U.t);
  const int NtPairs = (int)t_all.size();
  std::vector<ComplexD> vev_t(NtPairs);
  std::vector<std::vector<ComplexD>> C_t_raw(NtPairs), C_t_sub(NtPairs);
  for (int p = 0; p < NtPairs; ++p) {
    C_t_raw[p] = CorrelatorFromSlice(t_all[p], V4);
    vev_t[p]   = SliceMean(t_all[p]);
    C_t_sub[p] = VacSubtractCorrelator(C_t_raw[p], vev_t[p], V3);
    Scale(C_t_raw[p], lam4);
    Scale(C_t_sub[p], lam4);
  }
  std::vector<ComplexD> C_t_raw_flat(NtPairs * T);
  std::vector<ComplexD> C_t_sub_flat(NtPairs * T);
  for (int p = 0; p < NtPairs; ++p) {
    for (int t = 0; t < T; ++t) {
      C_t_raw_flat[p * T + t] = C_t_raw[p][t];
      C_t_sub_flat[p * T + t] = C_t_sub[p][t];
    }
  }

  // ----- Write h5 ----------------------------------------------------------
  std::string outfile = txqcd_data_dir() + "/aux_txqcd_" + std::to_string(traj) + ".h5";
  if (Grid.IsBoss()) {
    Hdf5Writer wr(outfile);

    // π iso-triplet
    write(wr, "aux_pi_raw", C_pi_raw);
    write(wr, "aux_pi_sub", C_pi_sub);
    write(wr, "vev_pi_tr_slice", vev_pi_tr);

    // π per-flavor matrix (flat [Nf², T])
    write(wr, "aux_pi_ab_raw", C_pi_ab_raw_flat);
    write(wr, "aux_pi_ab_sub", C_pi_ab_sub_flat);
    write(wr, "vev_pi_ab_slice", vev_pi_ab);

    // σ (Tr_flavor)
    write(wr, "aux_sigma_raw", C_sig);
    write(wr, "aux_sigma_sub", C_sig_sub);
    write(wr, "vev_sigma_slice", vev_sig);

    // s (Tr_color)
    write(wr, "aux_s_raw", C_s);
    write(wr, "aux_s_sub", C_s_sub);
    write(wr, "vev_s_slice", vev_s);

    // p (Tr_color)
    write(wr, "aux_p_raw", C_p);
    write(wr, "aux_p_sub", C_p_sub);
    write(wr, "vev_p_slice", vev_p);

    // t (Tr_color, 6 antisymmetric μν pairs in row-major mu<nu order)
    write(wr, "aux_t_raw", C_t_raw_flat);
    write(wr, "aux_t_sub", C_t_sub_flat);
    write(wr, "vev_t_slice", vev_t);

    // metadata
    write(wr, "traj", traj);
    write(wr, "lambda", lambda);
    write(wr, "Nf", TxqcdNf);
    write(wr, "Nc", Nc);
    write(wr, "Nd", Nd);
    write(wr, "T", T);
    write(wr, "V3", V3);
    write(wr, "V4", V4);
  }

  std::cout << GridLogMessage << "Written " << outfile << std::endl;
  Grid_finalize();
  return 0;
}
