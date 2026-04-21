#include "params.h"

using namespace TXQCDProduction;

static RealD vmean(const std::vector<RealD> &v) {
  RealD s = 0;
  for (auto x : v) s += x;
  return s / v.size();
}
static RealD vstderr(const std::vector<RealD> &v) {
  RealD m = vmean(v), s2 = 0;
  for (auto x : v) s2 += (x - m) * (x - m);
  return std::sqrt(s2 / (v.size() * (v.size() - 1)));
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

  Coordinate latt = lattice_size();
  int T = latt[Nd - 1];
  RealD V4 = 1.0;
  for (int mu = 0; mu < Nd; ++mu) V4 *= latt[mu];

  auto trajs = meas_trajs();
  std::string tx_dir  = txqcd_data_dir();
  std::string qcd_dir = qcd_data_dir();

  // Collect per-cfg data
  std::vector<std::vector<RealD>>    pion_conn_tx, pion_conn_qcd;
  std::vector<std::vector<ComplexD>> nucl_tx, nucl_qcd;
  std::vector<std::vector<ComplexD>> loop_ud_all;
  std::vector<RealD> trminv_tx, trminv_qcd;
  std::vector<RealD> trminv_strange_tx, trminv_strange_qcd;
  std::vector<RealD> vev_sigma_all, vev_s_all;
  std::vector<std::vector<ComplexD>> aux_pi_all, aux_sigma_all, aux_s_all;

  int n_tx = 0, n_qcd = 0;
  for (int traj : trajs) {
    std::string conn_tx = tx_dir + "/conn_txqcd_" + std::to_string(traj) + ".h5";
    std::string conn_qcd_f = qcd_dir + "/conn_qcd_" + std::to_string(traj) + ".h5";
    std::string disc_tx = tx_dir + "/disco_txqcd_" + std::to_string(traj) + ".h5";
    std::string disc_qcd_f = qcd_dir + "/disco_qcd_" + std::to_string(traj) + ".h5";
    std::string aux_f = tx_dir + "/aux_txqcd_" + std::to_string(traj) + ".h5";

    if (file_exists(conn_tx)) {
      Hdf5Reader rd(conn_tx);
      std::vector<RealD> pion;
      std::vector<ComplexD> nucl;
      read(rd, "pion_conn", pion);
      read(rd, "nucleon", nucl);
      pion_conn_tx.push_back(pion);
      nucl_tx.push_back(nucl);
    }
    if (file_exists(conn_qcd_f)) {
      Hdf5Reader rd(conn_qcd_f);
      std::vector<RealD> pion;
      std::vector<ComplexD> nucl;
      read(rd, "pion_conn", pion);
      read(rd, "nucleon", nucl);
      pion_conn_qcd.push_back(pion);
      nucl_qcd.push_back(nucl);
    }
    if (file_exists(disc_tx)) {
      Hdf5Reader rd(disc_tx);
      std::vector<ComplexD> loop;
      RealD tm, tms, vs, vsig;
      read(rd, "loop_ud", loop);
      read(rd, "trminv", tm);
      read(rd, "trminv_strange", tms);
      read(rd, "vev_sigma", vsig);
      read(rd, "vev_s", vs);
      loop_ud_all.push_back(loop);
      trminv_tx.push_back(tm);
      trminv_strange_tx.push_back(tms);
      vev_sigma_all.push_back(vsig);
      vev_s_all.push_back(vs);
      n_tx++;
    }
    if (file_exists(disc_qcd_f)) {
      Hdf5Reader rd(disc_qcd_f);
      RealD tm, tms;
      read(rd, "trminv", tm);
      read(rd, "trminv_strange", tms);
      trminv_qcd.push_back(tm);
      trminv_strange_qcd.push_back(tms);
      n_qcd++;
    }
    if (file_exists(aux_f)) {
      Hdf5Reader rd(aux_f);
      std::vector<ComplexD> ap, asig, as;
      read(rd, "aux_pi", ap);
      read(rd, "aux_sigma", asig);
      read(rd, "aux_s", as);
      aux_pi_all.push_back(ap);
      aux_sigma_all.push_back(asig);
      aux_s_all.push_back(as);
    }
  }

  int N = (int)pion_conn_tx.size();
  int M = (int)pion_conn_qcd.size();

  std::cout << GridLogMessage << "TXQCD samples: conn=" << N << " disco=" << n_tx
            << "   QCD samples: conn=" << M << " disco=" << n_qcd << std::endl;

  if (N == 0 || M == 0) {
    std::cout << GridLogMessage << "Insufficient data for comparison." << std::endl;
    Grid_finalize();
    return 1;
  }

  // Build full pion (conn - disc) for TXQCD
  int Ndisc = (int)loop_ud_all.size();
  int Nfull = std::min(N, Ndisc);
  std::vector<std::vector<RealD>> pion_full(Nfull);
  for (int c = 0; c < Nfull; ++c) {
    auto Cdisc = CorrelatorFromSlice(loop_ud_all[c], V4);
    pion_full[c].resize(T);
    for (int t = 0; t < T; ++t)
      pion_full[c][t] = pion_conn_tx[c][t] - Cdisc[t].real();
  }

  auto per_t_mean = [&](const std::vector<std::vector<RealD>> &X, int t) {
    std::vector<RealD> col;
    for (auto &row : X) col.push_back(row[t]);
    return std::make_pair(vmean(col), vstderr(col));
  };
  auto per_t_mean_c = [&](const std::vector<std::vector<ComplexD>> &X, int t) {
    std::vector<RealD> col;
    for (auto &row : X) col.push_back(row[t].real());
    return std::make_pair(vmean(col), vstderr(col));
  };

  int exitcode = 0;

  // Pion Fierz test (conn - disc)
  std::cout << GridLogMessage
            << "----- Pion correlator (conn - disc, Fierz test) -----\n";
  for (int t = 0; t < T; ++t) {
    auto [tm, te] = per_t_mean(pion_full, t);
    auto [qm, qe] = per_t_mean(pion_conn_qcd, t);
    RealD diff = tm - qm, derr = std::sqrt(te * te + qe * qe);
    RealD ns = (derr > 0) ? std::abs(diff) / derr : 0.0;
    bool pass = ns < 3.0;
    std::cout << GridLogMessage << "t=" << t << "  TX=" << tm << "+-" << te
              << "  QCD=" << qm << "+-" << qe << "  " << ns << "sig"
              << (pass ? "" : "  FAIL") << "\n";
    if (!pass) exitcode = 1;
  }

  // Nucleon Fierz test
  std::cout << GridLogMessage << "----- Nucleon correlator (Fierz test) -----\n";
  for (int t = 0; t < T; ++t) {
    auto [tm, te] = per_t_mean_c(nucl_tx, t);
    auto [qm, qe] = per_t_mean_c(nucl_qcd, t);
    RealD diff = tm - qm, derr = std::sqrt(te * te + qe * qe);
    RealD ns = (derr > 0) ? std::abs(diff) / derr : 0.0;
    bool pass = ns < 5.0;
    std::cout << GridLogMessage << "t=" << t << "  TX=" << tm << "+-" << te
              << "  QCD=" << qm << "+-" << qe << "  " << ns << "sig"
              << (pass ? "" : "  FAIL") << "\n";
    if (!pass) exitcode = 1;
  }

  // VEV checks
  std::cout << GridLogMessage << "----- VEV checks -----\n";
  if (!trminv_tx.empty() && !trminv_qcd.empty()) {
    auto sm = [](const std::vector<RealD> &v) {
      return std::make_pair(vmean(v), vstderr(v));
    };

    auto [sig_m, sig_e] = sm(vev_sigma_all);
    auto [s_m, s_e] = sm(vev_s_all);
    auto [mtx_m, mtx_e] = sm(trminv_tx);
    auto [mqc_m, mqc_e] = sm(trminv_qcd);
    auto [mstx_m, mstx_e] = sm(trminv_strange_tx);
    auto [msqc_m, msqc_e] = sm(trminv_strange_qcd);

    RealD mtx_pf = mtx_m / TxqcdNf, mtx_pf_e = mtx_e / TxqcdNf;
    RealD mqc_pf = mqc_m, mqc_pf_e = mqc_e;

    // SD identity: lambda^2 <Tr sigma>/V = <Tr M_TX^{-1}>/V
    {
      RealD lhs = lambda * lambda * sig_m, lhs_e = lambda * lambda * sig_e;
      RealD diff = lhs - mtx_m;
      RealD de = std::sqrt(lhs_e * lhs_e + mtx_e * mtx_e);
      RealD ns = (de > 0) ? std::abs(diff) / de : 0.0;
      bool pass = ns < 3.0;
      std::cout << GridLogMessage << "[SD] lam^2<sigma>=" << lhs << "+-" << lhs_e
                << "  <TrM^-1>=" << mtx_m << "+-" << mtx_e << "  " << ns << "sig"
                << (pass ? "  PASS" : "  FAIL") << "\n";
      if (!pass) exitcode = 1;
    }

    // Fierz VEV: TXQCD/Nf vs QCD
    {
      RealD diff = mtx_pf - mqc_pf;
      RealD de = std::sqrt(mtx_pf_e * mtx_pf_e + mqc_pf_e * mqc_pf_e);
      RealD ns = (de > 0) ? std::abs(diff) / de : 0.0;
      bool pass = ns < 3.0;
      std::cout << GridLogMessage << "[Fierz VEV] TX/Nf=" << mtx_pf << "+-" << mtx_pf_e
                << "  QCD=" << mqc_pf << "+-" << mqc_pf_e << "  " << ns << "sig"
                << (pass ? "  PASS" : "  FAIL") << "\n";
      if (!pass) exitcode = 1;
    }

    // Strange quark Fierz
    {
      RealD diff = mstx_m - msqc_m;
      RealD de = std::sqrt(mstx_e * mstx_e + msqc_e * msqc_e);
      RealD ns = (de > 0) ? std::abs(diff) / de : 0.0;
      bool pass = ns < 3.0;
      std::cout << GridLogMessage << "[Fierz strange] TX=" << mstx_m << "+-" << mstx_e
                << "  QCD=" << msqc_m << "+-" << msqc_e << "  " << ns << "sig"
                << (pass ? "  PASS" : "  FAIL") << "\n";
      if (!pass) exitcode = 1;
    }

    // sigma/s ratio = sqrt(2)
    {
      RealD ratio = sig_m / s_m;
      RealD ratio_e = std::abs(ratio) * std::sqrt(
          (sig_e / sig_m) * (sig_e / sig_m) + (s_e / s_m) * (s_e / s_m));
      RealD expected = std::sqrt(2.0);
      RealD ns = std::abs(ratio - expected) / ratio_e;
      bool pass = ns < 3.0;
      std::cout << GridLogMessage << "[ratio] <sigma>/<s>=" << ratio << "+-" << ratio_e
                << "  expect=" << expected << "  " << ns << "sig"
                << (pass ? "  PASS" : "  FAIL") << "\n";
      if (!pass) exitcode = 1;
    }
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED") << std::endl;
  Grid_finalize();
  return exitcode;
}
