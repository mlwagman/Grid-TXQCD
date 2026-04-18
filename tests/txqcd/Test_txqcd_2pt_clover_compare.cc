// Step 5 (Clover): Compare TXQCD and QCD 2pt measurements (Fierz test).
// Same analysis as Wilson version but reads from clover measurement directory.

#include "Test_txqcd_2pt_clover_utils.h"

using namespace TxqcdTest2ptClover;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::string dir = meas_dir();
  int T = 0;

  auto pion_conn_tx = ReadMeasReal(dir + "/pion_conn_txqcd.dat", T);
  int T2;
  auto pion_conn_qcd = ReadMeasReal(dir + "/pion_conn_qcd.dat", T2);
  auto nucl_tx  = ReadMeasComplex(dir + "/nucleon_txqcd.dat", T2);
  auto nucl_qcd = ReadMeasComplex(dir + "/nucleon_qcd.dat", T2);
  auto loop_ud  = ReadMeasComplex(dir + "/loop_ud_txqcd.dat", T2);
  auto plaq_tx  = ReadMeasScalar(dir + "/plaq_txqcd.dat");
  auto plaq_qcd = ReadMeasScalar(dir + "/plaq_qcd.dat");
  auto vev_sig  = ReadMeasScalar(dir + "/vev_sigma_txqcd.dat");
  auto vev_s    = ReadMeasScalar(dir + "/vev_s_txqcd.dat");
  auto trminv_tx  = ReadMeasScalar(dir + "/vev_trminv_txqcd.dat");
  auto trminv_qcd = ReadMeasScalar(dir + "/vev_trminv_qcd.dat");
  auto aux_pi     = ReadMeasComplex(dir + "/aux_pi_txqcd.dat", T2);
  auto aux_sigma  = ReadMeasComplex(dir + "/aux_sigma_txqcd.dat", T2);
  auto aux_s      = ReadMeasComplex(dir + "/aux_s_txqcd.dat", T2);

  int N = (int)pion_conn_tx.size();
  int M = (int)pion_conn_qcd.size();

  Coordinate latt = default_latt();
  RealD V4 = 1.0;
  for (int mu = 0; mu < Nd; ++mu) V4 *= latt[mu];

  std::vector<std::vector<RealD>> pion_disc(N);
  for (int c = 0; c < N; ++c) {
    auto Cdisc = CorrelatorFromSlice(loop_ud[c], V4);
    pion_disc[c].resize(T);
    for (int t = 0; t < T; ++t) pion_disc[c][t] = Cdisc[t].real();
  }

  std::vector<std::vector<RealD>> pion_full(N);
  for (int c = 0; c < N; ++c) {
    pion_full[c].resize(T);
    for (int t = 0; t < T; ++t)
      pion_full[c][t] = pion_conn_tx[c][t] - pion_disc[c][t];
  }

  auto per_t_mean = [&](const std::vector<std::vector<RealD>> &X, int t) {
    std::vector<RealD> col;
    col.reserve(X.size());
    for (auto &row : X) col.push_back(row[t]);
    return std::make_pair(vmean(col), vstderr(col));
  };
  auto per_t_mean_c = [&](const std::vector<std::vector<ComplexD>> &X, int t) {
    std::vector<RealD> col;
    col.reserve(X.size());
    for (auto &row : X) col.push_back(row[t].real());
    return std::make_pair(vmean(col), vstderr(col));
  };

  int exitcode = 0;

  std::cout << GridLogMessage << "TXQCD clover samples: " << N
            << "   QCD clover samples: " << M << std::endl;

  std::cout << GridLogMessage
            << "----- Pion correlator (connected only, diagnostic) -----\n";
  std::cout << GridLogMessage
            << "t    TXQCD C_conn(t)          QCD C_pi(t)             nsigma\n";
  for (int t = 0; t < T; ++t) {
    auto [tm, te] = per_t_mean(pion_conn_tx, t);
    auto [qm, qe] = per_t_mean(pion_conn_qcd, t);
    RealD diff = tm - qm, derr = std::sqrt(te * te + qe * qe);
    RealD ns = (derr > 0) ? std::abs(diff) / derr : 0.0;
    std::cout << GridLogMessage << t << "    " << tm << " +/- " << te
              << "    " << qm << " +/- " << qe << "    " << ns << "\n";
  }

  std::cout << GridLogMessage << "----- Pion disconnected (diagnostic) -----\n";
  for (int t = 0; t < T; ++t) {
    auto [dm, de] = per_t_mean(pion_disc, t);
    std::cout << GridLogMessage << t << "    " << dm << " +/- " << de << "\n";
  }

  std::cout << GridLogMessage
            << "----- Pion correlator (conn - disc, Fierz test) -----\n";
  std::cout << GridLogMessage
            << "t    TXQCD C_full(t)          QCD C_pi(t)             nsigma\n";
  for (int t = 0; t < T; ++t) {
    auto [tm, te] = per_t_mean(pion_full, t);
    auto [qm, qe] = per_t_mean(pion_conn_qcd, t);
    RealD diff = tm - qm, derr = std::sqrt(te * te + qe * qe);
    RealD ns = (derr > 0) ? std::abs(diff) / derr : 0.0;
    bool pass = ns < 3.0;
    std::cout << GridLogMessage << t << "    " << tm << " +/- " << te
              << "    " << qm << " +/- " << qe << "    " << ns
              << (pass ? "  PASS" : "  FAIL") << "\n";
    if (!pass) exitcode = 1;
  }

  std::cout << GridLogMessage << "----- Proton correlator (Re, Fierz test) -----\n";
  std::cout << GridLogMessage
            << "t    TXQCD C_N(t)             QCD C_N(t)              nsigma\n";
  for (int t = 0; t < T; ++t) {
    auto [tm, te] = per_t_mean_c(nucl_tx, t);
    auto [qm, qe] = per_t_mean_c(nucl_qcd, t);
    RealD diff = tm - qm, derr = std::sqrt(te * te + qe * qe);
    RealD ns = (derr > 0) ? std::abs(diff) / derr : 0.0;
    bool pass = ns < 5.0;
    std::cout << GridLogMessage << t << "    " << tm << " +/- " << te
              << "    " << qm << " +/- " << qe << "    " << ns
              << (pass ? "  PASS" : "  FAIL") << "\n";
    if (!pass) exitcode = 1;
  }

  std::cout << GridLogMessage
            << "----- Aux pi I=1 (Tr[pi pi^dag] - Tr[pi]Tr[pi^dag]) -----\n";
  std::cout << GridLogMessage
            << "t    C_pi_aux(t)              C_pi_conn_quark(t)      nsigma\n";
  for (int t = 0; t < T; ++t) {
    auto [am, ae] = per_t_mean_c(aux_pi, t);
    auto [qm, qe] = per_t_mean(pion_conn_tx, t);
    RealD diff = am - qm, derr = std::sqrt(ae * ae + qe * qe);
    RealD ns = (derr > 0) ? std::abs(diff) / derr : 0.0;
    std::cout << GridLogMessage << t << "    " << am << " +/- " << ae
              << "    " << qm << " +/- " << qe << "    " << ns << "\n";
  }

  std::cout << GridLogMessage << "----- Aux sigma (Tr_f[sigma] Tr_f[sigma]*) -----\n";
  for (int t = 0; t < T; ++t) {
    auto [am, ae] = per_t_mean_c(aux_sigma, t);
    std::cout << GridLogMessage << t << "    " << am << " +/- " << ae << "\n";
  }

  std::cout << GridLogMessage << "----- Aux s (Tr_c[s] Tr_c[s]*) -----\n";
  for (int t = 0; t < T; ++t) {
    auto [am, ae] = per_t_mean_c(aux_s, t);
    std::cout << GridLogMessage << t << "    " << am << " +/- " << ae << "\n";
  }

  std::cout << GridLogMessage << "----- VEV checks -----\n";

  auto sm = [](const std::vector<RealD> &v) {
    return std::make_pair(vmean(v), vstderr(v));
  };

  auto [sig_m, sig_e]   = sm(vev_sig);
  auto [s_m, s_e]       = sm(vev_s);
  auto [mtx_m, mtx_e]   = sm(trminv_tx);
  auto [mqc_m, mqc_e]   = sm(trminv_qcd);

  const int Nf_tx = TxqcdNf;
  const RealD sig_pf = sig_m / Nf_tx,  sig_pf_e = sig_e / Nf_tx;
  const RealD mtx_pf = mtx_m / Nf_tx,  mtx_pf_e = mtx_e / Nf_tx;
  const RealD mqc_pf = mqc_m,           mqc_pf_e = mqc_e;

  std::cout << GridLogMessage << "<Tr sigma>/V = " << sig_m << " +/- " << sig_e
            << "  (per flavor: " << sig_pf << " +/- " << sig_pf_e << ")\n";
  std::cout << GridLogMessage << "<Tr s>/V     = " << s_m << " +/- " << s_e << "\n";
  std::cout << GridLogMessage << "<Re Tr M_TX^{-1}>/V = " << mtx_m << " +/- " << mtx_e
            << "  (per flavor: " << mtx_pf << " +/- " << mtx_pf_e << ")\n";
  std::cout << GridLogMessage << "<Re Tr M_WC^{-1}>/V = " << mqc_m << " +/- " << mqc_e
            << "\n";

  {
    RealD lhs = lambda * lambda * sig_m, lhs_e = lambda * lambda * sig_e;
    RealD rhs = mtx_m, rhs_e = mtx_e;
    RealD diff = lhs - rhs;
    RealD de = std::sqrt(lhs_e * lhs_e + rhs_e * rhs_e);
    RealD ns = (de > 0) ? std::abs(diff) / de : 0.0;
    bool pass = ns < 3.0;
    std::cout << GridLogMessage << "[SD] lam^2 <Tr sigma>/V = " << lhs << " +/- "
              << lhs_e << "  vs  <Tr M_TX^{-1}>/V = " << rhs << " +/- " << rhs_e
              << "  (" << ns << " sigma)" << (pass ? "  PASS" : "  FAIL") << "\n";
    if (!pass) exitcode = 1;
  }

  {
    RealD diff = mtx_pf - mqc_pf;
    RealD de = std::sqrt(mtx_pf_e * mtx_pf_e + mqc_pf_e * mqc_pf_e);
    RealD ns = (de > 0) ? std::abs(diff) / de : 0.0;
    bool pass = ns < 3.0;
    std::cout << GridLogMessage << "[Fierz VEV] TXQCD/Nf = " << mtx_pf << " +/- "
              << mtx_pf_e << "  vs QCD = " << mqc_pf << " +/- " << mqc_pf_e
              << "  (" << ns << " sigma)" << (pass ? "  PASS" : "  FAIL") << "\n";
    if (!pass) exitcode = 1;
  }

  {
    RealD lhs = lambda * lambda * sig_pf, lhs_e = lambda * lambda * sig_pf_e;
    RealD diff = lhs - mqc_pf;
    RealD de = std::sqrt(lhs_e * lhs_e + mqc_pf_e * mqc_pf_e);
    RealD ns = (de > 0) ? std::abs(diff) / de : 0.0;
    bool pass = ns < 3.0;
    std::cout << GridLogMessage << "[aux vs QCD] lam^2 <Tr sigma>/(V Nf) = " << lhs
              << " +/- " << lhs_e << "  vs QCD = " << mqc_pf << " +/- " << mqc_pf_e
              << "  (" << ns << " sigma)" << (pass ? "  PASS" : "  FAIL") << "\n";
    if (!pass) exitcode = 1;
  }

  {
    RealD ratio = sig_m / s_m;
    RealD ratio_e = std::abs(ratio) * std::sqrt(
        (sig_e / sig_m) * (sig_e / sig_m) + (s_e / s_m) * (s_e / s_m));
    RealD expected = std::sqrt(2.0);
    RealD ns = std::abs(ratio - expected) / ratio_e;
    bool pass = ns < 3.0;
    std::cout << GridLogMessage << "[ratio] <Tr sigma>/<Tr s> = " << ratio
              << " +/- " << ratio_e << "  expected " << expected
              << "  (" << ns << " sigma)" << (pass ? "  PASS" : "  FAIL") << "\n";
    if (!pass) exitcode = 1;
  }

  {
    auto [ptx_m, ptx_e] = sm(plaq_tx);
    auto [pqc_m, pqc_e] = sm(plaq_qcd);
    RealD diff = ptx_m - pqc_m;
    RealD de = std::sqrt(ptx_e * ptx_e + pqc_e * pqc_e);
    RealD ns = (de > 0) ? std::abs(diff) / de : 0.0;
    bool pass = ns < 3.0;
    std::cout << GridLogMessage << "[plaq] TXQCD = " << ptx_m << " +/- " << ptx_e
              << "  vs QCD = " << pqc_m << " +/- " << pqc_e
              << "  (" << ns << " sigma)" << (pass ? "  PASS" : "  FAIL") << "\n";
    if (!pass) exitcode = 1;
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CLOVER 2pt CHECKS FAILED"
                         : "ALL CLOVER 2pt CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
