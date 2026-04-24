// Step 5: Compare TXQCD and QCD 2pt measurements (Fierz test).
//
// Reads measurement files from meas_2pt/ produced by steps 2-4 and performs:
//   - Full pion (conn - disc) TXQCD vs QCD: Fierz test
//   - Nucleon TXQCD vs QCD: Fierz test
//   - SD identity: lambda^2 <Tr sigma>/V = <Re Tr M_TX^{-1}>/V
//   - <Tr sigma>/<Tr s> = sqrt(2)
//   - Per-flavor VEV Fierz: TXQCD vs QCD
//   - Auxiliary correlator diagnostics: C_pi_aux, C_sigma, C_s

#include "Test_txqcd_2pt_optlam_utils.h"
#include <Grid/serialisation/Hdf5IO.h>

using namespace TxqcdTest2ptOptlam;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::string dir = meas_dir();

  std::vector<std::vector<RealD>>    pion_conn_tx, pion_conn_qcd;
  std::vector<std::vector<ComplexD>> nucl_tx, nucl_qcd, loop_ud;
  std::vector<RealD> plaq_tx, plaq_qcd, vev_sig, vev_s;
  std::vector<RealD> trminv_tx, trminv_qcd;
  std::vector<std::vector<ComplexD>> aux_pi, aux_sigma, aux_s;

  {
    Hdf5Reader rd(dir + "/meas_txqcd_conn.h5");
    read(rd, "pion_conn", pion_conn_tx);
    read(rd, "nucleon", nucl_tx);
    read(rd, "plaq", plaq_tx);
  }
  {
    Hdf5Reader rd(dir + "/meas_qcd_conn.h5");
    read(rd, "pion_conn", pion_conn_qcd);
    read(rd, "nucleon", nucl_qcd);
    read(rd, "plaq", plaq_qcd);
  }
  {
    Hdf5Reader rd(dir + "/meas_txqcd_disc.h5");
    read(rd, "loop_ud", loop_ud);
    read(rd, "vev_sigma", vev_sig);
    read(rd, "vev_s", vev_s);
    read(rd, "trminv", trminv_tx);
  }
  {
    Hdf5Reader rd(dir + "/meas_qcd_disc.h5");
    read(rd, "trminv", trminv_qcd);
  }
  {
    Hdf5Reader rd(dir + "/meas_txqcd_aux.h5");
    read(rd, "aux_pi", aux_pi);
    read(rd, "aux_sigma", aux_sigma);
    read(rd, "aux_s", aux_s);
  }

  int T = (int)pion_conn_tx[0].size();

  int N = (int)pion_conn_tx.size();
  int M = (int)pion_conn_qcd.size();

  // Build volume for disc correlator
  Coordinate latt = default_latt();
  RealD V4 = 1.0;
  for (int mu = 0; mu < Nd; ++mu) V4 *= latt[mu];

  // Disconnected pion per config: Disc(dt) = (1/V) sum_{t0} L_ud(t0+dt) conj(L_ud(t0))
  std::vector<std::vector<RealD>> pion_disc(N);
  for (int c = 0; c < N; ++c) {
    auto Cdisc = CorrelatorFromSlice(loop_ud[c], V4);
    pion_disc[c].resize(T);
    for (int t = 0; t < T; ++t) pion_disc[c][t] = Cdisc[t].real();
  }

  // Full pion = conn - disc
  std::vector<std::vector<RealD>> pion_full(N);
  for (int c = 0; c < N; ++c) {
    pion_full[c].resize(T);
    for (int t = 0; t < T; ++t)
      pion_full[c][t] = pion_conn_tx[c][t] - pion_disc[c][t];
  }

  // Stats helpers
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

  std::cout << GridLogMessage << "TXQCD samples: " << N
            << "   QCD samples: " << M << std::endl;

  // ---- Connected pion (diagnostic) ----
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

  // ---- Disconnected pion (diagnostic) ----
  std::cout << GridLogMessage << "----- Pion disconnected (diagnostic) -----\n";
  for (int t = 0; t < T; ++t) {
    auto [dm, de] = per_t_mean(pion_disc, t);
    std::cout << GridLogMessage << t << "    " << dm << " +/- " << de << "\n";
  }

  // ---- Full pion Fierz test ----
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

  // ---- Proton Fierz test ----
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

  // ---- Aux correlator diagnostics ----
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

  // ---- VEV checks ----
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
  std::cout << GridLogMessage << "<Re Tr M_W^{-1}>/V  = " << mqc_m << " +/- " << mqc_e
            << "\n";

  // SD: lambda^2 <Tr sigma>/V = <Re Tr M_TX^{-1}>/V
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

  // Fierz per-flavor: <Re Tr M_TX^{-1}>/(V Nf_tx) vs QCD
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

  // Chained: lambda^2 <Tr sigma>/(V Nf_tx) vs QCD
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

  // Ratio: <Tr sigma>/<Tr s> = sqrt(2)
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

  // Plaquette comparison: TXQCD vs QCD (Fierz test)
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
            << (exitcode ? "SOME 2pt CHECKS FAILED" : "ALL 2pt CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
