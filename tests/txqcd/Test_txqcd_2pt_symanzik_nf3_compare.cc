// Symanzik Nf=3 Fierz cross-check between TXQCD-Nf=3 (single operator with
// diag mass m={m_l, m_l, m_s}) and QCD Nf=2+1 reference.
// Compile with -DTXQCD_Nf=3.
//
// Checks:
//   1. Connected pion correlator: TXQCD vs QCD (Fierz; flavor-non-singlet,
//      no disc subtraction needed since both light quarks are sourced via
//      the same flavor block of the Nf=3 propagator).
//   2. Connected kaon correlator: TXQCD vs QCD.
//   3. Light Tr M^-1: TXQCD per-flavor light VEV vs QCD <Tr M_l^-1>.
//   4. Strange Tr M^-1: TXQCD flavor-2 VEV vs QCD <Tr M_s^-1>.
//   5. Aux Schwinger-Dyson identity: λ² <Tr σ>/(V·Nf) vs TXQCD light VEV
//      (TXQCD-internal consistency).
//   6. Ratio <Tr σ>/<Tr s> = √2 (TXQCD-internal).

#ifndef TXQCD_Nf
#error "Compile with -DTXQCD_Nf=3"
#endif
static_assert(TXQCD_Nf == 3, "expects TXQCD_Nf=3");

#include "Test_txqcd_2pt_symanzik_nf3_utils.h"

using namespace TxqcdTest2ptSymanzikNf3;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::string dir = meas_dir();

  std::vector<std::vector<RealD>> pion_tx, kaon_tx, pion_qcd, kaon_qcd;
  std::vector<RealD> trminv_l_tx, trminv_s_tx, trminv_l_qcd, trminv_s_qcd;
  std::vector<RealD> vev_sigma, vev_s;
  std::vector<std::vector<ComplexD>> aux_pi, aux_sigma, aux_s;

  {
    Hdf5Reader rd(dir + "/meas_txqcd_nf3_conn.h5");
    read(rd, "pion", pion_tx);
    read(rd, "kaon", kaon_tx);
  }
  {
    Hdf5Reader rd(dir + "/meas_qcd_conn.h5");
    read(rd, "pion", pion_qcd);
    read(rd, "kaon", kaon_qcd);
  }
  {
    Hdf5Reader rd(dir + "/meas_txqcd_nf3_disc.h5");
    read(rd, "vev_sigma", vev_sigma);
    read(rd, "vev_s",     vev_s);
    read(rd, "trminv_l",  trminv_l_tx);
    read(rd, "trminv_s",  trminv_s_tx);
  }
  {
    Hdf5Reader rd(dir + "/meas_qcd_disc.h5");
    read(rd, "trminv_l", trminv_l_qcd);
    read(rd, "trminv_s", trminv_s_qcd);
  }
  {
    Hdf5Reader rd(dir + "/meas_txqcd_nf3_aux.h5");
    read(rd, "aux_pi",    aux_pi);
    read(rd, "aux_sigma", aux_sigma);
    read(rd, "aux_s",     aux_s);
  }

  int T = (int)pion_tx[0].size();
  int N = (int)pion_tx.size();
  int M = (int)pion_qcd.size();
  std::cout << GridLogMessage
            << "TXQCD-Nf=3 cfgs: " << N << "   QCD-Nf=2+1 cfgs: " << M
            << std::endl;

  auto sm = [](const std::vector<RealD> &v) {
    return std::make_pair(vmean(v), vstderr(v));
  };
  auto per_t_real = [&](const std::vector<std::vector<RealD>> &X, int t) {
    std::vector<RealD> col;
    col.reserve(X.size());
    for (auto &row : X) col.push_back(row[t]);
    return sm(col);
  };
  auto per_t_complex = [&](const std::vector<std::vector<ComplexD>> &X, int t) {
    std::vector<RealD> col;
    col.reserve(X.size());
    for (auto &row : X) col.push_back(row[t].real());
    return sm(col);
  };

  int exitcode = 0;
  auto check = [&](const std::string &label, RealD a, RealD ae, RealD b,
                   RealD be, RealD threshold = 3.0) {
    RealD diff = a - b;
    RealD de = std::sqrt(ae * ae + be * be);
    RealD ns = (de > 0) ? std::abs(diff) / de : 0.0;
    bool pass = ns < threshold;
    std::cout << GridLogMessage << "[" << label << "] TXQCD=" << a << " +/- "
              << ae << "  QCD=" << b << " +/- " << be
              << "  (" << ns << " sigma)"
              << (pass ? "  PASS" : "  FAIL") << "\n";
    if (!pass) exitcode = 1;
  };

  std::cout << GridLogMessage
            << "----- Pion correlator (Fierz: TXQCD-Nf=3 vs QCD-Nf=2+1) -----\n";
  for (int t = 0; t < T; ++t) {
    auto [tm, te] = per_t_real(pion_tx, t);
    auto [qm, qe] = per_t_real(pion_qcd, t);
    check("pion t=" + std::to_string(t), tm, te, qm, qe);
  }

  std::cout << GridLogMessage
            << "----- Kaon correlator (Fierz: TXQCD-Nf=3 vs QCD-Nf=2+1) -----\n";
  for (int t = 0; t < T; ++t) {
    auto [tm, te] = per_t_real(kaon_tx, t);
    auto [qm, qe] = per_t_real(kaon_qcd, t);
    check("kaon t=" + std::to_string(t), tm, te, qm, qe);
  }

  std::cout << GridLogMessage << "----- VEVs -----\n";

  auto [trl_tx_m, trl_tx_e]     = sm(trminv_l_tx);
  auto [trs_tx_m, trs_tx_e]     = sm(trminv_s_tx);
  auto [trl_qcd_m, trl_qcd_e]   = sm(trminv_l_qcd);
  auto [trs_qcd_m, trs_qcd_e]   = sm(trminv_s_qcd);
  auto [vev_sig_m, vev_sig_e]   = sm(vev_sigma);
  auto [vev_s_m,   vev_s_e]     = sm(vev_s);

  // (3) Light Tr M^-1
  check("Fierz light VEV", trl_tx_m, trl_tx_e, trl_qcd_m, trl_qcd_e);
  // (4) Strange Tr M^-1
  check("Fierz strange VEV", trs_tx_m, trs_tx_e, trs_qcd_m, trs_qcd_e);

  // (5) TXQCD Schwinger-Dyson identity for light: λ² <σ_ll>/V = -<l̄l>.
  //     Per-flavor light σ vev = <Tr σ>/(V·Nf) for the 2 light flavors plus
  //     1 strange — for diag (m_l, m_l, m_s) the Tr σ averages all three so
  //     this is an approximate per-flavor comparison.  For the production
  //     analysis we'd extract σ_00 and σ_22 separately; here we just check
  //     the trace-averaged identity is consistent with the average light VEV.
  {
    const int Nf = TxqcdNf;
    RealD sig_pf_m = vev_sig_m / Nf;
    RealD sig_pf_e = vev_sig_e / Nf;
    RealD lhs = (lambda_runtime() * lambda_runtime()) * sig_pf_m;
    RealD lhs_e = (lambda_runtime() * lambda_runtime()) * sig_pf_e;
    // RHS = avg-over-flavors of <Tr M^-1>: (2*light + strange)/3.
    RealD rhs = (2.0 * trl_tx_m + trs_tx_m) / 3.0;
    RealD rhs_e = std::sqrt(4.0 * trl_tx_e * trl_tx_e + trs_tx_e * trs_tx_e) / 3.0;
    check("SD lam^2 <Tr sigma>/(V Nf) ~= avg <Tr M^-1>", lhs, lhs_e,
          rhs, rhs_e);
  }

  // (6) <Tr sigma> / <Tr s> = sqrt(2) for the lambda-tuned aux measure.
  {
    RealD ratio = vev_sig_m / vev_s_m;
    RealD ratio_e = std::abs(ratio) * std::sqrt(
        std::pow(vev_sig_e / vev_sig_m, 2) +
        std::pow(vev_s_e / vev_s_m, 2));
    RealD expected = std::sqrt(2.0);
    RealD ns = std::abs(ratio - expected) / ratio_e;
    bool pass = ns < 3.0;
    std::cout << GridLogMessage << "[ratio] <Tr sigma>/<Tr s> = " << ratio
              << " +/- " << ratio_e << "  expected " << expected
              << "  (" << ns << " sigma)" << (pass ? "  PASS" : "  FAIL")
              << std::endl;
    if (!pass) exitcode = 1;
  }

  // Aux pi vs connected pion (TXQCD-internal Fierz at the aux level).
  std::cout << GridLogMessage
            << "----- Aux pi correlator (TXQCD-internal Fierz) -----\n";
  for (int t = 0; t < T; ++t) {
    auto [am, ae] = per_t_complex(aux_pi, t);
    auto [pm, pe] = per_t_real(pion_tx, t);
    RealD diff = am - pm;
    RealD de = std::sqrt(ae * ae + pe * pe);
    RealD ns = (de > 0) ? std::abs(diff) / de : 0.0;
    std::cout << GridLogMessage << "  t=" << t << "  C_pi_aux=" << am
              << " +/- " << ae << "  C_pi_quark=" << pm << " +/- " << pe
              << "  (" << ns << " sigma)\n";
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME SYMANZIK Nf=3 CHECKS FAILED"
                         : "ALL SYMANZIK Nf=3 CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
