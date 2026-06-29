// Test_dtxqcd_multishift_quda_styleC: 4⁴ bit-equivalence between Stage B
// (DTXQCDMultiShiftCGQUDA_StageB on raw cudaMalloc'd device buffers + Option α
// host aux, the baseline already validated vs Stage A) and Style C
// (DTXQCDMultiShiftCGQUDA_StyleC on native ColorSpinorField + quda::blas inner
// loop + native FloatNOrder kernels for aux/Pre/Post/Gamma5).
//
// Per-shift gates:
//   ||(M†M + poles_s)·psi_C[s] − b|| / ||b||  ≤ 1e-9  (residual)
//   ||psi_C[s] − psi_B[s]|| / ||psi_B[s]||    ≤ 1e-9  (bit-equiv)
//
// Run:
//   QUDA_RESOURCE_PATH=/lustre2/nplqcd/cache/quda_resource_cuda12p2 \
//     ./Test_dtxqcd_multishift_quda_styleC --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCG.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGQUDA.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGQUDA_StageB.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGQUDA_StyleC.h>
#include <Grid/util/QudaInit.h>
#include <iomanip>
#include <iostream>

using namespace Grid;

namespace {
RealD rel_diff(const DTXQCDFermionDoubled &a, const DTXQCDFermionDoubled &b) {
  RealD num = 0.0, den = 0.0;
  for (int aa = 0; aa < DtxqcdNf; ++aa) {
    LatticeFermion du(a.upper.f[aa].Grid()), dl(a.lower.f[aa].Grid());
    du = a.upper.f[aa] - b.upper.f[aa];
    dl = a.lower.f[aa] - b.lower.f[aa];
    num += norm2(du) + norm2(dl);
    den += norm2(b.upper.f[aa]) + norm2(b.lower.f[aa]);
  }
  if (den == 0.0) return 0.0;
  return std::sqrt(num / den);
}

void randomize_doubled(GridParallelRNG &pRNG, DTXQCDFermionDoubled &v) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, v.upper.f[a]);
    gaussian(pRNG, v.lower.f[a]);
  }
}

template <class DTXQCDMop>
RealD shifted_residual(DTXQCDMop &Mop, RealD pole,
                       const DTXQCDFermionDoubled &psi,
                       const DTXQCDFermionDoubled &b) {
  GridBase *grid = b.Grid();
  DTXQCDFermionDoubled tmp(grid), mmp(grid), r(grid);
  Mop.M(psi, tmp);
  Mop.Mdag(tmp, mmp);
  for (int a = 0; a < DtxqcdNf; ++a) {
    mmp.upper.f[a] = mmp.upper.f[a] + pole * psi.upper.f[a];
    mmp.lower.f[a] = mmp.lower.f[a] + pole * psi.lower.f[a];
    r.upper.f[a]   = b.upper.f[a]   - mmp.upper.f[a];
    r.lower.f[a]   = b.lower.f[a]   - mmp.lower.f[a];
  }
  RealD num = norm2(r);
  RealD den = norm2(b);
  if (den == 0.0) return 0.0;
  return std::sqrt(num / den);
}
}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Grid::Quda::initialize();

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--grid" && i + 1 < argc) {
      std::vector<int> d;
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, '.')) d.push_back(std::stoi(tok));
      if (d.size() == (size_t)Nd) latt = Coordinate(d);
    }
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({601, 602, 603, 604});

  RealD mass = -0.245;
  RealD csw  = 1.24930970916466;
  RealD aux_scale = 1.0 / 3.0;

  LatticeGaugeField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  LatticeDtxqcdSigma sigma(&Grid);
  LatticeDtxqcdPi    pi(&Grid);
  LatticeDtxqcdD     d_field(&Grid);
  LatticeDtxqcdN     n_field(&Grid);
  LatticeDtxqcdS     s_field(&Grid);
  LatticeDtxqcdP     p_field(&Grid);

  DtxqcdHermitianCFGaussian(pRNG, sigma);
  DtxqcdHermitianCFGaussian(pRNG, pi);
  DtxqcdRealScalarGaussian(pRNG, s_field);
  DtxqcdRealScalarGaussian(pRNG, p_field);
  DtxqcdHermitianCFGaussian(pRNG, d_field);
  DtxqcdHermitianCFGaussian(pRNG, n_field);
  if (DtxqcdDnComplexSymmetric()) {
    DtxqcdRealSymmetricCFInPlace(sigma);
    DtxqcdRealSymmetricCFInPlace(pi);
    DtxqcdComplexSymmetricCFGaussian(pRNG, d_field);
    DtxqcdComplexSymmetricCFGaussian(pRNG, n_field);
  }
  sigma   = aux_scale * sigma;
  pi      = aux_scale * pi;
  s_field = aux_scale * s_field;
  p_field = aux_scale * p_field;
  d_field = aux_scale * d_field;
  n_field = aux_scale * n_field;

  DTXQCDWilsonCloverFermionEO Dw(U, Grid, RBGrid, mass, csw,
                                  sigma, pi, d_field, n_field, s_field, p_field);
  DTXQCDMOp Mop_grid(Dw);
  DTXQCDMpcOpQUDA Mop_quda(U, Grid, RBGrid, mass, csw,
                            sigma, pi, d_field, n_field, s_field, p_field);

  std::vector<RealD> poles = {0.01, 0.05, 0.1, 0.25, 0.5};
  std::vector<RealD> tol  (poles.size(), 1.0e-10);
  const int MaxIter = 3000;

  DTXQCDFermionDoubled b(&Grid);
  randomize_doubled(pRNG, b);

  // ============================================================
  // Pre-CG sanity: probe HermOp via Stage B's HermOp wrapper and via
  // (M_device_csf, Mdag_device_csf) on the same b.  This isolates whether
  // the divergence is in HermOp itself vs in the CG loop logic.
  // ============================================================
  {
    namespace SB = DtxqcdQudaStageB;
    namespace SC = DtxqcdQudaStyleC;
    std::size_t V_pr = Quda::local_volume(&Grid);
    std::size_t Nbuf_pr = SB::buf_doubles(V_pr);

    // Stage B path: pack b → device, call HermOp wrapper.
    SB::DoubledStateBuf p_sb, mmp_sb, tmp_sb, scratch_lo_sb;
    SB::allocate_state(p_sb, Nbuf_pr);
    SB::allocate_state(mmp_sb, Nbuf_pr);
    SB::allocate_state(tmp_sb, Nbuf_pr);
    SB::allocate_lower_only_state(scratch_lo_sb, Nbuf_pr);
    SB::pack_grid_to_state(b, p_sb, V_pr);
    RealD d_sb = DtxqcdHermOpQUDA_StageB(Mop_quda, p_sb, mmp_sb, tmp_sb,
                                         scratch_lo_sb, V_pr, Nbuf_pr, &Grid);
    RealD nmmp_sb = mmp_sb.norm2(Nbuf_pr, &Grid);
    std::cout << GridLogMessage << "[PROBE StageB] HermOp d=" << d_sb
              << " |mmp|^2=" << nmmp_sb << std::endl;
    SB::free_state(p_sb); SB::free_state(mmp_sb); SB::free_state(tmp_sb);
    SB::free_state(scratch_lo_sb);

    // Style C path: pack b → native CSF, call HermOp wrapper.
    quda::ColorSpinorParam param_tmpl = Mop_quda.MakeNativeCsfParam();
    int X_full_dims[4];
    for (int d_ = 0; d_ < 4; ++d_) X_full_dims[d_] = Mop_quda.GaugeParam().X[d_];
    SC::DoubledStateCSF p_sc, mmp_sc, tmp_sc, scratch_lo_sc;
    p_sc.allocate(param_tmpl);
    mmp_sc.allocate(param_tmpl);
    tmp_sc.allocate(param_tmpl);
    scratch_lo_sc.allocate_lower_only(param_tmpl);
    p_sc.copy_from_grid(b, Mop_quda.InvertParam(), X_full_dims);
    RealD d_sc = DtxqcdHermOpQUDA_StyleC(Mop_quda, p_sc, mmp_sc, tmp_sc,
                                         scratch_lo_sc);
    RealD nmmp_sc = mmp_sc.norm2();
    std::cout << GridLogMessage << "[PROBE StyleC] HermOp d=" << d_sc
              << " |mmp|^2=" << nmmp_sc << std::endl;

    // Direct M_device_csf-only probe (no Mdag wrap).
    quda::ColorSpinorField *p_up[DtxqcdNf] = {p_sc.csf[0].get(), p_sc.csf[1].get()};
    quda::ColorSpinorField *p_lo[DtxqcdNf] = {p_sc.csf[2].get(), p_sc.csf[3].get()};
    quda::ColorSpinorField *t_up[DtxqcdNf] = {tmp_sc.csf[0].get(), tmp_sc.csf[1].get()};
    quda::ColorSpinorField *t_lo[DtxqcdNf] = {tmp_sc.csf[2].get(), tmp_sc.csf[3].get()};
    quda::ColorSpinorField *s_lo[DtxqcdNf] = {scratch_lo_sc.csf[0].get(), scratch_lo_sc.csf[1].get()};

    // Call M_device_csf 3 times, each on a fresh-zeroed output.  If outputs
    // are deterministic, all 3 |M.b|^2 should match exactly.
    for (int rep = 0; rep < 3; ++rep) {
      tmp_sc.zero();
      Mop_quda.M_device_csf(p_up, p_lo, t_up, t_lo, s_lo, /*dagger=*/false);
      RealD ntmp_csf = tmp_sc.norm2();
      std::cout << GridLogMessage
                << "[PROBE StyleC] rep=" << rep
                << "  M_device_csf-only |M.b|^2=" << ntmp_csf << std::endl;
    }

    // Now call HermOp 3 times, checking determinism.
    for (int rep = 0; rep < 3; ++rep) {
      RealD d_rep = DtxqcdHermOpQUDA_StyleC(Mop_quda, p_sc, mmp_sc, tmp_sc,
                                            scratch_lo_sc);
      RealD nmmp_rep = mmp_sc.norm2();
      std::cout << GridLogMessage
                << "[PROBE StyleC] rep=" << rep
                << "  HermOp d=" << d_rep
                << "  |mmp|^2=" << nmmp_rep << std::endl;
    }
  }

  // Stage B reference (raw buffers + Option α host aux, validated bit-equiv
  // vs Stage A in Test_dtxqcd_multishift_quda_stageB).
  std::vector<DTXQCDFermionDoubled> psi_B;
  psi_B.reserve(poles.size());
  for (size_t s = 0; s < poles.size(); ++s) psi_B.emplace_back(&Grid);
  bool skip_stageB = std::getenv("SKIP_STAGEB") != nullptr;
  if (!skip_stageB) {
    std::cout << GridLogMessage << "---- Stage B (raw buffers + Option α) ----" << std::endl;
    DTXQCDMultiShiftCGQUDA_StageB(Mop_grid, Mop_quda, poles, tol, b, psi_B, MaxIter,
                                   /*ReliableUpdateFreq=*/50);
  } else {
    std::cout << GridLogMessage << "---- Stage B SKIPPED via SKIP_STAGEB env ----" << std::endl;
  }

  // Style C (native CSF + quda::blas inner loop).
  std::vector<DTXQCDFermionDoubled> psi_C;
  psi_C.reserve(poles.size());
  for (size_t s = 0; s < poles.size(); ++s) psi_C.emplace_back(&Grid);
  std::cout << GridLogMessage << "---- Style C (native CSF + quda::blas) ----" << std::endl;
  DTXQCDMultiShiftCGQUDA_StyleC(Mop_grid, Mop_quda, poles, tol, b, psi_C, MaxIter,
                                 /*ReliableUpdateFreq=*/50);

  const RealD gate_res = 1.0e-9;
  const RealD gate_rel = 1.0e-9;
  int exitcode = 0;
  std::cout << GridLogMessage << "---- Per-shift gates (C vs B) ----" << std::endl;
  for (size_t s = 0; s < poles.size(); ++s) {
    RealD res_B = shifted_residual(Mop_grid, poles[s], psi_B[s], b);
    RealD res_C = shifted_residual(Mop_grid, poles[s], psi_C[s], b);
    RealD rd    = rel_diff(psi_C[s], psi_B[s]);
    bool ok = (res_C < gate_res) && (rd < gate_rel);
    std::cout << GridLogMessage
              << "shift " << s << "  pole=" << std::setw(7) << std::setprecision(4) << poles[s]
              << "  res_B=" << std::scientific << std::setprecision(3) << res_B
              << "  res_C=" << res_C
              << "  ||psi_C-psi_B||/||psi_B||=" << rd
              << "  " << (ok ? "PASS" : "FAIL") << std::endl;
    if (!ok) exitcode = 1;
  }

  std::cout << GridLogMessage << "---- Overall ----" << std::endl;
  std::cout << GridLogMessage << "Test_dtxqcd_multishift_quda_styleC: "
            << (exitcode == 0 ? "PASS" : "FAIL") << std::endl;

  Grid::Quda::finalize();
  Grid_finalize();
  return exitcode;
}
