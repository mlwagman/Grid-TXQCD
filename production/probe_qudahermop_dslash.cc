// Phase M.1 — systematic convention probe for QUDA internal Dslash vs Grid Meooe.
//
// Phase I.1 baseline: cos(grid, quda) = -0.989, factor = -0.476.
// Magnitudes of |perpendicular| component are small (10%), so the mismatch is
// mostly a sign + scale that one of the convention knobs should resolve.
//
// Probe variables (set via env vars):
//   PROBE_GAMMA_BASIS = UKQCD | DEGRAND_ROSSI | CHIRAL
//   PROBE_DAGGER      = 0 (NO) | 1 (YES)
//   PROBE_IN_PARITY   = EVEN | ODD
//   PROBE_OUT_PARITY  = EVEN | ODD
//   PROBE_MATPC       = EE_ASYM | OO_ASYM | EE_SYM | OO_SYM
//   PROBE_BC_T        = +1 | -1   (boundary phase for time)
//   PROBE_SCALE       = floating-point scale to multiply QUDA output by before
//                       comparing to Grid (1.0, 2.0, 0.5, -1.0, -2.0, -0.5, ...)
//
// Output: a single line per invocation summarizing cos and factor so a shell
// driver can scan a 16+ configuration grid.

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>

#include <quda.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dirac_quda.h>

extern ::quda::GaugeField  *::gaugePrecise;
extern ::quda::CloverField *::cloverPrecise;

using namespace Grid;
typedef WilsonFermion<WilsonImplR> WilsonOp;

static const char *envstr(const char *name, const char *def) {
  const char *v = std::getenv(name);
  return (v && *v) ? v : def;
}
static double envdbl(const char *name, double def) {
  const char *v = std::getenv(name);
  return (v && *v) ? std::atof(v) : def;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt_size  = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();

  GridCartesian Grid4(latt_size, simd_layout, mpi_layout);
  GridRedBlackCartesian RBGrid4(&Grid4);

  GridParallelRNG pRNG(&Grid4);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});

  // === Decode probe variables ===
  // Host buffer gamma basis (what Grid's bytes are interpreted as).
  // Device CSF gamma basis stays UKQCD (required by QUDA's internal Dirac).
  std::string gb_s = envstr("PROBE_GAMMA_BASIS", "UKQCD");
  QudaGammaBasis gb = QUDA_UKQCD_GAMMA_BASIS;
  if (gb_s == "DEGRAND_ROSSI")  gb = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  else if (gb_s == "CHIRAL")    gb = QUDA_CHIRAL_GAMMA_BASIS;
  else if (gb_s == "UKQCD")     gb = QUDA_UKQCD_GAMMA_BASIS;
  // Device CSF basis: typically must be UKQCD for dirac->Dslash to work.
  std::string dev_gb_s = envstr("PROBE_DEV_GAMMA_BASIS", "UKQCD");
  QudaGammaBasis dev_gb = QUDA_UKQCD_GAMMA_BASIS;
  if (dev_gb_s == "DEGRAND_ROSSI")  dev_gb = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  else if (dev_gb_s == "CHIRAL")    dev_gb = QUDA_CHIRAL_GAMMA_BASIS;
  else if (dev_gb_s == "UKQCD")     dev_gb = QUDA_UKQCD_GAMMA_BASIS;

  int dagger = std::atoi(envstr("PROBE_DAGGER", "0"));
  std::string ip_s = envstr("PROBE_IN_PARITY", "EVEN");
  std::string op_s = envstr("PROBE_OUT_PARITY", "ODD");
  QudaParity in_p  = (ip_s == "EVEN") ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY;
  QudaParity out_p = (op_s == "EVEN") ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY;
  int g_in_p  = (ip_s == "EVEN") ? Even : Odd;
  int g_out_p = (op_s == "EVEN") ? Even : Odd;

  std::string matpc_s = envstr("PROBE_MATPC", "EE_ASYM");
  QudaMatPCType matpc = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  if (matpc_s == "EE_ASYM") matpc = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  else if (matpc_s == "OO_ASYM") matpc = QUDA_MATPC_ODD_ODD_ASYMMETRIC;
  else if (matpc_s == "EE_SYM")  matpc = QUDA_MATPC_EVEN_EVEN;
  else if (matpc_s == "OO_SYM")  matpc = QUDA_MATPC_ODD_ODD;

  double bc_t = envdbl("PROBE_BC_T", -1.0);
  double scale = envdbl("PROBE_SCALE", 1.0);

  Quda::initialize();
  RealD mass = envdbl("PROBE_MASS", -0.245);
  RealD csw  = envdbl("PROBE_CSW",  1.24930970916466);

  LatticeGaugeField U(&Grid4);
  std::string gauge_type = envstr("PROBE_GAUGE", "hot");
  if (gauge_type == "cold") {
    SU<Nc>::ColdConfiguration(U);
  } else {
    SU<Nc>::HotConfiguration(pRNG, U);
  }
  // Optional gauge transformation before passing to QUDA.
  // PROBE_GAUGE_XFORM = identity | transpose | conj | adj
  std::string xform = envstr("PROBE_GAUGE_XFORM", "identity");
  LatticeGaugeField U_quda(&Grid4);
  if (xform == "transpose") {
    for (int mu = 0; mu < Nd; ++mu) {
      auto Umu = PeekIndex<LorentzIndex>(U, mu);
      PokeIndex<LorentzIndex>(U_quda, transpose(Umu), mu);
    }
  } else if (xform == "conj") {
    for (int mu = 0; mu < Nd; ++mu) {
      auto Umu = PeekIndex<LorentzIndex>(U, mu);
      PokeIndex<LorentzIndex>(U_quda, conjugate(Umu), mu);
    }
  } else if (xform == "adj") {
    for (int mu = 0; mu < Nd; ++mu) {
      auto Umu = PeekIndex<LorentzIndex>(U, mu);
      PokeIndex<LorentzIndex>(U_quda, adj(Umu), mu);
    }
  } else {
    U_quda = U;
  }

  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  // PROBE_GRID_BC controls whether Grid's WilsonFermion applies AP-T phase.
  // QUDA always bakes the phase into its gauge at SetGauge() (when
  // anti_periodic_t=true).  Setting Grid_BC=+1 + QUDA bake = single-application.
  // Setting Grid_BC=-1 + QUDA bake = double-application (the bench's bug).
  double grid_bc_t = envdbl("PROBE_GRID_BC_T", bc_t);
  impl_p.boundary_phases[Nd - 1] = grid_bc_t;
  WilsonOp Dw(U, Grid4, RBGrid4, mass, impl_p);

  // === Grid Meooe baseline ===
  LatticeFermion src_full(&Grid4);
  random(pRNG, src_full);
  LatticeFermion src_in(&RBGrid4), out_grid(&RBGrid4);
  pickCheckerboard(g_in_p, src_in, src_full);
  out_grid.Checkerboard() = g_out_p;
  Dw.Meooe(src_in, out_grid);

  // === QUDA setup ===
  QudaCloverParams qp;
  qp.mass = mass;
  qp.csw  = csw;
  qp.anti_periodic_t = (bc_t < 0);
  qp.tol = 1e-10;
  qp.max_iter = 5000;
  qp.gamma_basis = gb;

  QudaCloverInverter quda_loader(&Grid4, qp);
  quda_loader.SetGauge(U_quda);

  QudaInvertParam &inv_param = quda_loader.InvertParam();
  inv_param.matpc_type = matpc;
  inv_param.dagger     = dagger ? QUDA_DAG_YES : QUDA_DAG_NO;
  inv_param.input_location  = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  using namespace ::quda;
  if (!::gaugePrecise) {
    std::cerr << "ERROR: gaugePrecise not loaded\n"; return 1;
  }
  QudaGaugeParam gauge_param = quda_loader.GaugeParam();
  GaugeFieldParam gParam(gauge_param, nullptr, QUDA_GENERAL_LINKS);

  ColorSpinorParam qParam(nullptr, inv_param, gParam.x, /*pc=*/false,
                          QUDA_CUDA_FIELD_LOCATION);
  qParam.setPrecision(gParam.Precision(), gParam.Precision(), true);
  qParam.create     = QUDA_NULL_FIELD_CREATE;
  qParam.gammaBasis = dev_gb;

  DiracParam diracParam;
  setDiracParam(diracParam, &inv_param, /*pc_solve=*/true);
  Dirac *dirac = Dirac::create(diracParam);

  ColorSpinorField x_dev(qParam);
  ColorSpinorField y_dev(qParam);

  // Pack src (full grid) via Grid's EO conversion machinery (same as
  // QudaCloverInverter::operator()).  This produces a buffer that QUDA reads
  // as: even half (V_eo sites, lex/2 order) followed by odd half (V_eo sites).
  int V    = Quda::local_volume(&Grid4);
  int V_eo = V / 2;
  std::vector<double> src_eo(24 * V, 0.0);
  // We need the FULL-volume input fermion for fermion_to_eo_buffer.  Build
  // src_full_in with src_in poked into its parity, zeros elsewhere.
  LatticeFermion src_full_in(&Grid4);
  src_full_in = Zero();
  setCheckerboard(src_full_in, src_in);
  Quda::fermion_to_eo_buffer(src_full_in, src_eo.data());

  {
    ColorSpinorParam cpuParam(src_eo.data() + (in_p == QUDA_EVEN_PARITY ? 0 : 24 * V_eo),
                              inv_param, gParam.x, /*pc=*/true,
                              QUDA_CPU_FIELD_LOCATION);
    cpuParam.gammaBasis = gb;
    ColorSpinorField cpuSrc(cpuParam);
    x_dev[in_p] = cpuSrc;
  }

  // Apply Dslash, in_p → out_p.
  dirac->Dslash(y_dev[out_p], x_dev[in_p], out_p);
  qudaDeviceSynchronize();

  // Pull QUDA result back via the same EO buffer machinery used for inversions.
  std::vector<double> out_eo(24 * V, 0.0);
  {
    ColorSpinorParam cpuParam(out_eo.data() + (out_p == QUDA_EVEN_PARITY ? 0 : 24 * V_eo),
                              inv_param, gParam.x, /*pc=*/true,
                              QUDA_CPU_FIELD_LOCATION);
    cpuParam.gammaBasis = gb;
    ColorSpinorField cpuOut(cpuParam);
    cpuOut = y_dev[out_p];
  }
  LatticeFermion out_quda_full(&Grid4);
  Quda::eo_buffer_to_fermion(out_eo.data(), out_quda_full);
  LatticeFermion out_quda(&RBGrid4);
  out_quda.Checkerboard() = g_out_p;
  pickCheckerboard(g_out_p, out_quda, out_quda_full);
  // Apply user-specified scale.
  out_quda = scale * out_quda;

  RealD n_grid = norm2(out_grid);
  RealD n_quda = norm2(out_quda);
  ComplexD ip  = innerProduct(out_grid, out_quda);
  RealD cos_v  = (n_grid > 0 && n_quda > 0)
                     ? real(ip) / std::sqrt(n_grid * n_quda) : 0.0;
  RealD factor = (n_quda > 0) ? real(ip) / n_quda : 0.0;
  RealD ratio  = std::sqrt(n_quda / n_grid);

  std::cout << "PROBE_RESULT"
            << " gb=" << gb_s
            << " dag=" << dagger
            << " in=" << ip_s
            << " out=" << op_s
            << " matpc=" << matpc_s
            << " bc_t=" << bc_t
            << " scale=" << scale
            << " cos=" << cos_v
            << " factor=" << factor
            << " |q|/|g|=" << ratio
            << std::endl;

  delete dirac;
  Quda::finalize();
  Grid_finalize();
  return 0;
}
