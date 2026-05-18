// Phase H.1: standalone bench for QUDA computeCloverOprod (Wilson-hop force
// primitive) with Schur-completed off-parity inputs vs Grid Path A's
// MpcDeriv+MpcDagDeriv chain.  Plain QCD with clover (no Δ, no TXQCD).
//
// What this prints:
//   - Per-direction cos(Ta(F_grid)[μ], Ta(F_quda)[μ]) and factor
//   - Aggregate cos+factor over all directions
//   - Norms of Ta-projected forces
//
// Gate (H.1 pass): cos ≥ 0.99999, factor ∈ [0.9999, 1.0001] for ALL μ.
//
// Convention knobs (env-var; only relevant for H.2 sweep):
//   BENCH_NEGATE_WO=1            negate W_o input
//   BENCH_NEGATE_ZO=1            negate Z_o input
//   BENCH_OFFPARITY_SCALE=<v>    override √(2κ) off-parity scale (default sqrt(2κ))
//   BENCH_FORCECOEFF_SIGN=-1     flip sign of force_coeff (kappa²→-kappa²)
//   BENCH_DAGGER_YES=1           toggle inv_param.dagger to YES
//   BENCH_QUDA_OUTPUT_SIGN=-1    flip overall QUDA output sign before compare
//   BENCH_NO_KAPPA_RESCALE=1     pack Y in mass-form (no 2κ scale)
//   BENCH_X_KAPPA_RESCALE=1      apply 2κ to X̂ (kappa-form X̂)
//   BENCH_GAMMA5_Y=1             apply γ5 to Y before pack
//   BENCH_GAMMA5_WO=1            apply γ5 to W_o before pack
//
// Run:
//   mpirun -np 1 ./bench_clover_oprod --grid 8.8.8.8 --mpi 1.1.1.1
//
// Recommended starting test on 4⁴ for fast turn-around once the bench builds:
//   mpirun -np 1 ./bench_clover_oprod --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/pseudofermion/EvenOddSchurDifferentiable.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>
#include <Grid/util/QudaForcePrimitives.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>

#include <quda.h>

using namespace Grid;

typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

static double envD(const char *k, double dflt) {
  const char *e = std::getenv(k);
  return (e && *e) ? std::atof(e) : dflt;
}
static int envI(const char *k, int dflt) {
  const char *e = std::getenv(k);
  return (e && *e) ? std::atoi(e) : dflt;
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

  Quda::initialize();

  // --- Plain Wilson-clover op (no Δ).
  RealD mass = -0.245;
  RealD csw  = 1.24930970916466;
  RealD kappa = 0.5 / (4.0 + mass);
  RealD two_kappa = 2.0 * kappa;
  RealD sqrt_two_kappa = std::sqrt(two_kappa);

  LatticeGaugeField U(&Grid4);
  SU<Nc>::HotConfiguration(pRNG, U);

  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  WCF Dw(U, Grid4, RBGrid4, mass, csw, csw, WilsonAnisotropyCoefficients(), impl_p);
  SchurDifferentiableOperator<WilsonImplR> Mpc(Dw);

  // --- Random X̂ on EVEN parity; bilinear inputs are self-consistent if
  // Y = M_pc·X̂, W_o/Z_o are exact Schur completions.  No CG needed for
  // convention validation since the comparison is bilinear in these inputs.
  LatticeFermion phi_full(&Grid4);
  random(pRNG, phi_full);
  LatticeFermion X_e(&RBGrid4), Y_e(&RBGrid4);
  pickCheckerboard(Even, X_e, phi_full);
  Y_e.Checkerboard() = Even;
  Mpc.Mpc(X_e, Y_e);                              // Y = M_pc·X̂ (mass form)

  // --- Schur completions on ODD parity.
  LatticeFermion W_o(&RBGrid4), Z_o(&RBGrid4), tmp1(&RBGrid4);
  Dw.Meooe(X_e, tmp1);     Dw.MooeeInv(tmp1, W_o);   // W_o = M_oo⁻¹·M_oe·X̂_e
  Dw.MeooeDag(Y_e, tmp1);  Dw.MooeeInvDag(tmp1, Z_o); // Z_o = M_oo⁻¹†·M_eo†·Y_e

  // --- Path A reference Wilson-piece force.
  LatticeGaugeField F_grid(&Grid4), tmp_force(&Grid4);
  F_grid = Zero();
  Mpc.MpcDeriv(tmp_force, Y_e, X_e);     F_grid = F_grid + tmp_force;
  Mpc.MpcDagDeriv(tmp_force, X_e, Y_e);  F_grid = F_grid + tmp_force;

  // --- QUDA gauge / clover loader.
  QudaCloverParams qp;
  qp.mass = mass;
  qp.csw  = csw;
  qp.anti_periodic_t = true;
  qp.tol = 1e-10;
  qp.max_iter = 5000;
  qp.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  QudaCloverInverter quda_loader(&Grid4, qp);
  quda_loader.SetGauge(U);

  QudaInvertParam &inv_param = quda_loader.InvertParam();
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  inv_param.dagger     = envI("BENCH_DAGGER_YES", 0) ? QUDA_DAG_YES : QUDA_DAG_NO;
  inv_param.input_location  = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  // --- Convention knobs.
  RealD off_scale = envD("BENCH_OFFPARITY_SCALE", sqrt_two_kappa);
  bool  neg_wo    = envI("BENCH_NEGATE_WO", 0);
  bool  neg_zo    = envI("BENCH_NEGATE_ZO", 0);
  bool  no_y_kappa = envI("BENCH_NO_KAPPA_RESCALE", 0);
  bool  x_kappa   = envI("BENCH_X_KAPPA_RESCALE", 0);
  bool  g5_y      = envI("BENCH_GAMMA5_Y", 0);
  bool  g5_wo     = envI("BENCH_GAMMA5_WO", 0);
  RealD force_coeff_sign = envD("BENCH_FORCECOEFF_SIGN", +1.0);
  RealD output_sign = envD("BENCH_QUDA_OUTPUT_SIGN", +1.0);

  std::cout << GridLogMessage << "Phase H.1 bench:" << std::endl
    << "  mass=" << mass << " csw=" << csw << " κ=" << kappa << std::endl
    << "  off_scale=" << off_scale << " neg_wo=" << neg_wo << " neg_zo=" << neg_zo
    << "  no_y_kappa=" << no_y_kappa << " x_kappa=" << x_kappa
    << "  g5_y=" << g5_y << " g5_wo=" << g5_wo
    << "  force_coeff_sign=" << force_coeff_sign
    << "  dagger=" << (inv_param.dagger == QUDA_DAG_YES ? "YES" : "NO") << std::endl;

  // --- Pack inputs.
  int V_eo = Quda::local_volume(&Grid4) / 2;
  using SiteSpinor = typename LatticeFermion::scalar_object;
  std::vector<double> x_buf(24*V_eo), y_buf(24*V_eo), w_buf(24*V_eo), z_buf(24*V_eo);

  auto pack_scaled = [&](const LatticeFermion &fld, RealD scale, double *buf,
                          bool apply_g5) {
    LatticeFermion scaled(fld.Grid());
    if (apply_g5) {
      Gamma g5(Gamma::Algebra::Gamma5);
      LatticeFermion g5fld(fld.Grid());
      g5fld = g5 * fld;
      scaled = scale * g5fld;
    } else {
      scaled = scale * fld;
    }
    scaled.Checkerboard() = fld.Checkerboard();
    std::vector<SiteSpinor> sv;
    unvectorizeToLexOrdArray(sv, scaled);
    std::memcpy(buf, sv.data(), V_eo * 24 * sizeof(double));
  };

  RealD x_scale = x_kappa ? two_kappa : 1.0;
  RealD y_scale = no_y_kappa ? 1.0 : two_kappa;
  RealD wo_scale = (neg_wo ? -1.0 : 1.0) * off_scale;
  RealD zo_scale = (neg_zo ? -1.0 : 1.0) * off_scale;

  pack_scaled(X_e, x_scale,  x_buf.data(), false);
  pack_scaled(Y_e, y_scale,  y_buf.data(), g5_y);
  pack_scaled(W_o, wo_scale, w_buf.data(), g5_wo);
  pack_scaled(Z_o, zo_scale, z_buf.data(), false);

  std::vector<void*> x_ptrs{x_buf.data()};
  std::vector<void*> y_ptrs{y_buf.data()};
  std::vector<void*> w_ptrs{w_buf.data()};
  std::vector<void*> z_ptrs{z_buf.data()};
  std::vector<double> coeff{1.0};

  RealD kappa2 = force_coeff_sign * (-kappa * kappa);
  RealD ck     = -csw * kappa / 8.0;
  RealD dt     = 1.0;

  int V = Quda::local_volume(&Grid4);
  constexpr int MOM_RECON = 10;
  std::vector<double> mom_buf(V * 4 * MOM_RECON, 0.0);

  QudaGaugeParam force_gauge_param = quda_loader.GaugeParam();
  force_gauge_param.type        = QUDA_GENERAL_LINKS;
  force_gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  force_gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
  force_gauge_param.overwrite_mom     = 1;
  force_gauge_param.use_resident_mom  = 0;
  force_gauge_param.make_resident_mom = 0;
  force_gauge_param.return_result_mom = 1;

  Quda::computeCloverWilsonForceWithSchurFields(
      mom_buf.data(),
      x_ptrs.data(), y_ptrs.data(),
      w_ptrs.data(), z_ptrs.data(),
      1, coeff,
      kappa2, ck, dt,
      &force_gauge_param,
      &inv_param);

  // --- Unpack mom_buf (MILC RECONSTRUCT_10 EO order, V·4·10 doubles)
  // → per-direction RECONSTRUCT_NO 18-double-per-site lex-ordered buffers
  // → Grid LatticeGaugeField via lex_buffers_to_gauge.
  Coordinate lc = Grid4.LocalDimensions();
  std::vector<std::vector<double>> dir_eo_18(4, std::vector<double>(18 * V));
  for (int mu = 0; mu < 4; ++mu) {
    double *dst = dir_eo_18[mu].data();
    for (int site = 0; site < V; ++site) {
      const double *m = &mom_buf[(site * 4 + mu) * MOM_RECON];
      double r01 = m[0], i01 = m[1];
      double r02 = m[2], i02 = m[3];
      double r12 = m[4], i12 = m[5];
      double a0  = m[6], a1  = m[7], a2 = m[8];
      double *d = &dst[site * 18];
      d[ 0] = 0.0;   d[ 1] = a0;
      d[ 2] = r01;   d[ 3] = i01;
      d[ 4] = r02;   d[ 5] = i02;
      d[ 6] = -r01;  d[ 7] = i01;
      d[ 8] = 0.0;   d[ 9] = a1;
      d[10] = r12;   d[11] = i12;
      d[12] = -r02;  d[13] = i02;
      d[14] = -r12;  d[15] = i12;
      d[16] = 0.0;   d[17] = a2;
    }
  }
  std::vector<std::vector<double>> dir_lex_18(4, std::vector<double>(18 * V));
  double *lex_ptrs[4];
  for (int mu = 0; mu < 4; ++mu) {
    Quda::eo_to_lex_permute(dir_eo_18[mu].data(), dir_lex_18[mu].data(),
                            V, 18, lc);
    lex_ptrs[mu] = dir_lex_18[mu].data();
  }
  LatticeGaugeField F_quda(&Grid4);
  Quda::lex_buffers_to_gauge(lex_ptrs, F_quda);

  // QUDA→Grid rescale used in the validated σ-only HYBRID path: -1/(8κ²).
  RealD quda_to_grid_factor = output_sign * (-1.0 / (8.0 * kappa * kappa));
  F_quda = quda_to_grid_factor * F_quda;

  // --- Compare per-direction on Ta-projected forces.
  auto Ta_of = [&](const LatticeGaugeField &G) {
    LatticeGaugeField T(&Grid4);
    for (int mu = 0; mu < Nd; ++mu) {
      PokeIndex<LorentzIndex>(T, Ta(PeekIndex<LorentzIndex>(G, mu)), mu);
    }
    return T;
  };
  LatticeGaugeField Ta_grid = Ta_of(F_grid);
  LatticeGaugeField Ta_quda = Ta_of(F_quda);

  std::cout << GridLogMessage
            << "===== Phase H bench: Schur-Wilson force comparison =====" << std::endl;
  RealD n_g_total = norm2(Ta_grid);
  RealD n_q_total = norm2(Ta_quda);
  ComplexD ip_total = innerProduct(Ta_grid, Ta_quda);
  RealD cos_total = (n_g_total > 0 && n_q_total > 0)
                       ? real(ip_total) / std::sqrt(n_g_total * n_q_total) : 0.0;
  RealD factor_total = (n_q_total > 0) ? real(ip_total) / n_q_total : 0.0;
  std::cout << GridLogMessage
            << "  Ta total: |grid|²=" << n_g_total
            << " |quda|²=" << n_q_total
            << " cos=" << cos_total
            << " factor=" << factor_total << std::endl;

  bool pass = std::abs(cos_total - 1.0) < 1e-5
              && std::abs(factor_total - 1.0) < 1e-3;

  for (int mu = 0; mu < Nd; ++mu) {
    auto Tg_mu = PeekIndex<LorentzIndex>(Ta_grid, mu);
    auto Tq_mu = PeekIndex<LorentzIndex>(Ta_quda, mu);
    RealD ng = norm2(Tg_mu);
    RealD nq = norm2(Tq_mu);
    ComplexD ip = innerProduct(Tg_mu, Tq_mu);
    RealD cos_mu = (ng > 0 && nq > 0) ? real(ip) / std::sqrt(ng * nq) : 0.0;
    RealD factor_mu = (nq > 0) ? real(ip) / nq : 0.0;
    std::cout << GridLogMessage << "  μ=" << mu
              << " |grid|²=" << ng
              << " |quda|²=" << nq
              << " cos=" << cos_mu
              << " factor=" << factor_mu
              << std::endl;
    if (std::abs(cos_mu - 1.0) > 1e-5
        || std::abs(factor_mu - 1.0) > 1e-3) pass = false;
  }

  std::cout << GridLogMessage << "===== H.1 gate: " << (pass ? "PASS" : "FAIL")
            << " =====" << std::endl;

  Quda::finalize();
  Grid_finalize();
  return pass ? 0 : 1;
}
