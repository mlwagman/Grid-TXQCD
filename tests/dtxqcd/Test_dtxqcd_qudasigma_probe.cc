// Phase H.0: convention probe for DTXQCD doubled-block σ-force via
// QUDA's computeCloverSigmaForceWithSchurFields.
//
// Reference (Path A): DtxqcdRatForceGpu::ExtractAll computes per-(μν)
// ColourMatrix CS[6] from doubled fermions (X, Y), then DTXQCD's existing
// Cmunu chain (mirrors DTXQCDWilsonCloverRationalEOAction:312-352) folds
// CS[6] into a LatticeGaugeField with Convention-A scale (-0.5).
//
// Candidate (Path B): pack a single block (upper OR lower) of doubled
// fermions as plain Wilson-clover Schur completions on gauge U, batched as
// DtxqcdNf RHS into QUDA's σ-force kernel.  Unpack with -1/(8κ²) scale.
//
// Setup choices:
//   - aux fields = 0 → the doubled operator reduces to upper ⊕ lower:
//     upper sees U, lower sees conj(U).  This isolates the upper block from
//     interference with off-diagonal terms.  Lower-block convention is a
//     follow-up (the QUDA gauge slot must be conj(U) for that, not U).
//   - 4⁴ hot gauge, csw = 1.24930970916466 (b6.1 production).
//   - Single parity probed: ODD (matpc_type = ODD_ODD_ASYMMETRIC).
//
// The probe runs for ONE block at a time (BLOCK=upper or lower).  The
// other block's fermions are zero so DtxqcdRatForceGpu's CS_upper or
// CS_lower contribution alone is the reference.
//
// Convention knobs (env var):
//   PROBE_BLOCK            "upper" (default) or "lower"
//   PROBE_NRHS             flavor count to use (default DtxqcdNf=2)
//   PROBE_KAPPA_FORM_Y     1: pack Y as 2κ·M_pc X̂ (default; matches TXQCD H)
//   PROBE_OFF_SCALE        off-parity scale (default 2.0, per TXQCD H)
//   PROBE_DAGGER_YES       1: invert inv_param.dagger (default 1, per TXQCD H)
//   PROBE_OUTPUT_SIGN      ±1 final QUDA gauge force sign (default +1)
//   PROBE_OUTPUT_CONJ      1: conj() the unpacked force entry-wise (default 0)
//   PROBE_OUTPUT_COLOR_T   1: transpose color of the unpacked force per μ
//   PROBE_SWAP_XY          1: swap (X̂, Y) and (W_o, Z_o) in pack
//
// Run:
//   ./tests/dtxqcd/Test_dtxqcd_qudasigma_probe --grid 4.4.4.4 --mpi 1.1.1.1
//
// Gate (H.0 pass): per-μ cos ≥ 0.99999, factor ∈ [0.9999, 1.0001] on hot
// gauge for at least one (BLOCK, knob-tuple) combination.  If no candidate
// achieves this, escalate to two-call fallback per HANDOFF_PHASE_H_LIGHT_SIGMA_FORCE.md.

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/pseudofermion/EvenOddSchurDifferentiable.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDRationalForceGpuKernel.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>
#include <Grid/util/QudaForcePrimitives.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>

#include <quda.h>

using namespace Grid;

typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

static int envI(const char *k, int dflt) {
  const char *e = std::getenv(k);
  return (e && *e) ? std::atoi(e) : dflt;
}
static double envD(const char *k, double dflt) {
  const char *e = std::getenv(k);
  return (e && *e) ? std::atof(e) : dflt;
}
static std::string envS(const char *k, const char *dflt) {
  const char *e = std::getenv(k);
  return (e && *e) ? std::string(e) : std::string(dflt);
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();

  GridCartesian Grid4(latt_size, simd_layout, mpi_layout);
  GridRedBlackCartesian RBGrid4(&Grid4);

  GridParallelRNG pRNG(&Grid4);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});

  Quda::initialize();

  // ----- Knobs -----
  const std::string block   = envS("PROBE_BLOCK", "upper");
  const int  Nrhs_use       = envI("PROBE_NRHS", DtxqcdNf);
  const bool kappa_form_y   = envI("PROBE_KAPPA_FORM_Y", 1);
  const double off_scale_in = envD("PROBE_OFF_SCALE", 2.0);
  const bool dagger_yes     = envI("PROBE_DAGGER_YES", 1);
  const double out_sign     = envD("PROBE_OUTPUT_SIGN", +1.0);
  const bool out_conj       = envI("PROBE_OUTPUT_CONJ", 0);
  const bool out_color_t    = envI("PROBE_OUTPUT_COLOR_T", 0);
  const bool swap_xy        = envI("PROBE_SWAP_XY", 0);
  const bool parity_odd     = envI("PROBE_PARITY_ODD", 0);  // Phase H.0 round 3
  const std::string basis_rotate = envS("PROBE_BASIS_ROTATE", "none");
  // Phase H.0 round 4: DR↔UKQCD σ-basis exploration.  QUDA enum_quda.h:370-371
  // shows DR has γ4 off-diagonal + γ5 diagonal; UKQCD has γ4 diagonal + γ5
  // off-diagonal (γ4↔γ5 swap), and γ2 carries opposite σ2 sign.  The unitary
  // V that swaps γ4↔γ5 is V = (γ4 + γ5)/√2 up to phase.
  // Candidates exposed (applied to X̂, Y, W_o, Z_o before QUDA pack):
  //   "none"    — identity (default)
  //   "g5"      — V = γ5
  //   "gT"      — V = γ4 (= GammaT in Grid)
  //   "g5gT"    — V = γ5·γ4
  //   "gTpg5"   — V = (γ4+γ5)/√2  [γ4↔γ5 swap candidate]
  //   "gTmg5"   — V = (γ4-γ5)/√2  [variant]
  const bool dump_site_bil  = envI("PROBE_DUMP_SITE_BIL", 0);
  const bool block_upper    = (block == "upper");
  const int  parity_int     = parity_odd ? Odd : Even;
  const int  other_int      = parity_odd ? Even : Odd;

  // ----- Plain Wilson-clover op on U (block reduces to QCD with aux=0) -----
  RealD mass = -0.245;
  RealD csw  = 1.24930970916466;
  RealD kappa = 0.5 / (4.0 + mass);
  RealD two_kappa = 2.0 * kappa;

  LatticeGaugeField U(&Grid4);
  SU<Nc>::HotConfiguration(pRNG, U);

  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  WCF Dw(U, Grid4, RBGrid4, mass, csw, csw, WilsonAnisotropyCoefficients(), impl_p);
  SchurDifferentiableOperator<WilsonImplR> Mpc(Dw);

  // ----- Build per-flavor Schur-completed fermions -----
  // Y_par = M_pc · X̂ (mass-form on Even parity slot per TXQCD-H matpc).
  // Then take their "full-grid extensions" so that DTXQCD's CS kernel
  // (which sums over BOTH parities) sees a self-consistent pair.
  std::vector<LatticeFermion> X_e_v, Y_e_v, W_o_v, Z_o_v;
  std::vector<LatticeFermion> X_full_v, Y_full_v;
  for (int a = 0; a < Nrhs_use; ++a) {
    LatticeFermion phi_full(&Grid4);
    random(pRNG, phi_full);

    LatticeFermion X_e(&RBGrid4), Y_e(&RBGrid4);
    pickCheckerboard(parity_int, X_e, phi_full);
    Y_e.Checkerboard() = parity_int;
    Mpc.Mpc(X_e, Y_e);                               // mass-form Y on kept parity

    LatticeFermion W_o(&RBGrid4), Z_o(&RBGrid4), tmp(&RBGrid4);
    W_o.Checkerboard() = other_int;
    Z_o.Checkerboard() = other_int;
    tmp.Checkerboard() = other_int;
    Dw.Meooe(X_e, tmp);     Dw.MooeeInv(tmp, W_o);    // W_o = M_oo⁻¹ M_oe X̂
    Dw.MeooeDag(Y_e, tmp);  Dw.MooeeInvDag(tmp, Z_o); // Z_o = M_oo⁻¹† M_eo† Y_e

    // Full-grid extension that self-consistently sums to the Schur picture:
    //   X_full = X̂_e on Even, W_o on Odd
    //   Y_full = Y_e on Even, Z_o on Odd
    // (DtxqcdRatForceGpu reads each parity site; the bilinear over both
    //  parities is the natural CS evaluation that QUDA computes.)
    LatticeFermion X_full(&Grid4), Y_full(&Grid4);
    X_full = Zero(); Y_full = Zero();
    setCheckerboard(X_full, X_e);
    setCheckerboard(X_full, W_o);
    setCheckerboard(Y_full, Y_e);
    setCheckerboard(Y_full, Z_o);

    X_e_v.push_back(X_e);  Y_e_v.push_back(Y_e);
    W_o_v.push_back(W_o);  Z_o_v.push_back(Z_o);
    X_full_v.push_back(X_full);
    Y_full_v.push_back(Y_full);
  }

  // ----- Path A: reference CS[6] via DtxqcdRatForceGpu::ExtractAll -----
  // Fill BLOCK with the Schur-extended fermions; the other block stays zero
  // so its contribution to CS vanishes.
  DTXQCDFermionDoubled X_d(&Grid4), Y_d(&Grid4);
  X_d = Zero(); Y_d = Zero();
  for (int a = 0; a < Nrhs_use; ++a) {
    if (block_upper) {
      X_d.upper.f[a] = X_full_v[a];
      Y_d.upper.f[a] = Y_full_v[a];
    } else {
      X_d.lower.f[a] = X_full_v[a];
      Y_d.lower.f[a] = Y_full_v[a];
    }
  }

  DtxqcdSpinMatrices spin(Grid4);
  LatticeDtxqcdSigma F_sig(&Grid4);
  LatticeDtxqcdPi    F_pi(&Grid4);
  LatticeDtxqcdD     F_d(&Grid4);
  LatticeDtxqcdN     F_n(&Grid4);
  LatticeDtxqcdS     F_s(&Grid4);
  LatticeDtxqcdP     F_p(&Grid4);
  std::array<LatticeColourMatrix, 6> F_cs{
      LatticeColourMatrix(&Grid4), LatticeColourMatrix(&Grid4),
      LatticeColourMatrix(&Grid4), LatticeColourMatrix(&Grid4),
      LatticeColourMatrix(&Grid4), LatticeColourMatrix(&Grid4)};

  DtxqcdRatForceGpu::ExtractAll(X_d, Y_d, csw, spin,
                                 F_sig, F_pi, F_d, F_n, F_s, F_p, F_cs);

  // ----- Path A: Cmunu fold (mirrors DTXQCDWilsonCloverRationalEOAction) -----
  typedef WilsonImplR Impl;
  std::vector<LatticeColourMatrix> Ulinks;
  Ulinks.reserve(Nd);
  for (int mu = 0; mu < Nd; ++mu) {
    LatticeColourMatrix Umu(&Grid4);
    Umu = PeekIndex<LorentzIndex>(U, mu);
    Ulinks.push_back(std::move(Umu));
  }
  LatticeGaugeField F_grid(&Grid4);
  F_grid = Zero();
  for (int mu = 0; mu < Nd; ++mu) {
    LatticeColourMatrix force_mu(&Grid4);
    force_mu = Zero();
    for (int nu = 0; nu < Nd; ++nu) {
      if (mu == nu) continue;
      int m = std::min(mu, nu);
      int n = std::max(mu, nu);
      int mn = 0;
      {
        int idx = 0;
        for (int mm = 0; mm < Nd; ++mm)
          for (int nn = mm + 1; nn < Nd; ++nn) {
            if (mm == m && nn == n) mn = idx;
            ++idx;
          }
      }
      RealD sign = (mu < nu) ? 1.0 : -1.0;
      force_mu += (0.25 * sign) *
          WilsonCloverHelpers<Impl>::Cmunu(Ulinks, F_cs[mn], mu, nu);
    }
    pokeLorentz(F_grid, Ulinks[mu] * force_mu, mu);
  }
  // Convention-A correction (-0.5), matching the EO action and TXQCD H.
  F_grid = ComplexD(-0.5, 0.0) * F_grid;

  // ----- Path B: QUDA σ-force kernel on UPPER-block Schur fields -----
  // (Lower-block case requires gauge = conj(U); flagged here, not yet probed.)
  if (!block_upper) {
    std::cout << GridLogMessage
              << "[WARN] lower-block probe needs gauge=conj(U); current build "
              << "feeds U.  Expect cos<0 unless OUTPUT_CONJ+sign tune lands."
              << std::endl;
  }

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
  inv_param.matpc_type   = parity_odd ? QUDA_MATPC_ODD_ODD_ASYMMETRIC
                                       : QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  inv_param.dagger       = dagger_yes ? QUDA_DAG_YES : QUDA_DAG_NO;
  inv_param.input_location  = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.twist_flavor = QUDA_TWIST_NO;
  // Sanity print: confirm the dagger plumbing (the prior sweep saw DAGGER_YES
  // appear as a no-op; print the env value AND the post-assignment field).
  std::cout << GridLogMessage
            << "[plumb] env(PROBE_DAGGER_YES)=" << dagger_yes
            << "  inv_param.dagger=" << (int)inv_param.dagger
            << " (YES=" << (int)QUDA_DAG_YES
            << " NO=" << (int)QUDA_DAG_NO << ")"
            << "  matpc_type=" << (int)inv_param.matpc_type
            << " (parity_odd=" << parity_odd << ")"
            << std::endl;

  int V_eo = Quda::local_volume(&Grid4) / 2;
  using SiteSpinor = typename LatticeFermion::scalar_object;
  std::vector<std::vector<double>> x_bufs(Nrhs_use, std::vector<double>(24 * V_eo));
  std::vector<std::vector<double>> y_bufs(Nrhs_use, std::vector<double>(24 * V_eo));
  std::vector<std::vector<double>> w_bufs(Nrhs_use, std::vector<double>(24 * V_eo));
  std::vector<std::vector<double>> z_bufs(Nrhs_use, std::vector<double>(24 * V_eo));

  // ---- Phase H.0 round 4: optional basis rotation V·ψ before packing. ----
  // QUDA σ-Oprod internally promotes the input CSF to QUDA_UKQCD_GAMMA_BASIS
  // (QudaForcePrimitives.h:409); if the auto-rotation between DR (our inv_param
  // setting) and UKQCD doesn't fire correctly inside QUDA's assignment, the
  // σ_μν tensor inside QUDA's kernel acts on UKQCD-basis spinors but we'd be
  // feeding raw-DR data — channel-dependent perp.  This knob applies V·ψ on
  // the host before pack so QUDA sees the equivalent rotated input.
  auto apply_rotation = [&](LatticeFermion &psi) {
    if (basis_rotate == "none") return;
    Gamma g5(Gamma::Algebra::Gamma5);
    Gamma gT(Gamma::Algebra::GammaT);
    LatticeFermion tmp(psi.Grid());
    tmp.Checkerboard() = psi.Checkerboard();
    if (basis_rotate == "g5") {
      tmp = g5 * psi;  psi = tmp;
    } else if (basis_rotate == "gT") {
      tmp = gT * psi;  psi = tmp;
    } else if (basis_rotate == "g5gT") {
      tmp = gT * psi;  psi = g5 * tmp;
    } else if (basis_rotate == "gTpg5") {
      LatticeFermion t1(psi.Grid()), t2(psi.Grid());
      t1.Checkerboard() = psi.Checkerboard();
      t2.Checkerboard() = psi.Checkerboard();
      t1 = gT * psi;
      t2 = g5 * psi;
      psi = ComplexD(1.0 / std::sqrt(2.0), 0.0) * (t1 + t2);
    } else if (basis_rotate == "gTmg5") {
      LatticeFermion t1(psi.Grid()), t2(psi.Grid());
      t1.Checkerboard() = psi.Checkerboard();
      t2.Checkerboard() = psi.Checkerboard();
      t1 = gT * psi;
      t2 = g5 * psi;
      psi = ComplexD(1.0 / std::sqrt(2.0), 0.0) * (t1 - t2);
    } else {
      std::cout << GridLogMessage
                << "[WARN] unknown PROBE_BASIS_ROTATE=" << basis_rotate
                << "; treating as 'none'" << std::endl;
    }
  };

  auto pack_scaled = [&](const LatticeFermion &fld, RealD scale, double *buf) {
    LatticeFermion scaled(fld.Grid());
    scaled.Checkerboard() = fld.Checkerboard();
    scaled = scale * fld;
    scaled.Checkerboard() = fld.Checkerboard();
    apply_rotation(scaled);
    std::vector<SiteSpinor> sv;
    unvectorizeToLexOrdArray(sv, scaled);
    std::memcpy(buf, sv.data(), V_eo * 24 * sizeof(double));
  };

  RealD x_scale = 1.0;
  RealD y_scale = kappa_form_y ? two_kappa : 1.0;
  for (int a = 0; a < Nrhs_use; ++a) {
    const LatticeFermion &Xe = swap_xy ? Y_e_v[a] : X_e_v[a];
    const LatticeFermion &Ye = swap_xy ? X_e_v[a] : Y_e_v[a];
    const LatticeFermion &Wo = swap_xy ? Z_o_v[a] : W_o_v[a];
    const LatticeFermion &Zo = swap_xy ? W_o_v[a] : Z_o_v[a];
    pack_scaled(Xe, x_scale,    x_bufs[a].data());
    pack_scaled(Ye, y_scale,    y_bufs[a].data());
    pack_scaled(Wo, off_scale_in, w_bufs[a].data());
    pack_scaled(Zo, off_scale_in, z_bufs[a].data());
  }

  // ---- Per-(α,β) σ_{0,1} bilinear dump at site (0,0,0,0), one RHS, ic=jc=0.
  // The CS reference kernel computes (for upper-only):
  //   CS[0,1](ic=0, jc=0) ∝ Σ_{α,β} σ_{0,1}(α,β) · bil((β,0),(α,0))
  //     where bil(R,C) = 0.5*(conj(Y[C])X[R] + conj(X[C])Y[R]).
  // We print the 4×4 grid of σ_{0,1}(α,β)·X(α,0)·conj(Y(β,0)) (one half of
  // the symmetrised bilinear) so the channel-by-channel content is visible.
  if (dump_site_bil) {
    Coordinate origin(Nd, 0);
    int origin_cb = ((0 + 0 + 0 + 0) & 1);
    if (origin_cb == parity_int) {
      SiteSpinor X0, Y0;
      peekSite(X0, X_e_v[0], origin);
      peekSite(Y0, Y_e_v[0], origin);
      const auto &smn = spin.sigma_munu[0];  // σ_{0,1}
      std::cout << GridLogMessage
                << "[bil-dump] site (0,0,0,0) parity=" << origin_cb
                << " σ_{0,1}(α,β)·X(α,0)·conj(Y(β,0)):" << std::endl;
      for (int alpha = 0; alpha < Ns; ++alpha) {
        std::ostringstream row;
        row << "  α=" << alpha << " :";
        for (int beta = 0; beta < Ns; ++beta) {
          ComplexD smn_ab(smn(alpha, beta).real(), smn(alpha, beta).imag());
          ComplexD X_a0 = ComplexD(X0()(alpha)(0).real(),
                                    X0()(alpha)(0).imag());
          ComplexD Y_b0 = ComplexD(Y0()(beta)(0).real(),
                                    Y0()(beta)(0).imag());
          ComplexD Y_b0_c(Y_b0.real(), -Y_b0.imag());
          ComplexD val = smn_ab * X_a0 * Y_b0_c;
          row << " (" << val.real() << "," << val.imag() << ")";
        }
        std::cout << GridLogMessage << row.str() << std::endl;
      }
      // Also print σ_{0,1} itself for reference.
      std::cout << GridLogMessage
                << "[bil-dump] Grid σ_{0,1} (4×4):" << std::endl;
      for (int alpha = 0; alpha < Ns; ++alpha) {
        std::ostringstream row;
        row << "  σ row α=" << alpha << " :";
        for (int beta = 0; beta < Ns; ++beta) {
          row << " (" << smn(alpha, beta).real()
              << "," << smn(alpha, beta).imag() << ")";
        }
        std::cout << GridLogMessage << row.str() << std::endl;
      }
    } else {
      std::cout << GridLogMessage
                << "[bil-dump] origin (0,0,0,0) is parity=" << origin_cb
                << " but kept parity=" << parity_int
                << "; skipping (run with PROBE_PARITY_ODD opposite to dump)."
                << std::endl;
    }
  }

  std::vector<void*> x_ptrs(Nrhs_use), y_ptrs(Nrhs_use),
                      w_ptrs(Nrhs_use), z_ptrs(Nrhs_use);
  std::vector<double> coeff(Nrhs_use, 1.0);
  for (int a = 0; a < Nrhs_use; ++a) {
    x_ptrs[a] = x_bufs[a].data();
    y_ptrs[a] = y_bufs[a].data();
    w_ptrs[a] = w_bufs[a].data();
    z_ptrs[a] = z_bufs[a].data();
  }

  RealD kappa2 = +kappa * kappa;             // TXQCD-H validated (positive)
  RealD ck     = -csw * kappa / 8.0;
  RealD dt     = 1.0;
  RealD sigma_trace_coeff = 0.0;

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

  // Pre-call banner: every scalar that lands in the kernel.  CRITICAL: kappa2
  // is taken by the kernel but UNUSED inside computeCloverSigmaForceWithSchurFields
  // (the body uses 1.0 at cloverDerivative line 488, and updateMomentum uses -1.0).
  // ck is the only kernel-internal coefficient knob (ferm_epsilon = 2·ck·coeff·dt).
  std::cout << GridLogMessage
            << "[plumb] kernel call:"
            << "  Nrhs=" << Nrhs_use
            << "  coeff[0]=" << coeff[0]
            << "  ck=" << ck
            << "  kappa2=" << kappa2 << " (UNUSED in σ kernel)"
            << "  dt=" << dt
            << "  sigma_trace_coeff=" << sigma_trace_coeff
            << "  inv_param.dagger=" << (int)inv_param.dagger
            << "  x_scale=" << x_scale
            << "  y_scale=" << y_scale
            << "  off_scale=" << off_scale_in
            << std::endl;
  Quda::computeCloverSigmaForceWithSchurFields(
      mom_buf.data(),
      x_ptrs.data(), y_ptrs.data(),
      w_ptrs.data(), z_ptrs.data(),
      Nrhs_use, coeff,
      kappa2, ck, dt,
      sigma_trace_coeff,
      &force_gauge_param,
      &inv_param);
  std::cout << GridLogMessage
            << "[plumb] post-kernel: inv_param.dagger=" << (int)inv_param.dagger
            << std::endl;

  // ----- Unpack mom_buf (MILC RECONSTRUCT_10 EO order, V·4·10 doubles)
  //       → per-dir RECONSTRUCT_NO 18-dpsite lex-ordered → Grid LatticeGaugeField
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

  // QUDA → Grid rescale used in TXQCD H σ-only HYBRID path.
  RealD quda_to_grid_factor = out_sign * (-1.0 / (8.0 * kappa * kappa));
  F_quda = quda_to_grid_factor * F_quda;

  // Optional knobs: entry-wise complex conjugation and color transpose per μ.
  if (out_conj || out_color_t) {
    LatticeGaugeField F_post(&Grid4);
    F_post = Zero();
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Mq = PeekIndex<LorentzIndex>(F_quda, mu);
      if (out_color_t) Mq = transpose(Mq);
      if (out_conj)    Mq = conjugate(Mq);
      pokeLorentz(F_post, Mq, mu);
    }
    F_quda = F_post;
  }

  // ----- Compare on Ta-projected forces (per-direction) -----
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
            << "===== Phase H.0 DTXQCD σ-force probe =====" << std::endl
            << "  BLOCK=" << block << " Nrhs=" << Nrhs_use
            << " kappa_form_Y=" << kappa_form_y << " off_scale=" << off_scale_in
            << " dagger=" << (dagger_yes ? "YES" : "NO")
            << " out_sign=" << out_sign << " out_conj=" << out_conj
            << " out_color_t=" << out_color_t << " swap_xy=" << swap_xy
            << " parity_odd=" << parity_odd
            << " basis_rotate=" << basis_rotate
            << std::endl
            << "  mass=" << mass << " csw=" << csw << " κ=" << kappa
            << " 2κ=" << two_kappa << std::endl;

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
    // Also print RAW (un-Ta'd) per-μ norms — if the raw scale ratio is
    // μ-uniform but the Ta-projected one isn't (or vice versa), that's a
    // σ-basis vs hermitian-mixing fingerprint.
    auto Fg_mu = PeekIndex<LorentzIndex>(F_grid, mu);
    auto Fq_mu = PeekIndex<LorentzIndex>(F_quda, mu);
    RealD ng_raw = norm2(Fg_mu);
    RealD nq_raw = norm2(Fq_mu);
    RealD ratio_raw = (ng_raw > 0) ? std::sqrt(nq_raw / ng_raw) : 0.0;
    RealD ratio_ta  = (ng > 0)     ? std::sqrt(nq     / ng    ) : 0.0;
    std::cout << GridLogMessage
              << "  μ=" << mu
              << "  |Ta(grid)|²=" << ng << "  |Ta(quda)|²=" << nq
              << "  cos=" << cos_mu << "  factor=" << factor_mu
              << "  |raw_grid|²=" << ng_raw << "  |raw_quda|²=" << nq_raw
              << "  |Q/G|_Ta=" << ratio_ta << "  |Q/G|_raw=" << ratio_raw
              << std::endl;
    if (std::abs(cos_mu - 1.0) > 1e-5 || std::abs(factor_mu - 1.0) > 1e-3)
      pass = false;
  }

  std::cout << GridLogMessage
            << (pass ? "[PROBE PASS]" : "[PROBE FAIL]") << std::endl;

  Quda::finalize();
  Grid_finalize();
  return pass ? 0 : 1;
}
