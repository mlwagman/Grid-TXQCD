#pragma once
// Per-operator Schur-complement clover force via QUDA fused kernels.
//
// Shared engine for the light-sector force-assembly port (Hasenbusch ratio
// rungs + tail determinant): given the ODD-parity pair (X, P) of a Schur
// pseudofermion force term, computes
//
//   F_wilson ≈ Ta( Mpc.MpcDeriv(P, X) + Mpc.MpcDagDeriv(X, P) )      [hop]
//   F_sigma  ≈ Ta( Op.MooDeriv(P,X,No) + Op.MooDeriv(X,P,Yes)
//                + Op.MeeDeriv(Z,W,No) + Op.MeeDeriv(W,Z,Yes) )      [clover]
//
// with W = Mee^{-1}·Meo·X and Z = Mee^{-†}·Moe†·P computed here in Grid
// (cheap operator applications), and everything expensive — the hopping-term
// outer products and the sigma_munu clover derivative — executed by QUDA's
// fused force primitives (Grid/util/QudaForcePrimitives.h).
//
// The two pseudofermion force pairings this serves:
//   determinant / tail:  P = Mpc·X               (X = (Mpc†Mpc)^{-1} Phi)
//   ratio, DenOp terms:  P = Mpc·X               (same)
//   ratio, NumOp terms:  P = PhiOdd              (Phi plays Y's role)
//
// CONVENTIONS — transplanted from the validated ODD-parity production
// precedent TXQCDWilsonCloverRationalEOActionQudaPrimitive.h (Phase H
// Wilson piece: cos=1.0, factor=1.0 vs Grid on 4^4/8^4/12^4; Phase B-prime
// sigma piece: production-validated with FD-test coverage):
//   Wilson call: x=X (·1), p=P (·1, MASS form), W/Z off-parity (·2.0),
//                dagger=QUDA_DAG_YES, kappa2=+kappa^2 (POSITIVE),
//                unpack factor -1/(8 kappa^2).
//   sigma call : x=X (·1), p=P (·2 kappa, KAPPA form), W/Z (·sqrt(2 kappa)),
//                dagger=QUDA_DAG_NO, ck=-csw·kappa/8, sigma_trace_coeff=0
//                (the M_ee LogDet is a separate monomial — never turn the
//                trace term on here), unpack factor -1/(8 kappa^2).
// The two pieces cannot share one fused call (different p/off-parity
// normalizations and dagger) — see the Phase H commentary in the TXQCD
// header, lines 160-177.
//
// Residency contract: the caller must have loaded the CURRENT smeared U
// into QUDA (loadGaugeQuda+loadCloverQuda — any rung/tail solver's
// SetGauge(U) at deriv entry does this).  These primitives read only
// ::gaugePrecise and ::extendedGaugeResident; ::cloverPrecise is
// null-checked but never read when sigma_trace_coeff=0, so kappa_den vs
// kappa_num need NO clover reload — kappa enters purely through the scalar
// coefficients and pack scales above.
//
// Empirical knobs for the first ODD-parity pinning of this action family
// (per-piece COMPARE decides; precedent says both +1):
//   QUDA_RUNG_FORCE_WILSON_SIGN=<double>  multiplies F_wilson  (default +1)
//   QUDA_RUNG_FORCE_SIGMA_SIGN=<double>   multiplies F_sigma   (default +1)
//   QUDA_FORCE_SCHUR_SCALE=<double>       overrides the sigma-call sqrt(2k)
//                                         off-parity scale (TXQCD-compatible)

#ifdef GRID_HAVE_QUDA

#include <Grid/GridCore.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaForcePrimitives.h>
#include <Grid/util/QudaPackGpu.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <vector>

NAMESPACE_BEGIN(Grid);
namespace Quda {

// Reusable device scratch + index tables + accumulated timers.  One instance
// per action monomial (grids are fixed for an action's lifetime).
struct QudaSchurForceScratch {
  deviceVector<int>    lex_table_dev;   // RB fermion grid → lex index
  bool                 lex_built = false;
  deviceVector<int>    eo_table_dev;    // full grid → EO mom-buffer site
  bool                 eo_built = false;
  deviceVector<double> pack_dev;        // 4 slots × V_eo × 24
  deviceVector<double> mom_dev;         // V × 4 × 10 (MILC RECONSTRUCT_10)
  // Accumulated timers (us) and call count, printed by the owning action.
  uint64_t t_wz_us = 0, t_pack_us = 0, t_wcall_us = 0, t_scall_us = 0,
           t_unpack_us = 0;
  uint64_t n_calls = 0;

  void Print(const std::string &tag) const {
    if (n_calls == 0) return;
    std::cout << GridLogMessage << "[QudaSchurOpForce/" << tag << "] "
              << n_calls << " op-force calls (ms/call): W/Z="
              << double(t_wz_us) * 1e-3 / n_calls
              << " pack=" << double(t_pack_us) * 1e-3 / n_calls
              << " Wcall=" << double(t_wcall_us) * 1e-3 / n_calls
              << " sigmacall=" << double(t_scall_us) * 1e-3 / n_calls
              << " unpack=" << double(t_unpack_us) * 1e-3 / n_calls
              << std::endl;
  }
};

// Compute the two force pieces of one Schur operator's (X, P) term.
// X, P are ODD-parity RB fermions; F_wilson/F_sigma are full-grid gauge
// fields, OVERWRITTEN (pure outputs).  `loader` supplies kappa/csw and the
// QUDA param structs for Op's mass (built with ODD_ODD_ASYMMETRIC matpc);
// it is used as a parameter source only — never solved with, never
// SetGauge'd here.
template <class FermionOp>
inline void computeSchurOpForceQuda(FermionOp &Op,
                                    const LatticeFermion &X,
                                    const LatticeFermion &P,
                                    QudaCloverMultiShiftInverter &loader,
                                    LatticeGaugeField &F_wilson,
                                    LatticeGaugeField &F_sigma,
                                    QudaSchurForceScratch &scratch) {
  GridBase *rbgrid    = X.Grid();
  GridBase *full_grid = F_wilson.Grid();
  assert(X.Checkerboard() == Odd && P.Checkerboard() == Odd);

  QudaInvertParam &inv_param = loader.InvertParam();
  assert(inv_param.matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC);
  // The sigma path's cloverDerivative reads the extended resident gauge,
  // created/refreshed by the caller's loadGaugeQuda/loadCloverQuda (any
  // solver SetGauge).  Fail loudly rather than deref null.
  assert(::gaugePrecise != nullptr &&
         "computeSchurOpForceQuda: no resident gauge (call a solver SetGauge first)");
  assert(::extendedGaugeResident != nullptr &&
         "computeSchurOpForceQuda: no extended resident gauge (call a solver SetGauge first)");

  // ---- Grid-side Schur completions (EVEN parity, cheap) -------------------
  auto t_wz0 = usecond();
  LatticeFermion W(rbgrid), Z(rbgrid), tmp(rbgrid);
  Op.Meooe(X, tmp);
  Op.MooeeInv(tmp, W);
  Op.MeooeDag(P, tmp);
  Op.MooeeInvDag(tmp, Z);
  scratch.t_wz_us += usecond() - t_wz0;

  // ---- Tables + scratch ---------------------------------------------------
  const int V    = local_volume(full_grid);
  const int V_eo = V / 2;
  const uint64_t per_rhs = uint64_t(V_eo) * 24;
  if (!scratch.lex_built) {
    BuildLexTable(rbgrid, scratch.lex_table_dev);
    scratch.lex_built = true;
  }
  if (!scratch.eo_built) {
    Coordinate lc = full_grid->LocalDimensions();
    BuildEoTable(full_grid, lc, scratch.eo_table_dev);
    scratch.eo_built = true;
  }
  if (scratch.pack_dev.size() < 4 * per_rhs) scratch.pack_dev.resize(4 * per_rhs);
  constexpr int MOM_RECON = 10;
  if (scratch.mom_dev.size() < uint64_t(V) * 4 * MOM_RECON)
    scratch.mom_dev.resize(uint64_t(V) * 4 * MOM_RECON);

  const int *lex_p = &scratch.lex_table_dev[0];
  double *dev_p    = &scratch.pack_dev[0];
  double *x_ptr = dev_p + 0 * per_rhs;
  double *p_ptr = dev_p + 1 * per_rhs;
  double *w_ptr = dev_p + 2 * per_rhs;
  double *z_ptr = dev_p + 3 * per_rhs;
  std::vector<void *> xs = {x_ptr}, ps = {p_ptr}, ws = {w_ptr}, zs = {z_ptr};
  std::vector<double> coeff = {1.0};

  const double kappa = inv_param.kappa;
  const double csw   = inv_param.clover_csw;
  const double ck    = -csw * kappa / 8.0;
  const double dt    = 1.0;
  const double quda_to_grid_factor = -1.0 / (8.0 * kappa * kappa);

  auto env_double = [](const char *name, double dflt) {
    if (const char *s = std::getenv(name); s && *s) return std::atof(s);
    return dflt;
  };
  const double wilson_sign = env_double("QUDA_RUNG_FORCE_WILSON_SIGN", 1.0);
  const double sigma_sign  = env_double("QUDA_RUNG_FORCE_SIGMA_SIGN", 1.0);

  // Param setup shared by both calls (save/restore pattern from the TXQCD
  // precedent — the loader's structs are reused across derivs).
  int saved_use_resident            = inv_param.use_resident_solution;
  QudaDagType saved_dagger          = inv_param.dagger;
  QudaTwistFlavorType saved_twist   = inv_param.twist_flavor;
  QudaFieldLocation saved_input_loc = inv_param.input_location;
  inv_param.use_resident_solution = 0;
  inv_param.twist_flavor          = QUDA_TWIST_NO;
  inv_param.input_location        = QUDA_CUDA_FIELD_LOCATION;  // device pack

  QudaGaugeParam force_gauge_param = loader.GaugeParam();
  force_gauge_param.type        = QUDA_GENERAL_LINKS;
  force_gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  force_gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
  force_gauge_param.location    = QUDA_CUDA_FIELD_LOCATION;    // device mom
  force_gauge_param.overwrite_mom     = 1;
  force_gauge_param.use_resident_mom  = 0;
  force_gauge_param.make_resident_mom = 0;
  force_gauge_param.return_result_mom = 1;

  // ---- Piece 1: Wilson hop (mass-form P, off-parity ×2, DAG_YES, +k^2) ----
  // accelerator_barrier() around each Grid<->QUDA buffer handoff: Grid's pack
  // kernels run on Grid's compute stream, the QUDA primitives on QUDA's own
  // streams, and (unlike QUDA's PUBLIC API entry points) the internal
  // primitives do no boundary synchronization -- without the fences the
  // multi-rank runs showed 1e-5-level nondeterministic per-piece deviations.
  auto t_pack0 = usecond();
  GpuPackFermionRbLex(X, 1.0, x_ptr, lex_p);
  GpuPackFermionRbLex(P, 1.0, p_ptr, lex_p);
  GpuPackFermionRbLex(W, 2.0, w_ptr, lex_p);
  GpuPackFermionRbLex(Z, 2.0, z_ptr, lex_p);
  accelerator_barrier();  // packs visible before QUDA reads
  scratch.t_pack_us += usecond() - t_pack0;

  inv_param.dagger = QUDA_DAG_YES;
  auto t_wcall0 = usecond();
  computeCloverWilsonForceWithSchurFields(
      &scratch.mom_dev[0], xs.data(), ps.data(), ws.data(), zs.data(),
      /*nvector=*/1, coeff, /*kappa2=*/+kappa * kappa, ck, dt,
      &force_gauge_param, &inv_param);
  accelerator_barrier();  // QUDA mom result visible before Grid unpack reads
  scratch.t_wcall_us += usecond() - t_wcall0;

  auto t_unpack0 = usecond();
  GpuUnpackMomToGauge(&scratch.mom_dev[0], &scratch.eo_table_dev[0], F_wilson,
                      wilson_sign * quda_to_grid_factor);
  scratch.t_unpack_us += usecond() - t_unpack0;

  // ---- Piece 2: sigma (kappa-form P ×2k, off-parity ×sqrt(2k), DAG_NO) ----
  const double two_kappa = 2.0 * kappa;
  const double scale_off = env_double("QUDA_FORCE_SCHUR_SCALE",
                                      std::sqrt(two_kappa));
  auto t_pack1 = usecond();
  GpuPackFermionRbLex(P, two_kappa, p_ptr, lex_p);   // X slot unchanged
  GpuPackFermionRbLex(W, scale_off, w_ptr, lex_p);
  GpuPackFermionRbLex(Z, scale_off, z_ptr, lex_p);
  accelerator_barrier();  // packs visible before QUDA reads
  scratch.t_pack_us += usecond() - t_pack1;

  inv_param.dagger = QUDA_DAG_NO;
  auto t_scall0 = usecond();
  computeCloverSigmaForceWithSchurFields(
      &scratch.mom_dev[0], xs.data(), ps.data(), ws.data(), zs.data(),
      /*nvector=*/1, coeff, /*kappa2=*/-kappa * kappa, ck, dt,
      /*sigma_trace_coeff=*/0.0, &force_gauge_param, &inv_param);
  accelerator_barrier();  // QUDA mom result visible before Grid unpack reads
  scratch.t_scall_us += usecond() - t_scall0;

  auto t_unpack1 = usecond();
  GpuUnpackMomToGauge(&scratch.mom_dev[0], &scratch.eo_table_dev[0], F_sigma,
                      sigma_sign * quda_to_grid_factor);
  scratch.t_unpack_us += usecond() - t_unpack1;

  inv_param.use_resident_solution = saved_use_resident;
  inv_param.dagger                = saved_dagger;
  inv_param.twist_flavor          = saved_twist;
  inv_param.input_location        = saved_input_loc;

  scratch.n_calls++;
}

// ---- COMPARE-mode utilities (shared by the ratio and tail actions) --------
// QUDA's updateMomentum output is traceless-anti-Hermitian, so the Grid
// reference piece must be Ta-projected before comparison (the strange-sector
// COMPARE established the same convention: cos(Ta(A), B)).

inline void TaProjectGaugeField(LatticeGaugeField &f) {
  for (int mu = 0; mu < Nd; ++mu) {
    LatticeColourMatrix l = PeekIndex<LorentzIndex>(f, mu);
    l = Ta(l);
    PokeIndex<LorentzIndex>(f, l, mu);
  }
}

// A taken by value: Ta-projected locally.  Prints cos, magnitude factor,
// and both norms; per-piece cos = -1 means flip the corresponding sign knob.
inline void CompareForcePiece(const std::string &tag,
                              LatticeGaugeField A,
                              const LatticeGaugeField &B) {
  TaProjectGaugeField(A);
  ComplexD ip = innerProduct(A, B);
  RealD nA = norm2(A), nB = norm2(B);
  RealD cosv = (nA > 0.0 && nB > 0.0) ? real(ip) / std::sqrt(nA * nB) : 0.0;
  RealD fac  = (nA > 0.0) ? std::sqrt(nB / nA) : 0.0;
  std::cout << GridLogMessage << "[QudaForceCompare] " << tag
            << " cos(Ta(A),B)= " << std::setprecision(10) << cosv
            << " |B|/|Ta(A)|= " << fac
            << " |Ta(A)|^2= " << nA << " |B|^2= " << nB << std::endl;
}

}  // namespace Quda
NAMESPACE_END(Grid);

#endif  // GRID_HAVE_QUDA
