#pragma once
// QUDA-accelerated rational action with Grid-Y-injected QUDA-primitive force.
//
// Phase B variant — replaces Phase 7's monolithic computeCloverForceQuda
// call with our QudaForcePrimitives wrapper which lets us inject the
// host-supplied Y = M_pc_grid · X intermediate field.  Empirically Phase 7's
// computeCloverForceQuda gave cos(Ta(A),B)=0.897 (10% perpendicular gap);
// the gap most likely originates from QUDA's internal γ5+Dslash+M derivation
// of the equivalent of Y, which differs from Grid's mass-form M_pc·X.
//
// This class targets pure Wilson-clover (no TXQCD Δ).  Phase C extends to
// TXQCD by inheriting from a TXQCD action and feeding the TXQCD-modified
// M_pc·X output instead.
//
// Built on top of OneFlavourSchurCloverRationalActionEven for refresh/S
// consistency with the Phase 6 EVEN-parity reference.

#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionEven.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>
#include <Grid/util/QudaFieldConvert.h>
#include <Grid/util/QudaForcePrimitives.h>

#include <quda.h>
#include <cstdlib>
#include <cstring>

namespace Grid {

template <class ImplD, class ImplF,
          class FermOpD_ = WilsonCloverFermion<ImplD, CloverHelpers<ImplD>>,
          class FermOpF_ = WilsonCloverFermion<ImplF, CloverHelpers<ImplF>>>
class OneFlavourSchurCloverQudaPrimitiveActionMP
    : public OneFlavourSchurCloverRationalActionEven<ImplD, FermOpD_> {
 public:
  typedef OneFlavourSchurCloverRationalActionEven<ImplD, FermOpD_> Base;
  typedef typename Base::FermionField FermionField;
  typedef FermOpD_ FermOpD;
  typedef FermOpF_ FermOpF;
  typedef typename ImplD::GaugeField GaugeField;

  OneFlavourSchurCloverQudaPrimitiveActionMP(
      FermOpD &opD, FermOpF & /*opF*/, GridBase * /*sp_rbgrid*/,
      OneFlavourRationalParams &p,
      const QudaCloverParams &qp,
      int /*reliable_update_freq*/ = 50)
      : Base(opD, p), qp_(qp) {

    QudaCloverMultiShiftSpec spec;
    spec.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
    auto &poles = this->PowerNegHalf.poles;
    spec.shifts.resize(poles.size());
    spec.tols.assign(poles.size(), p.tolerance);
    for (size_t k = 0; k < poles.size(); ++k) spec.shifts[k] = poles[k];

    // Pass the gauge grid -> QUDA inherits Grid's MPI comm + rank map (MPI build);
    // see OneFlavourSchurCloverQudaForceRationalActionMP.h for the rationale.
    Quda::initialize(/*device=*/-1, /*mpi_dims=*/nullptr, opD.GaugeGrid());
    quda_ms_.reset(new QudaCloverMultiShiftInverter(
        opD.GaugeGrid(), qp_, spec));
    std::cout << GridLogMessage
              << "[OneFlavourSchurCloverQudaPrimitiveActionMP] "
              << "built with " << poles.size()
              << " rational shifts, matpc=EVEN_EVEN_ASYMMETRIC" << std::endl;
  }

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    auto &FermOp = this->FermOp;
    auto &PhiEven = this->PhiEven;
    auto &PowerNegHalf = this->PowerNegHalf;
    const int Npole = PowerNegHalf.poles.size();
    GridBase *fcbgrid = FermOp.FermionRedBlackGrid();
    GridBase *ggrid   = FermOp.GaugeGrid();

    FermOp.ImportGauge(U);
    quda_ms_->SetGauge(U);

    // Multishift solve via QUDA → MPhi_k (Even parity, mass-form X).
    std::vector<FermionField> MPhi_k(Npole, fcbgrid);
    quda_ms_->solve_rb_even(PhiEven, MPhi_k, /*make_resident=*/false);

    // ----------------------------------------------------------------------
    // Compute Y_k = M_pc_grid · MPhi_k via Grid (Even parity, mass-form).
    // Used directly by Path A; for Path D, we rescale by 2κ at pack time
    // since QUDA's internal P uses kappa-form (M_pc_kappa = 2κ·M_pc_grid).
    // ----------------------------------------------------------------------
    SchurDifferentiableOperator<ImplD> Mpc(FermOp);
    std::vector<FermionField> Y_k(Npole, fcbgrid);       // Y = Mpc·X for Path A
    std::vector<FermionField> Yd_k(Npole, fcbgrid);      // Yd = MpcDag·X for Path D pack
    for (int k = 0; k < Npole; ++k) {
      Mpc.Mpc(MPhi_k[k], Y_k[k]);
      Mpc.MpcDag(MPhi_k[k], Yd_k[k]);
    }
    const RealD two_kappa = 2.0 * quda_ms_->InvertParam().kappa;
    bool use_mdag = std::getenv("QUDA_FORCE_Y_MDAG") != nullptr;

    // ----------------------------------------------------------------------
    // HYBRID MODE: Grid Wilson (exact, Path A) + QUDA σ-piece (cos=0.989).
    // Skips computeCloverOprod (the structurally-mismatched primitive) and
    // skips Grid's MeeDeriv+MooDeriv (replaced by QUDA σ).  Phase B-prime.
    //
    // Activated by QUDA_FORCE_HYBRID=1.  Independent of QUDA_FORCE_KERNEL etc.
    // ----------------------------------------------------------------------
    bool hybrid_mode = std::getenv("QUDA_FORCE_HYBRID") != nullptr;
    // Phase D: full QUDA force (Wilson + σ).  Skips entire Path A loop in
    // production (compare mode still computes Path A as ground truth).
    bool hybrid_full_top = std::getenv("QUDA_FORCE_HYBRID_FULL") != nullptr;

    // ----------------------------------------------------------------------
    // Path A: Grid deriv chain.  Run when QUDA_FORCE_KERNEL_COMPARE is set
    // to compare against the new primitive path.
    // ----------------------------------------------------------------------
    bool path_a_compare = std::getenv("QUDA_FORCE_KERNEL_COMPARE") != nullptr;
    GaugeField dSdU_pathA(ggrid);
    GaugeField dSdU_pathA_wilson(ggrid);   // term split: Mpc Wilson-hop only (sum)
    GaugeField dSdU_pathA_wilsonMpc(ggrid);// term split: just MpcDeriv(Y,X)
    GaugeField dSdU_pathA_wilsonMpcDag(ggrid);// term split: just MpcDagDeriv(X,Y)
    GaugeField dSdU_pathA_mee(ggrid);      // term split: clover deriv on EVEN
    GaugeField dSdU_pathA_moo(ggrid);      // term split: clover deriv on ODD
    // Skip the entire Path A loop in production hybrid_full mode (Phase D).
    // In compare mode, still compute Path A as ground truth.
    bool skip_pathA_for_full = hybrid_full_top && !path_a_compare;
    if (!skip_pathA_for_full &&
        (hybrid_mode || std::getenv("QUDA_FORCE_KERNEL") == nullptr || path_a_compare)) {
      FermionField X(fcbgrid), Y(fcbgrid);
      GaugeField tmp(ggrid);
      dSdU = Zero();
      dSdU_pathA_wilson = Zero();
      dSdU_pathA_wilsonMpc = Zero();
      dSdU_pathA_wilsonMpcDag = Zero();
      dSdU_pathA_mee    = Zero();
      dSdU_pathA_moo    = Zero();
      for (int k = 0; k < Npole; ++k) {
        RealD ak = PowerNegHalf.residues[k];
        X = MPhi_k[k];
        Y = Y_k[k];

        // Wilson-hop part of M_pc derivative (∂(M_eo·M_oo^-1·M_oe)).
        Mpc.MpcDeriv(tmp, Y, X);     dSdU_pathA_wilsonMpc    = dSdU_pathA_wilsonMpc + ak * tmp;
        Mpc.MpcDagDeriv(tmp, X, Y);  dSdU_pathA_wilsonMpcDag = dSdU_pathA_wilsonMpcDag + ak * tmp;
        dSdU_pathA_wilson = dSdU_pathA_wilsonMpc + dSdU_pathA_wilsonMpcDag;

        // Skip MeeDeriv/MooDeriv when in hybrid mode (production) — QUDA σ
        // replaces them.  In compare mode, always compute full Path A to
        // serve as ground truth for the comparator.
        if (!hybrid_mode || path_a_compare) {
          // Clover deriv on EVEN parity (∂M_ee).
          FermOp.MeeDeriv(tmp, Y, X, DaggerNo);    dSdU_pathA_mee = dSdU_pathA_mee + ak * tmp;
          FermOp.MeeDeriv(tmp, X, Y, DaggerYes);   dSdU_pathA_mee = dSdU_pathA_mee + ak * tmp;

          // Clover deriv on ODD parity (∂M_oo with Schur-completed off-parity fields).
          FermionField W_o(fcbgrid), Z_o(fcbgrid), tmp1(fcbgrid);
          FermOp.Meooe(X, tmp1);          FermOp.MooeeInv(tmp1, W_o);
          FermOp.MeooeDag(Y, tmp1);       FermOp.MooeeInvDag(tmp1, Z_o);
          FermOp.MooDeriv(tmp, Z_o, W_o, DaggerNo);   dSdU_pathA_moo = dSdU_pathA_moo + ak * tmp;
          FermOp.MooDeriv(tmp, W_o, Z_o, DaggerYes);  dSdU_pathA_moo = dSdU_pathA_moo + ak * tmp;
        }
      }
      dSdU = dSdU_pathA_wilson + dSdU_pathA_mee + dSdU_pathA_moo;
      // In hybrid_mode: dSdU now holds Wilson piece only; we'll add QUDA σ below.
      // In Path-A-only mode (no compare, no kernel): return now.
      if (!path_a_compare && !hybrid_mode) return;
      if (path_a_compare) {
        std::cout << GridLogMessage
                  << "[QudaPrim] PathA dSdU norm2=" << norm2(dSdU)
                  << " |Wilson|²=" << norm2(dSdU_pathA_wilson)
                  << " |Mee|²="    << norm2(dSdU_pathA_mee)
                  << " |Moo|²="    << norm2(dSdU_pathA_moo)
                  << std::endl;
        dSdU_pathA = dSdU;
      }
    }

    // ----------------------------------------------------------------------
    // Path D: QudaForcePrimitives with Grid-Y injection.
    // ----------------------------------------------------------------------

    // Pack X_k and Y_k as parity-subset host buffers (Even parity, V_eo sites).
    int V_eo = Quda::local_volume(ggrid) / 2;
    using SiteSpinor = typename FermionField::scalar_object;
    static_assert(sizeof(SiteSpinor) == 24 * sizeof(double),
                  "expected 24 doubles/site for fermion");

    std::vector<std::vector<double>> x_bufs(Npole, std::vector<double>(24 * V_eo));
    std::vector<std::vector<double>> y_bufs(Npole, std::vector<double>(24 * V_eo));
    // Hybrid / Phase D: also pack Schur-completed off-parity fields W_o, Z_o
    // so QUDA's σ-Oprod and (Phase D) computeCloverOprod consume the same
    // bilinear inputs as Path A's Mee/Moo/Mpc derivative chain.
    bool hybrid_full_pack = std::getenv("QUDA_FORCE_HYBRID_FULL") != nullptr;
    bool wilson_probe_pack = std::getenv("QUDA_FORCE_WILSON_PROBE") != nullptr;
    bool need_schur_pack = hybrid_mode || hybrid_full_pack || wilson_probe_pack;
    std::vector<std::vector<double>> wo_bufs, zo_bufs;
    if (need_schur_pack) {
      wo_bufs.assign(Npole, std::vector<double>(24 * V_eo));
      zo_bufs.assign(Npole, std::vector<double>(24 * V_eo));
    }
    std::vector<void *> x_ptrs(Npole), y_ptrs(Npole);
    std::vector<void *> wo_ptrs(Npole), zo_ptrs(Npole);
    // QUDA_FORCE_NO_KAPPA_RESCALE skips the 2κ rescale on Y.
    // QUDA_FORCE_X_KAPPA_RESCALE applies 2κ to X̂ (kappa-form X̂).
    // QUDA_FORCE_X_RESCALE=<v> overrides the X̂ scale to a custom value.
    bool no_kappa_rescale = std::getenv("QUDA_FORCE_NO_KAPPA_RESCALE") != nullptr;
    bool x_kappa_rescale = std::getenv("QUDA_FORCE_X_KAPPA_RESCALE") != nullptr;
    RealD x_scale = x_kappa_rescale ? two_kappa : 1.0;
    if (const char *xs = std::getenv("QUDA_FORCE_X_RESCALE"); xs && *xs)
      x_scale = std::atof(xs);
    for (int k = 0; k < Npole; ++k) {
      FermionField X_kappa(fcbgrid), Y_kappa(fcbgrid);
      RealD y_scale = no_kappa_rescale ? 1.0 : two_kappa;
      X_kappa = x_scale * MPhi_k[k];
      Y_kappa = y_scale * (use_mdag ? Yd_k[k] : Y_k[k]);
      std::vector<SiteSpinor> sX, sY;
      unvectorizeToLexOrdArray(sX, X_kappa);
      unvectorizeToLexOrdArray(sY, Y_kappa);
      std::memcpy(x_bufs[k].data(), sX.data(), V_eo * 24 * sizeof(double));
      std::memcpy(y_bufs[k].data(), sY.data(), V_eo * 24 * sizeof(double));
      x_ptrs[k] = x_bufs[k].data();
      y_ptrs[k] = y_bufs[k].data();

      if (need_schur_pack) {
        // Compute Path A's exact Schur-completed off-parity fields:
        //   W_o = M_oo^{-1}·M_oe·X̂_e
        //   Z_o = M_oo^{-1†}·M_eo†·Y_e   (Y = M_pc·X̂_e, Path A's mass-form)
        // Empirical scale_off = √(2κ) gives b=1 in σ_quda = a·Mee + b·Moo
        // decomposition.  With scale=2κ on off-parity, b≈2κ (Moo undercounted);
        // with scale=1, b≈1/(2κ) (Moo overcounted).  So b = scale²/(2κ),
        // requiring scale = √(2κ).  Configurable via QUDA_FORCE_SCHUR_SCALE.
        const RealD sqrt_two_kappa = std::sqrt(two_kappa);
        RealD scale_off = sqrt_two_kappa;
        if (const char *s = std::getenv("QUDA_FORCE_SCHUR_SCALE"); s && *s)
          scale_off = std::atof(s);
        FermionField W_o(fcbgrid), Z_o(fcbgrid), tmp1(fcbgrid);
        FermOp.Meooe(MPhi_k[k], tmp1);     FermOp.MooeeInv(tmp1, W_o);
        FermOp.MeooeDag(Y_k[k], tmp1);     FermOp.MooeeInvDag(tmp1, Z_o);
        FermionField W_scaled(fcbgrid), Z_scaled(fcbgrid);
        W_scaled = scale_off * W_o;
        Z_scaled = scale_off * Z_o;
        std::vector<SiteSpinor> sW, sZ;
        unvectorizeToLexOrdArray(sW, W_scaled);
        unvectorizeToLexOrdArray(sZ, Z_scaled);
        std::memcpy(wo_bufs[k].data(), sW.data(), V_eo * 24 * sizeof(double));
        std::memcpy(zo_bufs[k].data(), sZ.data(), V_eo * 24 * sizeof(double));
        wo_ptrs[k] = wo_bufs[k].data();
        zo_ptrs[k] = zo_bufs[k].data();
      }
    }

    int V = Quda::local_volume(ggrid);
    constexpr int MOM_RECON = 10;
    std::vector<double> mom_buf(V * 4 * MOM_RECON, 0.0);

    // Set up gauge_param for the force-call mom output (matches
    // Phase 7 setup: GENERAL_LINKS, MILC gauge_order, RECONSTRUCT_NO,
    // overwrite_mom=1, return_result_mom=1).
    QudaInvertParam &inv_param = quda_ms_->InvertParam();
    int saved_use_resident = inv_param.use_resident_solution;
    QudaDagType saved_dagger = inv_param.dagger;
    QudaTwistFlavorType saved_twist = inv_param.twist_flavor;
    inv_param.use_resident_solution = 0;
    inv_param.dagger        = QUDA_DAG_NO;
    inv_param.twist_flavor  = QUDA_TWIST_NO;

    QudaGaugeParam force_gauge_param = quda_ms_->GaugeParam();
    force_gauge_param.type        = QUDA_GENERAL_LINKS;
    force_gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
    force_gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
    force_gauge_param.overwrite_mom     = 1;
    force_gauge_param.use_resident_mom  = 0;
    force_gauge_param.make_resident_mom = 0;
    force_gauge_param.return_result_mom = 1;

    // Coefficients: feed residues raw — wrapper internally applies
    // 2·dt·coeff·κ² scaling for the Wilson-hop force_coeff.
    std::vector<double> coeff(Npole);
    for (int k = 0; k < Npole; ++k) coeff[k] = PowerNegHalf.residues[k];

    const double kappa  = inv_param.kappa;
    const double kappa2 = -kappa * kappa;          // matches MILC convention
    const double ck     = -inv_param.clover_csw * kappa / 8.0;  // Phase 7 sign
    const double dt     = 1.0;
    const double sigma_trace_coeff = 0.0;          // Grid handles LogDet

    bool hybrid_full_mode = std::getenv("QUDA_FORCE_HYBRID_FULL") != nullptr;
    bool wilson_probe     = std::getenv("QUDA_FORCE_WILSON_PROBE") != nullptr;
    if (hybrid_full_mode) {
      // Phase D: full QUDA force (Wilson + σ) with Schur-completed off-parity.
      Quda::computeCloverFullForceWithSchurFields(
          mom_buf.data(),
          x_ptrs.data(), y_ptrs.data(),
          wo_ptrs.data(), zo_ptrs.data(),
          Npole, coeff,
          kappa2, ck, dt,
          sigma_trace_coeff,
          &force_gauge_param,
          &inv_param);
    } else if (wilson_probe) {
      // Phase D D.1: Wilson-hop only (skip σ).  Used in compare mode to
      // verify the Wilson-hop primitive matches Path A's MpcDeriv+MpcDagDeriv.
      Quda::computeCloverWilsonForceWithSchurFields(
          mom_buf.data(),
          x_ptrs.data(), y_ptrs.data(),
          wo_ptrs.data(), zo_ptrs.data(),
          Npole, coeff,
          kappa2, ck, dt,
          &force_gauge_param,
          &inv_param);
    } else if (hybrid_mode) {
      // Phase B-prime: σ-only force with Schur-completed fields.
      Quda::computeCloverSigmaForceWithSchurFields(
          mom_buf.data(),
          x_ptrs.data(), y_ptrs.data(),
          wo_ptrs.data(), zo_ptrs.data(),
          Npole, coeff,
          kappa2, ck, dt,
          sigma_trace_coeff,
          &force_gauge_param,
          &inv_param);
    } else {
      Quda::computeCloverForceWithGridY(
          mom_buf.data(),
          x_ptrs.data(), y_ptrs.data(),
          Npole, coeff,
          kappa2, ck, dt,
          sigma_trace_coeff,
          &force_gauge_param,
          &inv_param);
    }

    inv_param.use_resident_solution = saved_use_resident;
    inv_param.dagger                = saved_dagger;
    inv_param.twist_flavor          = saved_twist;

    // Diagnostic: mom_buf NaN check + first10 values.
    {
      double mom_norm = 0.0;
      int n_nan = 0;
      for (size_t i = 0; i < mom_buf.size(); ++i) {
        if (std::isnan(mom_buf[i])) ++n_nan;
        else mom_norm += mom_buf[i] * mom_buf[i];
      }
      std::cout << GridLogMessage
                << "[QudaPrim] mom_buf size=" << mom_buf.size()
                << " norm2(non-NaN)=" << mom_norm << " n_nan=" << n_nan
                << " first10=";
      for (int i = 0; i < 10; ++i) std::cout << mom_buf[i] << " ";
      std::cout << std::endl;
    }

    // Unpack QUDA's MILC mom layout → Grid's anti-Hermitian gauge field.
    // Same per-link layout as Phase 7's class.
    Coordinate lc = ggrid->LocalDimensions();
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
    GaugeField dSdU_quda(ggrid);
    Quda::lex_buffers_to_gauge(lex_ptrs, dSdU_quda);

    // Sign + scale (Phase 7 conventions).
    const double quda_to_grid_factor = -1.0 / (8.0 * kappa * kappa);
    dSdU_quda = quda_to_grid_factor * dSdU_quda;

    if (hybrid_full_top && !path_a_compare) {
      // Phase D production: dSdU is Zero (Path A skipped).  Replace with QUDA.
      dSdU = dSdU_quda;
      return;
    }
    if (hybrid_mode && !path_a_compare) {
      // Production hybrid (σ-only): dSdU has Wilson only.  Add σ.
      dSdU = dSdU + dSdU_quda;
      return;
    }
    if ((hybrid_mode || hybrid_full_top) && path_a_compare) {
      // Compare hybrid/full: dSdU currently holds full Path A.  Below the
      // comparator runs against dSdU_quda (= σ-only or Wilson-only or full
      // depending on mode).  Output stays Path A by default.
    } else {
      // Non-hybrid Path D: replace dSdU with the QUDA result.
      dSdU = dSdU_quda;
    }

    if (path_a_compare) {
      // Compare against PathA via Ta projection.  Report cos vs full PathA
      // and vs each component (Wilson-hop / Mee / Moo / Mee+Moo) so that
      // when QUDA_FORCE_SKIP_OPROD or QUDA_FORCE_SKIP_SIGMA is set, we can
      // localize which piece of QUDA primitives matches which Path A piece.
      auto Ta_of = [&](const GaugeField &G) {
        GaugeField T(ggrid);
        for (int mu = 0; mu < Nd; ++mu) {
          PokeIndex<LorentzIndex>(T, Ta(PeekIndex<LorentzIndex>(G, mu)), mu);
        }
        return T;
      };
      GaugeField TaA       = Ta_of(dSdU_pathA);
      GaugeField TaA_w     = Ta_of(dSdU_pathA_wilson);
      GaugeField TaA_wMpc  = Ta_of(dSdU_pathA_wilsonMpc);     // MpcDeriv(Y,X) only
      GaugeField TaA_wMpcD = Ta_of(dSdU_pathA_wilsonMpcDag);  // MpcDagDeriv(X,Y) only
      GaugeField TaA_mee   = Ta_of(dSdU_pathA_mee);
      GaugeField TaA_moo   = Ta_of(dSdU_pathA_moo);
      GaugeField TaA_clov(ggrid);
      TaA_clov = TaA_mee + TaA_moo;
      // For comparator: in hybrid/hybrid_full/wilson_probe + compare mode,
      // dSdU holds full Path A (computed for ground truth).  We want to
      // compare the QUDA output (σ-only / Wilson-only / full) against
      // Path A's pieces — use dSdU_quda directly.
      const bool quda_substituted = hybrid_mode || hybrid_full_top
                                  || std::getenv("QUDA_FORCE_WILSON_PROBE") != nullptr;
      const GaugeField &dSdU_compare = quda_substituted ? dSdU_quda : dSdU;
      double n2D = norm2(dSdU_compare);
      auto report = [&](const char *name, const GaugeField &T) {
        double n2T = norm2(T);
        auto in = innerProduct(T, dSdU_compare);
        double cos = (n2T > 0 && n2D > 0) ? real(in)/std::sqrt(n2T*n2D) : 0.0;
        std::cout << GridLogMessage
                  << "[QudaPrim/term] " << name
                  << "  |Ta|²=" << n2T
                  << "  |D|²=" << n2D
                  << "  ⟨Ta,D⟩=" << real(in)
                  << "  factor=" << (n2T>0 ? real(in)/n2T : 0.0)
                  << "  cos=" << cos
                  << std::endl;
      };
      report("Ta(A)full ", TaA);
      report("Ta(A)wils ", TaA_w);
      report("Ta(A)wMpc ", TaA_wMpc);   // MpcDeriv(Y,X) only
      report("Ta(A)wMpcD", TaA_wMpcD);  // MpcDagDeriv(X,Y) only
      report("Ta(A)mee  ", TaA_mee);
      report("Ta(A)moo  ", TaA_moo);
      report("Ta(A)clov ", TaA_clov);
      // By default, output PathA so FD test passes.  Set
      // QUDA_FORCE_OUTPUT_PATHD=1 to instead OUTPUT PathD (so FD test can
      // grade PathD itself).
      if (std::getenv("QUDA_FORCE_OUTPUT_PATHD") == nullptr) {
        dSdU = dSdU_pathA;  // override with PathA's correct force
      }
    }
  }

 private:
  QudaCloverParams qp_;
  std::unique_ptr<QudaCloverMultiShiftInverter> quda_ms_;
};

}  // namespace Grid
