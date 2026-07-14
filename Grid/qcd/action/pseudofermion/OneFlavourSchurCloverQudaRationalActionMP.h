#pragma once
// QUDA-backed variant of OneFlavourSchurCloverRationalActionMP.
//
// Overrides deriv() to call QudaCloverMultiShiftInverter for the inner
// rational multishift CG.  Everything else (refresh, S, the rest of the
// force assembly) is inherited from the base.
//
// Only the inner multishift gets accelerated — the outer force-assembly
// gauge derivatives still go through Grid's MpcDeriv / MeooeDeriv chain.
// This is the right granularity: 95% of HMC time is the multishift;
// the gauge-deriv chain is cheap and would require porting Grid's
// derivative ops to QUDA to accelerate further.

#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>

namespace Grid {

template <class ImplD, class ImplF,
          class FermOpD_ = WilsonCloverFermion<ImplD, CloverHelpers<ImplD>>,
          class FermOpF_ = WilsonCloverFermion<ImplF, CloverHelpers<ImplF>>>
class OneFlavourSchurCloverQudaRationalActionMP
    : public OneFlavourSchurCloverRationalActionMP<ImplD, ImplF, FermOpD_, FermOpF_> {
 public:
  typedef OneFlavourSchurCloverRationalActionMP<ImplD, ImplF, FermOpD_, FermOpF_> Base;
  typedef typename Base::FermionField FermionField;
  typedef FermOpD_ FermOpD;
  typedef FermOpF_ FermOpF;
  typedef typename ImplD::GaugeField GaugeField;

  OneFlavourSchurCloverQudaRationalActionMP(
      FermOpD &opD, FermOpF &opF, GridBase *sp_rbgrid,
      OneFlavourRationalParams &p,
      const QudaCloverParams &qp, int reliable_update_freq = 50)
    : Base(opD, opF, sp_rbgrid, p, reliable_update_freq), qp_(qp) {

    // Build the QUDA multishift inverter from the rational poles.  Needs
    // matpc_type=ODD_ODD to match Grid's SchurDifferentiableOperator which
    // asserts U.Checkerboard()==Odd in its MpcDeriv.
    QudaCloverMultiShiftSpec spec;
    // ASYMMETRIC matches Grid's SchurDiagMooee form M_pc = M_oo − M_oe·M_ee^-1·M_eo.
    // The plain ODD_ODD ("symmetric" in QUDA) is M_oo^-1·M_pc_grid which gives a
    // different M_pc^†M_pc spectrum and produces forces that don't FD-match
    // Grid's deriv (verified empirically: ~46× ratio with symmetric vs ~1.0 with asym).
    spec.matpc_type = QUDA_MATPC_ODD_ODD_ASYMMETRIC;
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
              << "[OneFlavourSchurCloverQudaRationalActionMP] built with "
              << poles.size() << " rational shifts, matpc=ODD_ODD"
              << std::endl;
  }

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    auto &FermOp = this->FermOp;
    auto &PhiOdd = this->PhiOdd;
    auto &PowerNegHalf = this->PowerNegHalf;
    const int Npole = PowerNegHalf.poles.size();

    GridBase *fcbgrid = FermOp.FermionRedBlackGrid();
    GridBase *ggrid   = FermOp.GaugeGrid();

    std::vector<FermionField> MPhi_k(Npole, fcbgrid);
    FermionField X(fcbgrid), Y(fcbgrid);
    GaugeField tmp(ggrid);

    FermOp.ImportGauge(U);
    quda_ms_->SetGauge(U);  // re-upload after each MD step

    SchurDifferentiableOperator<ImplD> Mpc(FermOp);

    // QUDA multishift on the odd-parity Schur operator.  Direct half-volume
    // pack: Grid's RB cb-site order matches QUDA's cb_site = full_lex >> 1
    // within a parity, so unvectorize → memcpy into the odd-half slab of
    // QUDA's EO buffer.  Avoids the full-volume setCheckerboard
    // intermediate (which fails on SIMD layouts where outer blocks span
    // both parities).
    quda_ms_->solve_rb_odd(PhiOdd, MPhi_k);

    // Rest of the force assembly: identical to base class deriv().
    dSdU = Zero();
    for (int k = 0; k < Npole; k++) {
      RealD ak = PowerNegHalf.residues[k];
      X = MPhi_k[k];
      Mpc.Mpc(X, Y);

      Mpc.MpcDeriv(tmp, Y, X);          dSdU = dSdU + ak * tmp;
      Mpc.MpcDagDeriv(tmp, X, Y);       dSdU = dSdU + ak * tmp;
      FermOp.MooDeriv(tmp, Y, X, DaggerNo);  dSdU = dSdU + ak * tmp;
      FermOp.MooDeriv(tmp, X, Y, DaggerYes); dSdU = dSdU + ak * tmp;

      FermionField W_e(fcbgrid), Z_e(fcbgrid), tmp1(fcbgrid);
      FermOp.Meooe(X, tmp1);      FermOp.MooeeInv(tmp1, W_e);
      FermOp.MeooeDag(Y, tmp1);   FermOp.MooeeInvDag(tmp1, Z_e);
      FermOp.MeeDeriv(tmp, Z_e, W_e, DaggerNo);  dSdU = dSdU + ak * tmp;
      FermOp.MeeDeriv(tmp, W_e, Z_e, DaggerYes); dSdU = dSdU + ak * tmp;
    }
  }

 private:
  QudaCloverParams qp_;
  std::unique_ptr<QudaCloverMultiShiftInverter> quda_ms_;
};

}  // namespace Grid
