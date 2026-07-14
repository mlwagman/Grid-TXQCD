#pragma once
// Common interface for the QUDA-backed inner solvers usable as a
// TwoFlavourSchurCloverRatioActionQuda DerivativeSolver/ActionSolver:
// QudaMGSchurSolver (MG, full-volume zero-pad + gamma5 two-solve, needed
// because QUDA's MG-as-preconditioner only supports QUDA_DIRECT_SOLVE) and
// QudaCGSchurSolver (plain CG, direct half-volume Schur solve via
// QudaCloverMultiShiftInverter). Both need re-uploading the gauge on every
// smearing update; SetGauge() is not part of Grid's OperatorFunction
// interface, so this adds it as a virtual, letting
// TwoFlavourSchurCloverRatioActionQuda hold either concrete type behind one
// reference without a template parameter.

#include <Grid/Grid.h>
#include <Grid/algorithms/LinearOperator.h>

namespace Grid {

class QudaRungSolverBase : public OperatorFunction<LatticeFermion> {
 public:
  virtual void SetGauge(const LatticeGaugeField &U) = 0;
  virtual ~QudaRungSolverBase() = default;
};

}  // namespace Grid
