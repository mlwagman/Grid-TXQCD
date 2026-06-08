#pragma once
// Gaussian prior for the DTXQCD auxiliary fields (paper Eqs. 268-273):
//
//   S_aux = (lambda^2 / 2) sum_x [ sigma^A sigma^A + pi^A pi^A
//                                + (1/2) t^A_{mu,nu} t^A_{mu,nu}
//                                + Tr(d^2) + Tr(n^2) ]
//
// Storage / inner product conventions:
//   sigma^A, pi^A : real triplet stored in iVector<vComplex, 3>; the (real)
//                   variance contributes via norm2 directly.
//   t^A_{mu,nu}   : antisymmetric in (mu,nu) and a triplet in A.  Grid's
//                   norm2 sums both (mu<nu) and (mu>nu); each unique pair
//                   appears squared twice -- so 0.5 * norm2(t) gives the
//                   sum sum_{mu<nu, A} (t^A_{mu,nu})^2 we want, and the
//                   action's overall (1/2) makes the t coefficient 1/4
//                   * norm2(t) = (1/2) * sum_{mu<nu} t^2.
//   d, n          : Hermitian color matrices.  norm2(Hermitian) =
//                   Sum_i d_ii^2 + 2 Sum_{i<j} |d_ij|^2 = Tr(d^2) exactly
//                   (real-valued for Hermitian), so norm2 IS the trace.
//
// dS/dX = lambda^2 * X on each aux slot; gauge slot is zero.  Acts on the
// composite DTXQCDField so it can sit at the innermost MD level alongside
// the pseudofermion + log-det actions.  Direct port of TXQCD's
// AuxiliaryFieldGaussianAction with the field roster updated for the
// diquark-tensor variant.

#include <Grid/qcd/action/dtxqcd/DTXQCDField.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>

NAMESPACE_BEGIN(Grid);

class DTXQCDAuxiliaryFieldGaussianAction : public Action<DTXQCDField> {
 public:
  explicit DTXQCDAuxiliaryFieldGaussianAction(RealD lambda_) : lambda(lambda_) {}

  std::string action_name() override {
    return "DTXQCDAuxiliaryFieldGaussianAction";
  }

  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage
       << "[DTXQCDAuxiliaryFieldGaussianAction] lambda = " << lambda
       << std::endl;
    return os.str();
  }

  void refresh(const DTXQCDField &, GridSerialRNG &,
               GridParallelRNG &) override {}

  RealD S(const DTXQCDField &U) override {
    DTXQCDField &Unc = const_cast<DTXQCDField &>(U);
    // The factor 0.5 on the t term undoes Grid's norm2 double-count over
    // (mu, nu) vs (nu, mu) for the antisymmetric tensor -- so that the
    // outer (lambda^2 / 2) gives the action's (lambda^2 / 2) sum_{mu<nu} t^2.
    RealD n2 = norm2(Unc.sigma)
             + norm2(Unc.pi)
             + 0.5 * norm2(Unc.t)
             + norm2(Unc.d)
             + norm2(Unc.n);
    return 0.5 * lambda * lambda * n2;
  }

  void deriv(const DTXQCDField &U, DTXQCDField &dSdU) override {
    const RealD c = lambda * lambda;
    dSdU.U     = Zero();
    dSdU.sigma = c * U.sigma;
    dSdU.pi    = c * U.pi;
    // t coefficient is c (not 2c) because Grid's norm2(t) already includes
    // BOTH (mu,nu) and (nu,mu) of the antisymmetric tensor -- d/dt(t_{mu,nu})
    // of (1/2) norm2(t) = t_{mu,nu}, so the action coefficient (lambda^2/2)
    // * (1/2 norm2) gives dS/dt = (lambda^2 / 2) * t.  Following the
    // norm2-only convention here for consistency with the FD test's
    // AuxInnerReal which uses the same 0.5 fold.
    dSdU.t     = c * U.t;
    // For Hermitian d, n the natural-Wirtinger convention (matching
    // AuxInnerReal's Re trace(F * Y^T)) requires the transpose: F_kl gets
    // d/dd_kl(0.5 lambda^2 |d_kl|^2) under the Hermitian constraint, which
    // for d_lk = conj(d_kl) folds in BOTH the d_kl and d_lk Wirtinger
    // contributions and gives F_kl = lambda^2 conj(d_kl) = lambda^2 d_lk
    // = lambda^2 transpose(d)_kl.  Without the transpose, AuxInnerReal /
    // the integrator picks up the wrong sign on the off-diagonal imaginary
    // (anti-symmetric) part -- the Gaussian FD test catches this.
    dSdU.d     = c * transpose(U.d);
    dSdU.n     = c * transpose(U.n);
  }

 private:
  RealD lambda;
};

NAMESPACE_END(Grid);
