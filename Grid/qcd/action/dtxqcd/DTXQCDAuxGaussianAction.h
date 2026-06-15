#pragma once
// Gaussian prior for the v2 DTXQCD auxiliary fields (dtxqcd_v2.tex
// Eqs. 116-128 with the four-quark cross terms integrated out):
//
//   S_aux = (lambda^2 / 2) sum_x [ Tr(sigma^2) + Tr(pi^2)
//                                + Tr(d^2)     + Tr(n^2) ]
//         + (lambda^2 / 4) sum_x [ s^2 + p^2 ]
//
// 2026-06-15: halved coefficient for s, p following the (s, Tr σ)
// redundancy resolution — see project_dtxqcd_s_sigma_redundancy.md.
// σ, π are simultaneously enforced traceless (in CompositeImpl) so the
// single-field Pf coupling has the natural saddle ⟨s⟩ = 2·N_F·Σ/λ².
//
// Sum-of-squares, manifestly positive (the v1 negative-tensor problem is
// gone -- the t channel has been Fierz-eliminated in the v2 derivation).
//
// Storage / inner product conventions:
//   sigma, pi, d, n : color-flavor 6x6 Hermitian traceless matrices in
//                     iScalar<iMatrix<iMatrix<vComplex, Nc>, Nf>>.  Grid's
//                     norm2 sums all NfNc^2 = 36 complex entries; for
//                     Hermitian M this equals Tr(M^dag M) = Tr(M^2).
//   s, p            : singlet scalars (real DOF in real part of vComplex).
//                     norm2 sums (re)^2 + (im)^2, but imag is kept at zero
//                     by DtxqcdRealScalarProjectInPlace, so norm2 = s^2.
//
// dS/dX = lambda^2 * X on each Hermitian aux slot (with the same
// natural-Wirtinger transpose() that v1 used for d, n -- carries
// over verbatim for the 6x6 case); dS/ds = lambda^2 * s, dS/dp = lambda^2 * p.
// Gauge slot is zero.

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
    RealD n2_matrix = norm2(Unc.sigma)
                    + norm2(Unc.pi)
                    + norm2(Unc.d)
                    + norm2(Unc.n);
    RealD n2_scalar = norm2(Unc.s) + norm2(Unc.p);
    return 0.5  * lambda * lambda * n2_matrix
         + 0.25 * lambda * lambda * n2_scalar;
  }

  void deriv(const DTXQCDField &U, DTXQCDField &dSdU) override {
    const RealD c_matrix = lambda * lambda;        // dS/dX = λ² X for σ, π, d, n
    const RealD c_scalar = 0.5 * lambda * lambda;  // dS/ds = (λ²/2) s for s, p
    dSdU.U     = Zero();
    // For Hermitian CF matrix sigma_ab^ij the action is
    //   S = (lambda^2/2) * Sum_{a,b,i,j} |sigma_ab^ij|^2
    // and the gradient under the integrator's update P -= F * eps (which
    // expects F = dS/dX as the actual position-space gradient) is
    //   dS/dsigma_ab^ij = lambda^2 * sigma_ab^ij,
    // i.e. F = lambda^2 * X, NOT lambda^2 * transpose(X).  The pre-fix
    // code used transpose, which is conj() on Hermitian fields; that
    // flipped the imaginary part of the off-diagonal force.  The
    // resulting integrator was non-symplectic and dH leaked linearly in
    // eps even though the FD-vs-analytic test passed (the test used the
    // same transpose convention in its inner product, so both sides
    // were self-consistent with each other but not with the true
    // gradient flow that the HMC integrator runs).
    dSdU.sigma = c_matrix * U.sigma;
    dSdU.pi    = c_matrix * U.pi;
    dSdU.d     = c_matrix * U.d;
    dSdU.n     = c_matrix * U.n;
    dSdU.s     = c_scalar * U.s;
    dSdU.p     = c_scalar * U.p;
  }

 private:
  RealD lambda;
};

NAMESPACE_END(Grid);
