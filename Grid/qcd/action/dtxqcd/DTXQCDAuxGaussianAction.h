#pragma once
// Gaussian prior for the v2 DTXQCD auxiliary fields (dtxqcd_v2.tex
// Eqs. 116-128 with the four-quark cross terms integrated out):
//
//   S_aux = (lambda^2 / 2) sum_x [ Tr(sigma^2) + Tr(pi^2)
//                                + Tr(d^2)     + Tr(n^2)
//                                + s^2 + p^2 ]
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
    RealD n2 = norm2(Unc.sigma)
             + norm2(Unc.pi)
             + norm2(Unc.d)
             + norm2(Unc.n)
             + norm2(Unc.s)
             + norm2(Unc.p);
    return 0.5 * lambda * lambda * n2;
  }

  void deriv(const DTXQCDField &U, DTXQCDField &dSdU) override {
    const RealD c = lambda * lambda;
    dSdU.U     = Zero();
    // For Hermitian matrix fields, the natural-Wirtinger derivative of
    // (1/2)|M_kl|^2 (matched against an AuxInnerReal that uses Re Tr(F Y^T))
    // requires transpose(M) on the right side -- same convention v1 used for
    // d, n and now extended to sigma, pi for the 6x6 CF Hermitian case.
    dSdU.sigma = c * transpose(U.sigma);
    dSdU.pi    = c * transpose(U.pi);
    dSdU.d     = c * transpose(U.d);
    dSdU.n     = c * transpose(U.n);
    // Singlet scalars: real-valued, no transpose subtlety; dS/ds = lambda^2 s.
    dSdU.s     = c * U.s;
    dSdU.p     = c * U.p;
  }

 private:
  RealD lambda;
};

NAMESPACE_END(Grid);
