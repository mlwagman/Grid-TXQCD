#pragma once
// Gaussian prior for the TXQCD auxiliary fields (notes Eq. 4):
//
//   S_aux[sigma,pi,s,p,t] = (lambda^2 / 2) sum_x { Tr sigma^2 + Tr pi^2
//                                                 + Tr s^2 + Tr p^2
//                                                 + sum_{mu<nu} Tr t_{mu,nu}^2 }.
//
// dS/dX = lambda^2 * X on each aux slot; gauge slot is zero. Acts on the
// composite TXQCDField so it can sit at the innermost MD level alongside the
// pseudofermion action.

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>

NAMESPACE_BEGIN(Grid);

class AuxiliaryFieldGaussianAction : public Action<TXQCDField> {
 public:
  explicit AuxiliaryFieldGaussianAction(RealD lambda_) : lambda(lambda_) {}

  std::string action_name() override { return "AuxiliaryFieldGaussianAction"; }

  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage
       << "[AuxiliaryFieldGaussianAction] lambda = " << lambda << std::endl;
    return os.str();
  }

  void refresh(const TXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {}

  RealD S(const TXQCDField &U) override {
    TXQCDField &Unc = const_cast<TXQCDField &>(U);
    RealD n2 = HermitianFieldSquareNorm(Unc.sigma)
             + HermitianFieldSquareNorm(Unc.pi)
             + HermitianFieldSquareNorm(Unc.s)
             + HermitianFieldSquareNorm(Unc.p)
             + TensorFieldSquareNorm(Unc.t);
    return 0.5 * lambda * lambda * n2;
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    RealD c = lambda * lambda;
    dSdU.U = Zero();
    dSdU.sigma = c * U.sigma;
    dSdU.pi    = c * U.pi;
    dSdU.s     = c * U.s;
    dSdU.p     = c * U.p;
    dSdU.t     = c * U.t;
  }

 private:
  RealD lambda;
};

NAMESPACE_END(Grid);
