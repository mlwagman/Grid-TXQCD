#pragma once
// Log-determinant action for the even-site diagonal block in the
// EO-preconditioned Wilson-Clover operator.
//
// det(M) = det(Mee) * det(Mpc)
//
// For Nf=2: S_logdet = -ln det(Mee†Mee) = -2 Σ_{x∈even} ln|det(Mee(x))|
//
// This is paired with TwoFlavourSchurCloverAction for the Schur complement.
//
// The force arises from the gauge dependence of the clover term on even sites,
// computed via Cmunu staples (same machinery as WilsonCloverFermion::MDeriv).

#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

NAMESPACE_BEGIN(Grid);

template <class Impl, class CloverHelpers = Grid::CloverHelpers<Impl>>
class QCDLogDetCloverEOAction : public Action<typename Impl::GaugeField> {
public:
  INHERIT_IMPL_TYPES(Impl);
  INHERIT_CLOVER_TYPES(Impl);

  typedef WilsonCloverFermion<Impl, CloverHelpers> FermionOperator;

  QCDLogDetCloverEOAction(FermionOperator &Op, int nf = 2)
      : FermOp(Op), Nf(nf) {}

  std::string action_name() override { return "QCDLogDetCloverEOAction"; }

  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=" << FermOp.diag_mass - 4.0
       << " csw_r=" << 2.0 * FermOp.csw_r << " csw_t=" << 2.0 * FermOp.csw_t << std::endl;
    return os.str();
  }

  void refresh(const GaugeField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
  }

  RealD S(const GaugeField &U) override {
    FermOp.ImportGauge(U);

    int DimRep = Impl::Dimension;
    int lvol = FermOp.CloverTermEven.Grid()->lSites();

    std::vector<typename SiteClover::scalar_object> ct_lex(lvol);
    unvectorizeToLexOrdArray(ct_lex, FermOp.CloverTermEven);

    RealD logdet = 0.0;
    for (int site = 0; site < lvol; ++site) {
      Eigen::MatrixXcd EigenM = Eigen::MatrixXcd::Zero(Ns * DimRep, Ns * DimRep);
      for (int j = 0; j < Ns; j++)
        for (int k = 0; k < Ns; k++)
          for (int a = 0; a < DimRep; a++)
            for (int b = 0; b < DimRep; b++)
              EigenM(a + j * DimRep, b + k * DimRep) =
                  std::complex<double>(ct_lex[site]()(j, k)(a, b));
      logdet += std::log(std::abs(EigenM.determinant()));
    }

    FermOp.GaugeGrid()->GlobalSum(logdet);
    RealD action = -RealD(Nf) * logdet;
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action << std::endl;
    return action;
  }

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    FermOp.ImportGauge(U);

    GridBase *fgrid = FermOp.GaugeGrid();

    // Mee^{-1} on even sites, zero on odd — acts as the "propagator" Lambda.
    CloverField Lambda(fgrid);
    Lambda = Zero();
    setCheckerboard(Lambda, FermOp.CloverTermInvEven);

    std::vector<GaugeLinkField> Ulinks(Nd, fgrid);
    for (int mu = 0; mu < Nd; ++mu)
      Ulinks[mu] = PeekIndex<LorentzIndex>(FermOp.Umu, mu);

    // Sigma loop: same structure as WilsonCloverFermion::MDeriv.
    // Uses 12-element sigma array (including MinusSigma for nu<mu pairs).
    Gamma::Algebra sigma[] = {
        Gamma::Algebra::SigmaXY,
        Gamma::Algebra::SigmaXZ,
        Gamma::Algebra::SigmaXT,
        Gamma::Algebra::MinusSigmaXY,
        Gamma::Algebra::SigmaYZ,
        Gamma::Algebra::SigmaYT,
        Gamma::Algebra::MinusSigmaXZ,
        Gamma::Algebra::MinusSigmaYZ,
        Gamma::Algebra::SigmaZT,
        Gamma::Algebra::MinusSigmaXT,
        Gamma::Algebra::MinusSigmaYT,
        Gamma::Algebra::MinusSigmaZT};

    GaugeLinkField force_mu(fgrid), lambda(fgrid);
    GaugeField clover_force(fgrid);
    int count = 0;
    clover_force = Zero();
    for (int mu = 0; mu < 4; mu++) {
      force_mu = Zero();
      for (int nu = 0; nu < 4; nu++) {
        if (mu == nu) continue;

        RealD factor = (nu == 3 || mu == 3) ? 2.0 * FermOp.csw_t
                                             : 2.0 * FermOp.csw_r;
        CloverField Slambda = Gamma(sigma[count]) * Lambda;
        lambda = Zero();
        for (int s = 0; s < Ns; ++s)
          lambda = lambda + PeekIndex<SpinIndex>(Slambda, s, s);
        force_mu -= factor * CloverHelpers::Cmunu(Ulinks, lambda, mu, nu);
        count++;
      }
      pokeLorentz(clover_force, Ulinks[mu] * force_mu, mu);
    }

    // S = -Nf * ln|det Mee|.
    // clover_force = Tr(Mee^{-1} dMee/dU) in UdSdU convention.
    // Grid pseudofermion convention: UdSdU = -dS/dU (force opposes gradient).
    // dS/dU = -Nf * clover_force, so UdSdU = +Nf * clover_force.
    dSdU = RealD(Nf) * clover_force;
  }

private:
  FermionOperator &FermOp;
  int Nf;
};

NAMESPACE_END(Grid);
