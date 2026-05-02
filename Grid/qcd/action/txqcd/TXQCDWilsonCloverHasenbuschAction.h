#pragma once
// Hasenbusch mass preconditioning (Nf=1 rational-ratio form) for the TXQCD
// Wilson-Clover pseudofermion:
//
//   S_ratio = Phi^dag  (V^dag V)^{1/4}  (M^dag M)^{-1/2}  (V^dag V)^{1/4}  Phi
//
// where:
//   M = M_light = TXQCDWilsonCloverFermionEO at mass = m_light   (denominator)
//   V = M_heavy = TXQCDWilsonCloverFermionEO at mass = m_heavy   (numerator)
//
// Choosing m_heavy > m_light (i.e. m_heavy less-negative, further from
// the critical hopping parameter), the combined HMC weight
//
//   |det M_light| = |det M_heavy| * (|det M_light| / |det M_heavy|)
//
// splits into two monomials: this Hasenbusch ratio (weight
// |det M_light|/|det M_heavy|) and a plain rational at m_heavy
// (weight |det M_heavy|, implemented by the existing
// TXQCDWilsonCloverRationalEOAction class at m_heavy).  The ratio
// force is reduced roughly by (m_heavy - m_light)^2 relative to the
// original single-monomial rational force -- the standard Hasenbusch
// force reduction.
//
// Implementation follows Grid's GeneralEvenOddRatioRationalPseudoFermionAction
// (Grid/qcd/action/pseudofermion/GeneralEvenOddRationalRatio.h), adapted to
// the TXQCD composite field: force contributions go to BOTH the gauge slot
// (via TXQCDWilsonCloverFermionEO::Wilson().M{eo,oe}Deriv and clover-Cmunu)
// and the aux slots (via per-site scalar bilinear pattern, same as the
// existing TXQCDWilsonCloverRationalEOAction::AccumulateAuxForce).
//
// Validation status (tests/txqcd/Test_txqcd_hasenbusch_force.cc, 4^4 lattice):
//   csw=0, Δm=0.2:  all 6 force-FD checks PASS at rel ~1e-5.
//   csw=1, Δm=0:    sanity -- all forces ~5e-5 (rational residual), PASS.
//   csw=1, Δm=0.2:  σ, π, s, p, t all PASS at rel ~1e-5;
//                   gauge PASS at rel ~2e-7 (matches rational-action reference).
// Clover gauge force uses the hand-rolled isig·(Y*·X + X*·Y) pattern (same
// code as TXQCDWilsonCloverRationalEOAction's validated csw!=0 path) for both
// the symmetric (part 1) and asymmetric (parts 2+3) bilinear structures.
//
// Refresh:
//   eta ~ Gaussian; eta is on the odd sub-lattice
//   tmp  = (M^dag M)^{1/4} eta                     [multishift rational]
//   Phi  = (V^dag V)^{-1/4} tmp                    [multishift rational]
//   => <Phi| (V^dag V)^{1/4} (M^dag M)^{-1/2} (V^dag V)^{1/4} |Phi>
//      = <eta|eta>  (Gaussian-sampled action stored as RefreshAction)
//
// Action:
//   X = (V^dag V)^{1/4} Phi
//   Y = (M^dag M)^{-1/4} X
//   S = |Y|^2
//
// Derivative (with  n_f = degree of (M^dag M)^{-1/2},
//                   n_pv= degree of (V^dag V)^{+1/4}):
//
//   X_k    = (V^dag V + b_pv_k)^{-1} Phi                [npv poles]
//   MpvPhi = sum_k a_pv_k X_k     ~= (V^dag V)^{1/4} Phi
//   W_k    = (M^dag M + b_f_k)^{-1} MpvPhi              [nf poles]
//   MfMpvPhi = sum_k a_f_k W_k   ~= (M^dag M)^{-1/2} MpvPhi
//   Z_k    = (V^dag V + b_pv_k)^{-1} MfMpvPhi           [npv poles]
//
//   dS/dU = -sum_k a_f_k  W_k^dag  [dM^dag M + M^dag dM]  W_k
//           -sum_k a_pv_k Z_k^dag  [dV^dag V + V^dag dV]  X_k
//           -sum_k a_pv_k X_k^dag  [dV^dag V + V^dag dV]  Z_k
//
// Each [dM^dag M + M^dag dM] (Schur version) yields both gauge-link force and
// aux-slot forces.  We reuse the same per-pole bilinear routines that
// TXQCDWilsonCloverRationalEOAction uses, applied twice (once to M-level
// residues, once to V-level residues).

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDCloverSchurOp.h>
#include <Grid/qcd/action/txqcd/TXQCDSolvers.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonPseudoFermionAction.h>
#include <Grid/qcd/action/fermion/WilsonCloverHelpers.h>

NAMESPACE_BEGIN(Grid);

class TXQCDWilsonCloverHasenbuschAction : public Action<TXQCDField> {
 public:
  typedef OneFlavourRationalParams Params;

  // mu rescales Delta in the inner TXQCDWilsonCloverFermionEO operators and
  // the aux-field forces.  Default mu=1 reproduces the original action.
  TXQCDWilsonCloverHasenbuschAction(GridCartesian &grid,
                                    GridRedBlackCartesian &rbgrid,
                                    RealD mass_light, RealD mass_heavy,
                                    Params &p, RealD csw = 0.0,
                                    RealD mu = 1.0)
      : grid_(grid), rbgrid_(rbgrid),
        mass_l_(mass_light), mass_h_(mass_heavy), csw_(csw), mu_(mu),
        param(p), Phi(&rbgrid) {
    AlgRemez remez(param.lo, param.hi, param.precision);
    std::cout << GridLogMessage
              << "[TXQCDWilsonCloverHasenbusch] degree " << param.degree
              << " rational for x^(1/2)" << std::endl;
    remez.generateApprox(param.degree, 1, 2);
    PowerHalf.Init(remez, param.tolerance, false);
    PowerNegHalf.Init(remez, param.tolerance, true);
    std::cout << GridLogMessage
              << "[TXQCDWilsonCloverHasenbusch] degree " << param.degree
              << " rational for x^(1/4)" << std::endl;
    remez.generateApprox(param.degree, 1, 4);
    PowerQuarter.Init(remez, param.tolerance, false);
    PowerNegQuarter.Init(remez, param.tolerance, true);
  }

  std::string action_name() override {
    return "TXQCDWilsonCloverHasenbuschAction";
  }

  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "]"
       << " m_light=" << mass_l_ << " m_heavy=" << mass_h_
       << " lo=" << param.lo << " hi=" << param.hi
       << " degree=" << param.degree << " tol=" << param.tolerance
       << " MaxIter=" << param.MaxIter << std::endl;
    return os.str();
  }

  // Refresh: sample Phi so that <Phi|A|Phi> = <eta|eta> with eta Gaussian,
  // where A = (V^dag V)^{1/4} (M^dag M)^{-1/2} (V^dag V)^{1/4}.
  // Phi = (V^dag V)^{-1/4} (M^dag M)^{1/4} eta.
  void refresh(const TXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    TXQCDFermionNf eta(&rbgrid_);
    const RealD scale = std::sqrt(0.5);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, eta.f[a]);
      eta.f[a] = scale * eta.f[a];
      eta.f[a].Checkerboard() = Odd;
    }

    auto EOp_l = MakeEOp(U, mass_l_);
    auto EOp_h = MakeEOp(U, mass_h_);
    TXQCDCloverSchurOp Schur_l(EOp_l);
    TXQCDCloverSchurOp Schur_h(EOp_h);

    // tmp = (M^dag M)^{1/4} eta   [apply PowerQuarter on the denominator op]
    TXQCDFermionNf tmp(&rbgrid_);
    ApplyRational(Schur_l, PowerQuarter, eta, tmp);

    // Phi = (V^dag V)^{-1/4} tmp   [apply PowerNegQuarter on the numerator op]
    ApplyRational(Schur_h, PowerNegQuarter, tmp, Phi);

    RefreshAction_ = norm2(eta);
    std::cout << GridLogMessage << "[" << action_name() << "]"
              << " refresh: RefreshAction = " << RefreshAction_ << std::endl;
  }

  // S(U) = |Y|^2 with Y = (M^dag M)^{-1/4} (V^dag V)^{1/4} Phi.
  RealD S(const TXQCDField &U) override {
    auto EOp_l = MakeEOp(U, mass_l_);
    auto EOp_h = MakeEOp(U, mass_h_);
    TXQCDCloverSchurOp Schur_l(EOp_l);
    TXQCDCloverSchurOp Schur_h(EOp_h);

    TXQCDFermionNf X(&rbgrid_);
    ApplyRational(Schur_h, PowerQuarter, Phi, X);  // X = (V†V)^{1/4} Phi

    TXQCDFermionNf Y(&rbgrid_);
    ApplyRational(Schur_l, PowerNegQuarter, X, Y); // Y = (M†M)^{-1/4} X

    RealD action = norm2(Y);
    std::cout << GridLogMessage << "[" << action_name() << "]"
              << " S = " << action << std::endl;
    return action;
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    auto EOp_l = MakeEOp(U, mass_l_);
    auto EOp_h = MakeEOp(U, mass_h_);
    TXQCDCloverSchurOp Schur_l(EOp_l);
    TXQCDCloverSchurOp Schur_h(EOp_h);

    const int n_f  = static_cast<int>(PowerNegHalf.poles.size());
    const int n_pv = static_cast<int>(PowerQuarter.poles.size());

    // Solve (V†V + b_pv_k)^{-1} Phi  -> MpvPhi_k   (X_k in notes)
    std::vector<TXQCDFermionNf> MpvPhi_k; MpvPhi_k.reserve(n_pv);
    for (int k = 0; k < n_pv; ++k) MpvPhi_k.emplace_back(&rbgrid_);
    std::vector<RealD> md_tol_pv(n_pv, param.mdtolerance);
    TXQCDMultiShiftCGSchur MSCG_pv(param.MaxIter);
    MSCG_pv(Schur_h, PowerQuarter.poles, md_tol_pv, Phi, MpvPhi_k);

    // MpvPhi = norm·Phi + sum_k r_k MpvPhi_k  ~= (V†V)^{1/4} Phi
    TXQCDFermionNf MpvPhi(&rbgrid_);
    for (int a = 0; a < TxqcdNf; ++a) {
      MpvPhi.f[a] = PowerQuarter.norm * Phi.f[a];
      MpvPhi.f[a].Checkerboard() = Odd;
    }
    for (int k = 0; k < n_pv; ++k)
      for (int a = 0; a < TxqcdNf; ++a)
        MpvPhi.f[a] = MpvPhi.f[a] + PowerQuarter.residues[k] * MpvPhi_k[k].f[a];

    // Solve (M†M + b_f_k)^{-1} MpvPhi -> MfMpvPhi_k  (W_k in notes)
    std::vector<TXQCDFermionNf> MfMpvPhi_k; MfMpvPhi_k.reserve(n_f);
    for (int k = 0; k < n_f; ++k) MfMpvPhi_k.emplace_back(&rbgrid_);
    std::vector<RealD> md_tol_f(n_f, param.mdtolerance);
    TXQCDMultiShiftCGSchur MSCG_f(param.MaxIter);
    MSCG_f(Schur_l, PowerNegHalf.poles, md_tol_f, MpvPhi, MfMpvPhi_k);

    // MfMpvPhi = norm·MpvPhi + sum_k r_k MfMpvPhi_k  ~= (M†M)^{-1/2} MpvPhi
    TXQCDFermionNf MfMpvPhi(&rbgrid_);
    for (int a = 0; a < TxqcdNf; ++a) {
      MfMpvPhi.f[a] = PowerNegHalf.norm * MpvPhi.f[a];
      MfMpvPhi.f[a].Checkerboard() = Odd;
    }
    for (int k = 0; k < n_f; ++k)
      for (int a = 0; a < TxqcdNf; ++a)
        MfMpvPhi.f[a] = MfMpvPhi.f[a] +
                        PowerNegHalf.residues[k] * MfMpvPhi_k[k].f[a];

    // Solve (V†V + b_pv_k)^{-1} MfMpvPhi -> MpvMfMpvPhi_k  (Z_k in notes)
    std::vector<TXQCDFermionNf> MpvMfMpvPhi_k; MpvMfMpvPhi_k.reserve(n_pv);
    for (int k = 0; k < n_pv; ++k) MpvMfMpvPhi_k.emplace_back(&rbgrid_);
    MSCG_pv(Schur_h, PowerQuarter.poles, md_tol_pv, MfMpvPhi, MpvMfMpvPhi_k);

    // ---- Zero the force accumulator ----
    dSdU.sigma = Zero();
    dSdU.pi    = Zero();
    dSdU.s     = Zero();
    dSdU.p     = Zero();
    dSdU.t     = Zero();
    dSdU.U     = Zero();

    const ComplexD inv_sqrt2(1.0 / std::sqrt(2.0), 0.0);
    SpinTable Id{IdentitySpinMatrix()};
    SpinTable G5{Gamma5Matrix()};
    Gamma g5(Gamma::Algebra::Gamma5);

    //
    // Part (1): M-side.  dS_1/dU = sum_k a_f_k d/dU[W_k† Mpc_l† Mpc_l W_k].
    // Per pole this is the same symmetric-bilinear structure as the ordinary
    // rational action (W_k plays the role of X in that derivation): one
    // AccumulateAuxForce call for odd, one for even, plus the usual 2-call
    // MpcDeriv/MpcDagDeriv gauge force.
    //
    for (int k = 0; k < n_f; ++k) {
      const RealD ak = PowerNegHalf.residues[k];
      TXQCDFermionNf &W = MfMpvPhi_k[k];
      TXQCDFermionNf Y(&rbgrid_);
      Schur_l.Mpc(W, Y);

      TXQCDFermionNf W_e(&rbgrid_), Z_e(&rbgrid_), tmp_e(&rbgrid_);
      EOp_l.Meooe(W, tmp_e);
      EOp_l.MooeeInv(tmp_e, W_e);      // W_e = Mee_l^{-1} Meo_l W
      EOp_l.MeooeDag(Y, tmp_e);
      EOp_l.MooeeInvDag(tmp_e, Z_e);   // Z_e = Mee_l^{-1†} Moe_l† Y

      AccumulateAuxForceDerivH(dSdU, ak * mu_, Y,   W,   g5, inv_sqrt2, Id, G5);
      AccumulateAuxForceDerivH(dSdU, ak * mu_, Z_e, W_e, g5, inv_sqrt2, Id, G5);
      AddSymmetricGaugeForce(dSdU, ak, EOp_l, Y, W, W_e, Z_e);
    }

    //
    // Part (2)+(3): V-side.  Combined,
    //   dS_{2+3}/dU = sum_k a_pv_k d/dU[Z_k† Mpc_h† Mpc_h X_k + h.c.]
    //               = sum_k a_pv_k d/dU[ 2 Re( Y_Z† Y_X ) ]
    // with  X_k = MpvPhi_k, Z_k = MpvMfMpvPhi_k, Y_X = Mpc_h X_k, Y_Z = Mpc_h Z_k.
    //
    // Grid's GeneralEvenOddRatioRational decomposes this into FOUR Schur
    // chain-rule calls per pole (MpcDagDeriv(Z,Y_X)+MpcDeriv(Y_X,Z) for (2),
    // MpcDeriv(Y_Z,X)+MpcDagDeriv(X,Y_Z) for (3)).  For TXQCD aux force we
    // pair them: each of (2a,2b) and (3a,3b) combines into a single
    // Hermitian-projected AccumulateAuxForce call thanks to the (A,B)<->(B,A)
    // symmetry of HermitianFlavorForce.  Result per pole: 4 aux calls
    // (2 odd-site + 2 even-site) and 4 gauge chain-rule calls (one each for
    // MpcDeriv(Y_X,Z), MpcDagDeriv(Z,Y_X), MpcDeriv(Y_Z,X), MpcDagDeriv(X,Y_Z)).
    //
    for (int k = 0; k < n_pv; ++k) {
      const RealD ak = PowerQuarter.residues[k];
      TXQCDFermionNf &X = MpvPhi_k[k];
      TXQCDFermionNf &Z = MpvMfMpvPhi_k[k];

      TXQCDFermionNf Y_X(&rbgrid_);  // = Mpc_h X
      Schur_h.Mpc(X, Y_X);
      TXQCDFermionNf Y_Z(&rbgrid_);  // = Mpc_h Z
      Schur_h.Mpc(Z, Y_Z);

      TXQCDFermionNf W_X_e(&rbgrid_), W_Z_e(&rbgrid_);
      TXQCDFermionNf Ze_X_e(&rbgrid_), Ze_Z_e(&rbgrid_);
      TXQCDFermionNf tmp_e(&rbgrid_);
      // W_X_e = Mee_h^{-1} Meo_h X
      EOp_h.Meooe(X, tmp_e); EOp_h.MooeeInv(tmp_e, W_X_e);
      // W_Z_e = Mee_h^{-1} Meo_h Z
      EOp_h.Meooe(Z, tmp_e); EOp_h.MooeeInv(tmp_e, W_Z_e);
      // Ze_X_e = Mee_h^{-1†} Moe_h† Y_X
      EOp_h.MeooeDag(Y_X, tmp_e); EOp_h.MooeeInvDag(tmp_e, Ze_X_e);
      // Ze_Z_e = Mee_h^{-1†} Moe_h† Y_Z
      EOp_h.MeooeDag(Y_Z, tmp_e); EOp_h.MooeeInvDag(tmp_e, Ze_Z_e);

      // Aux force (odd): bilinear(Y_X, Z) from (2a)+(2b);
      //                  bilinear(Y_Z, X) from (3a)+(3b).
      AccumulateAuxForceDerivH(dSdU, ak * mu_, Y_X, Z, g5, inv_sqrt2, Id, G5);
      AccumulateAuxForceDerivH(dSdU, ak * mu_, Y_Z, X, g5, inv_sqrt2, Id, G5);

      // Aux force (even): bilinear(Ze_X, W_Z) from (2a)+(2b);
      //                   bilinear(Ze_Z, W_X) from (3a)+(3b).
      AccumulateAuxForceDerivH(dSdU, ak * mu_, Ze_X_e, W_Z_e, g5, inv_sqrt2, Id, G5);
      AccumulateAuxForceDerivH(dSdU, ak * mu_, Ze_Z_e, W_X_e, g5, inv_sqrt2, Id, G5);

      // Gauge force: 4 MpcDeriv-style chain-rule calls per pole.
      AddAsymmetricGaugeForce(dSdU, ak, EOp_h,
                              /*Y_X=*/Y_X, /*X=*/X,
                              /*Y_Z=*/Y_Z, /*Z=*/Z,
                              /*W_X_e=*/W_X_e, /*W_Z_e=*/W_Z_e,
                              /*Ze_X_e=*/Ze_X_e, /*Ze_Z_e=*/Ze_Z_e);
    }
  }

 private:
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  RealD mass_l_, mass_h_, csw_;
  RealD mu_;
  Params &param;
  TXQCDFermionNf Phi;
  MultiShiftFunction PowerHalf;       // x^{1/2}
  MultiShiftFunction PowerNegHalf;    // x^{-1/2}
  MultiShiftFunction PowerQuarter;    // x^{1/4}
  MultiShiftFunction PowerNegQuarter; // x^{-1/4}
  RealD RefreshAction_{0.0};

  TXQCDWilsonCloverFermionEO MakeEOp(const TXQCDField &U, RealD mass) {
    TXQCDField &Unc = const_cast<TXQCDField &>(U);
    return TXQCDWilsonCloverFermionEO(
        Unc.U, grid_, rbgrid_, mass, Unc.sigma, Unc.pi, Unc.s, Unc.p, Unc.t,
        csw_, TXQCDWilsonCloverFermionEO::DefaultImplParams(), mu_);
  }

  // Apply a rational approximation  r(x) = c0 + sum_k ck / (x + pk)  to an
  // input fermion field, via multi-shift CG on the Schur operator's (M†M+σ).
  void ApplyRational(TXQCDCloverSchurOp &schurOp, const MultiShiftFunction &rat,
                     const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    const int nshift = static_cast<int>(rat.poles.size());
    std::vector<TXQCDFermionNf> xk;
    xk.reserve(nshift);
    for (int k = 0; k < nshift; ++k) xk.emplace_back(&rbgrid_);
    TXQCDMultiShiftCGSchur MSCG(param.MaxIter);
    MSCG(schurOp, rat.poles, rat.tolerances, in, xk);
    for (int a = 0; a < TxqcdNf; ++a) {
      out.f[a] = rat.norm * in.f[a];
      out.f[a].Checkerboard() = Odd;
    }
    for (int k = 0; k < nshift; ++k) {
      for (int a = 0; a < TxqcdNf; ++a)
        out.f[a] = out.f[a] + rat.residues[k] * xk[k].f[a];
    }
  }

  // Accumulate the aux-field force from a single (Y, X) bilinear pair, at
  // weight ak.  Same structure as TXQCDWilsonCloverRationalEOAction::
  // AccumulateAuxForce (duplicated here to avoid coupling to its private
  // member).  Populates sigma/pi/s/p/t slots of dSdU.
  void AccumulateAuxForceDerivH(TXQCDField &dSdU, RealD ak,
                                 const TXQCDFermionNf &Y,
                                 const TXQCDFermionNf &X,
                                 const Gamma &g5, ComplexD inv_sqrt2,
                                 const SpinTable &Id, const SpinTable &G5) {
    GridBase *grid = Y.Grid();
    int cb = Y.f[0].Checkerboard();

    // sigma
    {
      auto G = FlavorBilinear(Y, X);
      G.Checkerboard() = cb;
      LatticeSigmaField F = HermitianFlavorForce(G);
      F.Checkerboard() = cb;
      LatticeSigmaField tmp(&grid_);
      tmp = Zero();
      setCheckerboard(tmp, F);
      dSdU.sigma = dSdU.sigma + ak * tmp;
    }
    // pi
    {
      TXQCDFermionNf g5X(grid);
      for (int a = 0; a < TxqcdNf; ++a) {
        g5X.f[a] = g5 * X.f[a];
        g5X.f[a].Checkerboard() = cb;
      }
      auto G = FlavorBilinear(Y, g5X);
      G.Checkerboard() = cb;
      LatticePiField F = HermitianFlavorForce(G);
      F.Checkerboard() = cb;
      LatticePiField tmp(&grid_);
      tmp = Zero();
      setCheckerboard(tmp, F);
      dSdU.pi = dSdU.pi + ak * tmp;
    }
    // s
    {
      auto G = ColorBilinearSpinOp(Y, X, Id);
      G.Checkerboard() = cb;
      G = inv_sqrt2 * G;
      LatticeSFieldC F = HermitianColorForce(G);
      F.Checkerboard() = cb;
      LatticeSFieldC tmp(&grid_);
      tmp = Zero();
      setCheckerboard(tmp, F);
      dSdU.s = dSdU.s + ak * tmp;
    }
    // p
    {
      auto G = ColorBilinearSpinOp(Y, X, G5);
      G.Checkerboard() = cb;
      G = inv_sqrt2 * G;
      LatticePFieldC F = HermitianColorForce(G);
      F.Checkerboard() = cb;
      LatticePFieldC tmp(&grid_);
      tmp = Zero();
      setCheckerboard(tmp, F);
      dSdU.p = dSdU.p + ak * tmp;
    }
    // t_{mu,nu}
    LatticeTField Tk(&grid_);
    Tk = Zero();
    {
      autoView(tkv, Tk, CpuWrite);
      for (int mu = 0; mu < Nd; ++mu) {
        for (int nu = mu + 1; nu < Nd; ++nu) {
          SpinTable iSig{ISigmaMatrix(mu, nu)};
          auto Gt = ColorBilinearSpinOp(Y, X, iSig);
          Gt.Checkerboard() = cb;
          LatticeSFieldC Ft = HermitianColorForce(Gt);
          Ft.Checkerboard() = cb;
          LatticeSFieldC Ftfull(&grid_);
          Ftfull = Zero();
          setCheckerboard(Ftfull, Ft);
          autoView(src, Ftfull, CpuRead);
          thread_for(ss, grid_.oSites(), {
            for (int i = 0; i < Nc; ++i) {
              for (int j = 0; j < Nc; ++j) {
                tkv[ss]()(mu, nu)(i, j) =  src[ss]()()(i, j);
                tkv[ss]()(nu, mu)(i, j) = -src[ss]()()(i, j);
              }
            }
          });
        }
      }
    }
    dSdU.t = dSdU.t + ak * Tk;
  }

  // Symmetric-bilinear gauge force (part 1): chain rule through
  // d/dU[X†·Mpc†·Mpc·X].  Combines MpcDeriv(Y,X) + MpcDagDeriv(X,Y) as the
  // existing rational action does, where Y = Mpc·X.  Handles hopping AND
  // clover Cmunu when csw != 0.
  void AddSymmetricGaugeForce(TXQCDField &dSdU, RealD ak,
                               TXQCDWilsonCloverFermionEO &EOp,
                               const TXQCDFermionNf &Y,
                               const TXQCDFermionNf &X,
                               const TXQCDFermionNf &W_e,
                               const TXQCDFermionNf &Z_e) {
    LatticeGaugeField gforce(&grid_); gforce = Zero();
    LatticeGaugeField gtmp(&grid_);
    LatticeGaugeField ForceO(&rbgrid_), ForceE(&rbgrid_);

    for (int a = 0; a < TxqcdNf; ++a) {
      EOp.Wilson().MoeDeriv(ForceO, Y.f[a], W_e.f[a], DaggerNo);
      EOp.Wilson().MeoDeriv(ForceE, Z_e.f[a], X.f[a], DaggerNo);
      setCheckerboard(gtmp, ForceO);
      setCheckerboard(gtmp, ForceE);
      gforce = gforce - gtmp;

      EOp.Wilson().MoeDeriv(ForceO, X.f[a], Z_e.f[a], DaggerYes);
      EOp.Wilson().MeoDeriv(ForceE, W_e.f[a], Y.f[a], DaggerYes);
      setCheckerboard(gtmp, ForceO);
      setCheckerboard(gtmp, ForceE);
      gforce = gforce - gtmp;
    }

    if (csw_ != 0.0) {
      AddSymmetricCloverForce(gforce, EOp, Y, X, W_e, Z_e);
    }

    dSdU.U = dSdU.U + ak * gforce;
  }

  // Clover (Cmunu) contribution for the symmetric case.  Build a flavor-
  // summed spin-color Lambda = outer(X,Y) + outer(Y,X) on odd, and
  // outer(Z_e,W_e) + outer(W_e,Z_e) on even, then drive the 12-sigma Cmunu
  // iteration.  Equivalent to the hand-rolled isig·(Y*·X + X*·Y) pattern but
  // uses Grid's native Gamma(sigma)·Lambda spin contraction -- same approach
  // as QCDLogDetCloverEOAction, which is the clover-force path we've
  // carefully validated against finite difference.
  void AddSymmetricCloverForce(LatticeGaugeField &gforce,
                                TXQCDWilsonCloverFermionEO &EOp,
                                const TXQCDFermionNf &Y,
                                const TXQCDFermionNf &X,
                                const TXQCDFermionNf &W_e,
                                const TXQCDFermionNf &Z_e) {
    // Hand-rolled pattern copied verbatim from the validated
    // TXQCDWilsonCloverRationalEOAction (which passes the rational FD check
    // at rel ~1e-6 with csw=1).  Kept to isolate the Hasenbusch-specific
    // gauge-force discrepancy from any subtle translation issue in the
    // QCDLogDet-style Gamma(sigma)*Lambda path.
    typedef TXQCDSiteMatrixUtil SMU;
    typedef typename LatticeColourMatrix::vector_object::scalar_object CMsobj;
    typedef typename LatticeFermion::vector_object::scalar_object Fsobj;
    SMU::SpinMatrices sm;

    std::array<std::vector<Fsobj>, TxqcdNf> Xv, Yv, Wv, Zv;
    for (int a = 0; a < TxqcdNf; ++a) {
      unvectorizeToLexOrdArray(Xv[a], X.f[a]);
      unvectorizeToLexOrdArray(Yv[a], Y.f[a]);
      unvectorizeToLexOrdArray(Wv[a], W_e.f[a]);
      unvectorizeToLexOrdArray(Zv[a], Z_e.f[a]);
    }
    uint64_t nsites_odd = Xv[0].size();
    uint64_t nsites_even = Wv[0].size();

    std::vector<LatticeColourMatrix> Sigma_full;
    int fk = 0;
    for (int rho = 0; rho < Nd; ++rho) {
      for (int sig = rho + 1; sig < Nd; ++sig) {
        std::vector<CMsobj> sig_odd(nsites_odd);
        thread_for(x, nsites_odd, {
          for (int i = 0; i < Nc; ++i)
            for (int j = 0; j < Nc; ++j) {
              std::complex<double> val(0, 0);
              for (int a = 0; a < TxqcdNf; ++a)
                for (int alpha = 0; alpha < Ns; ++alpha)
                  for (int beta = 0; beta < Ns; ++beta) {
                    auto isig = sm.isigma[rho][sig](alpha, beta);
                    if (isig == std::complex<double>(0, 0)) continue;
                    std::complex<double> Yaj(Yv[a][x]()(alpha)(j).real(),
                                              Yv[a][x]()(alpha)(j).imag());
                    std::complex<double> Xbi(Xv[a][x]()(beta)(i).real(),
                                              Xv[a][x]()(beta)(i).imag());
                    std::complex<double> Xaj(Xv[a][x]()(alpha)(j).real(),
                                              Xv[a][x]()(alpha)(j).imag());
                    std::complex<double> Ybi(Yv[a][x]()(beta)(i).real(),
                                              Yv[a][x]()(beta)(i).imag());
                    val += isig * (std::conj(Yaj) * Xbi +
                                    std::conj(Xaj) * Ybi);
                  }
              std::complex<double> cv(0.0, -0.5 * csw_);
              std::complex<double> cval = cv * val;
              sig_odd[x]()()(i, j) = ComplexD(cval.real(), cval.imag());
            }
        });
        std::vector<CMsobj> sig_even(nsites_even);
        thread_for(x, nsites_even, {
          for (int i = 0; i < Nc; ++i)
            for (int j = 0; j < Nc; ++j) {
              std::complex<double> val(0, 0);
              for (int a = 0; a < TxqcdNf; ++a)
                for (int alpha = 0; alpha < Ns; ++alpha)
                  for (int beta = 0; beta < Ns; ++beta) {
                    auto isig = sm.isigma[rho][sig](alpha, beta);
                    if (isig == std::complex<double>(0, 0)) continue;
                    std::complex<double> Zaj(Zv[a][x]()(alpha)(j).real(),
                                              Zv[a][x]()(alpha)(j).imag());
                    std::complex<double> Wbi(Wv[a][x]()(beta)(i).real(),
                                              Wv[a][x]()(beta)(i).imag());
                    std::complex<double> Waj(Wv[a][x]()(alpha)(j).real(),
                                              Wv[a][x]()(alpha)(j).imag());
                    std::complex<double> Zbi(Zv[a][x]()(beta)(i).real(),
                                              Zv[a][x]()(beta)(i).imag());
                    val += isig * (std::conj(Zaj) * Wbi +
                                    std::conj(Waj) * Zbi);
                  }
              std::complex<double> cv(0.0, -0.5 * csw_);
              std::complex<double> cval = cv * val;
              sig_even[x]()()(i, j) = ComplexD(cval.real(), cval.imag());
            }
        });
        LatticeColourMatrix lam_odd(&rbgrid_);
        vectorizeFromLexOrdArray(sig_odd, lam_odd);
        lam_odd.Checkerboard() = Odd;
        LatticeColourMatrix lam_even(&rbgrid_);
        vectorizeFromLexOrdArray(sig_even, lam_even);
        lam_even.Checkerboard() = Even;
        Sigma_full.emplace_back(&grid_);
        Sigma_full.back() = Zero();
        setCheckerboard(Sigma_full[fk], lam_odd);
        setCheckerboard(Sigma_full[fk], lam_even);
        ++fk;
      }
    }
    std::vector<LatticeColourMatrix> Ulinks(Nd, &grid_);
    for (int mu = 0; mu < Nd; ++mu)
      Ulinks[mu] = PeekIndex<LorentzIndex>(EOp.Gauge(), mu);
    LatticeGaugeField clover_gforce(&grid_);
    clover_gforce = Zero();
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix force_mu(&grid_);
      force_mu = Zero();
      for (int nu = 0; nu < Nd; ++nu) {
        if (mu == nu) continue;
        int mn = (mu < nu) ? SMU::FmnIndex(mu, nu) : SMU::FmnIndex(nu, mu);
        LatticeColourMatrix lam =
            (mu < nu) ? Sigma_full[mn] : (-1.0) * Sigma_full[mn];
        force_mu = force_mu + 0.25 *
                   WilsonCloverHelpers<WilsonImplR>::Cmunu(Ulinks, lam, mu, nu);
      }
      pokeLorentz(clover_gforce, Ulinks[mu] * force_mu, mu);
    }
    gforce = gforce + (-0.5) * clover_gforce;
  }

  // Asymmetric-bilinear gauge force (parts 2+3): chain rule through
  // d/dU[Z†·Mpc†·Mpc·X + X†·Mpc†·Mpc·Z].  Four MpcDeriv/MpcDagDeriv analogs
  // per pole, using Y_X = Mpc·X and Y_Z = Mpc·Z along with their respective
  // even-site intermediates.
  void AddAsymmetricGaugeForce(TXQCDField &dSdU, RealD ak,
                                TXQCDWilsonCloverFermionEO &EOp,
                                const TXQCDFermionNf &Y_X,
                                const TXQCDFermionNf &X,
                                const TXQCDFermionNf &Y_Z,
                                const TXQCDFermionNf &Z,
                                const TXQCDFermionNf &W_X_e,
                                const TXQCDFermionNf &W_Z_e,
                                const TXQCDFermionNf &Ze_X_e,
                                const TXQCDFermionNf &Ze_Z_e) {
    LatticeGaugeField gforce(&grid_); gforce = Zero();
    LatticeGaugeField gtmp(&grid_);
    LatticeGaugeField ForceO(&rbgrid_), ForceE(&rbgrid_);

    for (int a = 0; a < TxqcdNf; ++a) {
      // (2a) MpcDagDeriv(Z, Y_X):   MoeDeriv(Z, Ze_X, Yes), MeoDeriv(W_Z, Y_X, Yes)
      EOp.Wilson().MoeDeriv(ForceO, Z.f[a],     Ze_X_e.f[a], DaggerYes);
      EOp.Wilson().MeoDeriv(ForceE, W_Z_e.f[a], Y_X.f[a],    DaggerYes);
      setCheckerboard(gtmp, ForceO);
      setCheckerboard(gtmp, ForceE);
      gforce = gforce - gtmp;

      // (2b) MpcDeriv(Y_X, Z):       MoeDeriv(Y_X, W_Z, No), MeoDeriv(Ze_X, Z, No)
      EOp.Wilson().MoeDeriv(ForceO, Y_X.f[a],    W_Z_e.f[a], DaggerNo);
      EOp.Wilson().MeoDeriv(ForceE, Ze_X_e.f[a], Z.f[a],     DaggerNo);
      setCheckerboard(gtmp, ForceO);
      setCheckerboard(gtmp, ForceE);
      gforce = gforce - gtmp;

      // (3a) MpcDeriv(Y_Z, X):       MoeDeriv(Y_Z, W_X, No), MeoDeriv(Ze_Z, X, No)
      EOp.Wilson().MoeDeriv(ForceO, Y_Z.f[a],    W_X_e.f[a], DaggerNo);
      EOp.Wilson().MeoDeriv(ForceE, Ze_Z_e.f[a], X.f[a],     DaggerNo);
      setCheckerboard(gtmp, ForceO);
      setCheckerboard(gtmp, ForceE);
      gforce = gforce - gtmp;

      // (3b) MpcDagDeriv(X, Y_Z):    MoeDeriv(X, Ze_Z, Yes), MeoDeriv(W_X, Y_Z, Yes)
      EOp.Wilson().MoeDeriv(ForceO, X.f[a],     Ze_Z_e.f[a], DaggerYes);
      EOp.Wilson().MeoDeriv(ForceE, W_X_e.f[a], Y_Z.f[a],    DaggerYes);
      setCheckerboard(gtmp, ForceO);
      setCheckerboard(gtmp, ForceE);
      gforce = gforce - gtmp;
    }

    if (csw_ != 0.0) {
      AddAsymmetricCloverForce(gforce, EOp, Y_X, X, Y_Z, Z,
                                W_X_e, W_Z_e, Ze_X_e, Ze_Z_e);
    }

    dSdU.U = dSdU.U + ak * gforce;
  }

  // Clover (Cmunu) contribution for the asymmetric (parts 2+3) case.
  // Hand-rolled pattern analogous to the validated rational-action clover
  // force, but summing TWO bilinear pairs per (rho,sig) since we collapse
  // four MpcDeriv/MpcDagDeriv calls per pole into a single Cmunu loop.
  void AddAsymmetricCloverForce(LatticeGaugeField &gforce,
                                 TXQCDWilsonCloverFermionEO &EOp,
                                 const TXQCDFermionNf &Y_X,
                                 const TXQCDFermionNf &X,
                                 const TXQCDFermionNf &Y_Z,
                                 const TXQCDFermionNf &Z,
                                 const TXQCDFermionNf &W_X_e,
                                 const TXQCDFermionNf &W_Z_e,
                                 const TXQCDFermionNf &Ze_X_e,
                                 const TXQCDFermionNf &Ze_Z_e) {
    typedef TXQCDSiteMatrixUtil SMU;
    typedef typename LatticeColourMatrix::vector_object::scalar_object CMsobj;
    typedef typename LatticeFermion::vector_object::scalar_object Fsobj;
    SMU::SpinMatrices sm;

    std::array<std::vector<Fsobj>, TxqcdNf> Xv, YXv, Zv, YZv;
    std::array<std::vector<Fsobj>, TxqcdNf> WXv, WZv, ZeXv, ZeZv;
    for (int a = 0; a < TxqcdNf; ++a) {
      unvectorizeToLexOrdArray(Xv[a],    X.f[a]);
      unvectorizeToLexOrdArray(YXv[a],   Y_X.f[a]);
      unvectorizeToLexOrdArray(Zv[a],    Z.f[a]);
      unvectorizeToLexOrdArray(YZv[a],   Y_Z.f[a]);
      unvectorizeToLexOrdArray(WXv[a],   W_X_e.f[a]);
      unvectorizeToLexOrdArray(WZv[a],   W_Z_e.f[a]);
      unvectorizeToLexOrdArray(ZeXv[a],  Ze_X_e.f[a]);
      unvectorizeToLexOrdArray(ZeZv[a],  Ze_Z_e.f[a]);
    }
    uint64_t nsites_odd  = Xv[0].size();
    uint64_t nsites_even = WXv[0].size();

    std::vector<LatticeColourMatrix> Sigma_full;
    int fk = 0;
    for (int rho = 0; rho < Nd; ++rho) {
      for (int sig = rho + 1; sig < Nd; ++sig) {
        std::vector<CMsobj> sig_odd(nsites_odd);
        thread_for(x, nsites_odd, {
          for (int i = 0; i < Nc; ++i)
            for (int j = 0; j < Nc; ++j) {
              std::complex<double> val(0, 0);
              for (int a = 0; a < TxqcdNf; ++a)
                for (int alpha = 0; alpha < Ns; ++alpha)
                  for (int beta = 0; beta < Ns; ++beta) {
                    auto isig = sm.isigma[rho][sig](alpha, beta);
                    if (isig == std::complex<double>(0, 0)) continue;
                    // (2a)+(2b) with (A,B) = (Y_X, Z):
                    std::complex<double> YXaj(YXv[a][x]()(alpha)(j).real(),
                                               YXv[a][x]()(alpha)(j).imag());
                    std::complex<double> Zbi (Zv [a][x]()(beta)(i).real(),
                                               Zv [a][x]()(beta)(i).imag());
                    std::complex<double> Zaj (Zv [a][x]()(alpha)(j).real(),
                                               Zv [a][x]()(alpha)(j).imag());
                    std::complex<double> YXbi(YXv[a][x]()(beta)(i).real(),
                                               YXv[a][x]()(beta)(i).imag());
                    val += isig * (std::conj(YXaj) * Zbi +
                                    std::conj(Zaj)  * YXbi);
                    // (3a)+(3b) with (A,B) = (Y_Z, X):
                    std::complex<double> YZaj(YZv[a][x]()(alpha)(j).real(),
                                               YZv[a][x]()(alpha)(j).imag());
                    std::complex<double> Xbi (Xv [a][x]()(beta)(i).real(),
                                               Xv [a][x]()(beta)(i).imag());
                    std::complex<double> Xaj (Xv [a][x]()(alpha)(j).real(),
                                               Xv [a][x]()(alpha)(j).imag());
                    std::complex<double> YZbi(YZv[a][x]()(beta)(i).real(),
                                               YZv[a][x]()(beta)(i).imag());
                    val += isig * (std::conj(YZaj) * Xbi +
                                    std::conj(Xaj)  * YZbi);
                  }
              std::complex<double> cv(0.0, -0.5 * csw_);
              std::complex<double> cval = cv * val;
              sig_odd[x]()()(i, j) = ComplexD(cval.real(), cval.imag());
            }
        });
        std::vector<CMsobj> sig_even(nsites_even);
        thread_for(x, nsites_even, {
          for (int i = 0; i < Nc; ++i)
            for (int j = 0; j < Nc; ++j) {
              std::complex<double> val(0, 0);
              for (int a = 0; a < TxqcdNf; ++a)
                for (int alpha = 0; alpha < Ns; ++alpha)
                  for (int beta = 0; beta < Ns; ++beta) {
                    auto isig = sm.isigma[rho][sig](alpha, beta);
                    if (isig == std::complex<double>(0, 0)) continue;
                    // (2a)+(2b) with (A,B) = (Ze_X, W_Z):
                    std::complex<double> ZeXaj(ZeXv[a][x]()(alpha)(j).real(),
                                                ZeXv[a][x]()(alpha)(j).imag());
                    std::complex<double> WZbi (WZv [a][x]()(beta)(i).real(),
                                                WZv [a][x]()(beta)(i).imag());
                    std::complex<double> WZaj (WZv [a][x]()(alpha)(j).real(),
                                                WZv [a][x]()(alpha)(j).imag());
                    std::complex<double> ZeXbi(ZeXv[a][x]()(beta)(i).real(),
                                                ZeXv[a][x]()(beta)(i).imag());
                    val += isig * (std::conj(ZeXaj) * WZbi +
                                    std::conj(WZaj)  * ZeXbi);
                    // (3a)+(3b) with (A,B) = (Ze_Z, W_X):
                    std::complex<double> ZeZaj(ZeZv[a][x]()(alpha)(j).real(),
                                                ZeZv[a][x]()(alpha)(j).imag());
                    std::complex<double> WXbi (WXv [a][x]()(beta)(i).real(),
                                                WXv [a][x]()(beta)(i).imag());
                    std::complex<double> WXaj (WXv [a][x]()(alpha)(j).real(),
                                                WXv [a][x]()(alpha)(j).imag());
                    std::complex<double> ZeZbi(ZeZv[a][x]()(beta)(i).real(),
                                                ZeZv[a][x]()(beta)(i).imag());
                    val += isig * (std::conj(ZeZaj) * WXbi +
                                    std::conj(WXaj)  * ZeZbi);
                  }
              std::complex<double> cv(0.0, -0.5 * csw_);
              std::complex<double> cval = cv * val;
              sig_even[x]()()(i, j) = ComplexD(cval.real(), cval.imag());
            }
        });
        LatticeColourMatrix lam_odd(&rbgrid_);
        vectorizeFromLexOrdArray(sig_odd, lam_odd);
        lam_odd.Checkerboard() = Odd;
        LatticeColourMatrix lam_even(&rbgrid_);
        vectorizeFromLexOrdArray(sig_even, lam_even);
        lam_even.Checkerboard() = Even;
        Sigma_full.emplace_back(&grid_);
        Sigma_full.back() = Zero();
        setCheckerboard(Sigma_full[fk], lam_odd);
        setCheckerboard(Sigma_full[fk], lam_even);
        ++fk;
      }
    }
    std::vector<LatticeColourMatrix> Ulinks(Nd, &grid_);
    for (int mu = 0; mu < Nd; ++mu)
      Ulinks[mu] = PeekIndex<LorentzIndex>(EOp.Gauge(), mu);
    LatticeGaugeField clover_gforce(&grid_);
    clover_gforce = Zero();
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix force_mu(&grid_);
      force_mu = Zero();
      for (int nu = 0; nu < Nd; ++nu) {
        if (mu == nu) continue;
        int mn = (mu < nu) ? SMU::FmnIndex(mu, nu) : SMU::FmnIndex(nu, mu);
        LatticeColourMatrix lam =
            (mu < nu) ? Sigma_full[mn] : (-1.0) * Sigma_full[mn];
        force_mu = force_mu + 0.25 *
                   WilsonCloverHelpers<WilsonImplR>::Cmunu(Ulinks, lam, mu, nu);
      }
      pokeLorentz(clover_gforce, Ulinks[mu] * force_mu, mu);
    }
    gforce = gforce + (-0.5) * clover_gforce;
  }
};

NAMESPACE_END(Grid);
