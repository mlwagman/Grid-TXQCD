#pragma once
// Schur complement operator for the EO-preconditioned TXQCD Möbius operator.
//
// Mpc = Moo - Moe * Mee^{-1} * Meo, operating on TXQCDFermionNf on the
// odd 5D red-black sublattice.  HermOp computes Mpc†Mpc for CG.
//
// Structurally identical to TXQCDSchurOp (Wilson) but with the Möbius EO
// operator.  The TXQCDFermionNf free functions (conformable, axpy_norm,
// operator +/-/*, innerProduct) are shared from TXQCDSchurOp.h — including
// it here reuses them and avoids ODR duplication.

#include <Grid/qcd/action/txqcd/TXQCDSchurOp.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusFermionEO.h>

NAMESPACE_BEGIN(Grid);

class TXQCDMobiusSchurOp : public LinearOperatorBase<TXQCDFermionNf> {
 public:
  TXQCDMobiusFermionEO &_Mat;

  TXQCDMobiusSchurOp(TXQCDMobiusFermionEO &Mat) : _Mat(Mat) {}

  // Mpc = Moo - Moe Mee^{-1} Meo  (odd sublattice)
  void Mpc(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    GridBase *rbgrid = in.f[0].Grid();
    TXQCDFermionNf tmp(rbgrid), tmp2(rbgrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      tmp.f[a].Checkerboard()  = Even;
      tmp2.f[a].Checkerboard() = Even;
      out.f[a].Checkerboard()  = Odd;
    }
    _Mat.Meooe(in, tmp);        // Meo:  odd → even
    _Mat.MooeeInv(tmp, tmp2);   // Mee^{-1}: even → even
    _Mat.Meooe(tmp2, tmp);      // Moe:  even → odd
    _Mat.Mooee(in, out);        // Moo:  odd → odd
    for (int a = 0; a < TxqcdNf; ++a)
      out.f[a] = out.f[a] - tmp.f[a];
  }

  void MpcDag(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    GridBase *rbgrid = in.f[0].Grid();
    TXQCDFermionNf tmp(rbgrid), tmp2(rbgrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      tmp.f[a].Checkerboard()  = Even;
      tmp2.f[a].Checkerboard() = Even;
      out.f[a].Checkerboard()  = Odd;
    }
    _Mat.MeooeDag(in, tmp);
    _Mat.MooeeInvDag(tmp, tmp2);
    _Mat.MeooeDag(tmp2, tmp);
    _Mat.MooeeDag(in, out);
    for (int a = 0; a < TxqcdNf; ++a)
      out.f[a] = out.f[a] - tmp.f[a];
  }

  // --- LinearOperatorBase interface ---
  void OpDiag(const TXQCDFermionNf &, TXQCDFermionNf &) override {
    GRID_ASSERT(0 && "TXQCDMobiusSchurOp::OpDiag not implemented");
  }
  void OpDir(const TXQCDFermionNf &, TXQCDFermionNf &, int, int) override {
    GRID_ASSERT(0 && "TXQCDMobiusSchurOp::OpDir not implemented");
  }
  void OpDirAll(const TXQCDFermionNf &,
                std::vector<TXQCDFermionNf> &) override {
    GRID_ASSERT(0 && "TXQCDMobiusSchurOp::OpDirAll not implemented");
  }
  void Op(const TXQCDFermionNf &in, TXQCDFermionNf &out) override {
    Mpc(in, out);
  }
  void AdjOp(const TXQCDFermionNf &in, TXQCDFermionNf &out) override {
    MpcDag(in, out);
  }
  void HermOpAndNorm(const TXQCDFermionNf &in, TXQCDFermionNf &out,
                     RealD &n1, RealD &n2) override {
    HermOp(in, out);
    ComplexD dot = innerProduct(in, out);
    n1 = real(dot);
    n2 = norm2(out);
  }
  // HermOp = Mpc† Mpc
  void HermOp(const TXQCDFermionNf &in, TXQCDFermionNf &out) override {
    TXQCDFermionNf tmp(in.f[0].Grid());
    for (int a = 0; a < TxqcdNf; ++a) tmp.f[a].Checkerboard() = Odd;
    Mpc(in, tmp);
    MpcDag(tmp, out);
  }
};

NAMESPACE_END(Grid);
