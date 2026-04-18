#pragma once
// Schur complement operator for the EO-preconditioned TXQCD Wilson-Clover
// operator. Identical interface to TXQCDSchurOp but wraps
// TXQCDWilsonCloverFermionEO instead of TXQCDWilsonFermionEO.

#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>

NAMESPACE_BEGIN(Grid);

class TXQCDCloverSchurOp : public LinearOperatorBase<TXQCDFermionNf> {
 public:
  TXQCDWilsonCloverFermionEO &_Mat;

  TXQCDCloverSchurOp(TXQCDWilsonCloverFermionEO &Mat) : _Mat(Mat) {}

  void Mpc(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    GridBase *rbgrid = in.Grid();
    TXQCDFermionNf tmp(rbgrid);
    TXQCDFermionNf tmp2(rbgrid);

    _Mat.Meooe(in, tmp);
    _Mat.MooeeInv(tmp, tmp2);
    _Mat.Meooe(tmp2, tmp);
    _Mat.Mooee(in, out);
    for (int a = 0; a < TxqcdNf; ++a)
      out.f[a] = out.f[a] - tmp.f[a];
  }

  void MpcDag(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    GridBase *rbgrid = in.Grid();
    TXQCDFermionNf tmp(rbgrid);
    TXQCDFermionNf tmp2(rbgrid);

    _Mat.MeooeDag(in, tmp);
    _Mat.MooeeInvDag(tmp, tmp2);
    _Mat.MeooeDag(tmp2, tmp);
    _Mat.MooeeDag(in, out);
    for (int a = 0; a < TxqcdNf; ++a)
      out.f[a] = out.f[a] - tmp.f[a];
  }

  void OpDiag(const TXQCDFermionNf &in, TXQCDFermionNf &out) override {
    GRID_ASSERT(0 && "TXQCDCloverSchurOp::OpDiag not implemented");
  }
  void OpDir(const TXQCDFermionNf &in, TXQCDFermionNf &out, int dir,
             int disp) override {
    GRID_ASSERT(0 && "TXQCDCloverSchurOp::OpDir not implemented");
  }
  void OpDirAll(const TXQCDFermionNf &in,
                std::vector<TXQCDFermionNf> &out) override {
    GRID_ASSERT(0 && "TXQCDCloverSchurOp::OpDirAll not implemented");
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

  void HermOp(const TXQCDFermionNf &in, TXQCDFermionNf &out) override {
    TXQCDFermionNf tmp(in.Grid());
    Mpc(in, tmp);
    MpcDag(tmp, out);
  }
};

NAMESPACE_END(Grid);
