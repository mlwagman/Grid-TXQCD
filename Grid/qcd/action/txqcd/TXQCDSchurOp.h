#pragma once
// Schur complement operator for the EO-preconditioned TXQCD Wilson operator.
//
// Mpc = Moo - Moe * Mee^{-1} * Meo, operating on TXQCDFermionNf on the
// odd red-black sublattice. HermOp computes Mpc†Mpc for CG.
//
// Implements LinearOperatorBase<TXQCDFermionNf> so that Grid's stock
// ConjugateGradient and ConjugateGradientMultiShift can be used directly.

#include <Grid/qcd/action/txqcd/TXQCDWilsonFermionEO.h>

NAMESPACE_BEGIN(Grid);

class TXQCDSchurOp : public LinearOperatorBase<TXQCDFermionNf> {
 public:
  TXQCDWilsonFermionEO &_Mat;

  TXQCDSchurOp(TXQCDWilsonFermionEO &Mat) : _Mat(Mat) {}

  // Mpc = Moo - Moe Mee^{-1} Meo
  void Mpc(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    GridBase *rbgrid = in.Grid();
    TXQCDFermionNf tmp(rbgrid);
    TXQCDFermionNf tmp2(rbgrid);

    _Mat.Meooe(in, tmp);        // Meo: odd → even
    _Mat.MooeeInv(tmp, tmp2);   // Mee^{-1}: even → even
    _Mat.Meooe(tmp2, tmp);      // Moe: even → odd
    _Mat.Mooee(in, out);        // Moo: odd → odd
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

  // --- LinearOperatorBase interface ---

  void OpDiag(const TXQCDFermionNf &in, TXQCDFermionNf &out) override {
    GRID_ASSERT(0 && "TXQCDSchurOp::OpDiag not implemented");
  }
  void OpDir(const TXQCDFermionNf &in, TXQCDFermionNf &out, int dir,
             int disp) override {
    GRID_ASSERT(0 && "TXQCDSchurOp::OpDir not implemented");
  }
  void OpDirAll(const TXQCDFermionNf &in,
                std::vector<TXQCDFermionNf> &out) override {
    GRID_ASSERT(0 && "TXQCDSchurOp::OpDirAll not implemented");
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

  // HermOp = Mpc† Mpc (for CG on the normal equations).
  void HermOp(const TXQCDFermionNf &in, TXQCDFermionNf &out) override {
    TXQCDFermionNf tmp(in.Grid());
    Mpc(in, tmp);
    MpcDag(tmp, out);
  }
};

// Free functions required by Grid's CG templates for TXQCDFermionNf.
inline void conformable(const TXQCDFermionNf &a, const TXQCDFermionNf &b) {
  for (int i = 0; i < TxqcdNf; ++i) conformable(a.f[i], b.f[i]);
}

inline RealD axpy_norm(TXQCDFermionNf &r, const ComplexD &a,
                       const TXQCDFermionNf &x, const TXQCDFermionNf &y) {
  RealD n = 0;
  for (int i = 0; i < TxqcdNf; ++i) {
    r.f[i] = a * x.f[i] + y.f[i];
    n += norm2(r.f[i]);
  }
  return n;
}

inline ComplexD innerProduct(const TXQCDFermionNf &a,
                             const TXQCDFermionNf &b);  // already in TXQCDDeltaOp.h

// operator- for TXQCDFermionNf (needed by CG residual computation)
inline TXQCDFermionNf operator-(const TXQCDFermionNf &a,
                                const TXQCDFermionNf &b) {
  TXQCDFermionNf r(a.Grid());
  for (int i = 0; i < TxqcdNf; ++i) r.f[i] = a.f[i] - b.f[i];
  return r;
}

inline TXQCDFermionNf operator+(const TXQCDFermionNf &a,
                                const TXQCDFermionNf &b) {
  TXQCDFermionNf r(a.Grid());
  for (int i = 0; i < TxqcdNf; ++i) r.f[i] = a.f[i] + b.f[i];
  return r;
}

// Scalar multiply
inline TXQCDFermionNf operator*(const ComplexD &a, const TXQCDFermionNf &x) {
  TXQCDFermionNf r(x.Grid());
  for (int i = 0; i < TxqcdNf; ++i) r.f[i] = a * x.f[i];
  return r;
}
inline TXQCDFermionNf operator*(const RealD &a, const TXQCDFermionNf &x) {
  TXQCDFermionNf r(x.Grid());
  for (int i = 0; i < TxqcdNf; ++i) r.f[i] = a * x.f[i];
  return r;
}

NAMESPACE_END(Grid);
