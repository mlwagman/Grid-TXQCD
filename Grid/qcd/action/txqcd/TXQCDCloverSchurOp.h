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

  // Phase M.4.b: fused multi-RHS Mpc/MpcDag/HermOp.  Cuts cuBLAS Mooee/MooeeInv
  // work to one fused gemm per Mpc step (instead of N_RHS separate gemv calls);
  // Wilson hop loops sequentially.
  //
  // Phase M.4.c: scratch buffers (`scratch_mpc_*`) are cached across calls,
  // keyed by (rb, NRHS).  Naive per-call `std::vector<TXQCDFermionNf>(N,...)`
  // construction was responsible for ~36s/cfg overhead at 16³×48 NRHS=24
  // (5 vectors × 643 HermOpN calls × 24 RHS × 2 fields × 19 MB).
  void EnsureScratch(GridBase *rb, int N) {
    if (scratch_rb_ != rb) {
      // Grid changed (rare) — invalidate everything.
      scratch_rb_ = rb;
      scratch_tmp_.clear();
      scratch_tmp2_.clear();
      scratch_h_tmp_.clear();
    }
    while ((int)scratch_tmp_.size()   < N) scratch_tmp_.emplace_back(rb);
    while ((int)scratch_tmp2_.size()  < N) scratch_tmp2_.emplace_back(rb);
    while ((int)scratch_h_tmp_.size() < N) scratch_h_tmp_.emplace_back(rb);
    // ApplyMooeeCublasN asserts ins.size() == outs.size(); shrink on demand.
    // Use erase() instead of resize(N) — TXQCDFermionNf has no default ctor.
    if ((int)scratch_tmp_.size()   > N)
      scratch_tmp_.erase(scratch_tmp_.begin() + N, scratch_tmp_.end());
    if ((int)scratch_tmp2_.size()  > N)
      scratch_tmp2_.erase(scratch_tmp2_.begin() + N, scratch_tmp2_.end());
    if ((int)scratch_h_tmp_.size() > N)
      scratch_h_tmp_.erase(scratch_h_tmp_.begin() + N, scratch_h_tmp_.end());
  }

  // Phase M.4.d: fused per-RHS per-flavor subtract.  Single accelerator_for
  // over (oSite, j*Nf+a) replaces 2*N separate Lattice subtractions (each was
  // its own kernel launch).  At 16³×48 NRHS=24 the un-fused subtract was
  // ~6 s/cfg of pure launch overhead.
  void FusedSubtractInPlaceN(std::vector<TXQCDFermionNf> &outs,
                             const std::vector<TXQCDFermionNf> &tmp) {
    int N = (int)outs.size();
    using vobj = typename LatticeFermion::vector_object;
    GridBase *rb = outs[0].f[0].Grid();
    uint64_t oSites = rb->oSites();
    constexpr int Nsimd = vobj::Nsimd();
    int total = N * TxqcdNf;

    std::vector<vobj*> outs_h(total), tmp_h(total);
    std::vector<LatticeView<vobj>> outs_v, tmp_v;
    outs_v.reserve(total); tmp_v.reserve(total);
    for (int j = 0; j < N; ++j) {
      for (int a = 0; a < TxqcdNf; ++a) {
        int k = j * TxqcdNf + a;
        outs_v.push_back(outs[j].f[a].View(AcceleratorWrite));
        tmp_v.push_back (tmp [j].f[a].View(AcceleratorRead));
        outs_h[k] = outs_v[k].getHostPointer();
        tmp_h [k] = tmp_v [k].getHostPointer();
      }
    }
    deviceVector<vobj*> outs_d(total), tmp_d(total);
    acceleratorCopyToDevice(outs_h.data(), &outs_d[0], total * sizeof(vobj*));
    acceleratorCopyToDevice(tmp_h.data(),  &tmp_d[0],  total * sizeof(vobj*));
    vobj **out_p = &outs_d[0];
    vobj **tmp_p = &tmp_d[0];
    accelerator_for(s, oSites, Nsimd, {
      for (int k = 0; k < total; ++k) {
        auto a = coalescedRead(out_p[k][s]);
        auto b = coalescedRead(tmp_p[k][s]);
        coalescedWrite(out_p[k][s], a - b);
      }
    });
    for (auto &v : outs_v) v.ViewClose();
    for (auto &v : tmp_v)  v.ViewClose();
  }

  void MpcN(const std::vector<TXQCDFermionNf> &ins,
            std::vector<TXQCDFermionNf> &outs) {
    int N = (int)ins.size();
    GridBase *rb = ins[0].Grid();
    EnsureScratch(rb, N);
    std::vector<TXQCDFermionNf> &tmp  = scratch_tmp_;
    std::vector<TXQCDFermionNf> &tmp2 = scratch_tmp2_;

    _Mat.MeooeN(ins, tmp);          // cb→!cb (per-RHS Wilson hop)
    _Mat.MooeeInvN(tmp, tmp2);      // M_ee^{-1} via fused cuBLAS
    _Mat.MeooeN(tmp2, tmp);         // !cb→cb
    _Mat.MooeeN(ins, outs);         // Mooee in via fused cuBLAS
    FusedSubtractInPlaceN(outs, tmp);
  }

  void MpcDagN(const std::vector<TXQCDFermionNf> &ins,
               std::vector<TXQCDFermionNf> &outs) {
    int N = (int)ins.size();
    GridBase *rb = ins[0].Grid();
    EnsureScratch(rb, N);
    std::vector<TXQCDFermionNf> &tmp  = scratch_tmp_;
    std::vector<TXQCDFermionNf> &tmp2 = scratch_tmp2_;

    _Mat.MeooeDagN(ins, tmp);
    _Mat.MooeeInvDagN(tmp, tmp2);
    _Mat.MeooeDagN(tmp2, tmp);
    _Mat.MooeeDagN(ins, outs);
    FusedSubtractInPlaceN(outs, tmp);
  }

  void HermOpN(const std::vector<TXQCDFermionNf> &ins,
               std::vector<TXQCDFermionNf> &outs) {
    int N = (int)ins.size();
    GridBase *rb = ins[0].Grid();
    EnsureScratch(rb, N);
    std::vector<TXQCDFermionNf> &tmp = scratch_h_tmp_;
    MpcN(ins, tmp);
    MpcDagN(tmp, outs);
  }

 private:
  // Phase M.4.c scratch — preallocated lazily on first HermOpN call.
  // tmp/tmp2 used by Mpc{N,DagN}; h_tmp used by HermOpN to chain Mpc·MpcDag.
  GridBase *scratch_rb_{nullptr};
  std::vector<TXQCDFermionNf> scratch_tmp_, scratch_tmp2_, scratch_h_tmp_;
};

NAMESPACE_END(Grid);
