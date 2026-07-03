#pragma once
// Packed single-Lattice view of the DTXQCD doubled fermion + a
// LinearOperatorBase adapter, enabling Grid's algebraic multigrid stack
// (Aggregation / GeneralCoarsenedMatrix / TwoLevelADEF2) on the Schur
// operator Mpc^dag Mpc.
//
// Grid's MG entry points (blockProject / Aggregation / CoarsenOperator)
// require the fine field to be a single Lattice<Fobj>; DTXQCDFermionDoubled
// is a struct of 2*Nf separate LatticeFermions.  vDtxqcdPackedVector stacks
// them as an outer iVector over k = 0..2*Nf-1 in the fixed order
//   k=0: upper.f[0], k=1: upper.f[1], k=2: lower.f[0], k=3: lower.f[1]
// (the same order as DoubledStateCSF::csf[]).  Pack/unpack is a pure
// per-site data shuffle on identical SIMD layout (both vComplex), so the
// round trip is bit-exact and checkerboard-preserving.
//
// DtxqcdPackedMpcHermOp exposes Op = AdjOp = HermOp = Mpc^dag Mpc (HPD) on
// packed odd-CB fields.  The matvec routes through the device-resident
// DTXQCDMpcOpQUDA_EO chain (M_device_csf then Mdag_device_csf, one
// host<->device round trip per HermOp) when a device op is supplied and
// DTXQCD_GRIDMG_QUDA_FINE != 0; otherwise through the Grid DTXQCDMpcOp
// reference.  OpDiag/OpDir are asserted out: only the algebraic coarsening
// family (GeneralCoarsenedMatrix, Op-probe only) is supported, the classic
// stencil CoarsenedMatrix is not.
//
// Checkerboard-flag convention: Grid's MG machinery (Aggregation noise
// fields, CoarsenOperator phase vectors, ADEF2 temporaries) creates fine
// fields with the DEFAULT checkerboard flag and never stamps the cb it was
// constructed with (see Aggregates.h CreateSubspace*).  The Schur Mpc is
// only defined on one fixed checkerboard, and Meooe dispatches on the input
// flag — so this adapter FORCES cb (ctor arg, default Odd) on the unpacked
// components and passes the packed field's own flag through untouched.
// Every packed fine field is thus interpreted as living on `cb` regardless
// of its bookkeeping flag, exactly analogous to how the HDCG reference test
// treats its Schur operator.

#include <cstdlib>

#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOp.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA_EO.h>
#include <Grid/qcd/action/dtxqcd/DoubledStateCSF.h>
#endif

NAMESPACE_BEGIN(Grid);

static constexpr int DtxqcdNDoubled = 2 * DtxqcdNf;

// 48 complex per site: [doubling x flavor][spin][color].
typedef iVector<iVector<iVector<vComplex, Nc>, Ns>, DtxqcdNDoubled>
    vDtxqcdPackedVector;
typedef Lattice<vDtxqcdPackedVector> LatticeDtxqcdPacked;

// The pack kernels index the four components explicitly (matching the
// DoubledStateCSF csf[0..3] convention); keep them honest for exotic Nf.
static_assert(DtxqcdNf == 2, "DTXQCDPackedField pack kernels assume Nf=2");

inline void DtxqcdPackDoubled(const DTXQCDFermionDoubled &in,
                              LatticeDtxqcdPacked &out) {
  GridBase *grid = in.Grid();
  conformable(grid, out.Grid());
  out.Checkerboard() = in.upper.f[0].Checkerboard();
  autoView(u0, in.upper.f[0], AcceleratorRead);
  autoView(u1, in.upper.f[1], AcceleratorRead);
  autoView(l0, in.lower.f[0], AcceleratorRead);
  autoView(l1, in.lower.f[1], AcceleratorRead);
  autoView(ov, out, AcceleratorWrite);
  accelerator_for(ss, grid->oSites(), vComplex::Nsimd(), {
    decltype(coalescedRead(ov[ss])) o;
    o(0) = coalescedRead(u0[ss])();
    o(1) = coalescedRead(u1[ss])();
    o(2) = coalescedRead(l0[ss])();
    o(3) = coalescedRead(l1[ss])();
    coalescedWrite(ov[ss], o);
  });
}

inline void DtxqcdUnpackDoubled(const LatticeDtxqcdPacked &in,
                                DTXQCDFermionDoubled &out) {
  GridBase *grid = in.Grid();
  conformable(grid, out.Grid());
  for (int a = 0; a < DtxqcdNf; ++a) {
    out.upper.f[a].Checkerboard() = in.Checkerboard();
    out.lower.f[a].Checkerboard() = in.Checkerboard();
  }
  autoView(iv, in, AcceleratorRead);
  autoView(u0, out.upper.f[0], AcceleratorWrite);
  autoView(u1, out.upper.f[1], AcceleratorWrite);
  autoView(l0, out.lower.f[0], AcceleratorWrite);
  autoView(l1, out.lower.f[1], AcceleratorWrite);
  accelerator_for(ss, grid->oSites(), vComplex::Nsimd(), {
    auto o = coalescedRead(iv[ss]);
    decltype(coalescedRead(u0[ss])) s0, s1, s2, s3;
    s0() = o(0);
    s1() = o(1);
    s2() = o(2);
    s3() = o(3);
    coalescedWrite(u0[ss], s0);
    coalescedWrite(u1[ss], s1);
    coalescedWrite(l0[ss], s2);
    coalescedWrite(l1[ss], s3);
  });
}

// HPD fine operator for Grid MG: Op = AdjOp = HermOp = Mpc^dag Mpc on packed
// odd-CB doubled fermions.  Mirrors HermOpAdaptor (Test_general_coarse_hdcg_
// phys.cc) so it plugs into Aggregation null-vector generation, the algebraic
// CoarsenOperator, TwoLevelADEF2 and plain ConjugateGradient alike.
class DtxqcdPackedMpcHermOp : public LinearOperatorBase<LatticeDtxqcdPacked> {
 public:
  typedef LatticeDtxqcdPacked Field;

  DtxqcdPackedMpcHermOp(DTXQCDMpcOp &mpc, GridBase *rbgrid, int cb = Odd)
      : mpc_(mpc), cb_(cb),
        tmp_in_(rbgrid), tmp_mid_(rbgrid), tmp_out_(rbgrid) {}

#ifdef GRID_HAVE_QUDA
  // Same, with the device-resident EO chain as the matvec fast path
  // (opt-out via env DTXQCD_GRIDMG_QUDA_FINE=0).
  DtxqcdPackedMpcHermOp(DTXQCDMpcOp &mpc, DTXQCDMpcOpQUDA_EO &Mq,
                        GridBase *rbgrid, int cb = Odd)
      : mpc_(mpc), Mq_(&Mq), cb_(cb),
        tmp_in_(rbgrid), tmp_mid_(rbgrid), tmp_out_(rbgrid) {
    if (const char *e = std::getenv("DTXQCD_GRIDMG_QUDA_FINE"))
      if (std::atoi(e) == 0) Mq_ = nullptr;
    if (const char *e = std::getenv("DTXQCD_GRIDMG_SKIP_ZERO"))
      skip_zero_ = std::atoi(e) != 0;
  }

  ~DtxqcdPackedMpcHermOp() {
    // Mirror DTXQCDMpcOpQUDA_EO::~: QUDA teardown order makes CSF
    // destructors unsafe at exit — drop ownership without destroying.
    for (auto *s : {&in_q_, &mid_q_, &out_q_, &scr_q_})
      for (auto &u : s->csf) (void)u.release();
  }
#endif

  void HermOp(const Field &in, Field &out) {
    double t0 = usecond();
    DtxqcdUnpackDoubled(in, tmp_in_);
    StampCb_(tmp_in_);
    double t1 = usecond();
    t_unpack_us += t1 - t0;
#ifdef GRID_HAVE_QUDA
    if (Mq_) {
      HermOpDevice_();
    } else
#endif
    {
      double tm = usecond();
      mpc_.M(tmp_in_, tmp_mid_);
      mpc_.Mdag(tmp_mid_, tmp_out_);
      t_matvec_us += usecond() - tm;
    }
    double t2 = usecond();
    DtxqcdPackDoubled(tmp_out_, out);
    out.Checkerboard() = in.Checkerboard();  // flag pass-through (see header)
    t_pack_us += usecond() - t2;
    ncalls++;
    t_herm_us += usecond() - t0;
  }
  void Op(const Field &in, Field &out) { HermOp(in, out); }
  void AdjOp(const Field &in, Field &out) { HermOp(in, out); }
  void HermOpAndNorm(const Field &in, Field &out, RealD &n1, RealD &n2) {
    HermOp(in, out);
    ComplexD dot = innerProduct(in, out);
    n1 = real(dot);
    n2 = norm2(out);
  }
  void OpDiag(const Field &in, Field &out) { GRID_ASSERT(0); }
  void OpDir(const Field &in, Field &out, int dir, int disp) { GRID_ASSERT(0); }
  void OpDirAll(const Field &in, std::vector<Field> &out) { GRID_ASSERT(0); }

  // Fine-matvec accounting for the coarsening/solve experiments.  Phase
  // accumulators (us): unpack = packed->struct Grid kernel (+cb stamps),
  // h2d/d2h = the host-mode copy_{from,to}_grid CSF crossings, zero = the
  // per-call CSF zeroing (skippable via DTXQCD_GRIDMG_SKIP_ZERO=1: zero only
  // once after allocation — measures whether the halo-padding guard is
  // load-bearing here), matvec = M_device_csf + Mdag_device_csf (or the
  // Grid-path M+Mdag), pack = struct->packed Grid kernel.
  size_t ncalls = 0;
  double t_herm_us = 0.0;
  double t_unpack_us = 0.0, t_pack_us = 0.0;
  double t_h2d_us = 0.0, t_d2h_us = 0.0;
  double t_zero_us = 0.0, t_matvec_us = 0.0;

  void ResetCounters() {
    ncalls = 0;
    t_herm_us = t_unpack_us = t_pack_us = 0.0;
    t_h2d_us = t_d2h_us = t_zero_us = t_matvec_us = 0.0;
  }
  void ReportCounters(const std::string &tag) const {
    if (!ncalls) return;
    double n = (double)ncalls;
    std::cout << GridLogMessage << "[packedop " << tag << "] " << ncalls
              << " HermOps, per-call ms: total " << t_herm_us / n * 1e-3
              << " = unpack " << t_unpack_us / n * 1e-3
              << " + h2d " << t_h2d_us / n * 1e-3
              << " + zero " << t_zero_us / n * 1e-3
              << " + matvec " << t_matvec_us / n * 1e-3
              << " + d2h " << t_d2h_us / n * 1e-3
              << " + pack " << t_pack_us / n * 1e-3 << std::endl;
  }

 private:
  void StampCb_(DTXQCDFermionDoubled &v) const {
    for (int a = 0; a < DtxqcdNf; ++a) {
      v.upper.f[a].Checkerboard() = cb_;
      v.lower.f[a].Checkerboard() = cb_;
    }
  }

#ifdef GRID_HAVE_QUDA
  void HermOpDevice_() {
    if (!csf_ready_) {
      quda::ColorSpinorParam ptmpl = Mq_->MakeNativeCsfParam();
      in_q_.allocate(ptmpl);
      mid_q_.allocate(ptmpl);
      out_q_.allocate(ptmpl);
      scr_q_.allocate_lower_only(ptmpl);
      for (int d = 0; d < 4; ++d) X_full_[d] = Mq_->GaugeParam().X[d];
      csf_ready_ = true;
    }
    double t0 = usecond();
    in_q_.copy_from_grid(tmp_in_, Mq_->InvertParam(), X_full_);
    double t1 = usecond();
    t_h2d_us += t1 - t0;
    // Zero output/intermediate buffers including halo padding: QUDA's dirac
    // gather can read uninitialised padded slots after NULL_FIELD_CREATE
    // (same guard as DtxqcdHermOpQUDA_StyleC).  DTXQCD_GRIDMG_SKIP_ZERO=1
    // zeroes only the first call after allocation (overhead probe).
    if (!skip_zero_ || !zeroed_once_) {
      mid_q_.zero();
      out_q_.zero();
      scr_q_.zero();
      zeroed_once_ = true;
    }
    double t2 = usecond();
    t_zero_us += t2 - t1;
    quda::ColorSpinorField *iu[DtxqcdNf] = {in_q_.csf[0].get(), in_q_.csf[1].get()};
    quda::ColorSpinorField *il[DtxqcdNf] = {in_q_.csf[2].get(), in_q_.csf[3].get()};
    quda::ColorSpinorField *mu[DtxqcdNf] = {mid_q_.csf[0].get(), mid_q_.csf[1].get()};
    quda::ColorSpinorField *ml[DtxqcdNf] = {mid_q_.csf[2].get(), mid_q_.csf[3].get()};
    quda::ColorSpinorField *ou[DtxqcdNf] = {out_q_.csf[0].get(), out_q_.csf[1].get()};
    quda::ColorSpinorField *ol[DtxqcdNf] = {out_q_.csf[2].get(), out_q_.csf[3].get()};
    quda::ColorSpinorField *sl[DtxqcdNf] = {scr_q_.csf[0].get(), scr_q_.csf[1].get()};
    double t3 = usecond();
    Mq_->M_device_csf(iu, il, mu, ml, sl);
    Mq_->Mdag_device_csf(mu, ml, ou, ol, sl);
    double t4 = usecond();
    t_matvec_us += t4 - t3;
    out_q_.copy_to_grid(tmp_out_, Mq_->InvertParam(), X_full_);
    StampCb_(tmp_out_);
    t_d2h_us += usecond() - t4;
  }
#endif

  DTXQCDMpcOp &mpc_;
  int cb_ = Odd;
#ifdef GRID_HAVE_QUDA
  DTXQCDMpcOpQUDA_EO *Mq_ = nullptr;
  DtxqcdQudaStyleC::DoubledStateCSF in_q_, mid_q_, out_q_, scr_q_;
  int X_full_[4];
  bool csf_ready_ = false;
  bool skip_zero_ = false;
  bool zeroed_once_ = false;
#endif
  DTXQCDFermionDoubled tmp_in_, tmp_mid_, tmp_out_;
};

NAMESPACE_END(Grid);
