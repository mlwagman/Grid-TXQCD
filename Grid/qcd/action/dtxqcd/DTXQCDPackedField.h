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
// DTXQCDMpcOpQUDA_EO chain (M_device_csf then Mdag_device_csf) when a device
// op is supplied and DTXQCD_GRIDMG_QUDA_FINE != 0; otherwise through the
// Grid DTXQCDMpcOp reference.  Under GRID_CUDA the packed ⇄ CSF conversion
// is fully device-resident by default (DTXQCD_GRIDMG_DEVICE_COPY=0 restores
// the host-mode copy_{from,to}_grid crossing, which costs 65% of a HermOp).  OpDiag/OpDir are asserted out: only the algebraic coarsening
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
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
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

// HPD fine operator for Grid MG: Op = AdjOp = HermOp = A^dag A on packed
// doubled fermions, templated on the (Grid-op, QUDA-op) pair so both the
// odd-CB Schur pair (DTXQCDMpcOp, DTXQCDMpcOpQUDA_EO) and the full-volume
// pair (DTXQCDMOp, DTXQCDMpcOpQUDA) instantiate — the M6 probe showed the
// full operator is the better-conditioned MG substrate at small lambda
// (the near-singular M_ee sites amplify only the Schur complement).
// Mirrors HermOpAdaptor (Test_general_coarse_hdcg_phys.cc) so it plugs into
// Aggregation null-vector generation, the algebraic CoarsenOperator,
// TwoLevelADEF2 and plain ConjugateGradient alike.
//
// The device-resident conversion table handles both fine-grid types: on a
// checkerboarded grid the staging index is the local lex site (Grid RB order
// == QUDA cb_site order); on a full grid it is QUDA's parity-blocked order
// eo_idx = parity*V/2 + lex/2 (replicating fermion_to_eo_buffer,
// QudaFieldConvert.h:133-140).
template <class MopGridT, class MopQudaT>
class DtxqcdPackedHermOpT : public LinearOperatorBase<LatticeDtxqcdPacked> {
 public:
  typedef LatticeDtxqcdPacked Field;

  DtxqcdPackedHermOpT(MopGridT &mpc, GridBase *finegrid, int cb = Odd)
      : mpc_(mpc), cb_(cb),
        tmp_in_(finegrid), tmp_mid_(finegrid), tmp_out_(finegrid) {}

#ifdef GRID_HAVE_QUDA
  // Same, with the device-resident chain as the matvec fast path
  // (opt-out via env DTXQCD_GRIDMG_QUDA_FINE=0).
  DtxqcdPackedHermOpT(MopGridT &mpc, MopQudaT &Mq,
                      GridBase *finegrid, int cb = Odd)
      : mpc_(mpc), Mq_(&Mq), cb_(cb),
        tmp_in_(finegrid), tmp_mid_(finegrid), tmp_out_(finegrid) {
    if (const char *e = std::getenv("DTXQCD_GRIDMG_QUDA_FINE"))
      if (std::atoi(e) == 0) Mq_ = nullptr;
    if (const char *e = std::getenv("DTXQCD_GRIDMG_SKIP_ZERO"))
      skip_zero_ = std::atoi(e) != 0;
#ifdef GRID_CUDA
    device_copy_ = true;  // default ON: 65% of a HermOp is the host crossing
    if (const char *e = std::getenv("DTXQCD_GRIDMG_DEVICE_COPY"))
      device_copy_ = std::atoi(e) != 0;
#endif
  }

  ~DtxqcdPackedHermOpT() {
    // Mirror DTXQCDMpcOpQUDA_EO::~: QUDA teardown order makes CSF
    // destructors unsafe at exit — drop ownership without destroying.
    for (auto *s : {&in_q_, &mid_q_, &out_q_, &scr_q_,
                    &in_sp_, &mid_sp_, &out_sp_, &scr_sp_})
      for (auto &u : s->csf) (void)u.release();
    for (auto &u : stage_csf_) (void)u.release();
    if (lex_tbl_d_) { acceleratorFreeDevice(lex_tbl_d_); lex_tbl_d_ = nullptr; }
    for (auto &p : stage_d_)
      if (p) { acceleratorFreeDevice(p); p = nullptr; }
  }
#endif

  void HermOp(const Field &in, Field &out) {
    double t0 = usecond();
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
    if (Mq_ && device_copy_) {
      // Device-resident conversion path: no struct roundtrip, no host-mode
      // copy_{from,to}_grid.  See PackedToCsfDevice_/CsfToPackedDevice_.
      HermOpDeviceDirect_(in, out);
      out.Checkerboard() = in.Checkerboard();
      ncalls++;
      t_herm_us += usecond() - t0;
      return;
    }
#endif
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

  // Single-leg device-resident applies (deformed-contour stack, 2026-07-13):
  // M-tilde = M_device + Grid-side per-timeslice hop corrections needs M and
  // Mdag SEPARATELY with the corrections applied in between, so the fused
  // HermOp is unusable there.  Same conversion/CSF machinery as
  // HermOpDeviceDirect_; non-QUDA builds (or device_copy off) fall back to
  // the struct-path mpc_ reference so the gates run everywhere.
  void MDeviceDirect(const Field &in, Field &out) {
    SingleLegDirect_(in, out, /*dagger=*/false);
  }
  void MdagDeviceDirect(const Field &in, Field &out) {
    SingleLegDirect_(in, out, /*dagger=*/true);
  }
  void SingleLegDirect_(const Field &in, Field &out, bool dagger) {
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
    if (Mq_ && device_copy_) {
      EnsureCsf_();
      InitDeviceCopy_(in.Grid());
      PackedToCsfDevice_(in, in_q_);
      if (!skip_zero_ || !zeroed_once_) {
        mid_q_.zero();
        out_q_.zero();
        scr_q_.zero();
        zeroed_once_ = true;
      }
      quda::ColorSpinorField *iu[DtxqcdNf] = {in_q_.csf[0].get(),
                                              in_q_.csf[1].get()};
      quda::ColorSpinorField *il[DtxqcdNf] = {in_q_.csf[2].get(),
                                              in_q_.csf[3].get()};
      quda::ColorSpinorField *ou[DtxqcdNf] = {out_q_.csf[0].get(),
                                              out_q_.csf[1].get()};
      quda::ColorSpinorField *ol[DtxqcdNf] = {out_q_.csf[2].get(),
                                              out_q_.csf[3].get()};
      quda::ColorSpinorField *sl[DtxqcdNf] = {scr_q_.csf[0].get(),
                                              scr_q_.csf[1].get()};
      if (dagger)
        Mq_->Mdag_device_csf(iu, il, ou, ol, sl);
      else
        Mq_->M_device_csf(iu, il, ou, ol, sl);
      CsfToPackedDevice_(out_q_, out);
      out.Checkerboard() = in.Checkerboard();
      ncalls++;
      return;
    }
#endif
    DtxqcdUnpackDoubled(in, tmp_in_);
    StampCb_(tmp_in_);
    if (dagger)
      mpc_.Mdag(tmp_in_, tmp_out_);
    else
      mpc_.M(tmp_in_, tmp_out_);
    DtxqcdPackDoubled(tmp_out_, out);
    out.Checkerboard() = in.Checkerboard();
    ncalls++;
  }

#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
  // Single-leg apply with a CSF-resident post-matvec hook: hook(in_csf[4],
  // out_csf[4]) runs after M/Mdag_device_csf and BEFORE the export, with
  // fields ordered {upper0, upper1, lower0, lower1}.  The deformed-operator
  // native hop-correction kernel rides here (out += corr(in) while both
  // sides are still device-native).
  template <class Hook>
  void SingleLegDirectHooked(const Field &in, Field &out, bool dagger,
                             Hook &&hook) {
    GRID_ASSERT(Mq_ && device_copy_ &&
                "SingleLegDirectHooked requires the QUDA device-copy path");
    EnsureCsf_();
    InitDeviceCopy_(in.Grid());
    PackedToCsfDevice_(in, in_q_);
    if (!skip_zero_ || !zeroed_once_) {
      mid_q_.zero();
      out_q_.zero();
      scr_q_.zero();
      zeroed_once_ = true;
    }
    quda::ColorSpinorField *iu[DtxqcdNf] = {in_q_.csf[0].get(),
                                            in_q_.csf[1].get()};
    quda::ColorSpinorField *il[DtxqcdNf] = {in_q_.csf[2].get(),
                                            in_q_.csf[3].get()};
    quda::ColorSpinorField *ou[DtxqcdNf] = {out_q_.csf[0].get(),
                                            out_q_.csf[1].get()};
    quda::ColorSpinorField *ol[DtxqcdNf] = {out_q_.csf[2].get(),
                                            out_q_.csf[3].get()};
    quda::ColorSpinorField *sl[DtxqcdNf] = {scr_q_.csf[0].get(),
                                            scr_q_.csf[1].get()};
    if (dagger)
      Mq_->Mdag_device_csf(iu, il, ou, ol, sl);
    else
      Mq_->M_device_csf(iu, il, ou, ol, sl);
    quda::ColorSpinorField *ics[4] = {iu[0], iu[1], il[0], il[1]};
    quda::ColorSpinorField *ocs[4] = {ou[0], ou[1], ol[0], ol[1]};
    hook(ics, ocs);
    CsfToPackedDevice_(out_q_, out);
    out.Checkerboard() = in.Checkerboard();
    ncalls++;
  }

  void EnsureCsf_() {
    if (!csf_ready_) {
      quda::ColorSpinorParam ptmpl = Mq_->MakeNativeCsfParam();
      in_q_.allocate(ptmpl);
      mid_q_.allocate(ptmpl);
      out_q_.allocate(ptmpl);
      scr_q_.allocate_lower_only(ptmpl);
      for (int d = 0; d < 4; ++d) X_full_[d] = Mq_->GaugeParam().X[d];
      csf_ready_ = true;
    }
  }
#endif

  // Fine-matvec accounting for the coarsening/solve experiments.  Phase
  // accumulators (us): unpack = packed->struct Grid kernel (+cb stamps),
  // h2d/d2h = the host-mode copy_{from,to}_grid CSF crossings, zero = the
  // per-call CSF zeroing (skippable via DTXQCD_GRIDMG_SKIP_ZERO=1: zero only
  // once after allocation — measures whether the halo-padding guard is
  // load-bearing here), matvec = M_device_csf + Mdag_device_csf (or the
  // Grid-path M+Mdag), pack = struct->packed Grid kernel.
  size_t ncalls = 0;
  size_t ncalls_sp = 0;
  double t_herm_us = 0.0;
  double t_herm_sp_us = 0.0;
  double t_unpack_us = 0.0, t_pack_us = 0.0;
  double t_h2d_us = 0.0, t_d2h_us = 0.0;
  double t_zero_us = 0.0, t_matvec_us = 0.0;

  void ResetCounters() {
    ncalls = 0;
    ncalls_sp = 0;
    t_herm_us = t_herm_sp_us = 0.0;
    t_unpack_us = t_pack_us = 0.0;
    t_h2d_us = t_d2h_us = t_zero_us = t_matvec_us = 0.0;
  }
  void ReportCounters(const std::string &tag) const {
    if (ncalls) {
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
    if (ncalls_sp) {
      double n = (double)ncalls_sp;
      std::cout << GridLogMessage << "[packedop " << tag << "] " << ncalls_sp
                << " SP HermOps, per-call ms: " << t_herm_sp_us / n * 1e-3
                << std::endl;
    }
  }

#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
  // NOTE: public only because nvcc forbids extended __device__ lambdas
  // inside private/protected member functions; not part of the interface.
  // ---- device-resident packed ⇄ CSF conversion ---------------------------
  // Replaces the host-mode copy_{from,to}_grid (host unvectorize + PCIe
  // round trip, measured 65% of a HermOp) with:
  //   packed Lattice --(one Grid gather kernel, lane->lex table)--> persistent
  //   SpaceSpinorColor device staging buffer --(the SAME QUDA CopyColorSpinor
  //   device kernel the host path used: UKQCD->native FloatN)--> CSF
  // The lex mapping replicates unvectorizeToLexOrdArray (Lattice_transfer.h):
  // lcoor = ocoor + rdim*icoor, lex = IndexFromCoor(lcoor, _ldimensions) —
  // within a parity, Grid's RB site order equals QUDA's cb_site order
  // (QudaFieldConvert.h:233), so the staging buffer is the exact byte image
  // the host path produced.  Slot k of the packed site object is contiguous
  // (12 complex, spin-major), matching SpaceSpinorColor per-site layout.
  void InitDeviceCopy_(GridBase *g) {
    if (lex_tbl_d_) return;
    osites_ = g->oSites();
    nsimd_ = g->Nsimd();
    vhalf_ = g->lSites();
    GRID_ASSERT((uint64_t)in_q_.csf[0]->Volume() == vhalf_);
    const bool cb_grid = g->_isCheckerBoarded;
    const int v_eo = (int)(vhalf_ / 2);  // full-grid parity-block size
    std::vector<int> tbl(osites_ * nsimd_);
    std::vector<Coordinate> icoor(nsimd_, Coordinate(g->Nd()));
    for (uint64_t l = 0; l < nsimd_; ++l) g->iCoorFromIindex(icoor[l], l);
    Coordinate ocoor(g->Nd()), lcoor(g->Nd());
    for (uint64_t o = 0; o < osites_; ++o) {
      g->oCoorFromOindex(ocoor, o);
      for (uint64_t l = 0; l < nsimd_; ++l) {
        int psum = 0;
        for (int mu = 0; mu < g->Nd(); ++mu) {
          lcoor[mu] = ocoor[mu] + g->_rdimensions[mu] * icoor[l][mu];
          psum += lcoor[mu];
        }
        int lex;
        Lexicographic::IndexFromCoor(lcoor, lex, g->_ldimensions);
        // CB grid: Grid RB order == QUDA cb_site order (identity on lex).
        // Full grid: QUDA parity-blocked order (fermion_to_eo_buffer).
        tbl[o * nsimd_ + l] =
            cb_grid ? lex : ((psum & 1) * v_eo + (lex >> 1));
      }
    }
    std::size_t tbytes = tbl.size() * sizeof(int);
    lex_tbl_d_ = (int *)acceleratorAllocDevice(tbytes);
    acceleratorCopyToDevice(tbl.data(), lex_tbl_d_, tbytes);
    std::size_t sbytes = (std::size_t)24 * vhalf_ * sizeof(double);
    quda::lat_dim_t X_full;
    for (int d = 0; d < 4; ++d) X_full[d] = X_full_[d];
    for (int d = 4; d < QUDA_MAX_DIM; ++d) X_full[d] = 1;
    inv_dev_ = Mq_->InvertParam();
    inv_dev_.input_location = QUDA_CUDA_FIELD_LOCATION;
    inv_dev_.output_location = QUDA_CUDA_FIELD_LOCATION;
    for (int k = 0; k < 4; ++k) {
      stage_d_[k] = (double *)acceleratorAllocDevice(sbytes);
      quda::ColorSpinorParam sp(stage_d_[k], inv_dev_, X_full,
                                /*pc=*/cb_grid, QUDA_CUDA_FIELD_LOCATION);
      stage_csf_[k] = std::make_unique<quda::ColorSpinorField>(sp);
    }
  }

  void PackedToCsfDevice_(const Field &in,
                          DtxqcdQudaStyleC::DoubledStateCSF &dst) {
    autoView(iv, in, AcceleratorRead);
    const int *tbl = lex_tbl_d_;
    double *s0 = stage_d_[0], *s1 = stage_d_[1];
    double *s2 = stage_d_[2], *s3 = stage_d_[3];
    const uint64_t ns = nsimd_;
    accelerator_for(n, osites_ * nsimd_, 1, {
      uint64_t o = n / ns;
      int lane = (int)(n % ns);
      auto sobj = extractLane(lane, iv[o]);
      const double *src = (const double *)&sobj;  // 4 x 24 doubles, k-major
      const std::size_t lex24 = (std::size_t)tbl[n] * 24;
      for (int j = 0; j < 24; ++j) {
        s0[lex24 + j] = src[0 * 24 + j];
        s1[lex24 + j] = src[1 * 24 + j];
        s2[lex24 + j] = src[2 * 24 + j];
        s3[lex24 + j] = src[3 * 24 + j];
      }
    });
    for (int k = 0; k < 4; ++k) dst.csf[k]->copy(*stage_csf_[k]);
    cudaDeviceSynchronize();
  }

  void CsfToPackedDevice_(const DtxqcdQudaStyleC::DoubledStateCSF &src,
                          Field &out) {
    for (int k = 0; k < 4; ++k) stage_csf_[k]->copy(*src.csf[k]);
    cudaDeviceSynchronize();
    autoView(ov, out, AcceleratorWrite);
    const int *tbl = lex_tbl_d_;
    const double *s0 = stage_d_[0], *s1 = stage_d_[1];
    const double *s2 = stage_d_[2], *s3 = stage_d_[3];
    const uint64_t ns = nsimd_;
    accelerator_for(n, osites_ * nsimd_, 1, {
      uint64_t o = n / ns;
      int lane = (int)(n % ns);
      typename Field::vector_object::scalar_object sobj;
      double *dstp = (double *)&sobj;
      const std::size_t lex24 = (std::size_t)tbl[n] * 24;
      for (int j = 0; j < 24; ++j) {
        dstp[0 * 24 + j] = s0[lex24 + j];
        dstp[1 * 24 + j] = s1[lex24 + j];
        dstp[2 * 24 + j] = s2[lex24 + j];
        dstp[3 * 24 + j] = s3[lex24 + j];
      }
      insertLane(lane, ov[o], sobj);  // lanes write disjoint words: race-free
    });
  }

  // SP-precision HermOp for smoother use (full-volume instantiation only —
  // the EO wrapper's SP path is unported and asserts).  Same device staging
  // as the DP path; QUDA's CopyColorSpinor converts DP staging <-> SP native.
  // The ADEF2 outer iteration stays DP; only the smoother's approximate
  // matvecs run SP, so solution accuracy is untouched (validated by the
  // in-run ADEF2-vs-CG gate).
  void HermOpSp(const Field &in, Field &out) {
    GRID_ASSERT(Mq_ && device_copy_);
    double t0 = usecond();
    if (!csf_ready_) {
      quda::ColorSpinorParam ptmpl = Mq_->MakeNativeCsfParam();
      in_q_.allocate(ptmpl);
      mid_q_.allocate(ptmpl);
      out_q_.allocate(ptmpl);
      scr_q_.allocate_lower_only(ptmpl);
      for (int d = 0; d < 4; ++d) X_full_[d] = Mq_->GaugeParam().X[d];
      csf_ready_ = true;
    }
    InitDeviceCopy_(in.Grid());
    if (!sp_ready_) {
      quda::ColorSpinorParam sp_tmpl = Mq_->MakeNativeCsfParamSp();
      in_sp_.allocate(sp_tmpl);
      mid_sp_.allocate(sp_tmpl);
      out_sp_.allocate(sp_tmpl);
      scr_sp_.allocate_lower_only(sp_tmpl);
      in_sp_.zero();
      mid_sp_.zero();
      out_sp_.zero();
      scr_sp_.zero();
      sp_ready_ = true;
    }
    // packed -> DP staging -> SP native
    {
      autoView(iv, in, AcceleratorRead);
      const int *tbl = lex_tbl_d_;
      double *s0 = stage_d_[0], *s1 = stage_d_[1];
      double *s2 = stage_d_[2], *s3 = stage_d_[3];
      const uint64_t ns = nsimd_;
      accelerator_for(n, osites_ * nsimd_, 1, {
        uint64_t o = n / ns;
        int lane = (int)(n % ns);
        auto sobj = extractLane(lane, iv[o]);
        const double *src = (const double *)&sobj;
        const std::size_t lex24 = (std::size_t)tbl[n] * 24;
        for (int j = 0; j < 24; ++j) {
          s0[lex24 + j] = src[0 * 24 + j];
          s1[lex24 + j] = src[1 * 24 + j];
          s2[lex24 + j] = src[2 * 24 + j];
          s3[lex24 + j] = src[3 * 24 + j];
        }
      });
      for (int k = 0; k < 4; ++k) in_sp_.csf[k]->copy(*stage_csf_[k]);
      cudaDeviceSynchronize();
    }
    quda::ColorSpinorField *iu[DtxqcdNf] = {in_sp_.csf[0].get(), in_sp_.csf[1].get()};
    quda::ColorSpinorField *il[DtxqcdNf] = {in_sp_.csf[2].get(), in_sp_.csf[3].get()};
    quda::ColorSpinorField *mu[DtxqcdNf] = {mid_sp_.csf[0].get(), mid_sp_.csf[1].get()};
    quda::ColorSpinorField *ml[DtxqcdNf] = {mid_sp_.csf[2].get(), mid_sp_.csf[3].get()};
    quda::ColorSpinorField *ou[DtxqcdNf] = {out_sp_.csf[0].get(), out_sp_.csf[1].get()};
    quda::ColorSpinorField *ol[DtxqcdNf] = {out_sp_.csf[2].get(), out_sp_.csf[3].get()};
    quda::ColorSpinorField *sl[DtxqcdNf] = {scr_sp_.csf[0].get(), scr_sp_.csf[1].get()};
    Mq_->M_device_csf_sp(iu, il, mu, ml, sl);
    Mq_->Mdag_device_csf_sp(mu, ml, ou, ol, sl);
    // SP native -> DP staging -> packed
    for (int k = 0; k < 4; ++k) stage_csf_[k]->copy(*out_sp_.csf[k]);
    cudaDeviceSynchronize();
    {
      autoView(ov, out, AcceleratorWrite);
      const int *tbl = lex_tbl_d_;
      const double *s0 = stage_d_[0], *s1 = stage_d_[1];
      const double *s2 = stage_d_[2], *s3 = stage_d_[3];
      const uint64_t ns = nsimd_;
      accelerator_for(n, osites_ * nsimd_, 1, {
        uint64_t o = n / ns;
        int lane = (int)(n % ns);
        typename Field::vector_object::scalar_object sobj;
        double *dstp = (double *)&sobj;
        const std::size_t lex24 = (std::size_t)tbl[n] * 24;
        for (int j = 0; j < 24; ++j) {
          dstp[0 * 24 + j] = s0[lex24 + j];
          dstp[1 * 24 + j] = s1[lex24 + j];
          dstp[2 * 24 + j] = s2[lex24 + j];
          dstp[3 * 24 + j] = s3[lex24 + j];
        }
        insertLane(lane, ov[o], sobj);
      });
    }
    out.Checkerboard() = in.Checkerboard();
    ncalls_sp++;
    t_herm_sp_us += usecond() - t0;
  }

  void HermOpDeviceDirect_(const Field &in, Field &out) {
    if (!csf_ready_) {
      quda::ColorSpinorParam ptmpl = Mq_->MakeNativeCsfParam();
      in_q_.allocate(ptmpl);
      mid_q_.allocate(ptmpl);
      out_q_.allocate(ptmpl);
      scr_q_.allocate_lower_only(ptmpl);
      for (int d = 0; d < 4; ++d) X_full_[d] = Mq_->GaugeParam().X[d];
      csf_ready_ = true;
    }
    InitDeviceCopy_(in.Grid());
    double t0 = usecond();
    PackedToCsfDevice_(in, in_q_);
    double t1 = usecond();
    t_h2d_us += t1 - t0;
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
    Mq_->M_device_csf(iu, il, mu, ml, sl);
    Mq_->Mdag_device_csf(mu, ml, ou, ol, sl);
    double t3 = usecond();
    t_matvec_us += t3 - t2;
    CsfToPackedDevice_(out_q_, out);
    t_d2h_us += usecond() - t3;
  }
#endif  // GRID_HAVE_QUDA && GRID_CUDA

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

  MopGridT &mpc_;
  int cb_ = Odd;
#ifdef GRID_HAVE_QUDA
  MopQudaT *Mq_ = nullptr;
  DtxqcdQudaStyleC::DoubledStateCSF in_q_, mid_q_, out_q_, scr_q_;
  DtxqcdQudaStyleC::DoubledStateCSF in_sp_, mid_sp_, out_sp_, scr_sp_;
  int X_full_[4];
  bool csf_ready_ = false;
  bool sp_ready_ = false;
  bool skip_zero_ = false;
  bool zeroed_once_ = false;
  bool device_copy_ = false;
  int *lex_tbl_d_ = nullptr;
  double *stage_d_[4] = {nullptr, nullptr, nullptr, nullptr};
  std::unique_ptr<quda::ColorSpinorField> stage_csf_[4];
  QudaInvertParam inv_dev_;
  uint64_t osites_ = 0, nsimd_ = 0, vhalf_ = 0;
#endif
  DTXQCDFermionDoubled tmp_in_, tmp_mid_, tmp_out_;
};

#ifdef GRID_HAVE_QUDA
// Odd-CB Schur pair (the original instantiation) and the full-volume pair.
typedef DtxqcdPackedHermOpT<DTXQCDMpcOp, DTXQCDMpcOpQUDA_EO>
    DtxqcdPackedMpcHermOp;
typedef DtxqcdPackedHermOpT<DTXQCDMOp, DTXQCDMpcOpQUDA>
    DtxqcdPackedFullHermOp;
#else
// Non-QUDA builds: dummy second parameter (never dereferenced).
typedef DtxqcdPackedHermOpT<DTXQCDMpcOp, void> DtxqcdPackedMpcHermOp;
typedef DtxqcdPackedHermOpT<DTXQCDMOp, void> DtxqcdPackedFullHermOp;
#endif

NAMESPACE_END(Grid);
