#pragma once
// Phase β Session B — SP-precision device-resident CG cleanup state.
//
// Mirror of DoubledStateCSF (DP) but allocated with QUDA_SINGLE_PRECISION
// FloatNOrder<float,4,3,4> CSF storage.  Used by DTXQCDSingleShiftCGSpCleanupCSF
// for the SP-only single-shift CG that polishes Style C's DP multishift
// solutions.
//
// Differences from DP DoubledStateCSF:
//   - param_tmpl must be SP (cuda_prec=QUDA_SINGLE_PRECISION) — obtained via
//     DTXQCDMpcOpQUDA::MakeNativeCsfParamSp().
//   - quda::blas norm2/redot return RealD regardless of CSF precision
//     (QUDA promotes the inner SP reduction to DP scalar).
//   - copy_from_grid / copy_to_grid removed: SP CSFs are populated from DP
//     CSFs via csf.copy() which handles precision conversion natively.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>

#ifndef GRID_HAVE_QUDA
#  error "DoubledStateCSFSp requires GRID_HAVE_QUDA"
#endif

#include <quda.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <memory>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaStyleC {

// 4-slot doubled state on native SP CSF.  Slot layout matches DoubledStateCSF:
//   slot[0]=upper.f[0], [1]=upper.f[1], [2]=lower.f[0], [3]=lower.f[1].
struct DoubledStateCSFSp {
  std::unique_ptr<quda::ColorSpinorField> csf[4];

  void allocate(const quda::ColorSpinorParam &param_tmpl_sp) {
    auto p = param_tmpl_sp;
    p.create = QUDA_NULL_FIELD_CREATE;
    for (int i = 0; i < 4; ++i) {
      csf[i] = std::make_unique<quda::ColorSpinorField>(p);
    }
  }

  void allocate_lower_only(const quda::ColorSpinorParam &param_tmpl_sp) {
    auto p = param_tmpl_sp;
    p.create = QUDA_NULL_FIELD_CREATE;
    csf[0] = std::make_unique<quda::ColorSpinorField>(p);
    csf[1] = std::make_unique<quda::ColorSpinorField>(p);
    csf[2].reset();
    csf[3].reset();
  }

  RealD norm2() const {
    RealD acc = 0.0;
    for (int i = 0; i < 4; ++i)
      if (csf[i]) acc += quda::blas::norm2(*csf[i]);
    return acc;
  }

  static RealD redot(const DoubledStateCSFSp &x, const DoubledStateCSFSp &y) {
    RealD acc = 0.0;
    for (int i = 0; i < 4; ++i) {
      if (x.csf[i] && y.csf[i])
        acc += quda::blas::reDotProduct(*x.csf[i], *y.csf[i]);
    }
    return acc;
  }

  void zero() {
    for (int i = 0; i < 4; ++i) {
      if (csf[i]) csf[i]->zero();
    }
  }

  // y += a · x  (4-slot axpy).  Coefficient passed as double; QUDA handles
  // the down-cast to SP internally on the SP CSF.
  void axpy(double a, const DoubledStateCSFSp &x) {
    quda::vector<double> a_vec{a};
    for (int i = 0; i < 4; ++i) {
      if (!csf[i] || !x.csf[i]) continue;
      quda::vector_ref<const quda::ColorSpinorField> xr{*x.csf[i]};
      quda::vector_ref<quda::ColorSpinorField> yr{*csf[i]};
      quda::blas::axpy(a_vec, xr, yr);
    }
  }

  // y = scale · y + add  (matches DP DoubledStateCSF::scale_add semantics).
  void scale_add(double scale, const DoubledStateCSFSp &add) {
    quda::vector<double> ones{1.0};
    quda::vector<double> bs{scale};
    for (int i = 0; i < 4; ++i) {
      if (!csf[i] || !add.csf[i]) continue;
      quda::vector_ref<const quda::ColorSpinorField> xr{*add.csf[i]};
      quda::vector_ref<quda::ColorSpinorField> yr{*csf[i]};
      quda::blas::axpby(ones, xr, bs, yr);
    }
  }

  void copy_from(const DoubledStateCSFSp &src) {
    for (int i = 0; i < 4; ++i) {
      if (csf[i] && src.csf[i]) csf[i]->copy(*src.csf[i]);
    }
  }

  // DP → SP precision change: csf.copy handles the precision conversion when
  // src is DP and *this is SP.
  void copy_from_dp(const struct DoubledStateCSF &src);

  // SP → DP precision change.
  void copy_to_dp(struct DoubledStateCSF &dst) const;
};

}  // namespace DtxqcdQudaStyleC
NAMESPACE_END(Grid);

// Out-of-line precision-change methods — defined in a follow-up include that
// has DoubledStateCSF visible.
#include <Grid/qcd/action/dtxqcd/DoubledStateCSF.h>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaStyleC {

inline void DoubledStateCSFSp::copy_from_dp(const DoubledStateCSF &src) {
  for (int i = 0; i < 4; ++i) {
    if (csf[i] && src.csf[i]) csf[i]->copy(*src.csf[i]);
  }
}

inline void DoubledStateCSFSp::copy_to_dp(DoubledStateCSF &dst) const {
  for (int i = 0; i < 4; ++i) {
    if (csf[i] && dst.csf[i]) dst.csf[i]->copy(*csf[i]);
  }
}

}  // namespace DtxqcdQudaStyleC
NAMESPACE_END(Grid);
