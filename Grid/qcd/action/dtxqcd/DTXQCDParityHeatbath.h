#pragma once
// HmcObservable that applies the full DTXQCD parity transformation to the
// unsmeared (thin-link) DTXQCDField with a Bernoulli probability after each
// completed trajectory.
//
// Purpose: break ergodicity failure between the two parity sectors at low λ
// where the fermion determinant supports a spontaneously broken Tr[π]
// (isosinglet parity-odd) condensate.  See
// [[project_dtxqcd_low_lambda_isosinglet_condensate]] +
// [[reference_dtxqcd_parity_transform]] for the physics and the validated
// transformation.
//
// Correctness:
//   S(U, aux) = S(P·U, P·aux) exactly (verified 4^4 bit-exact by
//     Test_dtxqcd_parity_invariance, commit 98b6fdc4).
//   Independent Bernoulli(prob=0.5) coin flip after each accepted trajectory
//     is equivalent to a heatbath on the two-element orbit {X, P·X} because
//     the action-invariant symmetry gives equal detailed-balance weight to
//     both sides:
//         ρ(X) · T(X → P·X) = ρ(X) · 0.5,
//         ρ(P·X) · T(P·X → X) = ρ(X) · 0.5,
//   which is trivially satisfied.  Any Bernoulli probability p ∈ (0, 1)
//   preserves the stationary distribution; p = 0.5 gives the fastest
//   mixing between the two P-sectors.
//
// Wiring notes:
// - The observer mutates *SmartConfig.get_U(false) IN PLACE — the smearer
//   holds a pointer to the driver's Ucur, so the mutation is visible to the
//   next iteration of the HMC evolve loop.
// - The smearer's cached SmearedField_ becomes stale after this observer
//   flips; that's fine because the integrator calls Smearer.set_Field(Ucur)
//   at the start of the very next trajectory's refresh() (see
//   Integrator.h:415), which repopulates the smeared cache from the flipped
//   Ucur.
// - Uses GridSerialRNG only so the coin flip is bit-identical across all
//   MPI ranks (parity is a global transformation; every rank must agree).
// - Order relative to other observers matters: put the parity heatbath
//   BEFORE the checkpointer so the checkpoint reflects the post-flip state
//   consistent with the sampled ensemble.

#include <Grid/qcd/action/dtxqcd/DTXQCDField.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDParity.h>
#include <Grid/qcd/observables/hmc_observable.h>

NAMESPACE_BEGIN(Grid);

class DTXQCDParityHeatbath : public HmcObservable<DTXQCDField> {
 public:
  explicit DTXQCDParityHeatbath(RealD prob = 0.5)
      : prob_(prob) {}

  // Smearer-aware overload: mutates the underlying thin-link field so the
  // flip propagates to the next trajectory.
  void TrajectoryComplete(int traj,
                          ConfigurationBase<DTXQCDField> &SmartConfig,
                          GridSerialRNG &sRNG,
                          GridParallelRNG & /*pRNG*/) override {
    RealD r;
    random(sRNG, r);
    const bool flip = (r < prob_);
    if (flip) {
      DTXQCDField &U = SmartConfig.get_U(false);   // unsmeared thin links
      DtxqcdParity::ApplyField(U);
    }
    std::cout << GridLogMessage
              << "[DTXQCDParityHeatbath] traj=" << traj
              << " r=" << r
              << " prob=" << prob_
              << " flip=" << (flip ? "YES" : "no")
              << std::endl;
  }

  // Fallback pure-field overload: same coin flip on the passed field.
  void TrajectoryComplete(int traj, DTXQCDField &U,
                          GridSerialRNG &sRNG,
                          GridParallelRNG & /*pRNG*/) override {
    RealD r;
    random(sRNG, r);
    const bool flip = (r < prob_);
    if (flip) DtxqcdParity::ApplyField(U);
    std::cout << GridLogMessage
              << "[DTXQCDParityHeatbath] traj=" << traj
              << " r=" << r
              << " prob=" << prob_
              << " flip=" << (flip ? "YES" : "no")
              << std::endl;
  }

 private:
  RealD prob_;
};

NAMESPACE_END(Grid);
