# MG for the deformed operator: coarsening M̃ itself (design sketch)

*2026-07-15, following the one-night normal-op MG campaign (commit e8dfc4c7 and
`dtxqcd-deform-mg` memory). Status: DESIGN — nothing here is implemented.*

## Why the normal-op route hit a wall (measured, not conjectured)

Every level of `Mpc̃†Mpc̃` aggregation re-fights a denser band:

- 4⁴ fine blocks: outer plateau ~0.1, invariant under null-vector order
  (600/1500), REFINE, nbasis (32/48), coarse tol. The aggregation loses a
  fixed slice of the band to within-block averaging.
- Small blocks (2⁴, 4.4.4.2): the transfer captures the band — proven by
  inheritance: the level-1 coarse op becomes CG-intractable in double
  precision (κ ~ 1e10; residual diverges).
- 3-level: the L2 aggregation (32 vectors / 768-dim blocks) faces the same
  ratio the fine 4⁴ failure had, against a *relatively denser* band
  (hundreds of modes below 1e-3 in a 73k-dim space — the coarse census
  0/48 mush measured this). Stock windows diverge (fPcg 636 → 9,730).

Root cause hypothesis: we aggregate the **normal operator**, whose relevant
low space is the *singular vectors* of M̃. For strongly non-normal operators
(and the t-graded deformation is exactly that) singular vectors are rougher
and less block-representable than eigenvectors. Wilson-clover MG (QUDA,
DDαAMG, HDCR) succeeds by coarsening **M itself** with a γ5-compatible
transfer — the construction this note adapts.

## The two big structural wins of coarsening M̃

1. **κ un-squared.** CGNR works on M̃†M̃: κ ≈ 2.7e8 at λ=5 η=0.8. Solving M̃
   directly faces κ(M̃) ~ √that ≈ 1.6e4. The band that is *numerically
   impossible* squared (coarse CG divergence in double precision) is merely
   hard unsquared. Every level of the hierarchy benefits identically.
2. **Eigenvector (not singular-vector) transfer.** The near-kernel of M̃ is
   what blocked aggregation is empirically good at for Wilson-type
   operators; the doubled/aux structure rides along as extra internal DOF
   per site (48 instead of 12 — same construction, bigger blocks).

## Hermiticity structure available to us

The undeformed doubled operator is γ5-Hermitian: `Γ5 M Γ5 = M†` (the
sigmaHerm convention gates guarantee this; Γ5 = γ5 ⊗ 1 on the doubled
index). The deformation breaks it in a CONTROLLED way — the measured dagger
identity from the QUDA port (`corr† = corr with upper↔lower coefficient
arrays swapped`, i.e. w → 1/w):

    Γ5 M̃(w) Γ5 = M̃(1/w)†        (quasi-γ5-Hermiticity)

Consequence: **left null vectors of M̃(w) are Γ5 × right null vectors of
M̃(1/w)**. We never need a separate left-space setup: generate right
near-kernel vectors for the w and 1/w operators (the 1/w apply already
exists — `deform_recip_` toggling is how MdagDeviceDirect works), and the
Petrov–Galerkin restrictor comes for free:

    P  = blockAggregate( {v_i : M̃(w) v_i ≈ 0} )
    R  = ( blockAggregate( {Γ5 u_i : M̃(1/w) u_i ≈ 0} ) )†

For η=0: w=1/w=1, R = (Γ5 P)†, and the construction degenerates to the
standard γ5-compatible transfer — the η→0 limit is the correctness gate.

## Coarse operator and its solver

    A_c = R M̃ P      (Petrov–Galerkin; NOT Hermitian, NOT normal)

- Coarsening machinery: `GeneralCoarsenedMatrix::CoarsenOperator`
  reconstructs the coarse stencil from phase-modulated probes of a
  black-box `linop.Op`. Inspect whether it assumes Hermiticity anywhere
  (suspect: it may symmetrize or only probe one side). If it does, the
  non-Hermitian variant needs probing both M̃P (columns) and R M̃
  (rows) — same phase-probe trick, two passes. This is the main new
  in-tree machinery.
- Coarse-level solver: GMRES/GCR (Grid: `GeneralisedMinimalResidual`,
  `PrecGeneralisedConjugateResidual`) — no HPD assumption. At κ(A_c) ~
  κ(M̃_band) ~ 1e4-5 *unsquared*, plain GMRES on a 73k-dim system is
  minutes, and a K-cycle (third level, same construction recursed) is the
  production form.
- Outer solver: right-preconditioned GCR on M̃ x = b with the V-cycle as
  preconditioner. Replaces CGNR entirely. (The measurement consumers call
  `SolveTilEO` — swap-in point is `CgnrEO_`'s call site.)
- Smoother: a few iterations of GMRES on M̃, or Chebyshev on M̃†M̃ (both
  fine; QUDA uses damped Jacobi/GCR — cheapest that damps the bulk).

## Null-vector generation (the part the deformation makes interesting)

Near-kernel of M̃, not of M̃†M̃. Options in increasing cost:
1. Relaxation: v ← smoother-iterate on M̃ v = 0 from random starts (the
   DDαAMG bootstrap). Cheap, needs only the M̃ apply.
2. Inverse iteration at loose tol via the existing CGNR solver (expensive
   per vector at η=0.8 — the 60k problem — but ~1e-1 tol suffices for
   null-vector quality and costs a fraction).
3. Bootstrap-AMG style: build a crude 2-level from (1), use its V-cycle to
   polish the vectors, re-coarsen (one round usually enough).

The tent structure argues for t-refined blocks from the start (4.4.4.2),
and the doubled index should NOT be aggregated away: keep all 48 internal
components in the block (standard — aggregation is site-blocked only).

## Prototype plan (2–4 days)

- **Step 0** (half day): packed single-leg applies. `MpcTil/MpcTilDag` on
  `LatticeDtxqcdPacked` wrappers (EoLeg_ already does struct→packed→device;
  expose as `Op`/`AdjOp` of a NON-Hermitian `LinearOperatorBase`).
  Gate: adjointness ⟨u, M̃v⟩ = ⟨M̃†u, v⟩ on random packed fields (exists in
  spirit as the deform device-adjointness gate).
- **Step 1** (half day): null-vector generation by relaxation + the
  Γ5/recip left-space trick. Gate: ‖M̃v_i‖/‖v_i‖ ≲ 1e-2·λmax^(1/2)-scale,
  and at η=0 verify left set == Γ5 × right set exactly.
- **Step 2** (1 day): non-Hermitian coarsening. Either verify
  `GeneralCoarsenedMatrix` probes are Hermiticity-agnostic and feed it
  Op=M̃ with the Petrov-Galerkin R via a wrapper, or write the two-sided
  phase-probe variant. Gate: coarse-op action matches R M̃ P applied
  directly on random coarse vectors (exact identity, cheap).
- **Step 3** (1 day): two-level solve. Outer PGCR(M̃, V-cycle), GMRES
  coarse solves, smoother scan. Measurements: outer iterations + wall
  clock vs the 59,931-iteration/24-min CGNR baseline, at η=0.6 (easy
  cross-check) and η=0.8 (the target).
- **Gates throughout**: η=0 must reproduce the undeformed γ5-compatible
  construction; solution h5 A/B vs the certified CGNR path at 1e-7.

## Deliberately out of scope for the prototype

Multi-rank (the deform native kernels are single-rank; same constraint as
today), K-cycle/3-level (add only if the 2-level coarse solve is the
bottleneck — at 73k DOF unsquared it likely is not), mrhs batching, and
HMC-side use (measurement solves only).

## Kill criteria

If Step 3's two-level outer does not beat plain CGNR by ≥3× in wall clock
at η=0.8 with nbasis ≤ 48 — or if Step 1 relaxation cannot produce vectors
with ‖M̃v‖ decaying against the bulk (i.e., the near-kernel itself is not
block-local even for M̃) — stop: the operator is telling us deformed-frame
propagator solves are intrinsically Krylov-priced, and the program should
optimize the plain-CG path (mrhs, mixed precision) instead.
