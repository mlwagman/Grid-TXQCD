# TXQCD Implementation in Grid: Architecture and Design

**Author:** Michael Wagman (Fermilab), with implementation assistance from Claude  
**Audience:** Peter Boyle, Anthony, and other Grid/LQCD experts  
**Date:** April 2026

## 1. Physics Summary

TXQCD introduces auxiliary tensor fields coupled to quarks through site-diagonal bilinear interactions. By a Fierz identity, the quartic auxiliary-quark interactions cancel exactly, so TXQCD is observationally identical to QCD — the auxiliary field path integral is Gaussian and integrates out to unity. The key property is that the fermion determinant `det(M)` remains **real** in the presence of these auxiliary fields, which opens the door to signal-to-noise improvements via correlated sampling.

The auxiliary fields are:
- `sigma_{ab}` (Nf x Nf Hermitian, flavor) — scalar channel
- `pi_{ab}` (Nf x Nf Hermitian, flavor) — pseudoscalar channel (gamma5-odd)
- `s_{ij}` (Nc x Nc Hermitian, color) — scalar channel
- `p_{ij}` (Nc x Nc Hermitian, color) — pseudoscalar channel (gamma5-odd)
- `t^{ij}_{mu,nu}` (Nc x Nc Hermitian, antisymmetric in mu,nu) — tensor channel

These enter the Dirac operator as a site-diagonal perturbation `Delta(x)` added to the Wilson-Clover operator, preserving gamma5-Hermiticity.

## 2. How TXQCD Plugs Into Grid's HMC

Grid's HMC is templated on a `FieldImplementation` that provides static methods for momentum generation, field updates, force projection, and kinetic energy. The integrator never sees the field's internal structure — it calls `FieldImplementation::update_field(P, U, dt)` and doesn't care what `U` is.

**The central design decision:** we define a single composite type `TXQCDField` that bundles `LatticeGaugeField U` with the five auxiliary lattice fields, and a `TXQCDCompositeImpl` that implements all static methods by delegating:

- **Gauge component** → `PeriodicGimplR` (exponential map via `Ta` projection, standard SU(3) update)
- **Auxiliary components** → additive updates (`X += P_X * dt`), Hermitian projection for forces

This means the existing `HybridMonteCarlo`, `LeapFrog`, `ForceGradient`, `Integrator`, smearing, and checkpointing infrastructure all work without modification. The TXQCD extension is ~4700 lines of new headers under `Grid/qcd/action/txqcd/`, plus ~120 lines of clover force helpers added to `WilsonCloverFermionImplementation.h`. No existing Grid code was changed beyond two `#include` additions.

### 2.1 Composite Field and Momentum

```
TXQCDField {
  LatticeGaugeField U;          // standard SU(3) gauge links
  LatticeFlavorMatrix sigma;    // Nf x Nf Hermitian (site-diagonal)
  LatticeFlavorMatrix pi;       // Nf x Nf Hermitian
  LatticeColorMatrix  s;        // Nc x Nc Hermitian
  LatticeColorMatrix  p;        // Nc x Nc Hermitian
  LatticeTensorField   t;       // Nd x Nd matrix of Nc x Nc, antisymmetric in mu,nu
}
```

The momentum `P` is of the same type. Gauge momenta are antihermitian (as in standard Grid); auxiliary momenta are Hermitian (natural conjugate variables for Hermitian fields). The kinetic energy combines both signs:

```
K = -Tr(P_U^2) / denom               [antihermitian → negative trace]
  - sum_aux Tr(P_aux^2) / (2*denom)   [Hermitian → positive trace, with explicit -]
```

The factor-of-2 difference from the gauge convention arises because auxiliary fields are real-valued scalars (one d.o.f. per matrix element, not the two from antihermitian traceless), and the `1/2` matches the standard `p^2/2` kinetic term for scalar fields.

Momentum generation scales by `sqrt(HMC_MOMENTUM_DENOMINATOR)` to match Grid's convention where the integrator divides by this denominator in the kinetic energy.

### 2.2 Integration

`update_field(P, U, dt)` delegates to `PeriodicGimplR::update_field` for the gauge link (exponential map: `U_mu(x) *= exp(i P_mu(x) dt)`) and uses simple additive updates for auxiliary fields. This is exact — auxiliary fields live in a flat Hermitian-matrix space, not a group manifold.

`projectForce` applies `Ta` projection to the gauge force (as in standard Grid) and Hermitian + antisymmetry projection to auxiliary forces.

All standard integrators (LeapFrog, ForceGradient, MinimumNorm2) work unchanged.

## 3. Action Decomposition

The TXQCD partition function for Nf=2 light + Nf=1 strange is:

```
Z = int dU d(aux) exp(-S_gauge[U_smeared] - S_aux[aux] - S_logdet[U,aux] - S_PF[U,aux] - S_strange[U_smeared])
```

Each piece is a separate `Action<TXQCDField>` in the HMC action set:

| Action class | What it computes | Force targets |
|---|---|---|
| `GaugeActionAdapter<SymanzikR>` | Symanzik gauge action on smeared U | dS/dU only |
| `AuxiliaryFieldGaussianAction` | (lambda^2/2) sum Tr(X^2) | dS/d(aux) only |
| `TXQCDLogDetCloverEOAction` | -ln det(M_ee) on even sites | dS/d(aux) on even sites + dS/dU from clover |
| `TXQCDWilsonCloverRationalEOAction` | RHMC for det(M_pc)^{1/2} on odd sublattice | dS/d(aux) on both parities + dS/dU (hopping + clover) |
| `QCDActionAdapter(StrangeLogDet)` | -ln det(M_ee) for strange (standard QCD) | dS/dU only (extracts U from TXQCDField) |
| `QCDActionAdapter(StrangeSchurPF)` | RHMC for det(M_pc)^{1/2} strange | dS/dU only |

### 3.1 Adapter Pattern

Two adapter classes allow mixing TXQCD and standard QCD actions in the same HMC:

- **`GaugeActionAdapter<T>`**: Wraps any `GaugeAction<PeriodicGimplR>`. Extracts `U` from `TXQCDField`, calls the inner action's `S()`/`deriv()`, packs the gauge force back into a composite force field (aux components zeroed).

- **`QCDActionAdapter`**: Wraps any `Action<LatticeGaugeField>`. Same extraction/repacking pattern. This is how the strange quark (a standard QCD fermion with no auxiliary coupling) participates in TXQCD HMC. The strange Dirac operator depends only on the gauge field, so its force has no auxiliary components.

### 3.2 Stout Smearing

`TXQCDSmearedConfiguration` wraps Grid's `SmearedConfiguration<PeriodicGimplR>`:
- `set_Field(U)`: Smears the gauge component through the standard stout chain. Auxiliary fields pass through unchanged (they don't get smeared).
- `smeared_force(F)`: Applies the stout force chain rule to `F.U`. Auxiliary force components pass through unchanged.
- Actions with `is_smeared = true` see the smeared gauge field; the integrator handles force backpropagation.

## 4. Even-Odd Preconditioning

The TXQCD Dirac operator `M = D_W + Delta + Clover` is site-diagonal on even and odd sublattices (the auxiliary fields and clover term are site-diagonal; only the Wilson hopping term couples sites). This gives the standard EO factorization:

```
det(M) = det(M_ee) * det(M_oo - M_oe M_ee^{-1} M_eo)
       = det(M_ee) * det(M_ee) * det(M_pc)    [by Schur complement]
```

where `M_pc = 1 - M_ee^{-1} M_eo M_oo^{-1} M_oe` acts on the odd sublattice (half volume).

### 4.1 TXQCDWilsonCloverFermionEO (Fermion Operator)

This is the workhorse operator. It stores:
- A `WilsonFermion<WilsonImplR>` for the hopping kernel
- References to all five auxiliary fields (checkerboarded)
- Precomputed LU-factored 24x24 site matrices for `M_ee^{-1}` and `M_oo^{-1}`

The 24x24 comes from the internal flavor-spin-color structure: Nf(2) x Ns(4) x Nc(3) = 24. At each site, `M_site` is:

```
M_site = (4 + m) * I_24
       + sigma_{ab} (x) delta_{alpha,beta} delta_{ij}    [flavor]
       + pi_{ab}    (x) (gamma5)_{alpha,beta} delta_{ij} [flavor, gamma5-odd]
       + s_{ij}     (x) delta_{ab} delta_{alpha,beta} / sqrt(2)  [color]
       + p_{ij}     (x) delta_{ab} (gamma5)_{alpha,beta} / sqrt(2)  [color]
       + t^{ij}_{mu,nu}(x) delta_{ab} (i*sigma_{mu,nu})_{alpha,beta} / sqrt(2) [tensor]
       + csw * clover_{ij}^{alpha,beta}(x)  [standard clover]
```

On field import, the operator unvectorizes the auxiliary fields to per-site scalar arrays and builds+LU-factors these 24x24 matrices for both parities using Eigen. This is the main setup cost; subsequent `Mooee`/`MooeeInv` calls just apply the precomputed factors.

**gamma5-Hermiticity** is preserved: `gamma5 M gamma5 = M†` holds because each auxiliary coupling is individually gamma5-Hermitian (sigma and s are gamma5-even; pi, p contribute with gamma5; t contributes with i*sigma_{mu,nu} which anticommutes with gamma5 in pairs that cancel).

### 4.2 LogDet Action

`TXQCDLogDetCloverEOAction` computes `S = -ln |det(M_ee)|` exactly (no stochastic estimation). The force requires inverting `M_ee` at each even site (reuses the precomputed LU factors) and contracting:

```
dS/d(sigma_{ab}) = -sum_{alpha,i} [M_ee^{-1}]_{(a,alpha,i),(b,alpha,i)}
dS/d(pi_{ab})    = same with gamma5 projection
dS/d(s_{ij})     = -sum_{a,alpha} [M_ee^{-1}]_{(a,alpha,i),(a,alpha,j)} / sqrt(2)
... etc for p, t
```

The clover contribution to the gauge force goes through the standard `C_{mu,nu}` staple construction (see Section 5).

### 4.3 Schur Complement RHMC

`TXQCDWilsonCloverRationalEOAction` handles `det(M_pc†M_pc)^{1/2}` via rational HMC with multishift CG on the odd sublattice (half volume). The rational approximation to `x^{1/4}` is computed by `AlgRemez` at construction time.

The force for each rational pole `k` with shift `sigma_k` and residue `alpha_k`:

1. Multishift CG solves `(M_pc† M_pc + sigma_k) X_k = Phi` for all poles simultaneously
2. `Y_k = M_pc X_k` (odd-site application)
3. Even-site intermediates: `W_e = M_ee^{-1} M_eo X_k`, `Z_e = (M_ee^{-1})† M_oe† Y_k`
4. Auxiliary forces accumulate bilinear contractions from `(Y_k, X_k)` on odd sites and `(Z_e, W_e)` on even sites
5. Gauge hopping force: `MoeDeriv` / `MeoDeriv` applied to the appropriate fermion pairs
6. Gauge clover force: per-site `sigma_{mu,nu}` construction from the bilinears

## 5. Clover Force Implementation

The clover force is the most complex piece. When `csw != 0`, both the LogDet and RHMC actions contribute gauge forces through the field-strength tensor `F_{mu,nu}`.

The implementation follows Grid's convention for `WilsonCloverFermion`:

1. **Bilinear construction:** For each fermion pair `(psi, chi)` contributing to the force, compute the site-local matrix `sigma_{mu,nu} = (1/2) * [gamma_mu, gamma_nu] * outer(psi, chi)` in spin-color space.

2. **Convention conversion:** Grid's `WilsonCloverFermion` uses "Convention B" internally (the clover term as a sum over plaquette-like paths), but the force needs "Convention A" (derivative w.r.t. each link). The conversion factor is `-1/2`.

3. **Staple contraction:** The `Cmunu` helper constructs the clover staple — the sum of the three paths around each plaquette that don't include the link being differentiated — and contracts it against the `sigma_{mu,nu}` matrix.

Added to `WilsonCloverFermionImplementation.h`:
- `MooeeDeriv` / `MooeeInvDeriv`: compute dM_ee/dU and d(M_ee^{-1})/dU contributions to the gauge force from even-site clover terms. These were not present in stock Grid (which doesn't need them for standard EO pseudofermion actions).

## 6. Per-Site Scalar Operations

Several TXQCD operations require per-site scalar linear algebra on 24x24 matrices that don't vectorize naturally across SIMD lanes (LU factorization, matrix inversion, determinant). These are centralized in `TXQCDSiteMatrix.h`:

- `AuxSiteArrays`: Unvectorizes all five auxiliary fields from SIMD-packed lattice format to per-site `Eigen::Matrix<ComplexD, N, N>` arrays. One array per outer SIMD site; inner SIMD lanes are expanded.
- `BuildSiteMatrix`: Assembles the 24x24 matrix at each site from the aux + clover contributions.
- `SpinMatrices`: Precomputed gamma5 and i*sigma_{mu,nu} in 4x4 spin space, cached to avoid repeated construction.

This is currently the main performance bottleneck for the LogDet and RHMC setup phases, and the most obvious target for GPU optimization (see Section 8).

## 7. Checkpoint I/O

`TXQCDCheckpointer` saves configurations in two parts:
- **Gauge:** Standard NERSC format via `NerscIO` — fully interoperable with stock Grid and other LQCD codes.
- **Auxiliary:** Custom binary sidecar file (`ckpoint_lat_aux.<traj>`) with a packed Hermitian representation.

The packing stores only the independent real components of each Hermitian matrix:
- Nf x Nf Hermitian: Nf diagonal reals + Nf*(Nf-1)/2 complex upper-triangle = Nf^2 reals
- Nc x Nc Hermitian: Nc^2 reals
- Tensor: 6 upper-triangle (mu,nu) blocks, each Nc^2 reals → 6*Nc^2 reals

Total per site: 2*4 + 2*9 + 6*9 = 80 doubles = 640 bytes. Header contains magic bytes (0x54585141 = "TXQA") and version tag. Data is stored big-endian (IEEE64BIG) for portability.

## 8. Performance Considerations and Optimization Opportunities

### What we've measured

On a 4^3 x 8 test lattice (single node, CPU, no MPI):
- **EO preconditioning gives 2.6x CG iteration reduction** (88.7 → 34.2 iterations/solve for the Schur complement vs full volume)
- The LogDet eigendecomposition adds fixed overhead per trajectory, which dominates on tiny lattices but becomes negligible as volume grows
- TXQCD HMC is ~5x slower per trajectory than QCD EO on the test lattice, dominated by the multishift CG with auxiliary-field-dependent operator and the per-site 24x24 setup

### Obvious optimization targets

**Per-site 24x24 operations (highest priority):**
The LU factorization, inversion, and determinant of the site-diagonal blocks are currently done with Eigen on unvectorized scalar data. On GPU, these are embarrassingly parallel across sites — each site is independent. Options:
- Batched LAPACK/cuBLAS (`cublasDgetrfBatched` etc.) for the 24x24 inversions
- Hand-written GPU kernels that exploit the known block structure (the 24x24 decomposes into 2x2 flavor blocks of 12x12 spin-color matrices when sigma/pi are diagonal, which they often are in practice)
- For the force, the contraction patterns are fixed at compile time — template-generated kernels could avoid the Eigen overhead entirely

**Auxiliary field SIMD:**
The auxiliary fields use `vComplex` SIMD types but the per-site operations (BuildSiteMatrix, force contractions) unvectorize to scalar. The lattice-level operations (additive updates, Hermitian projection, Gaussian generation) do vectorize. The question is whether the unvectorize/revectorize overhead matters relative to the actual computation — on GPU it likely doesn't (plenty of parallelism across sites), but on CPU/AVX-512 it may be worth exploring SIMD-batched small matrix routines.

**Multishift CG convergence:**
The TXQCD Dirac operator has a wider eigenvalue spread than standard Wilson-Clover (the auxiliary fields add to the diagonal), which can increase the condition number. The rational approximation bounds (currently `[1e-4, 200]`) may benefit from tuning per-ensemble. Mixed-precision CG would help here — the inner solver could run in single precision with a double-precision outer correction.

**Stout smearing force backpropagation:**
Currently wraps Grid's `SmearedConfiguration` which was designed for gauge-only forces. The auxiliary force passthrough is trivial (identity), so there's no wasted computation, but the gauge force chain rule could potentially be fused with the clover force computation to reduce memory traffic.

**Memory layout:**
`TXQCDField` stores 6 separate lattice fields (U, sigma, pi, s, p, t). Actions that need all of them at a site (the 24x24 construction) must gather from 6 different memory locations. A structure-of-arrays → array-of-structures transformation (packing all aux data contiguously per site) could improve cache behavior for the site-local operations, at the cost of complicating the lattice-level operations that only touch one component.

### What probably doesn't need optimization

- **HMC integration loop:** The integrator overhead is negligible; all time is in action evaluation and force computation.
- **Auxiliary field updates:** Additive updates are trivially fast (one lattice-scale axpy per field).
- **Checkpointing:** I/O is infrequent (every `meas_skip` trajectories) and the packed format is already compact.
- **Gaussian action and force:** `S_aux = (lambda^2/2) Tr(X^2)` and `dS/dX = lambda^2 X` are single lattice-scale operations.

## 9. File Map

All TXQCD-specific code lives under `Grid/qcd/action/txqcd/`:

| File | Lines | Purpose |
|------|-------|---------|
| `AuxFieldTypes.h` | ~60 | Type aliases for auxiliary lattice/site types |
| `TXQCDField.h` | ~120 | Composite field struct with arithmetic operators |
| `TXQCDCompositeImpl.h` | ~250 | FieldImplementation: momenta, update, kinetic energy |
| `TXQCDDeltaOp.h` | ~200 | Site-diagonal auxiliary insertion into fermion operator |
| `TXQCDWilsonCloverFermionEO.h` | ~500 | EO-preconditioned Dirac operator with 24x24 site blocks |
| `TXQCDSiteMatrix.h` | ~400 | Per-site matrix construction, LU factorization, utilities |
| `TXQCDWilsonCloverRationalEOAction.h` | ~700 | RHMC pseudofermion action (multishift CG + force) |
| `TXQCDLogDetCloverEOAction.h` | ~500 | Even-site log-determinant action |
| `AuxiliaryFieldGaussianAction.h` | ~100 | Quadratic auxiliary kinetic term |
| `GaugeActionAdapter.h` | ~80 | Wraps gauge actions for composite field HMC |
| `QCDActionAdapter.h` | ~80 | Wraps QCD fermion actions for composite field HMC |
| `TXQCDCheckpointer.h` | ~350 | Packed Hermitian checkpoint I/O |
| `TXQCDSmearedConfiguration.h` | ~150 | Stout smearing with auxiliary passthrough |

Modified existing files:
- `WilsonCloverFermionImplementation.h`: +120 lines for `MooeeDeriv`/`MooeeInvDeriv` (clover force helpers needed by EO actions)
- `PseudoFermion.h`: +2 lines (`#include` for new EO pseudofermion headers)

New QCD EO pseudofermion actions (also useful independent of TXQCD):
- `QCDLogDetCloverEOAction.h`: S = -Nf * ln|det(M_ee)| for standard Wilson-Clover
- `TwoFlavourSchurCloverAction.h`: Nf=2 Schur complement PF with clover force corrections
- `OneFlavourSchurCloverRationalAction.h`: Nf=1 RHMC on Schur complement

## 10. Current Status

- All correctness tests pass (operator identity checks, force tests via finite differences, Fierz identity end-to-end tests comparing TXQCD and QCD correlators across three action variants)
- Production code compiles and runs for Nf=2+1 with Symanzik gauge + stout smearing + EO preconditioning
- First target ensemble: 16^3 x 48, Wilson-Clover, m = -0.245, csw = 1.249, beta = 6.1
- Lambda tuning workflow: measure VEV (Sigma) from QCD configs, compute lambda_opt = sqrt(4*Sigma/M_N), then launch TXQCD streams (directory paths encode lambda for parallel runs)
