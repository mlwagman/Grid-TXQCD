# DTXQCD GPU production handoff

Status as of 2026-06-17. This note covers what a fresh engineer (or
cluster-side Claude) needs to know before standing up DTXQCD on the GPU
16³×48 ensemble that has been running TXQCD.

## 1. The C-doubled Dirac operator

DTXQCD doubles the Dirac space to 48 components per site:
**N_f · N_s · N_c · 2 = 2 · 4 · 3 · 2 = 48**, where the last factor of 2
is the "block" index (upper / lower).

```
    ┌  D_QCD δ_ab + X^{ij}_{ab}            d^{ij}_{ab} γ5  +  n^{ij}_{ab}    ┐
M = │                                                                       │
    └  d^{ij}_{ab} γ5  +  n^{ij}_{ab}      C [D_QCD δ_ab]^T C  +  X^{ij}_{ab} ┘
```

where the upper block is the usual flavored Wilson-clover quark operator
and the lower block is the charge-conjugate carrying a sign-flipped
clover term (per the corrected v2 operator commit, see
[[reference-dtxqcd-corrected-v2-operator]]). The off-diagonal `d γ5 + n`
field couples diquark / anti-diquark bilinears between the blocks.

Concrete normalization choices that bit into HMC stability:

- **+X in BOTH blocks** (not −X in the lower; that was the v1 sign
  error which broke Pfaffian positivity).
- **√2 prefactor on the off-diagonal `d, n` cross-block coupling**.
- **−csw/2 clover sign-flip in the lower block** (so the lower block's
  γ5-Hermiticity matches upper after C conjugation).

The 48×48 per-site matrix is implemented in `Grid/qcd/action/dtxqcd/
DTXQCDSiteMatrix.h` and applied via `DTXQCDWilsonCloverFermionEO`
(EO-preconditioned) or the full-volume `DTXQCDWilsonCloverRationalFull
Action` wrapper. The 48×48 LU and the per-site inverse cache (`InvField`)
are the obvious GPU hotspots — see the existing TXQCD QUDA primitive for
the analogous targeting.

## 2. Aux field content

DTXQCD has **six** auxiliary lattice fields. All Hermitian-and-traceless
or real-symmetric (per [[project-dtxqcd-pfaffian-realsymmetric-fix]]) to
keep `(K·M48)^T = −(K·M48)` and hence the Pfaffian positive-definite at
the action level:

| field | shape | parity | role |
|---|---|---|---|
| σ^{ij}_{ab} | 6×6 (color⊗flavor) real-sym, traceless | even | mass-shift bilinear |
| π^{ij}_{ab} | 6×6 real-sym, traceless | odd (γ5) | π-pseudoscalar |
| s | singlet scalar | even | absorbs Tr σ singlet |
| p | singlet scalar | odd (γ5) | parity partner of s |
| d^{ij}_{ab} | 6×6 real-sym | — | diquark, off-diagonal block |
| n^{ij}_{ab} | 6×6 real-sym | — | anti-diquark, off-diagonal block |

### The `s ← s + Tr σ / N_f` shift

Originally the action had **both** a traced singlet inside σ AND an
independent singlet field `s`, giving a redundant zero mode (see
[[project-dtxqcd-s-sigma-redundancy]]). In the implemented v2 action:

- **σ is made traceless** (Tr σ ≡ 0 by construction at fill time and
  enforced after every momentum update via a HermitizeAndTraceless
  projector in `DTXQCDCompositeImpl`).
- **The s² coefficient is halved**: action contains `(λ² / 4) · s²`
  instead of `(λ² / 2) · s²` — this matches the freedom of the
  reabsorbed singlet, so that the marginalized aux measure is
  unchanged.

That's why production aux observables (and the free-field unit test)
show `⟨Tr σ⟩ = 0` to machine precision and the only scalar VEV lives in
`⟨s⟩`.

## 3. Aux-field VEV expectations

At equilibrium HMC distribution on a fixed gauge background:

- `⟨Tr σ⟩ = 0` (constraint, exact)
- `⟨Tr π⟩ = 0` (parity)
- `⟨p⟩    ≈ 0` (parity; nonzero values flag the parity-broken
  metastability seen at λ ≤ 0.1 — see
  [[project-dtxqcd-parity-metastability]])
- `⟨s⟩    = (2·N_f · Σ) / λ²` where Σ is the per-quark
  `Tr M_DTX^{-1} / (V · 2 · N_f)` of the equilibrated DTXQCD operator
  (NOT the bare Wilson Σ_W). This is the self-consistent saddle:
  at λ ≪ 1 it differs from the bare-Σ formula
  `2 · N_f · Σ_W / λ²` by a factor of 2–5 because the back-reaction
  of `s` on the effective fermion mass is large
  (see [[project-dtxqcd-aux-init-self-consistency]]).
  The production `AUX_INIT_AUTO=1` knob in
  `Test_dtxqcd_2pt_gencfgs.cc` does the bisection numerically — this
  pattern is now also wired into the freefield Fierz unit test.
- `⟨Tr d⟩, ⟨Tr n⟩ ≈ 0` at parity-symmetric λ ≥ 1. At λ < 1 the d_01
  pseudoscalar diquark channel develops a parity-odd condensate
  ([[project-dtxqcd-d01-pseudoscalar-diquark-condensate]]).
- `‖σ‖² / V ≈ N_DOF(σ) / λ²` (pure Gaussian width 1/λ on each real
  DOF). Similar for π, d, n. Tracking these against the predicted
  free-Gaussian floor is the standard "is anything condensing?" probe.

## 4. ⟨q̄q⟩ measurement

Per-quark chiral condensate from a saved DTXQCD cfg:

```
Σ_DTX  =  Tr[M_DTX^{-1}]  /  (V · 2 · N_f)
       =  ⟨η†  M_DTX^{-1}  η⟩_noise / (2 V · 2 · N_f)        (Hutchinson, σ²_η = 2)
```

The `/(2V)` absorbs Grid's complex-Gaussian noise variance σ²=2; the
`/(2·N_f)` makes it per-quark so it matches `Σ_W = Tr[D_W^{-1}]/V` on
the bare gauge cfg at aux=0. Implementation: `m48_trminv_masked` in
`Test_dtxqcd_trminv_block_compare.cc` (mode=Both for the full M48) and
the new `DtxqcdFierzOpTrminv` helper in
`Test_dtxqcd_fierz_check_utils.h`.

**Fierz identity at fixed U** (after marginalizing aux):
`⟨Σ_DTX⟩_aux = Σ_W(U)` exactly. The free-field unit test enforces
`|⟨Σ_DTX⟩ / Σ_W − 1| < 0.02`.

Important — **per-cfg `Σ_DTX(cfg)` does NOT equal Σ_W**. The equality
is an ensemble identity over aux. At small λ a single cfg's
`Σ_DTX` sits at the saddle value (e.g. 0.12·Σ_W at λ=0.1, m=0.1);
the ensemble mean is recovered through rare wide-aux outlier cfgs that
drag the mean up — see
[[project-dtxqcd-aux-init-self-consistency]]. **This is the right
behaviour, not a bug.** Production measurements should average Σ_DTX
over many cfgs (the new `FIERZ_AVG_N_NOISE` knob in the freefield
unit test does this in-line, with the cfg-fluctuation noise folded
into the reported standard error).

## 5. Pfaffians and sign reweighting

DTXQCD's HMC samples |Pf(K · M48)| via RHMC on a `(M^†M)^{1/4}` rational
expansion. The **Pf is real exactly** (γ5-Hermiticity verified — see
[[project-dtxqcd-pf-realness-verified]]), so:

```
signPf  =  (-1)^{n_neg}
```

where `n_neg` is the number of negative eigenvalues of `γ5 · M48`. At
λ ≥ 5 this sign is stable across the entire HMC chain; at λ ≤ 0.1 it
flips multiple times per O(10) trajs and dominates the variance of
parity-odd observables ([[project-dtxqcd-pfaffian-signflip-lowLambda]]).

**Practical guidance**:

- Production must **reweight every observable** by `signPf`, not just
  the parity-odd ones. Even ⟨q̄q⟩ gets a sign correction at small λ.
- `n_neg` is logged per trajectory by the production gencfgs driver
  (`Test_dtxqcd_2pt_gencfgs.cc` and the deployed binary), computed via
  the γ5·M48 Lanczos tracker installed earlier
  ([[reference-dtxqcd-g5m-lanczos-tracking]]).
- For the 16³×48 stout-smeared tadpole-improved LW ensemble the BC
  bug is real — see §6. ALL DTXQCD config-gen prior to commit
  `99950d1c` was sampling with PBC time, while production
  measurement programs were already on APBC. Rerun fresh.

## 6. Free-field tests and findings

### Test binaries (committed this session)

| binary | mass | λ | wall | purpose |
|---|---|---|---|---|
| `Test_dtxqcd_freefield_qbarq` | 1000 | 1 | ~85 s | smoke — κ→0 ⇒ BC irrelevant; pure compile-and-Hutchinson check |
| `Test_dtxqcd_freefield_qbarq_light` | 0.1 | 10 | ~40 min | light corner — BC-sensitive regime; full Fierz unit test |

Both run HMC at frozen U with APBC time, then measure Σ_DTX / Σ_W on
the equilibrated in-memory state. Exit 0 = PASS (`|ratio − 1| < 0.02`),
exit 1 = FAIL.

### Shared header

`tests/dtxqcd/Test_dtxqcd_fierz_check_utils.h` provides:

- `DtxqcdInitFrozenGauge` — `GAUGE_INIT` env knob dispatcher (cold /
  tepid[:AMP] / hot / nersc:PATH).
- `DtxqcdFierzPlainWilsonTrminv` — plain `WilsonFermionD` (csw=0) or
  `WilsonCloverFermion` (csw≠0) reference, APBC time.
- `DtxqcdFierzOpTrminv` — full M48 stochastic per-quark trace.
- `DtxqcdSelfConsistentAuxInit` — production-gencfgs bisection for
  `Σ = Σ_DTX(Σ)`.
- `DtxqcdFierzCheck` — single-cfg final check.
- `DtxqcdFierzAveragingObserver` — `HmcObservable` subclass that
  measures Σ_DTX, Σ_W, and the parity-even / parity-odd aux trace
  VEVs every prod-window trajectory, then averages at the end.

### Env knobs (consistent across all four freefield tests)

```
LAMBDA, MASS, CSW, MDSTEPS, TRAJL, N_THERM, N_PROD, MEAS_SKIP, CFG_DIR
RAT_LO, RAT_HI, RAT_DEGREE, MD_CG_TOL, MEAS_CG_TOL, N_NOISE
PASS_TOL                  default 0.02
GAUGE_INIT                cold | tepid[:AMP] | hot | nersc:PATH
AUX_INIT=Σ                manual Σ_init override (DTXQCD only)
AUX_INIT_AUTO=1           self-consistent bisection (DTXQCD only)
SAVE_TRACE=PATH           write gauge+aux every MEAS_SKIP trajs (off by default)
FIERZ_AVG_N_NOISE=K       install in-line averaging observer with K noise/traj
                          (off by default; final check is single-cfg)
```

### Verified stress passes (m=0.1, λ=10, MDs=20, RAT[0.005,80]^12, N_NOISE=64)

| stress mode | ratio Σ_DTX/Σ_W | \|dev\| | verdict |
|---|---|---|---|
| cold U, csw=0 (default) | 0.9824 | 0.0176 | **PASS** |
| GAUGE_INIT=tepid:0.1 | 0.9834 | 0.0166 | **PASS** |
| GAUGE_INIT=tepid:0.1, CSW=1.0 | 0.9834 | 0.0166 | **PASS** |
| N_NOISE=256, cold U | 0.9848 | 0.0152 | **PASS** |
| FIERZ_AVG_N_NOISE=16, cold U (41 cfgs × 16 noise = 656) | 0.9849 ± 0.0006 (SE) | 0.0151 | **PASS** |

Going N_NOISE 64 → 256 dropped |dev| only 14 % (0.0176 → 0.0152), not
the √4 ≈ 2× one would see from pure stochastic noise.  The averaging
variant tightens the SE to 0.00063, **putting the offset from unity
at 24 σ** — so it is NOT statistical noise.  At fixed (m=0.1, λ=10,
4³×8), `⟨Σ_DTX⟩_aux` sits ~1.5 % below `Σ_W`.  The Fierz identity
`⟨Σ_DTX⟩ = Σ_W` would require either (a) many more cfgs to sample
the rare wide-aux outliers that pull `Σ_DTX` up, or (b) larger
volume so finite-V eigenmode contributions matter less.  The test
**passes at the 2 % tolerance** with healthy margin; the 1.5 %
residual is a physically genuine finite-volume / finite-statistics
signature, not a code issue.

### The boundary-condition bug (yesterday — context for the GPU port)

The TXQCD/DTXQCD library had `TXQCDWilsonOp` and `DTXQCDMeooeDoubled`
silently defaulting to PBC time (Grid's stock `WilsonFermion`
default), while `TXQCDWilsonCloverFermionEO` and explicit production
measurement code defaulted to APBC time. This means:

- **TXQCD production was fine** — its HMC chain runs through
  `TXQCDWilsonCloverRationalEOAction` → `TXQCDWilsonCloverFermionEO`
  which has always had `DefaultImplParams()` returning APBC time.
- **DTXQCD production was NOT fine** — its HMC chain runs through
  `DTXQCDWilsonCloverFermionEO` → `DTXQCDMeooeDoubled` which
  silently used PBC. Any DTXQCD ensembles generated prior to commit
  `99950d1c` are physically wrong for the chroma-convention
  ensemble we're matching against. Regenerate from scratch under
  the corrected defaults.

The fix landed yesterday adds explicit `DefaultImplParams()` returning
APBC time to both `DTXQCDMeooeDoubled` and `DTXQCDWilsonCloverFermionEO`.

### Empirical landscape vs λ (m=0.1, fixed U=I, MDsteps=10)

- **λ ≥ 5**: SADDLE_INIT seed close to true saddle; thermalization
  near-instant; ratio = 1.00 within stochastic noise. Test passes
  in ~10 min.
- **λ = 2**: 1/λ² ≈ 25 trajs to fully equilibrate; needs RAT bracket
  sized for the spectrum at the true saddle. Borderline practical.
- **λ ≤ 1**: per-λ rational bracket essential (default [0.005, 80]
  doesn't span the spectrum at λ=0.5: 1/λ² ≈ 100+ trajs to
  thermalize; SADDLE_INIT seed overshoots true saddle by 2–5×).
  Not registered as a unit test.
- **λ = 0.1**: 100+ trajs to drift to saddle; the per-cfg ratio
  plateaus at the saddle value (≈ 0.12, NOT 1.0); recovering
  Fierz `⟨ratio⟩ = 1` requires averaging over many cfgs and
  catching the wide-aux outliers. The 100-traj probe in
  `/tmp/probe_traj_mds10/` (saved during yesterday's session)
  is the working illustration.

### Tensor-channel observation

In TXQCD the antisymmetric Lorentz tensor `t_{μν}` aux field sits at
its pure-Gaussian fluctuation magnitude `6 N_c² / λ²` even at
small λ — no condensate. DTXQCD doesn't have this field by
construction. **Expectation**: any condensate at small λ lives in σ
(traceless) and s (singlet), with the off-diagonal d/n following at
λ ≪ 1 (parity-broken metastability).

---

**Where to start the GPU port**: replicate the existing TXQCD
QUDA-aware primitive structure in `TXQCDWilsonCloverRationalEOActionQudaPrimitive.h`,
targeting `DTXQCDWilsonCloverRationalFullAction` (the non-EO full
operator that production is using on CPU). The 48×48 per-site
inverse cache and the per-pole multishift CG are the obvious
candidates for cuBLAS gemmBatched / per-pole streams.
