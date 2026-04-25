# HMC integrator convention dictionary: Grid ↔ chroma

This document maps Grid-TXQCD HMC parameters to/from chroma so production
runs in this fork can match chroma reference ensembles bit-for-bit (in
physical MD dynamics, not literal floats).

## TL;DR

**Grid's `eps_grid = (1/4) × chroma's `dt_chroma`** for the same physical
step. So to reproduce chroma's `n_steps=7 tau0=√2`:

```
INTEGRATOR=MinimumNorm2  LAMBDA_MN2=0.1789  GAUGE_INNER_MULT=4
TRAJL=0.353553390593274     # = √2 / 4
MDSTEPS=7
```

Empirically verified on `cl3_16_48_b6p1_m0p2450`: dH ~ O(1), plaq holds
at chroma's 0.5135.  4× cheaper per trajectory than naively using
`TRAJL=√2 MDSTEPS=28` (which corresponds to chroma `tau0=4√2` — a 4×
longer physical trajectory).

## Why factor 4

Two compounding conventions:

### 1. Momentum variance (factor 2 in σ, 4 in variance)

| | Grid | chroma |
|---|---|---|
| Refresh | `P *= sqrt(2)` | `P *= sqrt(0.5)` |
| Var(c_a) per Lie-alg generator | 2 | 1/2 |
| σ(c_a) ratio | 2 | 1/(2) |

(Grid uses `CPS_MD_TIME` defined in `Grid/qcd/action/gauge/GaugeImplTypes.h`
which sets `HMC_MOMENTUM_DENOMINATOR=2`.)

### 2. Implicit kinetic mass (factor 4)

The Hamiltonian's `H_kin` has different normalization in each:

- Grid: `H_kin = (1/2) tr(P²)/MOM_DENOM = (1/2)|P_grid|²/2 = (1/4)|P_grid|²`
  with `Var(P_grid)=2` ⇒ ⟨H_kin⟩ = N_dof × 2/4 = N_dof/2 (canonical).
- Chroma: `H_kin = |P_chroma|²` (in `mesKE` of `exact_hamiltonian.h`,
  with the −4VNd offset for log readability), `Var(P_chroma)=1/2`
  ⇒ ⟨H_kin⟩ = N_dof × 1/2 (canonical).

Both give canonical ⟨H_kin⟩=N_dof/2, but in terms of the implicit kinetic
mass `m` defined by `H_kin = (1/(2m))|P|²`:

- `m_grid = 2`
- `m_chroma = 1/2`
- ratio = 4

For the equations of motion `dQ/dt = P/m`, the same physical step `dQ`
corresponds to:

- Grid: `dQ = (P_grid/m_grid)·dt_phys = (P_grid/2)·dt_phys`. Grid's
  `update_field` does `dQ = P_grid · eps_grid`, so `eps_grid = dt_phys/2`.
- Chroma: `dQ = (P_chroma/m_chroma)·dt_phys = 2·P_chroma·dt_phys`. Chroma's
  `leapQ` does `dQ = P_chroma · dt_chroma`, so `dt_chroma = 2·dt_phys`.

Therefore `eps_grid / dt_chroma = (dt_phys/2)/(2·dt_phys) = 1/4`.

## Empirical confirmation

dt-scan from `cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime`, NO_METROP=1, single
trajectory. ForceGradient unless noted.

| Grid MDs at trajL=√2 | eps_grid | implied chroma dt | dH | plaq |
|---|---|---|---|---|
| 7  (chroma's MDs *naïvely matched*) | 0.20 | 0.81 | 1.5M | 0.42 |
| 14 | 0.10 | 0.40 | 22k | 0.51 |
| 28 (= chroma's *physical* dt of 0.20) | 0.05 | 0.20 ✓ | -0.18 | 0.5135 ✓ |
| 56 | 0.025 | 0.10 | 0.029 | 0.5135 ✓ |

**Same scan with MN2 (Grid default λ=0.1932):**

| MDs | eps | dH (FG) | dH (MN2) |
|---|---|---|---|
| 14 | 0.10 | 52 | 10764 |
| 28 | 0.05 | -0.18 | -0.23 |
| 56 | 0.025 | 0.029 | -0.069 |

MN2 ≈ FG at dt=0.05 in dH; MN2 is ~35% cheaper per trajectory (2 force
evals/step vs ~3 for FG).  At dt > 0.05 MN2 fails earlier than FG (lower-
order error blows up); at dt ≤ 0.05 they're equivalent and MN2 wins on cost.

## Chroma's exact production setup (cl3_16_48_b6p1_m0p2450)

Read from the LIME header of any chroma config in
`/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/`:

```xml
<MDIntegrator>
  <tau0>1.414</tau0>
  <Integrator>
    <Name>LCM_STS_MIN_NORM_2</Name>
    <n_steps>7</n_steps>
    <lambda>0.1789</lambda>
    <SubIntegrator>
      <Name>LCM_STS_MIN_NORM_2</Name>
      <n_steps>4</n_steps>
      <monomial_ids>HMC::lw_tree_gauge</monomial_ids>
    </SubIntegrator>
  </Integrator>
</MDIntegrator>
```

Translation to Grid:

```bash
INTEGRATOR=MinimumNorm2     # LCM_STS_MIN_NORM_2
LAMBDA_MN2=0.1789           # chroma's λ (Grid default = 0.1932)
GAUGE_INNER_MULT=4          # multi-rate gauge sub-integrator
TRAJL=0.353553390593274     # √2 / 4 (chroma's tau0 / 4)
MDSTEPS=7                   # outer fermion steps
```

Plus the gauge action:
```cpp
PlaqPlusRectangleAction(beta, -beta/(20*u0*u0))   // chroma's LW_TREE_GAUGEACT
GaugeAction.is_smeared = false                     // gauge sees thin links
```

And antiperiodic time BC for fermions (`boundary_phases = {1,1,1,-1}`).

## Implications for chroma logs

When reading chroma logs that mention `tau0=1.414 n_steps=7`, the Grid
equivalent is **NOT** `TRAJL=1.414 MDSTEPS=7`.  Multiply chroma's `tau0` by
`1/4` and keep `n_steps` as-is.  Equivalently, multiply chroma's `dt` by
`1/4` to get Grid's `eps`, then pick any `(TRAJL, MDSTEPS)` whose ratio is
this `eps`.

## What `is_smeared = false` for the gauge action

Chroma's `LW_TREE_GAUGEACT` operates on the thin (unsmeared) gauge links
even when `STOUT_FERM_STATE` is wrapping the fermion action.  In Grid, this
is achieved by

```cpp
PlaqRectR GaugeAction(beta, -beta/(20.0 * u0 * u0));
GaugeAction.is_smeared = false;   // gauge sees thin U; fermions see smeared
```

Without this fix the gauge action evaluates on the stout-smeared links,
which gives a 0.12 plaq equilibrium shift vs chroma's 0.5135 reference.

## Force-FD diagnostic

`production/gen_qcd_cfgs` accepts `TEST_FORCE_FD=1` to skip HMC and instead
verify `(S(U+εp)−S(U−εp))/2ε` matches `Σ_μ Re tr(p_μ·∂S/∂U_μ)` for each
action via the projected analytic deriv.  At `CG_TOL=1e-13` the ratio is 1
to 8 digits for all 5 production actions.  This confirmed the integrator
instability at chroma's nominal MDs=7 is a true symplectic-stability
boundary, not a force/action mismatch.
