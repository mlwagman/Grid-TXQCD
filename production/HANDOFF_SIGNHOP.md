# HANDOFF: DTXQCD/TXQCD sign-sector ("signhop") scans — results, tools, and how to run new ones locally

Written 2026-08-14 for a fresh Claude context working on a laptop/CPU build.
Everything in this tarball is self-contained except the Grid-DTXQCD source
tree itself (branch `dtxqcd-v2`), which you build once (see §4).

## 1. What the campaign established (the physics you are extending)

The DTXQCD fermion determinant is a Pfaffian, `Pf(K·M48)`; HMC samples
`|Pf|·e^{−S_aux}` and the sign `signPf = (−1)^{n_neg/2}` of the dense
γ5·M48 spectrum (Kramers-doubled, so n_neg is always even — using
`(−1)^{n_neg}` was a bug that could never see a flip) must be reweighted.
At production volume (16³×48) the dense spectrum is unmeasurable, so the
question "where does the sign problem set in as a function of λ" was
answered on 4⁴ chains where the 48·V = 12288-dimensional dense spectrum is
cheap, and the plaquette deficit was validated as the production-volume
proxy.

**Result table (standard aux action, 4⁴, β=6.1 point m=−0.245 csw=1.2493;
`signhop_table.tex`, ntherm=50, 20-traj blocked errors):**

| λ | N_traj | ⟨signPf⟩ | ⟨P⟩ | ΔP vs λ=5 | λ²⟨s⟩ |
|---|---|---|---|---|---|
| 0.1 | 783 | +0.02(4) | 0.4415 | +0.064 | 0.40 |
| 0.5 | 734 | +0.01(4) | 0.4402 | +0.065 | 1.01 |
| 1 | 637 | +0.03(5) | 0.4419 | +0.063 | 2.86 |
| 2 | 955 | −0.06(4) | 0.4641 | +0.041 | 4.41 |
| 2.25 | 155 | −0.07(3) | 0.4660 | +0.039 | 6.02 |
| 2.5 | 868 | +0.07(3) | 0.4750 | +0.030 | 6.13 |
| 2.75 | 896 | +0.07(3) | 0.4765 | +0.029 | 6.21 |
| 3 | 455 | **+0.49(4)** | 0.4758 | +0.029 | 6.24 |
| 4 | 208 | **+1.000(0)** | 0.4968 | +0.008 | 6.16 |
| 5 | 355 | **+1.000(0)** | 0.5050 | 0 | 6.13 |

Key conclusions (full narrative in `notes/dtxqcd-smalllambda-sign-catastrophe.md`):
- ⟨signPf⟩ ≈ 0 for λ ≤ 2.25, first positive nudge at 2.5–2.75, sharp rise
  2.75 → 3 → 4, saturated (zero flips) at λ ≥ 4.
- **Plaquette deficit is the validated proxy**: ΔP → 0 exactly as ⟨signPf⟩ → 1.
  At production volume, plaq-vs-QCD is therefore the sign diagnostic.
- The 4⁴ curve is a *mechanism map* quantitatively distorted at intermediate
  λ (deconfined 4⁴ background + rational near-wall bias overstate mixing):
  production-volume forks are the authoritative boundary — λ=2 forks decay
  away from QCD, λ=3 holds at QCD (plaq 0.51339(15) vs λ=5 0.51370(6)).
- Mechanism is Wilson-specific: aux fields drag unprotected real Wilson
  modes through zero (dense-spectrum scans over aux prior draws:
  `Test_dtxqcd_sign_aux_draws`, `analyze_sign_draws.py`).
- 4⁴ |Pf| HMC **self-flips** sign sectors (walls are close at 4⁴), so
  ⟨signPf⟩ is measurable from a plain chain; the explicit Metropolis "hops"
  (s_tslice0 / negate_sigma, `dtxqcd_sign_hop.h`) exist but are dead at the
  saddle aux for λ ≤ 1 — keep `DTXQCD_SIGN_HOP_PROB` on anyway, it is free.
- Parity heatbath does NOT cross sectors (η ≡ +1 structurally).

## 2. Inventory of this tarball

```
HANDOFF_SIGNHOP.md            this note
dtxqcd_sign_hop.h             the on-chain signPf observer (HmcObservable):
                              dense γ5·M48 assembly + eigensolve each traj,
                              [SIGNHOP] record + optional Metropolis hops.
                              cuSOLVER on CUDA builds, **Eigen CPU fallback
                              added 2026-08-14** (this tarball's version).
Test_dtxqcd_sign_aux_draws.cc dense spectra over aux PRIOR draws at fixed
                              gauge (weak/tepid/hot), the mechanism probe
                              (Eigen fallback added).
Test_txqcd_sign_aux_draws.cc  TXQCD analog (24×24 site matrix, Wilson vs
                              Möbius) — the dense-spectrum kernel you reuse
                              for a TXQCD on-chain sign record (Eigen
                              fallback added).
Test_dtxqcd_sign_flip.cc      sign-flip transform menu (which aux moves
                              cross sectors); sign formula fixed 07-04.
Test_qcd_quenched_plaq.cc     pure-gauge anchor (quenched plaq at the
                              production action point) — the "quasi-
                              quenching" end of the deficit story.
analyze_signhop.py            ⟨signPf⟩ per log, blocked errors, flip rate
make_signhop_table.py         the table above (+ plaq, VEVs) → signhop_table.tex
analyze_sign_draws.py         prior-draw spectra analysis
analyze_sign_plaq_corr.py     sign⊗plaq correlation test (uncorrelated at 4⁴)
analyze_sign_reweighting.py   production-volume sign observer consumer
slurm_dtxqcd_signhop_4t4.sh   THE launcher (env contract in §5)
slurm_dtxqcd_signdraws.sh, slurm_dtxqcd_signflip*.sh, slurm_sign_*.sh,
slurm_txqcd_sign_dwf.sh       the other campaign launchers (cluster sbatch;
                              read for the env contracts, run locally via §5)
signhop_table.tex             the result table (auto-generated; regenerate)
cfgs/                         all 13 4⁴ chains: checkpoints (ckpoint_lat,
                              _daux, _rng every 50 trajs) + hmc_diagnostics
                              h5 (aux walls/VEVs).  ~90 MB.  Resume any chain
                              by pointing DTXQCD_SUFFIX at it.
logs/                         per-chain slurm logs reduced to the analysis
                              lines ([SIGNHOP], "Traj N plaq", headers).  The
                              analyzers run on these unchanged.  Full 1.5 GB
                              originals: lq2:/lustre2/nplqcd/Grid-DTXQCD/
                              production/slurm-logs/dtx_signhop_*.out
notes/                        the narrative records: sign catastrophe (the
                              campaign log), λ-stability map, Fierz-exactness
                              rule, TXQCD tensor-condensed-phase / EO-artifact
                              note, DTXQCD_SIGN_REWEIGHTING.md (June design
                              doc — predates the campaign, historical),
                              PARITY_HB_SIGN_FLIP_STATUS.md, TXQCD non-EO
                              clover rational handoff.
```

## 3. The two new scans you are being asked to run

### 3a. β=6.5 (the fine-spacing point) λ-scan — β/λ_crit interplay
Same 4⁴ recipe with the b6.5 ensemble parameters: `BETA=6.5
CSW=1.170082389372972 MASS_LIGHT_DTXQCD=-0.1788 MASS_STRANGE=-0.1788`.
(Chroma reference for these: cl3_32_64_b6p5_m0p1788; the production DTXQCD
b6.5 chain launched 2026-08-14 on lq2.)  Scan λ ∈ {0.5, 1, 2, 2.5, 3, 4, 5}
first, then refine around the transition.  Like-for-like caveat: 4⁴ at
β=6.5 is an even smaller physical box than at 6.1 — the comparison is
mechanism-map vs mechanism-map, not physics vs physics; the production-
volume plaq-deficit proxy on the b6.5 chain is the physics check.

### 3b. Clean TXQCD sign runs at the empirically-found production region
TXQCD (24×24 site matrix, `Grid/qcd/action/txqcd/`) has det(M) REAL, so
the sign is sign(det M) = (−1)^{n_neg} of the dense γ5·M24 spectrum
(NOT Kramers-doubled — do not halve n_neg there).  History: the TXQCD
production plaquette deficit / "tensor-condensed phase" at small λ was
found at production scale; `notes/project_tensor_condensed_phase.md`
records that much of the original λ≲2 "instability" was an EO-Schur
preconditioner artifact (Mee near-zero mode ↔ Mpc near-infinity), so the
clean runs MUST be non-EO (`USE_FULL_PF=1`-style full operator) exactly as
the DTXQCD 4⁴ chains were.  There is no TXQCD on-chain sign observer yet:
build it by cloning `dtxqcd_sign_hop.h` and swapping the assembler for the
TXQCD 24×24 site matrix (the dense-assembly + eigensolve code you need is
already in `Test_txqcd_sign_aux_draws.cc`); wire it into the TXQCD 4⁴
driver (`HMC/TXQCD_Wilson_small.cc` pattern).  Then run the same λ ladder
(the TXQCD region of interest was λ ≲ 2 at production; the aux width is
1/λ in both theories so the ladders are comparable).

## 4. Building locally (CPU)

```
git clone <Grid-DTXQCD>  &&  git checkout dtxqcd-v2
./bootstrap.sh
mkdir build-cpu && cd build-cpu
../configure --enable-simd=AVX2 --enable-comms=none \
             --disable-fermion-reps --disable-gparity     # (or mpi; AVX512 if you have it)
make -j                                                   # libGrid
cd ../production
make GRID_BUILD_DIR=../build-cpu gen_dtxqcd_cfgs Test_dtxqcd_sign_aux_draws \
     Test_dtxqcd_sign_flip Test_txqcd_sign_aux_draws
```
No LIME needed (4⁴ chains start from weak field + AUX_INIT_AUTO, not a
chroma import).  The observer's dense eigensolve on CPU is Eigen
`SelfAdjointEigenSolver` on a 12288² Hermitian matrix: ~1–3 min per
trajectory on a laptop core set; the 4⁴ non-EO HMC step itself is seconds.
So ~1000 trajectories ≈ 1–2 days per λ — run the λ ladder as parallel
processes.  (A 4³×2 or 2⁴ lattice would be 10–100× faster but is NOT
comparable to the existing table; if you shrink, redo the β=6.1 baseline
at the same size.)

## 5. Running a scan (the env contract, from `slurm_dtxqcd_signhop_4t4.sh`)

```
cd production      # cfg_dir is RELATIVE — the resume scan looks under ./cfgs/
env LATT=4.4.4.4 LAMBDA_DTXQCD=2.5 USE_FULL_PF=1 \
    MDSTEPS=10 TRAJL=0.353553390593274 RHMC_DEG=8 \
    STOUT_NSMEAR=1 STOUT_RHO=0.125 AUX_INIT_AUTO=1 RAT_AUTO_HI=1 \
    DTXQCD_SIGN_HOP_PROB=0.5 \
    TRAJ=1005 NO_METROP=5 N_SKIP=50 \
    DTXQCD_SUFFIX="_4t4_signhop_lam2.5" \
    [BETA=6.5 CSW=1.170082389372972 MASS_LIGHT_DTXQCD=-0.1788 MASS_STRANGE=-0.1788] \
  ./gen_dtxqcd_cfgs --grid 4.4.4.4 --mpi 1.1.1.1 > logs/dtx_signhop_l2p5_b65.out 2>&1
```
- `TRAJ` is a TARGET (executed = TRAJ − NO_METROP − start); rerunning with
  a larger TRAJ resumes from the latest checkpoint in the suffix dir.
- `USE_FULL_PF=1` = non-EO full-operator RHMC (mandatory at small λ: the
  EO-Schur path blows up there — see notes).  `RAT_AUTO_HI=1` keeps the
  rational bounds tracking the spectrum (the near-wall Remez distortion is
  a known systematic of the measure at small λ; note it, don't fight it).
- Every trajectory prints `[SIGNHOP] traj=N signPf=±1 n_neg=… logabsdet=…
  min_abs_eval=…`; the observer asserts single rank and V ≤ 4096.
- Analysis: `analyze_signhop.py --ntherm 50 logs/*.out`; the table:
  `make_signhop_table.py --ntherm 50` (reads `logs/` names `dtx_signhop_l<λ>`
  with `p` for the decimal point, and `cfgs/…/hmc_diagnostics.*.h5` for
  the VEV columns — keep those naming conventions).
- Thermalization: discard 50 (the table's convention).  Errors: 20-traj
  blocks (τ_int ~ 1 in the sign series at 4⁴, but verify with the flip-rate
  print at each new β/λ).

## 6. Gotchas that cost time before
- Sign formula: DTXQCD `(−1)^{n_neg/2}` (Kramers), TXQCD `(−1)^{n_neg}`.
- The λ=2 value came out slightly NEGATIVE (−0.06(4)); ⟨s⟩ must be > 0
  (Z_signed = det² > 0), so that is finite-stats + near-wall rational bias,
  not physics — expect the same wobble at the new β.
- Sign ⊗ plaq are uncorrelated at 4⁴ (`analyze_sign_plaq_corr.py`): the
  corrections live in exponentially suppressed correlations — do not read a
  "signed plaq" from 10³ trajectories as the QCD value.
- Observer destructor must not `cusolverDnDestroy` (CUDA teardown abort);
  irrelevant on CPU builds.
- Fierz exactness is the invariant (`notes/dtxqcd-fierz-exactness.md`):
  DTXQCD must reproduce QCD ⟨O⟩ at all λ once signs are included; a
  plaquette ≠ QCD is a sign/measure symptom, never "DTXQCD physics".
```
