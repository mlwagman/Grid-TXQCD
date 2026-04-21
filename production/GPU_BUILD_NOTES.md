# Grid-TXQCD GPU Build & Production Handoff

## What This Is

Grid-TXQCD is a fork of [Grid](https://github.com/paboyle/Grid) with TXQCD (Tensor eXtended QCD) extensions. The `production/` directory contains standalone programs for Nf=2+1 gauge generation and 2-point measurements, comparing QCD and TXQCD.

## Building Grid with GPU Support

From the repo root (`Grid-TXQCD/`):

```bash
./bootstrap.sh
mkdir build-gpu && cd build-gpu

# NVIDIA (CUDA):
../configure --enable-accelerator=cuda --enable-simd=GPU \
  --enable-comms=mpi-auto --enable-accelerator-aware-mpi=yes \
  CXX=nvcc MPICXX=mpicxx

# AMD (HIP):
../configure --enable-accelerator=hip --enable-simd=GPU \
  --enable-comms=mpi-auto --enable-accelerator-aware-mpi=yes \
  CXX=hipcc MPICXX=mpicxx

make -j$(nproc)
```

Required libraries: GMP, MPFR, HDF5 (for measurement I/O).

## Building Production Programs

```bash
cd production/
```

Edit `Makefile` line 1 to point to your GPU build:
```makefile
GRID_CONFIG = ../build-gpu/grid-config
```

Then:
```bash
make
```

This builds 8 programs: `gen_qcd_cfgs`, `gen_txqcd_cfgs`, `meas_conn_qcd`, `meas_conn_txqcd`, `meas_disco_qcd`, `meas_disco_txqcd`, `meas_aux_txqcd`, `compare`.

## Ensemble Parameters (in params.h)

All physics parameters live in `production/params.h`. Current target:

| Parameter | Value |
|-----------|-------|
| Lattice | 16³ × 48 |
| Nf | 2+1 (degenerate) |
| m_light = m_strange | -0.2450 |
| csw | 1.24930970916466 |
| Gauge action | Symanzik |
| beta | 6.1 |
| u0 | 0.832605301399891 |
| Stout (inversions) | rho=0.125, n=1 |
| Stout (sources) | rho=0.16, n=3 |
| Gaussian smearing | width=2.1, niter=20 |
| HMC trajectory | sqrt(2), ForceGradient, 10 MD steps |
| Thermalization | 300 trajectories |
| Production | 500 trajectories, checkpoint every 10 |

**lambda** (TXQCD auxiliary coupling) is the tuning parameter. It appears in directory names: `cfgs/txqcd_lam0.5000/`, `meas_2pt/txqcd_lam0.5000/`. Change it in params.h and `make` to run parallel streams.

## Run Sequence

### 1. QCD gauge generation (needed first for lambda tuning)
```bash
cd production/
mpirun -np <N> ./gen_qcd_cfgs --grid <Lx.Ly.Lz.Lt> --mpi <mx.my.mz.mt>
```
- Resumes automatically from the latest checkpoint in `cfgs/qcd/`
- Writes HMC diagnostics including `vev_trminv` (= Sigma, the per-flavor VEV)
- Use Sigma to compute: `lambda_opt = sqrt(4 * Sigma / 1.2035)`

### 2. TXQCD gauge generation
```bash
# After setting lambda in params.h and rebuilding:
mpirun -np <N> ./gen_txqcd_cfgs --grid <Lx.Ly.Lz.Lt> --mpi <mx.my.mz.mt>
```
- Also resumes from checkpoints
- Writes to `cfgs/txqcd_lam<X.XXXX>/`

### 3. Measurements (per-config, embarrassingly parallel)
```bash
# Single config:
mpirun -np <N> ./meas_conn_qcd <traj> --grid <Lx.Ly.Lz.Lt> --mpi <mx.my.mz.mt>
mpirun -np <N> ./meas_disco_qcd <traj> --grid ...
mpirun -np <N> ./meas_conn_txqcd <traj> --grid ...
mpirun -np <N> ./meas_disco_txqcd <traj> --grid ...
mpirun -np <N> ./meas_aux_txqcd <traj> --grid ...

# Or use driver scripts (sequential, handles skip logic):
./run_all_qcd_measurements.sh --grid <Lx.Ly.Lz.Lt> --mpi <mx.my.mz.mt>
LAMBDA=0.5000 ./run_all_txqcd_measurements.sh --grid <Lx.Ly.Lz.Lt> --mpi <mx.my.mz.mt>
```

### 4. Fierz consistency check
```bash
./compare --grid <Lx.Ly.Lz.Lt>
```
Reads from `meas_2pt/txqcd_lam<X.XXXX>/` and `meas_2pt/qcd/`, checks that TXQCD correlators satisfy Fierz identities vs QCD.

## Key Architecture Notes

- **EO preconditioning**: Both QCD and TXQCD gauge generators use even-odd preconditioned fermion actions (`QCDLogDetCloverEOAction` + `TwoFlavourSchurCloverAction` for Nf=2 light, `QCDLogDetCloverEOAction` + `OneFlavourSchurCloverRationalAction` for Nf=1 strange). This gives ~2.6x CG iteration reduction vs full-volume.
- **TXQCD strange quark**: QCD EO actions are wrapped via `QCDActionAdapter` to participate in TXQCD HMC (extracts gauge field `U` from the composite `TXQCDField`).
- **Smeared fermion action**: All fermion and gauge actions have `is_smeared = true` — the integrator applies stout smearing before action/force evaluation and back-projects forces.
- **Measurement inversions**: Use full-volume MdagM CG (not EO) for propagator inversions. This is intentional — the measurement CG is a one-shot solve, not an HMC force.
- **Connected sources**: 4³ spatial × 12 time = 768 Gaussian-smeared point sources per config, with source and sink smearing using a separate stout-smeared gauge field.
- **Checkpoint resume**: Both gauge generators scan for existing checkpoints and resume automatically.

## Directory Structure After Running

```
production/
├── cfgs/
│   ├── qcd/                          # QCD checkpoints + diagnostics
│   │   ├── ckpoint_lat.<traj>
│   │   ├── ckpoint_rng.<traj>
│   │   └── hmc_diagnostics.<traj>.h5
│   └── txqcd_lam0.5000/             # TXQCD checkpoints (one dir per lambda)
│       ├── ckpoint_lat.<traj>
│       ├── ckpoint_lat_aux.<traj>
│       ├── ckpoint_rng.<traj>
│       └── hmc_diagnostics.<traj>.h5
├── meas_2pt/
│   ├── qcd/                          # QCD measurement data
│   │   ├── conn_qcd_<traj>.h5
│   │   └── disco_qcd_<traj>.h5
│   └── txqcd_lam0.5000/             # TXQCD measurement data
│       ├── conn_txqcd_<traj>.h5
│       ├── disco_txqcd_<traj>.h5
│       └── aux_txqcd_<traj>.h5
└── logs/
```
