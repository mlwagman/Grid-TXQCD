#!/bin/bash
# N-level Hasenbusch wallclock scan at 16³×48 b6.1 (chroma equilibrated cfg).
# Picks the optimal N + mass list for production by measuring trajectory time
# at N=1 (no Hasenbusch), 2, 3, 4.  Runs on a single 4-GPU node (mpi=1.1.1.4).
#
# Output: per-N wallclock + per-level CG iter counts in logs/nscan_b6p1_N*.log.
# Decision rule: smallest "total light-CG iters × ms/iter" wins.

cd /lustre2/nplqcd/Grid-TXQCD/production
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

mkdir -p logs

# Reuse the production chroma 16³×48 cfg11100 as the seed.
CHROMA_CFG=/lustre2/nplqcd/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_cfg_11100.lime

# Common env — production b6.1 params + production QUDA stack + --shm-mpi 1.
export LATT=16.16.16.48
export MASS_LIGHT=-0.2450
export MASS_STRANGE=-0.2050
export CSW=1.24930970916466
export BETA=6.1
export LAMBDA=0.5
export SUFFIX=_nscan
export N_TRAJ=1
export NO_METROP=1
export IMPORT_CFG="$CHROMA_CFG"
export AUX_INIT=3.07
export INTEGRATOR=MinimumNorm2
export LAMBDA_MN2=0.1789
export MDSTEPS=2
export TRAJL=0.353553390593274
export GAUGE_MULT=4
export GAUGE_INNER_MULT=2
export AUX_MULT=1

# Full QUDA stack to exercise the path we'll use at production.
export QUDA_FORCE=1
export QUDA_FORCE_KERNEL=1
export QUDA_FORCE_LIGHT=1
export USE_HMC_MG=1
export TXQCD_QUDA_HYBRID=1
export TXQCD_QUDA_FULL=1
export TXQCD_PRECOMPUTE_GPU=1
export TXQCD_MOOEEINV_CUBLAS=1
export TXQCD_MOOEE_CUBLAS=1
export EIG_DIAG=0
export QUDA_ENABLE_MPS=1

# Production rationals (no RAT_* overrides this time — we want real perf).

run_one () {
  local label="$1"
  local logfile="logs/nscan_b6p1_${label}.log"
  echo "=== N-scan: $label ===" | tee "$logfile"
  date | tee -a "$logfile"
  T0=$(date +%s)
  srun --overlap --mpi=pmix --export=ALL \
       -N 1 -n 4 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=none \
       ./gen_txqcd_cfgs_2plus1 \
         --grid 16.16.16.48 --mpi 1.1.1.4 \
         --shm 1024 --shm-mpi 1 \
    >> "$logfile" 2>&1
  T1=$(date +%s)
  echo "WALL($label) = $((T1-T0)) seconds" | tee -a "$logfile"
  # Extract CG iters
  grep -E 'TXQCDMultiShiftCGSchur: converged|dH = ' "$logfile" | tee -a "$logfile.summary"
}

# N=1: no Hasenbusch (single rational at m_light).
unset HASEN_DM
unset HASEN_LADDER
SUFFIX="_nscan_N1" run_one N1

# N=2: HASEN_DM=0.05 (light → light+0.05).
HASEN_DM=0.05 SUFFIX="_nscan_N2_dm005" run_one N2_dm005

# N=2: HASEN_LADDER explicit; should match HASEN_DM=0.05.
unset HASEN_DM
HASEN_LADDER="-0.245,-0.195" SUFFIX="_nscan_N2_ladder" run_one N2_ladder

# N=3: light=-0.245, mid=-0.15, heavy=0.0.
unset HASEN_DM
HASEN_LADDER="-0.245,-0.15,0.0" SUFFIX="_nscan_N3" run_one N3

# N=4: light=-0.245, m2=-0.18, m3=-0.05, m4=+0.20.
HASEN_LADDER="-0.245,-0.18,-0.05,0.20" SUFFIX="_nscan_N4" run_one N4

# N=5 chroma-matched (b6.3 mass step sizes ported to b6.1 light=-0.245):
# scale chroma's Δm = {0.0016, 0.008, 0.014, 0.031} from b6.3 ladder.
HASEN_LADDER="-0.245,-0.2435,-0.2355,-0.2215,-0.1905" SUFFIX="_nscan_N5_chroma" run_one N5_chroma

# ─── Final summary ───────────────────────────────────────────────────────────
echo ""
echo "=== N-SCAN SUMMARY ==="
for f in logs/nscan_b6p1_*.log; do
  label=$(basename "$f" .log | sed 's/nscan_b6p1_//')
  WALL=$(grep -oE 'WALL\([^)]+\) = [0-9]+' "$f" | tail -1)
  DH=$(grep -E 'dH = ' "$f" | tail -1 | grep -oE 'dH = [^ ]+')
  ITERS=$(grep -cE 'TXQCDMultiShiftCGSchur: converged' "$f")
  echo "$label: $WALL  $DH  $ITERS CGs"
done
