#!/bin/bash
# N-level Hasenbusch ladder validation on 4⁴.
#
# Test 1 (bit-exact wiring regression):
#   HASEN_LADDER="m_l,m_h"   vs   HASEN_DM=(m_h - m_l)
#   Same seed, NO_METROP=1, MDS=4.  dH must match to 14 digits.
#
# Test 2 (3-level smoke):
#   HASEN_LADDER="m_l,m_mid,m_h"
#   Just confirm it runs without error and emits a finite dH.

cd /lustre2/nplqcd/Grid-TXQCD/production
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=4

mkdir -p logs

# Common 4⁴ env — match what the existing Hasenbusch 4⁴ FD tests use.
export LATT=4.4.4.4
# Match Test_txqcd_hasenbusch_force.cc validated 4⁴ params (mass_l=0.3, csw=0):
# heavy + clover-off + small aux → multishift CG converges in O(50) iters.
# This is a wiring-only correctness test, not physics realism.
export MASS_LIGHT=0.3
export MASS_STRANGE=1.5     # very heavy → strange RHMC CG converges in ~5 iters
export CSW=0.0
export BETA=6.1
export LAMBDA=2.0          # → σ = AUX_INIT/λ² = 0.025 (small perturbation)
export N_TRAJ=1
export NO_METROP=1
export INTEGRATOR=MinimumNorm2
export LAMBDA_MN2=0.1789
export MDSTEPS=2
export TRAJL=0.353553390593274
export GAUGE_MULT=4
export AUX_MULT=1
export AUX_INIT=0.1
# Fast wiring-test rationals: smaller pole range, fewer poles, looser tol.
# Production uses RAT_LO=1e-4 RAT_HI=100 RAT_DEGREE=20 RAT_TOL=1e-8 (unset).
export RAT_LO=0.01
export RAT_HI=10.0
export RAT_DEGREE=8
export RAT_TOL=1e-5
# Disable everything that adds memory/QUDA paths; this is a math correctness test.
unset TXQCD_QUDA_HYBRID
unset TXQCD_QUDA_FULL
unset USE_HMC_MG
export TXQCD_PRECOMPUTE_GPU=0
export TXQCD_MOOEE_CUBLAS=0
export TXQCD_MOOEEINV_CUBLAS=0
export EIG_DIAG=0

# Single-rank, fresh-start config (no IMPORT_CFG).
COMMON_ARGS="--grid 4.4.4.4 --mpi 1.1.1.1 --shm 256 --shm-mpi 1"

# ─────────────────────────────────── Test 1A ───────────────────────────────────
# 2-level via HASEN_DM=0.1.
LOG1=logs/test_hasen_4cube_v1_dm.log
echo "=== Test 1A: HASEN_DM=0.1 ===" | tee "$LOG1"
export HASEN_DM=0.1
unset HASEN_LADDER
export SUFFIX=_test_hasen_dm
srun --overlap --mpi=pmix --export=ALL -N 1 -n 1 --cpu-bind=none --gres=gpu:1 ./gen_txqcd_cfgs_2plus1 $COMMON_ARGS \
    >> "$LOG1" 2>&1 || echo "FAIL: HASEN_DM run errored"
DH_DM=$(grep -E 'dH = ' "$LOG1" | tail -1)
echo "HASEN_DM=0.1: $DH_DM"

# ─────────────────────────────────── Test 1B ───────────────────────────────────
# 2-level via HASEN_LADDER="m_light, m_light+0.1".
LOG2=logs/test_hasen_4cube_v1_ladder2.log
echo "=== Test 1B: HASEN_LADDER='-0.245,-0.145' ===" | tee "$LOG2"
unset HASEN_DM
export HASEN_LADDER="0.3,0.5"
export SUFFIX=_test_hasen_ladder2
srun --overlap --mpi=pmix --export=ALL -N 1 -n 1 --cpu-bind=none --gres=gpu:1 ./gen_txqcd_cfgs_2plus1 $COMMON_ARGS \
    >> "$LOG2" 2>&1 || echo "FAIL: HASEN_LADDER N=2 run errored"
DH_L2=$(grep -E 'dH = ' "$LOG2" | tail -1)
echo "HASEN_LADDER N=2: $DH_L2"

# ─────────────────────────────────── Test 2 ───────────────────────────────────
LOG3=logs/test_hasen_4cube_v2_ladder3.log
echo "=== Test 2: HASEN_LADDER='-0.245,-0.145,0.05' (3-level) ===" | tee "$LOG3"
unset HASEN_DM
export HASEN_LADDER="0.3,0.5,0.7"
export SUFFIX=_test_hasen_ladder3
srun --overlap --mpi=pmix --export=ALL -N 1 -n 1 --cpu-bind=none --gres=gpu:1 ./gen_txqcd_cfgs_2plus1 $COMMON_ARGS \
    >> "$LOG3" 2>&1 || echo "FAIL: HASEN_LADDER N=3 run errored"
DH_L3=$(grep -E 'dH = ' "$LOG3" | tail -1)
echo "HASEN_LADDER N=3: $DH_L3"

# ─────────────────────────────────── Verdict ───────────────────────────────────
echo ""
echo "=== Verdict ==="
echo "  Test 1A (HASEN_DM=0.1):    $DH_DM"
echo "  Test 1B (HASEN_LADDER N=2): $DH_L2"
echo "  Test 2  (HASEN_LADDER N=3): $DH_L3"
echo ""
echo "Expected: 1A and 1B should print identical dH to 14 digits."
echo "          Test 2 should print a finite dH (not nan/inf)."
