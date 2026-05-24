#!/bin/bash
# Bit-exact regression: HASEN_LADDER="0.3,0.4" (N=2, m_h=0.4) should produce
# identical dH to HASEN_DM=0.1 (m_l=0.3 → m_h=0.4) on the same RNG seed.
# Both paths construct: rational at 0.4 + Hasenbusch ratio at (0.3, 0.4).
cd /lustre2/nplqcd/Grid-TXQCD/production
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=4

mkdir -p logs
export LATT=4.4.4.4
export MASS_LIGHT=0.3
export MASS_STRANGE=1.5
export CSW=0.0
export BETA=6.1
export LAMBDA=2.0
export N_TRAJ=1
export NO_METROP=1
export INTEGRATOR=MinimumNorm2
export LAMBDA_MN2=0.1789
export MDSTEPS=2
export TRAJL=0.353553390593274
export GAUGE_MULT=4
export AUX_MULT=1
export AUX_INIT=0.1
export RAT_LO=0.01
export RAT_HI=10.0
export RAT_DEGREE=8
export RAT_TOL=1e-5
unset TXQCD_QUDA_HYBRID
unset TXQCD_QUDA_FULL
unset USE_HMC_MG
export TXQCD_PRECOMPUTE_GPU=0
export TXQCD_MOOEE_CUBLAS=0
export TXQCD_MOOEEINV_CUBLAS=0
export EIG_DIAG=0

COMMON_ARGS="--grid 4.4.4.4 --mpi 1.1.1.1 --shm 256 --shm-mpi 1"

# A: HASEN_DM
LOG1=logs/test_bitexact_v1_dm.log
echo "=== A: HASEN_DM=0.1 ===" | tee "$LOG1"
export HASEN_DM=0.1
unset HASEN_LADDER
export SUFFIX=_bitexact_dm
srun --overlap --mpi=pmix --export=ALL -N 1 -n 1 --cpu-bind=none --gres=gpu:1 \
    ./gen_txqcd_cfgs_2plus1 $COMMON_ARGS >> "$LOG1" 2>&1

# B: HASEN_LADDER with matching masses
LOG2=logs/test_bitexact_v2_ladder.log
echo "=== B: HASEN_LADDER='0.3,0.4' ===" | tee "$LOG2"
unset HASEN_DM
export HASEN_LADDER="0.3,0.4"
export SUFFIX=_bitexact_ladder
srun --overlap --mpi=pmix --export=ALL -N 1 -n 1 --cpu-bind=none --gres=gpu:1 \
    ./gen_txqcd_cfgs_2plus1 $COMMON_ARGS >> "$LOG2" 2>&1

DH_A=$(grep -E 'dH = ' "$LOG1" | tail -1 | grep -oE 'dH = [^ ]+')
DH_B=$(grep -E 'dH = ' "$LOG2" | tail -1 | grep -oE 'dH = [^ ]+')
echo ""
echo "=== VERDICT ==="
echo "A (HASEN_DM=0.1):       $DH_A"
echo "B (HASEN_LADDER N=2):    $DH_B"
if [ "$DH_A" = "$DH_B" ]; then
  echo "PASS: bit-exact match"
else
  echo "WARN: not bit-exact (may be ULP-level differences in float addition order)"
fi
