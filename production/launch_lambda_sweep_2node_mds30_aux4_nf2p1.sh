#!/bin/bash
# One-shot launcher for the active 8-λ Nf=2+1 TXQCD sweep on
# MN2 / MDSTEPS=30 / TRAJL=√2/4 / AUX_MULT=4 (cfg dirs suffixed
# _nf2p1_mds30_aux4).  Splits 8 λ values across 2 nodes (4 streams/node).
#
# Uses an ABSOLUTE final trajectory target (N_TRAJ).  Each job advances every
# stream toward that target until it is reached.  Use
# continue_lambda_sweep_2node_mds30_aux4_nf2p1.sh to chain follow-ups via
# --dependency=afterany on the parent job IDs.
#
# Usage:
#   bash launch_lambda_sweep_2node_mds30_aux4_nf2p1.sh             # default N_TRAJ=200
#   N_TRAJ=500 bash launch_lambda_sweep_2node_mds30_aux4_nf2p1.sh  # override target

set -e
cd "$(dirname "$0")"

N_TRAJ=${N_TRAJ:-200}
SUFFIX=_nf2p1_mds30_aux4

# λ=6 and λ=6.5 dropped — see continue_lambda_sweep for explanation.
NODE_A_LAMBDAS="4 5"
NODE_B_LAMBDAS="7 8 10 12"

COMMON_ENV="\
INTEGRATOR=MinimumNorm2,LAMBDA_MN2=0.1789,\
MDSTEPS=30,TRAJL=0.353553390593274,\
GAUGE_MULT=4,GAUGE_INNER_MULT=4,AUX_MULT=4,\
HASEN_DM=0,\
START_TYPE=thermal,WEAK_FIELD_SCALE=0.1,\
NO_METROP=0,\
QUDA_FORCE=1,QUDA_FORCE_KERNEL=1,\
TXQCD_QUDA_HYBRID=1,TXQCD_QUDA_FULL=1,\
TXQCD_PRECOMPUTE_GPU=1,TXQCD_MOOEEINV_CUBLAS=1,TXQCD_MOOEE_CUBLAS=1,\
EIG_DIAG=1,\
SUFFIX=${SUFFIX}"

echo "=== Node A (λ=${NODE_A_LAMBDAS}, N_TRAJ=${N_TRAJ}) ==="
sbatch --export=ALL,${COMMON_ENV},N_TRAJ=${N_TRAJ},LAMBDAS="${NODE_A_LAMBDAS}" \
       slurm_gen_txqcd_4stream_2plus1.sh

echo "=== Node B (λ=${NODE_B_LAMBDAS}, N_TRAJ=${N_TRAJ}) ==="
sbatch --export=ALL,${COMMON_ENV},N_TRAJ=${N_TRAJ},LAMBDAS="${NODE_B_LAMBDAS}" \
       slurm_gen_txqcd_4stream_2plus1.sh

echo
squeue -u "$USER" 2>&1 | head -10
