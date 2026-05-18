#!/bin/bash
# Reusable chain launcher for the active 8-λ Nf=2+1 TXQCD sweep on
# MN2 / MDSTEPS=30 / TRAJL=√2/4 / AUX_MULT=4 (cfg dirs suffixed
# _nf2p1_mds30_aux4).  Splits 8 λ values across 2 nodes (4 streams/node).
#
# Uses an ABSOLUTE final trajectory target (N_TRAJ).  Each chained job
# advances every stream toward that target until it's reached.  Earlier
# revisions used LATEST+STEP computed at submit time, which broke chains
# longer than 1 (children fired with N_TRAJ already met by the parent and
# exited as no-ops in ~2 minutes).
#
# Usage:
#   bash continue_lambda_sweep_2node_mds30_aux4_nf2p1.sh                       # default N_TRAJ=200
#   N_TRAJ=500 bash continue_lambda_sweep_2node_mds30_aux4_nf2p1.sh            # override target
#   bash continue_lambda_sweep_2node_mds30_aux4_nf2p1.sh <jobA_id> <jobB_id>   # chain after running parents
#
# If parent job IDs are passed, the new submissions wait via afterany; otherwise
# they queue immediately.

set -e
cd "$(dirname "$0")"

N_TRAJ=${N_TRAJ:-200}
SUFFIX=_nf2p1_mds30_aux4

# λ=6 and λ=6.5 dropped from this sweep — those streams stuck in BASIN_HI
# under weak-field MDS=30 start.  Recovery path: use slurm_nodeE_mixed.sh
# which forks λ=6/6.5 from the thermalized λ=7 BASIN_LO cfg at MDS=10
# (`_from_lam7_mds10` suffix).  Future lambda-sweep continues should only
# cover λ values that thermalize cleanly under weak-field start.
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
# NOTE 2026-05-09: brief MDSTEPS=20 experiment (job 1276457/8) reverted to
# MDSTEPS=30 — at the bistable window (λ=6, 6.5) the non-equilibrium gauge
# field plus larger eps gave 0% acceptance.  Two valid cfg.110 cfgs at λ=4, 5
# were written under MDS=20 and kept (mixed-MDS chain is fine).

DEP_A=""; DEP_B=""
[ -n "$1" ] && DEP_A="--dependency=afterany:$1"
[ -n "$2" ] && DEP_B="--dependency=afterany:$2"

echo "=== Node A (λ=${NODE_A_LAMBDAS}, N_TRAJ=${N_TRAJ} ${DEP_A}) ==="
sbatch $DEP_A --export=ALL,${COMMON_ENV},N_TRAJ=${N_TRAJ},LAMBDAS="${NODE_A_LAMBDAS}" \
       slurm_gen_txqcd_4stream_2plus1.sh

echo "=== Node B (λ=${NODE_B_LAMBDAS}, N_TRAJ=${N_TRAJ} ${DEP_B}) ==="
sbatch $DEP_B --export=ALL,${COMMON_ENV},N_TRAJ=${N_TRAJ},LAMBDAS="${NODE_B_LAMBDAS}" \
       slurm_gen_txqcd_4stream_2plus1.sh

echo
squeue -u "$USER" 2>&1 | head -10
