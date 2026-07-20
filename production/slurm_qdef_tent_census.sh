#!/bin/bash
#SBATCH --job-name=tent_census
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/tent_census.%j.out
# ============================================================================
# TENT-profile eigenvalue census at the solver-ladder points (lambda=5,
# traj 300, NEV=24, uniform ord=1001 instrument -- same bounds treatment
# as the cos census: values far below the window edge are upper bounds).
# Expected structure per eta: the isolated wall PAIR (apex/upper +
# trough/lower) + possibly secondary bound levels as the walls deepen +
# the subcritical bulk edge.  The mode COUNT vs eta prices the deflation
# nev for production (ladder job 1302963) and completes the tent-vs-cos
# spectral comparison for the note.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
ETAS_LIST=${ETAS_LIST:-"0.4 0.6 0.7 0.8"}
TAG=$(echo "$ETAS_LIST" | tr -d ' .')
rm -f $LOGD/tent_census_$TAG.DONE
GPU=0
for E in $ETAS_LIST; do
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        ETAS=$E NEV=24 IRL_LO_FRAC=3e-5 IRL_ORD=1001 \
        PROFILE=tent \
    $P/probe_deform_spectrum 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/tentcen_e$E.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for E in $ETAS_LIST; do
  echo "== eta=$E =="
  grep -aE "spectrum. eta.*(converged|ev\[0\]|ev\[1\]|ev\[2\]|ev\[23\])" $LOGD/tentcen_e$E.log | tail -5
done
touch $LOGD/tent_census_$TAG.DONE
