#!/bin/bash
#SBATCH --job-name=qdef_coscen
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_coscen_%j.out
# ============================================================================
# Cosine-profile MODE-COUNT census (lambda=5, traj 300): near-zero mode
# count vs eta traces the THRESHOLD OPENING N ~ (mu_max^2 - M*^2)^{3/2} x
# window volume (mu_max = eta*pi/2) — the Fermi-surface picture behind
# the MG window-volume law and the solver wall at cos eta >= 0.7.  NEV=48
# to resolve the shell, ord=1001 uniformly (identical instrument across
# the ladder; choose the counting cut in analysis from the full eval
# lists).  ETAS_GROUP env selects the 4-point group; one eta per GPU.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
GROUP=${ETAS_GROUP:-A}
case $GROUP in
  A) ETAS_LIST="0.55 0.6 0.65 0.7" ;;
  B) ETAS_LIST="0.35 0.4 0.45 0.5" ;;
  C) ETAS_LIST="0 0.1 0.2 0.3" ;;
esac
GPU=0
for E in $ETAS_LIST; do
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        ETAS=$E NEV=48 IRL_LO_FRAC=3e-5 IRL_ORD=1001 \
        PROFILE=cos \
    $P/probe_deform_spectrum 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/coscen_e$E.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for E in $ETAS_LIST; do
  echo "== eta=$E =="
  grep -aE "spectrum. eta.*(converged|ev\[0\]|ev\[47\])" $LOGD/coscen_e$E.log | tail -3
done
touch $LOGD/qdef_coscen_$GROUP.DONE
