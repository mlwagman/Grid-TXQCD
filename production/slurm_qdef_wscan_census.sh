#!/bin/bash
#SBATCH --job-name=qdef_wcensus
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_wcensus.%j.out
# ============================================================================
# Rounded-tent WIDTH SCAN census (lambda=5, eta=0.8): sigma_min vs apex
# rounding half-width w in {0(sharp),2,4,8}.  The tent apex is a mu-domain
# wall binding the t-localized near-zero mode that gates every eta=0.8
# solver; rounding spreads the sign flip into a linear crossing and should
# LIFT the bound state (Airy-ish).  Grading sacrifice: p_max - w/2, at the
# source antipode only.  Same instrument/settings as the edge census
# (ord=1001 lesson: the near-edge band needs the heavy filter).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"

for W in 0 2 4 8; do
  if [ "$W" = "0" ]; then PROF=tent; else PROF=tent_rounded; fi
  srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=16.16.16.48 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        ETAS=0.8 NEV=24 IRL_LO_FRAC=3e-5 IRL_ORD=1001 \
        PROFILE=$PROF PROFILE_W=$W \
    $P/probe_deform_spectrum 300 $GRID > $LOGD/wcensus_w$W.log 2>&1
  echo "== w=$W =="
  grep -aE "spectrum. eta.*(converged|ev\[0\]|ev\[1\]|ev\[2\])" $LOGD/wcensus_w$W.log | tail -4
done
touch $LOGD/qdef_wcensus.DONE
