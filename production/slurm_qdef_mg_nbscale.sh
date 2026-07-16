#!/bin/bash
#SBATCH --job-name=qdef_nbscale
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_nbscale.%j.out
# ============================================================================
# nbasis-scaling test of the smooth-profile M~-MG window-volume law:
# cosine eta=0.7 (~13 supercritical slices) STALLED at nb16 (1.4%/1000
# steps); does nb32/nb48 @ BLOCK=4.4.4.4 crack it?  If yes the law is a
# DIAL (nbasis ~ window size); if no it is a wall and the frontier stays
# at cos eta=0.6 + MG (5,187 fine Ops, differential 14.4).  Dense coarse
# LU: nb32 -> N=24,576 (9.7 GB), nb48 -> N=36,864 (21.7 GB), each on the
# arm's second GPU.  The last open solver cell before the reweighting
# handoff.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
R=$P/mtilde_runs

env PROFILE=cos ETA=0.7 GPUS=0,1 NB=32 BLOCK=4.4.4.4 \
  bash $R/run_mg_smooth.sh > $R/mg_cos07_nb32.log 2>&1 &
env PROFILE=cos ETA=0.7 GPUS=2,3 NB=48 BLOCK=4.4.4.4 \
  bash $R/run_mg_smooth.sh > $R/mg_cos07_nb48.log 2>&1 &
wait
for NB in 32 48; do
  echo "== nb$NB =="
  grep -aE "two-level PGCR|PASSED|FAILED" $R/mg_cos07_nb$NB.log | tail -2
done
touch $P/gridmg_setups/logs/qdef_nbscale.DONE
