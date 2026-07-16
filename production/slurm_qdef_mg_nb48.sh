#!/bin/bash
#SBATCH --job-name=qdef_nb48
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_nb48.%j.out
# nb48 retry with PGCR history halved (MMAX=8): the first attempt OOM'd a
# 75 MB alloc on the fine GPU right after the (successful, 48.7 s) 21.7 GB
# coarse LU — 96 basis fields + QUDA stack + 16-vector PGCR history is
# just over the line.  nb48@4.4.4.4 = 36,864 coarse DOF = the first TRUE
# dimension increase over the stalled nb16/nb32 (both 24,576): the actual
# dial-or-wall verdict for the window-volume law at cos eta=0.7.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
env PROFILE=cos ETA=0.7 GPUS=0,1 NB=48 BLOCK=4.4.4.4 MMAX=8 \
  bash $P/mtilde_runs/run_mg_smooth.sh > $P/mtilde_runs/mg_cos07_nb48b.log 2>&1
grep -aE "two-level PGCR|PASSED|FAILED" $P/mtilde_runs/mg_cos07_nb48b.log | tail -2
touch $P/gridmg_setups/logs/qdef_nb48.DONE
