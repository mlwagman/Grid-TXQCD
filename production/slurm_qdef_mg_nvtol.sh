#!/bin/bash
#SBATCH --job-name=qdef_nvtol
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_nvtol.%j.out
# Vector-QUALITY axis of the window-volume law: cos eta=0.7 MG stalls at
# nb16/32/48 with tol-1e-3 inverse-iteration vectors (dimension excluded
# as the fix).  Does NV_TOL=1e-4 / 1e-5 make a dent?  nb16 @ 4.4.4.2 =
# the original stall configuration, quality the only moving part.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
R=$P/mtilde_runs
env PROFILE=cos ETA=0.7 GPUS=0,1 NB=16 NVTOL=1e-4 \
  bash $R/run_mg_smooth.sh > $R/mg_cos07_nvtol4.log 2>&1 &
env PROFILE=cos ETA=0.7 GPUS=2,3 NB=16 NVTOL=1e-5 \
  bash $R/run_mg_smooth.sh > $R/mg_cos07_nvtol5.log 2>&1 &
wait
for T in 4 5; do
  echo "== NV_TOL=1e-$T =="
  grep -aE "two-level PGCR|PASSED|FAILED" $R/mg_cos07_nvtol$T.log | tail -2
done
touch $P/gridmg_setups/logs/qdef_nvtol.DONE
