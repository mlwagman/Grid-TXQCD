#!/bin/bash
#SBATCH --job-name=dtx_signw_tep
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/slurm-logs/dtx_signw_tep.%j.out
# |Pf|-weighted <sign> over aux prior draws, TEPID gauge, lambda grid extended
# to pin the clean-window boundary (first pass showed clean at 6, mixed at 3).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh

make -C /lustre2/nplqcd/Grid-DTXQCD/production \
     GRID_BUILD_DIR=/lustre2/nplqcd/Grid-DTXQCD/build-gpu -j8 \
     Test_dtxqcd_sign_aux_draws
echo "=== build rc=$? $(date) ==="

NDRAWS=12 LAMBDAS=6,5,4,3,2,1,0.5,0.25,0.1 GAUGE_KIND=tepid \
  /lustre2/nplqcd/Grid-DTXQCD/production/Test_dtxqcd_sign_aux_draws \
  --grid 4.4.4.4 --mpi 1.1.1.1 --device-mem 8000 \
  > /lustre2/nplqcd/Grid-DTXQCD/_sign_weighted_tepid.log 2>&1
echo "=== tepid weighted rc=$? $(date) ==="
