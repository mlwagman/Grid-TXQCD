#!/bin/bash
#SBATCH --job-name=dtx_signw_weak
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/slurm-logs/dtx_signw_weak.%j.out
# |Pf|-weighted <sign> over aux prior draws, WEAK-FIELD gauge (the production
# init roughness), extended lambda grid.  Waits 60 s so the tepid sibling's
# rebuild finishes first (running scans keep their cached inode).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
sleep 60

NDRAWS=12 LAMBDAS=6,5,4,3,2,1,0.5,0.25,0.1 GAUGE_KIND=weak \
  /lustre2/nplqcd/Grid-DTXQCD/production/Test_dtxqcd_sign_aux_draws \
  --grid 4.4.4.4 --mpi 1.1.1.1 --device-mem 8000 \
  > /lustre2/nplqcd/Grid-DTXQCD/_sign_weighted_weak.log 2>&1
echo "=== weak weighted rc=$? $(date) ==="
