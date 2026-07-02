#!/bin/bash
#SBATCH --job-name=txq_sign_dwf
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/slurm-logs/txq_sign_dwf.%j.out
# TXQCD Wilson vs naive-coupled Mobius DWF determinant-sign statistics over
# aux prior draws (tepid gauge, 4^4, Ls=4).  The go/no-go input for whether
# a DTXQCD-DWF port needs chiral aux couplings or whether even the naive
# broadcast coupling suppresses the small-lambda sign flips.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh

make -C /lustre2/nplqcd/Grid-DTXQCD/production \
     GRID_BUILD_DIR=/lustre2/nplqcd/Grid-DTXQCD/build-gpu -j8 \
     Test_txqcd_sign_aux_draws
echo "=== build rc=$? $(date) ==="

NDRAWS=8 LAMBDAS=3,1,0.5,0.1 LS=4 GAUGE_KIND=tepid \
  /lustre2/nplqcd/Grid-DTXQCD/production/Test_txqcd_sign_aux_draws \
  --grid 4.4.4.4 --mpi 1.1.1.1 --device-mem 12000 \
  > /lustre2/nplqcd/Grid-DTXQCD/_txqcd_sign_dwf_tepid.log 2>&1
echo "=== txqcd sign dwf rc=$? $(date) ==="
