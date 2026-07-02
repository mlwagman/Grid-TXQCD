#!/bin/bash
#SBATCH --job-name=dtx_sign_scan
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/slurm-logs/dtx_sign_scan.%j.out
# Pfaffian-sign statistics over aux prior draws (plaq-deficit investigation):
# rebuild Test_dtxqcd_sign_aux_draws, then tepid + weak gauge scans at
# lambda = {6,3,2,1,0.5,0.25,0.1}, 12 draws each, dense 4^4 gamma5.M48
# spectrum via cuSOLVER.  Hot-gauge reference scan (already done) saturated
# at 50/50 sign flips for lambda=2; these smoother backgrounds map where the
# aux-integral sign cleans up.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh

make -C /lustre2/nplqcd/Grid-DTXQCD/production \
     GRID_BUILD_DIR=/lustre2/nplqcd/Grid-DTXQCD/build-gpu -j8 \
     Test_dtxqcd_sign_aux_draws
echo "=== build rc=$? $(date) ==="

BIN=/lustre2/nplqcd/Grid-DTXQCD/production/Test_dtxqcd_sign_aux_draws

echo "######## tepid scan $(date) ########"
NDRAWS=12 LAMBDAS=6,3,2,1,0.5,0.25,0.1 GAUGE_KIND=tepid \
  "$BIN" --grid 4.4.4.4 --mpi 1.1.1.1 --device-mem 8000 \
  > /lustre2/nplqcd/Grid-DTXQCD/_sign_aux_draws_tepid.log 2>&1
echo "=== tepid rc=$? $(date) ==="

echo "######## weak scan $(date) ########"
NDRAWS=12 LAMBDAS=6,3,2,1,0.5,0.25,0.1 GAUGE_KIND=weak \
  "$BIN" --grid 4.4.4.4 --mpi 1.1.1.1 --device-mem 8000 \
  > /lustre2/nplqcd/Grid-DTXQCD/_sign_aux_draws_weak.log 2>&1
echo "=== weak rc=$? $(date) ==="
