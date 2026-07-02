#!/bin/bash
#SBATCH --job-name=dtx_sign_eo
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/slurm-logs/dtx_sign_eo.%j.out
# EO factorization of the Pfaffian sign over aux prior draws:
#   Pf(K M48) = prod_{x even} Pf(K M_ee(x)) * Pf(Schur_odd)
# Per draw the probe now also computes the exact site-local sign product
# (48x48 per even site) and the implied Schur sign.  If the small-lambda
# flips live in the site-local factor, production-volume sign tracking is
# exact and free (piggybacks on the EO LogDet per-site machinery).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh

make -C /lustre2/nplqcd/Grid-DTXQCD/production \
     GRID_BUILD_DIR=/lustre2/nplqcd/Grid-DTXQCD/build-gpu -j8 \
     Test_dtxqcd_sign_aux_draws
echo "=== build rc=$? $(date) ==="

srun --mpi=pmix -n 1 --cpu-bind=none \
  env NDRAWS=12 LAMBDAS=6,4,3,2,1,0.5,0.25,0.1 GAUGE_KIND=tepid \
  /lustre2/nplqcd/Grid-DTXQCD/production/Test_dtxqcd_sign_aux_draws \
  --grid 4.4.4.4 --mpi 1.1.1.1 --shm-mpi 1 --device-mem 8000 \
  > /lustre2/nplqcd/Grid-DTXQCD/_sign_eo_split_tepid.log 2>&1
echo "=== eo split scan rc=$? $(date) ==="
