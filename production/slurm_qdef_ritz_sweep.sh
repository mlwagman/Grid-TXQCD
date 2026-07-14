#!/bin/bash
#SBATCH --job-name=qdef_ritz
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_ritz.%j.out
# ============================================================================
# lambda x eta spectrum bounds via the RHMC-autoscale plain Lanczos
# (SPECTRUM_MODE=ritz, DtxqcdLanczosMinMax, NM=48): lambda_min/max Ritz
# estimates at ~1 s per (lambda, eta) point -> a FINE eta grid (delta 0.1).
# lambda=5/6 = the target (mild spectra, where the Chebyshev-IRL windows
# failed by sitting below the whole band); lambda=4 included to cross-
# calibrate Ritz vs the converged IRL census (at eta>=0.6 the Ritz
# lambda_min is an UPPER bound — underconverged on deep clusters).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
ETAS=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8

for lam in 5 6 4; do
  srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=16.16.16.48 CFG_DIR=$P/cfgs/dtxqcd_lam${lam}.0000_tla_fromchroma_cw \
        ETAS=$ETAS SPECTRUM_MODE=ritz NM=48 \
    $P/probe_deform_spectrum 300 $GRID > $LOGD/ritz_lam${lam}.log 2>&1
  echo "=== lambda=$lam ==="
  grep -a "\[ritz\]" $LOGD/ritz_lam${lam}.log
done
touch $LOGD/qdef_ritz.DONE
