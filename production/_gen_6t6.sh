#!/bin/bash
#SBATCH --job-name=dtx_gen_6t6
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/dtx_gen_6t6.%j.out
# 6^4 lambda=4 thermalized chain for the eta-deformation T-scaling test
# (6^4 scales ALL extents 4->6: the dense-tractable rung toward the confined
# regime).  Mirrors the 4^4 signhop generation settings minus the observer.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
cd /lustre2/nplqcd/Grid-DTXQCD/production
export OMP_NUM_THREADS=16
srun --mpi=pmix -n 1 --cpu-bind=none \
  env LATT=6.6.6.6 LAMBDA_DTXQCD=4 USE_FULL_PF=1 \
      MDSTEPS=10 TRAJL=0.353553390593274 RHMC_DEG=8 \
      STOUT_NSMEAR=1 STOUT_RHO=0.125 AUX_INIT_AUTO=1 RAT_AUTO_HI=1 \
      TRAJ=105 NO_METROP=5 N_SKIP=10 \
      DTXQCD_SUFFIX="_6t6_therm" \
  ./gen_dtxqcd_cfgs --grid 6.6.6.6 --mpi 1.1.1.1 --shm 512 --shm-mpi 1
echo "=== 6t6 gen done rc=$? $(date) ==="
