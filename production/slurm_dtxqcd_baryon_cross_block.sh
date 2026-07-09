#!/bin/bash
#SBATCH --job-name=dtx_baryon_xblk
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/%x.%j.out

# Cross-block baryon C-convention test (Test_dtxqcd_baryon_cross_block):
# does the C-even uu+ul+lu+ll averaging need an extra C in the cross terms?
# Arm 1: U=I + drawn aux (lambda=3)  -> per-config identities to 1e-7
# Arm 2: WEAK=1 weak gauge + aux     -> same identities (structural, must hold)
# Also verifies the block-grading theorem (S_cross odd in (d,n) => 3-quark
# cross correlators ensemble-average to ZERO).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
cd /lustre2/nplqcd/Grid-DTXQCD/production
export OMP_NUM_THREADS=16

echo "======== ARM 1: U=I + aux (free-field DTXQCD) ========"
srun --mpi=pmix -n 1 --gres=gpu:1 --cpu-bind=none \
  env LATT=4.4.4.4 LAMBDA=3 MASS=0.5 CSW=0 WEAK=0 SEED=1 \
  ./Test_dtxqcd_baryon_cross_block --grid 4.4.4.4 --mpi 1.1.1.1 --shm 512 --shm-mpi 1
rc1=$?
echo "======== ARM 1 rc=$rc1 ========"

echo "======== ARM 2: WEAK gauge + aux ========"
srun --mpi=pmix -n 1 --gres=gpu:1 --cpu-bind=none \
  env LATT=4.4.4.4 LAMBDA=3 MASS=0.5 CSW=0 WEAK=1 SEED=2 \
  ./Test_dtxqcd_baryon_cross_block --grid 4.4.4.4 --mpi 1.1.1.1 --shm 512 --shm-mpi 1
rc2=$?
echo "======== ARM 2 rc=$rc2 ========"
echo "=== cross-block test done rc1=$rc1 rc2=$rc2 $(date) ==="
