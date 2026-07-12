#!/bin/bash
#SBATCH --job-name=dtx_eta_pf
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/dtx_eta_pf.%j.out
# eta contour-deformation Pf-invariance test on two 4^4 configs (lambda=4
# signhop traj 250 + lambda=3 signhop traj 250): does the time-graded
# UR/LL rescaling preserve det(M48)?
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
srun --mpi=pmix -N1 -n1 --cpu-bind=none \
  env LATT=4.4.4.4 TRAJ=250 \
      CFG_DIR=$P/cfgs/dtxqcd_lam4.0000_4t4_signhop_lam4 OMP_NUM_THREADS=16 \
  $P/Test_dtxqcd_eta_deform_pf --grid 4.4.4.4 --mpi 1.1.1.1
echo "--- lambda=3 config ---"
srun --mpi=pmix -N1 -n1 --cpu-bind=none \
  env LATT=4.4.4.4 TRAJ=250 \
      CFG_DIR=$P/cfgs/dtxqcd_lam3.0000_4t4_signhop_lam3 OMP_NUM_THREADS=16 \
  $P/Test_dtxqcd_eta_deform_pf --grid 4.4.4.4 --mpi 1.1.1.1
echo "=== eta deform test done $(date) ==="
