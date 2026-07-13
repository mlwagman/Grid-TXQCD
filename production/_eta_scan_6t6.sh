#!/bin/bash
#SBATCH --job-name=dtx_eta_6t6
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/dtx_eta_6t6.%j.out
# TENT-profile eta scan on the 6^4 lambda=4 ensemble (dense, dim=62208,
# ~62 GB matrix -- fits one A100; ~30 min/config).  T-scaling test: compare
# per-hop response vs the 4^4 (hotter) result.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
free -g
for TRAJ in 30 50 70 90 100; do
  srun --mpi=pmix -N1 -n1 --cpu-bind=none \
    env LATT=6.6.6.6 TRAJ=$TRAJ PROFILE=tent \
        CFG_DIR=$P/cfgs/dtxqcd_lam4.0000_6t6_therm \
        ETA_SCAN="0.1,0.2,0.3,0.5,0.7,1.0,1.5" OMP_NUM_THREADS=16 \
    $P/Test_dtxqcd_eta_deform_pf --grid 6.6.6.6 --mpi 1.1.1.1
done
echo "=== 6t6 tent scan done $(date) ==="
