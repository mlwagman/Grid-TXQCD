#!/bin/bash
#SBATCH --job-name=build_eig_nf2
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=slurm-logs/build_eig_nf2.%j.out

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs
source ../env_lq2_grid.sh
echo "=== build eigspec_diag_nf2 ==="
date
make eigspec_diag_nf2
ls -la eigspec_diag_nf2
date
