#!/bin/bash
#SBATCH --job-name=build_test
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=slurm-logs/build_test.%j.out

cd "$SLURM_SUBMIT_DIR"
source ../env_lq2_grid.sh
export PATH=/srv/software/el8/x86_64/hpc/nvhpc/Linux_x86_64/23.7/cuda/12.2/bin:$PATH

echo "=== environment ==="
which nvcc mpicxx
echo ""
echo "=== building gen_qcd_cfgs ==="
rm -f gen_qcd_cfgs
make gen_qcd_cfgs 2>&1
echo ""
echo "=== build result ==="
ls -la gen_qcd_cfgs 2>&1
