#!/bin/bash
#SBATCH --job-name=dtx_meas_ab
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/slurm-logs/dtx_meas_ab.%j.out
# EO-vs-full measurement A/B at $LAM (see _meas_eo_ab_lam.sh).  24h limit:
# full-volume solves near the sign wall (lam=4: ~13 min/solve, lam=3 worse)
# need most of it; the EO side is expected to finish far earlier.
LAM=$LAM GPU_A=0 GPU_B=1 bash /lustre2/nplqcd/Grid-DTXQCD/_meas_eo_ab_lam.sh
echo "=== meas A/B lam=$LAM exited rc=$? $(date) ==="
