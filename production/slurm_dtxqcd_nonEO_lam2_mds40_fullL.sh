#!/bin/bash
#SBATCH --job-name=dtx_nEO_l2f40
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/dtx_nEO_l2f40.%j.out
# lambda=2 NON-EO weak-field, FULL trajL (sqrt2/4) at MDS=40 -> eps=0.0088.  This
# is the STIFFNESS PEAK (smallest |lambda|min ~0.56 / M^dagM lambda_min ~0.16), so
# the most demanding.  eps is IDENTICAL to the running halfL MDS=20 stream, so
# stability matches the proven step size at the same cost-per-MD-time (40 steps =
# 2x the MD time of halfL MDS=20).  Isolates trajL at the peak: one long traj vs
# two short ones for decorrelating the aux modes, matched compute.  eps kept fixed
# -- history check found no clean fullL dH at lambda=2 (fullL MDS=20 run 1289214
# was cancelled at traj 0), so 0.0088 is the only validated step here.  halfL
# MDS=20 (1294255) stays as the control; compare tau_int once equilibrated.
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader
LAMBDA_DTXQCD=2.0 USE_FULL_PF=1 MDSTEPS=40 TRAJL=0.353553390593274 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 DTXQCD_SUFFIX="_nonEO_fullL_mds40" \
TRAJ=1500 NO_METROP=5 N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== lam2 nonEO fullL MDS=40 exited rc=$? $(date) ==="
