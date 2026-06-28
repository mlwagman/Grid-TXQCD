#!/bin/bash
#SBATCH --job-name=dtx_nEO_l1f40
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/dtx_nEO_l1f40.%j.out
# lambda=1 NON-EO weak-field, FULL trajL (sqrt2/4) at MDS=40 -> eps=0.0088.
# eps is IDENTICAL to the running halfL MDS=20 stream (sqrt2/8 / 20 = 0.0088), so
# stability is guaranteed (same proven step size) and the cost-per-MD-time is the
# same (40 steps cover 2x the MD time of halfL MDS=20's 20 steps).  This isolates
# trajL: does one long trajectory decorrelate the slow aux modes (omega_aux~lambda)
# better than two short ones at matched compute?  Kept eps fixed deliberately --
# history check 2026-06-28 found NO clean fullL dH at lambda=1/2 (the fullL MDS=20
# runs 1289214/1289215 were cancelled at traj 0; the fullL MDS=10 chroma-fork blew
# up on the fork transient, dH 69->13), so eps=0.0088 is the only validated step
# at the stiffness peak -- don't gamble on coarser.  halfL MDS=20 (1294254) stays
# as the control.  Compare tau_int(aux VEVs) in MD-time once both equilibrate.
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader
LAMBDA_DTXQCD=1.0 USE_FULL_PF=1 MDSTEPS=40 TRAJL=0.353553390593274 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 DTXQCD_SUFFIX="_nonEO_fullL_mds40" \
TRAJ=1500 NO_METROP=5 N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== lam1 nonEO fullL MDS=40 exited rc=$? $(date) ==="
