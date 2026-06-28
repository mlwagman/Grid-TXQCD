#!/bin/bash
#SBATCH --job-name=dtx_nEO_l0p5f20
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/dtx_nEO_l0p5f20.%j.out
# lambda=0.5 NON-EO weak-field, FULL trajL (sqrt2/4) at MDS=20 -> eps=0.0177.
# eps-MATCHED replacement for the failed fullL MDS=10 (eps=0.0354) which blew up:
# first traj dH=11.4 (lambda=0.5 is too stiff for the doubled eps; dH~eps^4 took
# the ~0.5 halfL dH to ~11).  eps=0.0177 is the proven step at lambda=0.5 (the
# halfL MDS=10 control runs there fine, dH~0.5), so MDS=20 keeps that step size
# while doubling the trajectory -> cost-neutral per MD-time, isolates the trajL
# decorrelation benefit (same scheme as the lambda=1/2 MDS=40 tests).  RHMC_DEG=8
# as on the halfL control.  halfL MDS=10 (1294253) stays as the control until this
# is confirmed clean; compare tau_int(aux VEVs) in MD-time once equilibrated.
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader
LAMBDA_DTXQCD=0.5 USE_FULL_PF=1 MDSTEPS=20 TRAJL=0.353553390593274 RHMC_DEG=8 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 DTXQCD_SUFFIX="_nonEO_mds20_fullL" \
TRAJ=1500 NO_METROP=5 N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== lam0p5 nonEO fullL MDS=20 exited rc=$? $(date) ==="
