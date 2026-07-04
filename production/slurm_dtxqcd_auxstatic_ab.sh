#!/bin/bash
#SBATCH --job-name=dtx_auxstatic_ab
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/slurm_auxstatic_ab.%j.out
# ============================================================================
# Aux-static HMC bracket A/B (SetAuxStatic, 43c4d0ad + RunMultiShift bracket).
# Same-seed 1-traj non-EO gen at lambda=4 16^3x48 from the traj-120 chromafork
# checkpoint (copied into cfgs/dtxqcd_lam4.0000_auxstatic_{a,b}):
#   A: DTXQCD_AUX_STATIC=1 (default, pack aux once per solve)
#   B: DTXQCD_AUX_STATIC=0 (per-apply repack, the old behavior)
# dH must agree to fp precision (the pack is idempotent within a solve);
# the wallclock delta measures the saving.  Gate test already PASSED
# (hmc_gatetest.log).  Runs sequentially on one GPU.
# ============================================================================
cd /lustre2/nplqcd/Grid-DTXQCD/production
LOGD=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs
GENV="LATT=16.16.16.48 LAMBDA_DTXQCD=4.0 USE_FULL_PF=1 TRAJ=121 NO_METROP=0"

srun --mpi=pmix -n 1 --cpu-bind=none \
  env $GENV DTXQCD_SUFFIX=_auxstatic_a DTXQCD_AUX_STATIC=1 \
  ./gen_dtxqcd_cfgs --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/hmc_auxstatic_a.log 2>&1

srun --mpi=pmix -n 1 --cpu-bind=none \
  env $GENV DTXQCD_SUFFIX=_auxstatic_b DTXQCD_AUX_STATIC=0 \
  ./gen_dtxqcd_cfgs --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/hmc_auxstatic_b.log 2>&1

touch $LOGD/hmc_validate.DONE
echo "auxstatic A/B complete $(date)"
