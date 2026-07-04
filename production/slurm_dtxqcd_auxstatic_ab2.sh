#!/bin/bash
#SBATCH --job-name=dtx_auxstatic_ab2
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:30:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/slurm_auxstatic_ab2.%j.out
# ============================================================================
# Aux-static A/B, round 2 (job 1297234's B leg was walltime-doomed at 3h):
#   B2: DTXQCD_AUX_STATIC=0 full trajectory (the repack reference), AND
#   A2: DTXQCD_AUX_STATIC=1 rerun — the A-vs-A determinism CONTROL.
# Round-1 partials showed 5/21 solves wobbling +-1 CG iteration between
# A and B — the signature of QUDA tunecache/reduction-order run-to-run
# nondeterminism, not necessarily the bracket.  A2-vs-A calibrates the
# baseline: if A2 wobbles vs A the same way, the acceptance criterion is
# dH agreement at the Style-C precedent level (~7 decimals), not bit-match.
# Fresh suffixes (_a2/_b2) so nothing collides with job 1297234's dirs.
# ============================================================================
cd /lustre2/nplqcd/Grid-DTXQCD/production
LOGD=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs
GENV="LATT=16.16.16.48 LAMBDA_DTXQCD=4.0 USE_FULL_PF=1 TRAJ=121 NO_METROP=0"

srun --mpi=pmix -n 1 --cpu-bind=none \
  env $GENV DTXQCD_SUFFIX=_auxstatic_b2 DTXQCD_AUX_STATIC=0 \
  ./gen_dtxqcd_cfgs --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/hmc_auxstatic_b2.log 2>&1

srun --mpi=pmix -n 1 --cpu-bind=none \
  env $GENV DTXQCD_SUFFIX=_auxstatic_a2 DTXQCD_AUX_STATIC=1 \
  ./gen_dtxqcd_cfgs --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/hmc_auxstatic_a2.log 2>&1

touch $LOGD/hmc_validate2.DONE
echo "auxstatic A/B round 2 complete $(date)"
