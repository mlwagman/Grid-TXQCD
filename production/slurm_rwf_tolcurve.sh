#!/bin/bash
#SBATCH --job-name=rwf_tolcurve
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_tolcurve_%j.out
# Truncation-bias tolerance curve, same-seed vs the tol-1e-8 truth
# (dlogdet 3.8558, darg 4.7158): CG@1e-5, CG@1e-6, MG@1e-6 at [0.4,0.6].
# Measured at 1e-4: CG bias +3.18/-0.16, MG +4.18/+1.96.  Expect ~ tol^1
# scaling; pick the loosest tol with bias << 0.3, and keep MG iff its bias
# vanishes alongside.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
run_tol () {  # run_tol <tol> <mg> <tag>
  local MGENV="DTXQCD_DEFORM_MG=0"
  if [ "$2" = 1 ]; then
    MGENV="DTXQCD_DEFORM_MG=1 DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 DTXQCD_MG_REUSE_NV=1"
  fi
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
        ETA=0.6 ETA_LO=0.4 N_GL=2 N_NOISE=32 NOISE_SEED=2200 \
        CG_TOL=$1 \
        PROFILE=tent PROF_T0=0 DTXQCD_STOCHPF_ANTI=0 \
        DTXQCD_STOCHPF_DEFLATE=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 $MGENV \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_tolcurve_$3.log 2>&1
  echo "== tol=$1 MG=$2 =="
  grep -aE "stochpf-CG" $LOGD/rwf_tolcurve_$3.log | head -2
  grep -aE "STOCHPF traj" $LOGD/rwf_tolcurve_$3.log | tail -1
}
run_tol 1e-5 0 cg1e5
run_tol 1e-6 0 cg1e6
run_tol 1e-6 1 mg1e6
echo "== truth reference: dlogdet 3.8557927  darg 4.71577115 =="
