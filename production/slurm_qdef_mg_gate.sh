#!/bin/bash
#SBATCH --job-name=qdef_mggate
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_mggate.%j.out
# Production MG gate with the RHS discriminator: setup -> gaussian
# SELFTEST (the estimator's Hutchinson-noise RHS class = the standalone-
# validated case) -> wall-source solve -> h5 A/B vs the banked plain-CG
# reference (meas_mgab_0).  Wall solves measured a relres-0.64 stall on
# the same hierarchy; if the selftest converges (~5k Ops class) the MG is
# certified for the REWEIGHTING estimator and wall propagators simply
# keep plain CG (already fast at eta<=0.6).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
mkdir -p $P/meas_mgab_1
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
      PROFILE=cos ETA=0.6 PROF_T0=0 \
      CG_TOL=1e-7 CG_MAX=200000 DTXQCD_DEFORM_QUDA_CHECK=0 \
      MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=1 \
      DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_EO_DEFLATE=0 \
      DTXQCD_DEFORM_MG=1 DTXQCD_MG_SELFTEST=1 \
      DTXQCD_MG_NVTOL=${DTXQCD_MG_NVTOL:-1e-3} \
      DTXQCD_MG_NVMAX=${DTXQCD_MG_NVMAX:-50000} \
      DATA_DIR=$P/meas_mgab_1 \
  $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > $P/mtilde_runs/mgab_1c.$SLURM_JOB_ID.log 2>&1
echo "== selftest + wall =="
grep -aE "SELFTEST|deform-mg. solve|deform-CG.*iters=|NOT CONVERGED" \
  $P/mtilde_runs/mgab_1c.$SLURM_JOB_ID.log | tail -4
if [ -f $P/meas_mgab_1/baryon_deform_dtxqcd_300.h5 ]; then
  echo "== h5 A/B =="
  python3 $P/_qdef_cmp2.py $P/meas_mgab_0/baryon_deform_dtxqcd_300.h5 \
    $P/meas_mgab_1/baryon_deform_dtxqcd_300.h5 1e-4 | tail -1
fi
touch $P/gridmg_setups/logs/qdef_mggate.DONE
