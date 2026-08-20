#!/bin/bash
#SBATCH --job-name=rwf_truth
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_truth_%j.out
# TRUTH reference for the same-seed solver triangle at [0.4,0.6]:
# plain-CG stack at CG_TOL=1e-8, NOISE_SEED=2200 (identical noise to the
# MG-off tol-1e-4 arm 1310700 and the MG-on arm 1310841).  Then
#   |CG(1e-4) - CG(1e-8)| = sloppy-tolerance bias
#   |MG(1e-4) - CG(1e-8)| = MG + tolerance
# The same-seed A/B measured a 1.0 (dlogdet) / 2.1 (darg) solver-dependent
# shift -- this arm attributes it.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
      ETA=0.6 ETA_LO=0.4 N_GL=2 N_NOISE=32 NOISE_SEED=2200 \
      CG_TOL=1e-8 \
      PROFILE=tent PROF_T0=0 DTXQCD_STOCHPF_ANTI=0 \
      DTXQCD_STOCHPF_DEFLATE=0 \
      DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_MG=0 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > $LOGD/rwf_truth_seg0.4_0.6.log 2>&1
echo "== TRUTH CG 1e-8 same-seed [0.4,0.6] =="
grep -aE "stochpf-CG" $LOGD/rwf_truth_seg0.4_0.6.log | head -2
grep -aE "STOCHPF traj" $LOGD/rwf_truth_seg0.4_0.6.log | tail -1
