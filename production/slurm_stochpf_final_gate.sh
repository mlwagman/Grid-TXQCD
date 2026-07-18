#!/bin/bash
#SBATCH --job-name=stochpf_fin
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/stochpf_fin.%j.out
# ============================================================================
# Final-defaults regression: the estimator with NO overrides (comb-4 +
# Neyman + CG_TOL=1e-4 + schur defaults) at the 4^4 dense truth point
# (ep=0.4 via ETA=0.8 N_GL=1; truth -0.0356616 - 0.110153 i), N_NOISE=64.
# Plus a segment-consistency check: [0,0.4] in one piece vs
# [0,0.2]+[0.2,0.4] (means must agree within errors; N_GL=2 each).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/stochpf_fin.DONE
CFG=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=4.4.4.4 TRAJ=300 LAMBDA_DTXQCD=5 \
      CFG_DIR=$CFG ETA=0.8 N_GL=1 N_NOISE=64 NOISE_SEED=77 \
      PROFILE=cos PROF_T0=0 DTXQCD_DEFORM_MG=0 \
      DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 4.4.4.4 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > $LOGD/stochpf_fin_truth.log 2>&1
echo "== defaults vs dense truth =="
grep -aE "GLNODE|neyman" $LOGD/stochpf_fin_truth.log | tail -2
for SEG in 0.0:0.4 0.0:0.2 0.2:0.4; do
  LO=${SEG%%:*}
  HI=${SEG##*:}
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=4.4.4.4 TRAJ=300 LAMBDA_DTXQCD=5 \
        CFG_DIR=$CFG ETA=$HI ETA_LO=$LO N_GL=2 N_NOISE=48 NOISE_SEED=78 \
        PROFILE=cos PROF_T0=0 DTXQCD_DEFORM_MG=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
    $P/Test_dtxqcd_eta_stoch_pf --grid 4.4.4.4 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/stochpf_fin_seg_${LO}_${HI}.log 2>&1
  echo "== segment [$LO,$HI] =="
  grep -aE "STOCHPF" $LOGD/stochpf_fin_seg_${LO}_${HI}.log | tail -1
done
touch $LOGD/stochpf_fin.DONE
