#!/bin/bash
#SBATCH --job-name=rwf_tsrc
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_tsrc.%j.out
# ============================================================================
# TSRC-RATIO estimator first light: log[Pf(eta,t0)/Pf(eta,0)] for ALL
# offsets from one periodic quadrature in the continuous offset s
# (insertion O(eta p'') -- parametrically small vs the eta-insertion).
# Arm G4: 4^4 gate -- TSRC-DERIVCHECK (operator FD in s, machine
#   precision) + M=8 nodes x N=32: closure sum rule (must be 0 within
#   error) + node sigmas.
# Arm P16: production 16^3x48, eta=0.6, M=12 nodes x N=8: the node-sigma
#   measurement that prices TSRC coverage (expectation: sigma ~ 100x
#   below the eta-insertion's ~500-760 if the p'' scaling holds).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/rwf_tsrc.DONE
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env CUDA_VISIBLE_DEVICES=0 LATT=4.4.4.4 TRAJ=300 LAMBDA_DTXQCD=5 \
      CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5 \
      ETA=0.6 N_NOISE=32 NOISE_SEED=99 \
      PROFILE=cos PROF_T0=0 DTXQCD_DEFORM_MG=0 \
      DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
      DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_TSRC_RATIO=8 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 4.4.4.4 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > $LOGD/rwf_tsrc_G4.log 2>&1 &
# P16 at eta=0.4 (67 s solves): the sigma measurement transfers; the
# eta=0.6 version runs once MG-lite certifies single-GPU MG for the
# 26k-iter class.
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env CUDA_VISIBLE_DEVICES=1 LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 \
      CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
      ETA=0.4 N_NOISE=8 NOISE_SEED=99 \
      PROFILE=cos PROF_T0=0 DTXQCD_DEFORM_MG=0 \
      DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
      DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_TSRC_RATIO=12 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > $LOGD/rwf_tsrc_P16.log 2>&1 &
wait
for A in G4 P16; do
  echo "== arm $A =="
  grep -aE "TSRC-DERIVCHECK|TSRCCLOSURE" $LOGD/rwf_tsrc_$A.log
done
touch $LOGD/rwf_tsrc.DONE
