#!/bin/bash
#SBATCH --job-name=rwf_fmapL
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_fmapL.%j.out
# ============================================================================
# F(t,t') map LADDER at the PRODUCTION node strengths (16^3x48 lam5
# traj300, cos t0=0).  The reweighting targets eta=0.2/0.4/0.6 with
# N_GL=2 place their quadrature nodes at ep = eta*{0.211, 0.789}:
#   eta=0.2 -> {0.042, 0.158};  0.4 -> {0.085, 0.316};  0.6 -> {0.127, 0.473}
# Ladder ep = 0.05/0.16/0.32/0.47 covers all six to ~5%.  One ep per GPU
# (ETA=2*ep, N_GL=1 places the single node exactly at ep); probe pairs
# scaled to solve cost.  Offline: analyze_fmap.py per log -> per-ep
# optimal scheme + the ep-dependence of the t-structure (capture
# fractions, F(Dt) decay) that decides whether one scheme serves all
# production nodes or the dilution must be node-dependent.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/rwf_fmapL.DONE
GPU=0
run_ep () {
  EP=$1
  NP=$2
  GPUID=$3
  DEFL=$4   # deflation off at small ep: band marginal/absent (census) and
            # the production IRL window (LO=3e-4) would grind 0/16 there
  ETA2=$(python3 -c "print(2*$EP)")
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPUID LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        ETA=$ETA2 N_GL=1 N_NOISE=$NP NOISE_SEED=93 \
        PROFILE=cos PROF_T0=0 DTXQCD_DEFORM_MG=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
        DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_DEFLATE=$DEFL \
        DTXQCD_STOCHPF_TSLICE=2 \
        DTXQCD_DEFORM_EO_DEFLATE_CACHE=$P/defl_cache \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_fmapL_ep$EP.log 2>&1 &
}
run_ep 0.05 6 0 0
run_ep 0.16 6 1 0
run_ep 0.32 3 2 16
run_ep 0.47 2 3 16
wait
echo "== fmap ladder lines =="
grep -ac "FMAP" $LOGD/rwf_fmapL_ep0.05.log $LOGD/rwf_fmapL_ep0.16.log \
  $LOGD/rwf_fmapL_ep0.32.log $LOGD/rwf_fmapL_ep0.47.log
touch $LOGD/rwf_fmapL.DONE
