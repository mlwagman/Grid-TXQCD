#!/bin/bash
#SBATCH --job-name=rwf_mgx
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_mgx.%j.out
# ============================================================================
# MG-vs-CG CROSSOVER scan for the reweighting solves (16^3x48 lam5
# traj300 cos t0=0).  MG-lite L1 (single-GPU BLOCK=4.4.4.4 small LU)
# measured x5.2 at ep=0.47; this scan finds ep* where MG starts paying
# (sets DTXQCD_DEFORM_MG_ETAMIN for the battery) INCLUDING the ~3 min
# setup amortized over the arm's solves.  One ep per GPU; CG then MG
# run SEQUENTIALLY on the SAME GPU for fair wallclock.  N_NOISE=8.
# Read: per-arm total wallclock from the log timestamps + [deform-mg]
# setup lines; crossover = where MG total < CG total.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/rwf_mgx.DONE
GPU=0
for EP in 0.1 0.2 0.3 0.4; do
  ETA2=$(python3 -c "print(2*$EP)")
  (
    for ARM in CG MG; do
      case $ARM in
        CG) EX="DTXQCD_DEFORM_MG=0" ;;
        MG) EX="DTXQCD_DEFORM_MG=1 DTXQCD_DEFORM_MG_ETAMIN=0.05 DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0" ;;
      esac
      srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
        env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 TRAJ=300 \
            LAMBDA_DTXQCD=5 \
            CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
            ETA=$ETA2 N_GL=1 N_NOISE=8 NOISE_SEED=101 \
            PROFILE=cos PROF_T0=0 \
            DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
            DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_DEFLATE=0 $EX \
        $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
        --shm 512 --shm-mpi 1 > $LOGD/rwf_mgx_ep${EP}_$ARM.log 2>&1
    done
  ) &
  GPU=$((GPU+1))
done
wait
for EP in 0.1 0.2 0.3 0.4; do
  for ARM in CG MG; do
    echo "== ep=$EP $ARM =="
    grep -aE "GLNODE" $LOGD/rwf_mgx_ep${EP}_$ARM.log | tail -1
    tail -n 3 $LOGD/rwf_mgx_ep${EP}_$ARM.log | grep -aE "Message" | tail -1
  done
done
touch $LOGD/rwf_mgx.DONE
