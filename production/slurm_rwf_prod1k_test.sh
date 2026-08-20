#!/bin/bash
#SBATCH --job-name=rwf_prod1k
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_prod1k_%j.out
# ============================================================================
# ONE-CONFIG (traj 300) timing + noise test of the FULL production
# reweighting-factor stack at N_NOISE=1000 per segment, tent profile:
#   GPU0: [0.0,0.2] NA=1000            (projected ~15 h from tent smokes)
#   GPU1: [0.2,0.4] NA=1000            (projected ~19 h)
#   GPU2: [0.4,0.6] NA=500  seed job A (projected ~17 h)  } combine offline:
#   GPU3: [0.4,0.6] NA=500  seed job B (projected ~17 h)  } N=1000 total
# Stack: EO Schur + comb-4 dilution + Neyman + tol 1e-4 + MG-lite L1 +
# REUSE_NV (the slurm_rwf_battery_smoke_seg.sh contract, anchors only).
# Deliverable: measured sigma(dlogdet), sigma(darg), wall per segment at
# production noise counts -> the 30-config farm's per-segment budget.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
TRAJ=300
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
run_seg () {  # run_seg <gpu> <elo> <ehi> <na> <seed> <tag>
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$1 LATT=16.16.16.48 TRAJ=$TRAJ \
        LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
        ETA=$3 ETA_LO=$2 N_GL=2 N_NOISE=$4 NOISE_SEED=$5 \
        PROFILE=tent PROF_T0=0 DTXQCD_STOCHPF_ANTI=0 \
        DTXQCD_STOCHPF_DEFLATE=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
        DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 DTXQCD_MG_REUSE_NV=1 \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_prod1k_$6.log 2>&1
}
run_seg 0 0.0 0.2 1000 2001 seg00_02 &
run_seg 1 0.2 0.4 1000 2002 seg02_04 &
run_seg 2 0.4 0.6  500 2003 seg04_06_a &
run_seg 3 0.4 0.6  500 2004 seg04_06_b &
wait
for t in seg00_02 seg02_04 seg04_06_a seg04_06_b; do
  echo "== $t =="
  grep -aE "STOCHPF traj" $LOGD/rwf_prod1k_$t.log | tail -1
done
