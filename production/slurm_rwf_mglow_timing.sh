#!/bin/bash
#SBATCH --job-name=rwf_mglow
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_mglow_%j.out
# Forced-MG timing for the LOWER segments (companion to rwf_nomg 1310700 at
# identical N_NOISE=32/seeds-class): does the sharp-tent MG-lite 3.3x seen
# at [0.6,0.7] extend down?  MG enable threshold prints per run (nodes
# below ETAMIN fall back to plain stack solves -- expected for seg1 node 1).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
for SEG in "0.0 0.2" "0.2 0.4" "0.4 0.6"; do
  set -- $SEG
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
        ETA=$2 ETA_LO=$1 N_GL=2 N_NOISE=32 NOISE_SEED=2400 \
        PROFILE=tent PROF_T0=0 DTXQCD_STOCHPF_ANTI=0 \
        DTXQCD_STOCHPF_DEFLATE=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_MG=1 \
        DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 DTXQCD_MG_REUSE_NV=1 \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_mglow_seg${1}_${2}.log 2>&1
  echo "== segment $1-$2 (MG forced) =="
  grep -aE "deform-MG|stochpf-CG" $LOGD/rwf_mglow_seg${1}_${2}.log | head -5
  grep -aE "STOCHPF traj" $LOGD/rwf_mglow_seg${1}_${2}.log | tail -1
done
