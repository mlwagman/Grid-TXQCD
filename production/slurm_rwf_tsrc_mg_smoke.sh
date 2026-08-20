#!/bin/bash
#SBATCH --job-name=rwf_tsrcmg
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_tsrcmg_%j.out
# TSRC-half MG-forced timing smoke (1 GPU, traj 300, tent):
#   pass 1: TSRC at eta=0.6, MG OFF,    NT=8  (12 nodes x 8 x comb-4 = 384 solves)
#   pass 2: TSRC at eta=0.6, MG FORCED, NT=8
#   pass 3: TSRC at eta=0.7, MG FORCED, NT=4  (192 solves)
# All TSRC solves sit at the segment-top eta (the costliest), so this is
# where the MG 2.4-3.3x matters most for the farm budget.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
run_tsrc () {  # run_tsrc <eta> <nt> <mg> <tag>
  local MGENV="DTXQCD_DEFORM_MG=0"
  if [ "$3" = 1 ]; then
    MGENV="DTXQCD_DEFORM_MG=1 DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 DTXQCD_MG_REUSE_NV=1"
  fi
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
        ETA=$1 N_NOISE=$2 NOISE_SEED=2500 \
        PROFILE=tent PROF_T0=0 \
        DTXQCD_STOCHPF_TSRC_RATIO=12 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 $MGENV \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_tsrcmg_$4.log 2>&1
  echo "== TSRC eta=$1 MG=$3 =="
  grep -aE "TSRC-DERIVCHECK|deform-MG" $LOGD/rwf_tsrcmg_$4.log | head -2
  grep -aE "TSRCNODE m=(0|11) " $LOGD/rwf_tsrcmg_$4.log
  grep -aE "TSRCCLOSURE" $LOGD/rwf_tsrcmg_$4.log | tail -1
}
run_tsrc 0.6 8 0 e06_nomg
run_tsrc 0.6 8 1 e06_mg
run_tsrc 0.7 4 1 e07_mg
