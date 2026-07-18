#!/bin/bash
#SBATCH --job-name=rwf_tsrc06
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_tsrc06.%j.out
# ============================================================================
# TSRC-RATIO scan at PRODUCTION eta=0.6 (16^3x48 lam5 traj300 cos):
# all 48 offset ratios from M=12 periodic s-nodes x N_NOISE=8, riding
# the full locked stack -- MG-lite L1 per node (single GPU,
# BLOCK=4.4.4.4 LUDEV=0; setup ~200 s/node), CG_TOL=1e-4 (both now
# in-code defaults).  Gates in-log: TSRC-DERIVCHECK + TSRCCLOSURE.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/rwf_tsrc06.DONE
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env CUDA_VISIBLE_DEVICES=0 LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 \
      CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
      ETA=0.6 N_NOISE=8 NOISE_SEED=103 \
      PROFILE=cos PROF_T0=0 \
      DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
      DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_TSRC_RATIO=12 \
      DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > $LOGD/rwf_tsrc06_P16.log 2>&1
echo "== eta=0.6 tsrc-ratio =="
grep -aE "TSRC-DERIVCHECK|TSRCCLOSURE" $LOGD/rwf_tsrc06_P16.log
touch $LOGD/rwf_tsrc06.DONE
