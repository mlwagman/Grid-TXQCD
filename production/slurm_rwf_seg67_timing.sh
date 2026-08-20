#!/bin/bash
#SBATCH --job-name=rwf_seg67
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_seg67_%j.out
# [0.6,0.7] segment timing on 1 GPU, stack tol 1e-4 + comb-4 + Schur:
#   pass 1: MG-lite OFF
#   pass 2: MG-lite FORCED ON (DTXQCD_DEFORM_MG=1 overrides the sharp-tent
#           policy-off; nodes 0.62/0.68 > ETAMIN 0.5 so setup runs.  The MG
#           certification is for smooth profiles -- a stall here is itself
#           the answer, cf. the masked-basis NV law.)
# N_NOISE=16 = timing-grade (128 solves/pass with comb-4).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
common="LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG
        ETA=0.7 ETA_LO=0.6 N_GL=2 N_NOISE=16
        PROFILE=tent PROF_T0=0 DTXQCD_STOCHPF_ANTI=0
        DTXQCD_STOCHPF_DEFLATE=0
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1"
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $common NOISE_SEED=2300 DTXQCD_DEFORM_MG=0 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > $LOGD/rwf_seg67_nomg.log 2>&1
echo "== [0.6,0.7] MG OFF =="
grep -aE "stochpf-CG" $LOGD/rwf_seg67_nomg.log | head -3
grep -aE "STOCHPF traj" $LOGD/rwf_seg67_nomg.log | tail -1
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $common NOISE_SEED=2301 DTXQCD_DEFORM_MG=1 \
      DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 DTXQCD_MG_REUSE_NV=1 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > $LOGD/rwf_seg67_mg.log 2>&1
echo "== [0.6,0.7] MG ON =="
grep -aE "stochpf-CG|MG" $LOGD/rwf_seg67_mg.log | head -6
grep -aE "STOCHPF traj" $LOGD/rwf_seg67_mg.log | tail -1
