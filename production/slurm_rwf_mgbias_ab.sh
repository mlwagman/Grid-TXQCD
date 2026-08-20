#!/bin/bash
#SBATCH --job-name=rwf_mgAB
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_mgAB_%j.out
# SAME-SEED MG bias gate: [0.4,0.6] with MG forced, NOISE_SEED=2200 ==
# identical noise vectors to the MG-off arm rwf_nomg_seg0.4_0.6 (1310700).
# The dlogdet difference between the two runs cancels the stochastic noise
# and isolates any V-cycle bias at ~0.1 precision.  N_NOISE=32 matched.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
      ETA=0.6 ETA_LO=0.4 N_GL=2 N_NOISE=32 NOISE_SEED=2200 \
      PROFILE=tent PROF_T0=0 DTXQCD_STOCHPF_ANTI=0 \
      DTXQCD_STOCHPF_DEFLATE=0 \
      DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_MG=1 \
      DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 DTXQCD_MG_REUSE_NV=1 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > $LOGD/rwf_mgAB_seg0.4_0.6.log 2>&1
echo "== same-seed MG-ON [0.4,0.6] =="
grep -aE "deform-MG|stochpf-CG" $LOGD/rwf_mgAB_seg0.4_0.6.log | head -4
grep -aE "STOCHPF traj" $LOGD/rwf_mgAB_seg0.4_0.6.log | tail -1
echo "== MG-OFF same-seed reference (1310700) =="
grep -aE "STOCHPF traj" $LOGD/rwf_nomg_seg0.4_0.6.log | tail -1
