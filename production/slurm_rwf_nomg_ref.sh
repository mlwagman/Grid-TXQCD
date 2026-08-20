#!/bin/bash
#SBATCH --job-name=rwf_nomg
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_nomg_%j.out
# MG-lite ISOLATION arm for the prod1k stack test: CG_TOL=1e-4 sampling +
# comb-4 dilution + Schur + everything else ON, MG-lite explicitly OFF
# (DTXQCD_DEFORM_MG=0, no MG_* env).  N_NOISE=32/segment = timing-grade.
# vs the prod1k arms (MG env set): the per-solve delta IS MG-lite's
# contribution per segment.  (Sharp-tent policy may keep MG off anyway --
# identical timing would prove the tent stack's speed is tol+dilution.)
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
for SEG in "0.0 0.2" "0.2 0.4" "0.4 0.6"; do
  set -- $SEG
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
        ETA=$2 ETA_LO=$1 N_GL=2 N_NOISE=32 NOISE_SEED=2200 \
        PROFILE=tent PROF_T0=0 DTXQCD_STOCHPF_ANTI=0 \
        DTXQCD_STOCHPF_DEFLATE=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_MG=0 \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_nomg_seg${1}_${2}.log 2>&1
  echo "== segment $1-$2 (MG off) =="
  grep -aE "stochpf-CG" $LOGD/rwf_nomg_seg${1}_${2}.log | head -3
  grep -aE "STOCHPF traj" $LOGD/rwf_nomg_seg${1}_${2}.log | tail -1
done
