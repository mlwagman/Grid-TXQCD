#!/bin/bash
#SBATCH --job-name=rwf_truthlow
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=5:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_truthlow_%j.out
# Same-seed truth (CG 1e-8) for the LOW segments: is the tol-1e-4 bias
# (+3.2 at [0.4,0.6]) magnitude-scaled, i.e. are seg1/seg2 clean at 1e-4?
# Partners: rwf_nomg_seg0.0_0.2 (1.4048/1.6550) and seg0.2_0.4
# (2.1847/2.9690), both NOISE_SEED=2200 tol 1e-4.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
for SEG in "0.0 0.2" "0.2 0.4"; do
  set -- $SEG
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
        ETA=$2 ETA_LO=$1 N_GL=2 N_NOISE=32 NOISE_SEED=2200 \
        CG_TOL=1e-8 \
        PROFILE=tent PROF_T0=0 DTXQCD_STOCHPF_ANTI=0 \
        DTXQCD_STOCHPF_DEFLATE=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_MG=0 \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_truthlow_seg${1}_${2}.log 2>&1
  echo "== truth seg $1-$2 =="
  grep -aE "STOCHPF traj" $LOGD/rwf_truthlow_seg${1}_${2}.log | tail -1
done
