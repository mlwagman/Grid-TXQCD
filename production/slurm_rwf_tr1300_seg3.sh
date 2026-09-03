#!/bin/bash
#SBATCH --job-name=rwf_1300s3
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=3-6
#SBATCH --nice=5000
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_1300s3_%A_%a.out
# Targeted retry: traj 1300 [0.4,0.6] anchors.  This config's deformed op is
# ~2x worse-conditioned (~2,200 iters, 55 s/solve), so the standard N=250
# seed jobs need ~34h and both hit the 24h wall (the 2 dead farm items).
# Replacement: 4 seed jobs (j3..j6) x N=125 (~15h each) = N=500 total, same
# as the plan; analyzer combines all rwf_tr*_seg*_j*.log automatically.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
OUTD=$P/rwf_farmA10
SJ=$SLURM_ARRAY_TASK_ID
TAG="tr1300_seg3_j${SJ}"
if [ -f "$OUTD/done_$TAG" ]; then
  echo "item $TAG already done"
  exit 0
fi
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=1300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
      ETA=0.6 ETA_LO=0.4 N_GL=2 N_NOISE=125 \
      NOISE_SEED=$((3000 + 1300 + 100000 * 3 + 10000000 * SJ)) \
      CG_TOL=1e-5 \
      PROFILE=tent PROF_T0=0 DTXQCD_STOCHPF_ANTI=0 \
      DTXQCD_STOCHPF_DEFLATE=0 \
      DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_MG=0 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > "$OUTD/rwf_$TAG.log" 2>&1
if grep -aq "STOCHPF traj" "$OUTD/rwf_$TAG.log"; then
  touch "$OUTD/done_$TAG"
  grep -a "STOCHPF traj" "$OUTD/rwf_$TAG.log" | tail -1
else
  echo "item $TAG INCOMPLETE"
  exit 1
fi
