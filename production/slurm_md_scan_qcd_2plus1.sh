#!/bin/bash
#SBATCH --job-name=md_scan_qcd
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/md_scan_qcd.%j.out

# MDsteps sweep for the Nf=2+1 QCD action.  All 4 streams share
# integrator/action settings except MDsteps.  Per-stream cfg dirs distinguished
# by STREAM_ID + SUFFIX_PREFIX_mds<MDS> e.g. cfgs/qcd_s700_mdscan_mds15/.
#
# Required env vars:
#   MDS_LIST            space-separated list of 4 MDsteps values
#   SUFFIX_PREFIX       cfg-dir suffix prefix (per-stream MDs tag appended)
#   STREAM_ID_BASE      starting stream id (each stream gets BASE+i)
#
# Optional:
#   IMPORT_CFG, TRAJL, INTEGRATOR, LAMBDA_MN2, GAUGE_INNER_MULT, NO_METROP,
#   N_TRAJ, START_TYPE, WEAK_FIELD_SCALE

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh

export OMP_NUM_THREADS=16
read -r -a MDS_ARR <<< "$MDS_LIST"
N_STREAMS=${#MDS_ARR[@]}
STREAM_ID_BASE=${STREAM_ID_BASE:-700}

echo "=== Launching $N_STREAMS parallel QCD Nf=2+1 MD-scan streams ==="
echo "MDS_LIST: ${MDS_LIST}  SUFFIX_PREFIX: ${SUFFIX_PREFIX}  STREAM_ID_BASE: ${STREAM_ID_BASE}"
[ -n "${IMPORT_CFG-}" ] && echo "IMPORT_CFG: $IMPORT_CFG"
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

for i in $(seq 0 $((N_STREAMS-1))); do
  MDS="${MDS_ARR[$i]}"
  SID=$((STREAM_ID_BASE + i))
  SFX="${SUFFIX_PREFIX}_mds${MDS}"
  logfile="slurm-logs/md_scan_qcd_s${SID}_mds${MDS}.${SLURM_JOB_ID}.out"
  echo "[stream $i] GPU=$i  MDsteps=$MDS  STREAM_ID=$SID  SUFFIX=$SFX  log=$logfile"
  CUDA_VISIBLE_DEVICES=$i STREAM_ID=$SID \
      MDSTEPS="$MDS" SUFFIX="$SFX" \
      GAUGE_INNER_MULT="${GAUGE_INNER_MULT-}" \
      WEAK_FIELD_SCALE="${WEAK_FIELD_SCALE-}" NO_METROP="${NO_METROP-}" \
      START_TYPE="${START_TYPE-}" N_TRAJ="${N_TRAJ-}" \
      INTEGRATOR="${INTEGRATOR-}" TRAJL="${TRAJL-}" LAMBDA_MN2="${LAMBDA_MN2-}" \
      IMPORT_CFG="${IMPORT_CFG-}" \
      mpirun -np 1 --map-by ppr:1:socket:PE=16 \
          ./gen_qcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
      >"$logfile" 2>&1 &
  sleep 2
done

wait

echo "=== All streams exited ==="
date
