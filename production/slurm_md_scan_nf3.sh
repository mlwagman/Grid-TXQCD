#!/bin/bash
#SBATCH --job-name=md_scan
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/md_scan.%j.out

# MD-step sweep at fixed LAMBDA for the Nf=2+1 TXQCD setup.  All 4 streams
# share the same LAMBDA, TRAJL, action structure; only MDsteps differs.
#
# Required env vars:
#   LAMBDA              fixed lambda for all 4 streams
#   MDS_LIST            space-separated list of 4 MDsteps values
#   SUFFIX_PREFIX       cfg-dir suffix prefix; per-stream MDsteps tag is appended
#                       e.g. _nf2p1_l7_chrostart → cfgs/txqcd_lam7.0000_nf2p1_l7_chrostart_mds15/
#
# Optional env vars:
#   IMPORT_CFG          if set, all 4 streams import the same starting cfg
#   TRAJL, INTEGRATOR, LAMBDA_MN2, GAUGE_MULT, AUX_MULT, NO_METROP, N_TRAJ
#   START_TYPE, WEAK_FIELD_SCALE
#
# Example:
#   sbatch --export=ALL,LAMBDA=7,MDS_LIST="15 20 30 40",\
#          SUFFIX_PREFIX=_nf2p1_l7_mdscan,...  slurm_md_scan_nf3.sh

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh

export OMP_NUM_THREADS=16
read -r -a MDS_ARR <<< "$MDS_LIST"
N_STREAMS=${#MDS_ARR[@]}

echo "=== Launching $N_STREAMS parallel MD-scan streams (Nf=2+1, λ=$LAMBDA) ==="
echo "MDS_LIST: ${MDS_LIST}"
echo "SUFFIX_PREFIX: ${SUFFIX_PREFIX}"
[ -n "${IMPORT_CFG-}" ] && echo "IMPORT_CFG: $IMPORT_CFG"
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

for i in $(seq 0 $((N_STREAMS-1))); do
  MDS="${MDS_ARR[$i]}"
  SFX="${SUFFIX_PREFIX}_mds${MDS}"
  logfile="slurm-logs/md_scan_l${LAMBDA}_mds${MDS}.${SLURM_JOB_ID}.out"
  echo "[stream $i] GPU=$i  MDsteps=$MDS  SUFFIX=$SFX  log=$logfile"
  CUDA_VISIBLE_DEVICES=$i LAMBDA=$LAMBDA \
      MDSTEPS="$MDS" SUFFIX="$SFX" \
      AUX_MULT="${AUX_MULT-}" GAUGE_MULT="${GAUGE_MULT-}" \
      WEAK_FIELD_SCALE="${WEAK_FIELD_SCALE-}" NO_METROP="${NO_METROP-}" \
      START_TYPE="${START_TYPE-}" HASEN_DM="${HASEN_DM-}" \
      AUX_SIGMA_L="${AUX_SIGMA_L-}" N_TRAJ="${N_TRAJ-}" \
      INTEGRATOR="${INTEGRATOR-}" TRAJL="${TRAJL-}" LAMBDA_MN2="${LAMBDA_MN2-}" \
      IMPORT_CFG="${IMPORT_CFG-}" \
      mpirun -np 1 --map-by ppr:1:socket:PE=16 \
          ./gen_txqcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
      >"$logfile" 2>&1 &
  sleep 2
done

wait

echo "=== All streams exited ==="
date
