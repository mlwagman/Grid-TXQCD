#!/bin/bash
#SBATCH --job-name=qcd_tscan2
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/qcd_tscan2.%j.out

# trajL scan at fixed MDs (post dt-scan diagnosis).  Pin dt = trajL/MDs in
# the stable region (≤ 0.05 from chroma-cfg dt-scan).  Each stream picks MDs
# so dt ≈ 0.05.
#
#   GPU 0: trajL=0.5,   MDs=10  (dt=0.05)
#   GPU 1: trajL=1.0,   MDs=20  (dt=0.05)
#   GPU 2: trajL=1.4142,MDs=28  (dt=0.0505) — chroma's reference trajL
#   GPU 3: trajL=2.0,   MDs=40  (dt=0.05)

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

date
nvidia-smi --query-gpu=index,name --format=csv,noheader

launch_stream() {
  local gpu=$1 stream=$2 suffix=$3 trajL=$4 mds=$5 log=$6
  CUDA_VISIBLE_DEVICES=$gpu STREAM_ID=$stream SUFFIX="$suffix" \
    MDSTEPS="$mds" TRAJL="$trajL" \
    INTEGRATOR=ForceGradient NO_METROP=0 \
    WEAK_FIELD_SCALE=0.1 START_TYPE=tepid \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$log" 2>&1 &
}

launch_stream 0 850 _trajL0p5  0.5               10 "slurm-logs/qcd_s850_trajL0p5.${SLURM_JOB_ID}.out"
echo "[GPU 0] trajL=0.5  MDs=10"
sleep 2
launch_stream 1 851 _trajL1p0  1.0               20 "slurm-logs/qcd_s851_trajL1p0.${SLURM_JOB_ID}.out"
echo "[GPU 1] trajL=1.0  MDs=20"
sleep 2
launch_stream 2 852 _trajL1p4  1.4142135623730951 28 "slurm-logs/qcd_s852_trajL1p4.${SLURM_JOB_ID}.out"
echo "[GPU 2] trajL=√2  MDs=28"
sleep 2
launch_stream 3 853 _trajL2p0  2.0               40 "slurm-logs/qcd_s853_trajL2p0.${SLURM_JOB_ID}.out"
echo "[GPU 3] trajL=2.0  MDs=40"
sleep 2

wait
echo "=== all streams exited ==="
date
