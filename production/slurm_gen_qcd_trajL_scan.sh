#!/bin/bash
#SBATCH --job-name=qcd_tscan
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/qcd_tscan.%j.out

# Thermalization experiment: 4 parallel QCD streams with different trajL
# strategies, all Nf=2+1 Wilson-Clover on LWv2 gauge action (c_plaq=β,
# c_rect=-β/(20·u0²)), Metropolis on from traj 0, MDsteps=10, tepid start.
#
#   GPU 0: trajL=0.2                        cfgs/qcd_s800_trajL0p2
#   GPU 1: trajL=0.1                        cfgs/qcd_s801_trajL0p1
#   GPU 2: trajL=0.05                       cfgs/qcd_s802_trajL0p05
#   GPU 3: trajL=sqrt(2)≈1.414, m_l=m_s=-0.15  cfgs/qcd_s803_trajL1p4_m0p15
#
# Stream 3 is the mass-gradient warmup candidate (heavier mass with trajL≈1.4
# matches the chroma trajectory-length convention).

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

date
nvidia-smi --query-gpu=index,name --format=csv,noheader

common_env() {
  local gpu=$1 stream=$2 suffix=$3 trajL=$4 extra=$5 log=$6
  CUDA_VISIBLE_DEVICES=$gpu STREAM_ID=$stream SUFFIX="$suffix" \
    MDSTEPS=10 TRAJL="$trajL" INTEGRATOR=ForceGradient NO_METROP=0 \
    WEAK_FIELD_SCALE=0.1 START_TYPE=tepid $extra \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$log" 2>&1 &
}

# GPU 0: trajL=0.2
log0="slurm-logs/qcd_s800_trajL0p2.${SLURM_JOB_ID}.out"
echo "[GPU 0] trajL=0.2  log=$log0"
common_env 0 800 _trajL0p2 0.2 "" "$log0"
sleep 2

# GPU 1: trajL=0.1
log1="slurm-logs/qcd_s801_trajL0p1.${SLURM_JOB_ID}.out"
echo "[GPU 1] trajL=0.1  log=$log1"
common_env 1 801 _trajL0p1 0.1 "" "$log1"
sleep 2

# GPU 2: trajL=0.05
log2="slurm-logs/qcd_s802_trajL0p05.${SLURM_JOB_ID}.out"
echo "[GPU 2] trajL=0.05  log=$log2"
common_env 2 802 _trajL0p05 0.05 "" "$log2"
sleep 2

# GPU 3: trajL=sqrt(2), heavier mass m_l=m_s=-0.15
log3="slurm-logs/qcd_s803_trajL1p4_m0p15.${SLURM_JOB_ID}.out"
echo "[GPU 3] trajL=sqrt(2)  m_l=m_s=-0.15  log=$log3"
common_env 3 803 _trajL1p4_m0p15 1.4142135623730951 \
  "MASS_LIGHT=-0.15 MASS_STRANGE=-0.15" "$log3"
sleep 2

wait
echo "=== all streams exited ==="
date
