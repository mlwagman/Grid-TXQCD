#!/bin/bash
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --job-name=chroma_hmc
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=slurm-logs/chroma_hmc.%j.out

# Direct chroma HMC verification: 4 streams from weak-field, 16³×48,
# Wilson-Clover Nf=3 / Nf=2+1 at chroma's standard trajL=√2.
# Stream 0 GPU 0  Nf=3 chroma exact  MDs=7
# Stream 1 GPU 1  Nf=3 chroma exact  MDs=10
# Stream 2 GPU 2  Nf=2+1 our analog  MDs=7
# Stream 3 GPU 3  Nf=2+1 our analog  MDs=10
#
# Launch pattern adapted from /lustre2/nplqcd/nplqcd_production-cl3_32_48_b6p1_m0p2450/
# lq2_run_tshift.sh — uses NVHPC modules + srun --mpi=pmix.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs chroma_cfgs quda_resources

# Use the standard NPLQCD chroma env (gcc 12.3, cuda 12.2, nvhpc-openmpi3 23.11,
# adds chroma's own LLVM libs to LD_LIBRARY_PATH).
source /lustre2/nplqcd/env_lq2_chroma.sh

# `hmc` binary is the HMC driver; its input schema is rooted at <Params>.
# (`chroma` binary is for inline-measurement workflows with <chroma> root.)
CHROMA=/lustre2/nplqcd/chroma/install_oct_2025/chroma-quda-qdp-jit-double-nd4-cmake/bin/hmc
export OMP_NUM_THREADS=8
export QMT_NUM_THREADS=1
export CUDA_CACHE_PATH=/lustre2/nplqcd/cache

declare -A CFG=(
  [0]="hmc_nf3_mds7_wf"
  [1]="hmc_nf3_mds10_wf"
  [2]="hmc_nf2p1_mds7_wf"
  [3]="hmc_nf2p1_mds10_wf"
)

echo "=== chroma HMC 4-stream verification (weak-field start) ==="
echo "binary: $CHROMA"
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

for i in 0 1 2 3; do
  name="${CFG[$i]}"
  ini="chroma_inputs/${name}.ini.xml"
  out="slurm-logs/chroma_${name}.${SLURM_JOB_ID}.out"
  xml_out="slurm-logs/chroma_${name}.${SLURM_JOB_ID}.xml"
  qrp="quda_resources/${name}_${SLURM_JOB_ID}"
  mkdir -p "$qrp"
  echo "[stream $i] GPU=$i  cfg=$name  log=$out  qrp=$qrp"
  CUDA_VISIBLE_DEVICES=$i \
  QUDA_RESOURCE_PATH="$qrp" \
    srun --exact -N1 -n 1 -c 16 --gres=gpu:1 --mem=64G --cpu-bind=cores --mpi=pmix \
      $CHROMA -i $ini -o $xml_out \
              -geom 1 1 1 1 -qmpgeom 1 1 1 1 -iogeom 1 1 1 1 \
              -ptxdb $qrp/qdp_ptxdb \
      >"$out" 2>&1 &
  sleep 2
done

wait
echo "=== all streams exited ==="
date
