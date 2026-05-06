#!/bin/bash
#SBATCH --job-name=recov_4s
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/recov_4s.%j.out

# Recovery / comparison launcher for the λ=6, 6.5 stuck-basin investigation.
#
# Streams (one per GPU on an exclusive node, all NP=1):
#   GPU 0: TXQCD λ=6   fresh from chroma cfg  → cfgs/txqcd_lam6.0000_fromchroma_2026-05-06
#   GPU 1: TXQCD λ=6.5 fresh from chroma cfg  → cfgs/txqcd_lam6.5000_fromchroma_2026-05-06
#   GPU 2: TXQCD λ=7   fresh from chroma cfg  → cfgs/txqcd_lam7.0000_fromchroma_2026-05-06
#                       (smooth-basin reference; should match existing λ=7 stream)
#   GPU 3: vanilla QCD Nf=2+1 — extends qcd_s702_nf2p1_mdscan_mds30 from cfg 30
#                       (comparison stream for thermalization rate / plaq drift)
#
# All four use the same chroma cfg as starting gauge, with IMPORT_CFG so the
# TXQCD streams trigger AUX_INIT_AUTO (Σ measurement + smooth aux init).  The
# QCD stream auto-resumes from its latest checkpoint (cfg 30 currently).
#
# Existing λ=6 / λ=6.5 stuck-basin cfgs are LEFT IN PLACE so they can be
# compared against the fresh chroma-thermalized streams.
#
# Common production env (matches launch_lambda_sweep_2node_mds30_aux4_nf2p1.sh):
#   INTEGRATOR=MinimumNorm2 LAMBDA_MN2=0.1789
#   MDSTEPS=30 TRAJL=√2/4
#   GAUGE_MULT=4 GAUGE_INNER_MULT=4 AUX_MULT=4
#   HASEN_DM=0 NO_METROP=0 (real Metropolis)
#   QUDA_FORCE=1 QUDA_FORCE_KERNEL=1 TXQCD_QUDA_HYBRID=1 TXQCD_QUDA_FULL=1
#   TXQCD_PRECOMPUTE_GPU=1 TXQCD_MOOEEINV_CUBLAS=1 TXQCD_MOOEE_CUBLAS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh

export OMP_NUM_THREADS=16
N_TRAJ=${N_TRAJ:-100}
RECOV_SUFFIX="_fromchroma_2026-05-06"
CHROMA_CFG="/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime"

echo "=== Launching 3 TXQCD recov + 1 QCD comparison streams ==="
echo "CHROMA seed cfg: $CHROMA_CFG"
echo "TXQCD SUFFIX:    $RECOV_SUFFIX"
echo "N_TRAJ target:   $N_TRAJ"
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

# --- TXQCD streams on GPUs 0, 1, 2 ---
LAMS=("6" "6.5" "7")
for i in 0 1 2; do
  LAM="${LAMS[$i]}"
  logfile="slurm-logs/txqcd_lam${LAM}_fromchroma.${SLURM_JOB_ID}.out"
  echo "[stream $i] GPU=$i  TXQCD λ=$LAM  log=$logfile"
  # NOTE: env-var assignments must be on the same line as the command (or use
  # `env` prefix) — array expansion of "VAR=val" tokens treats them as command
  # names, NOT environment variables (this caused 94-byte crash logs in v1).
  CUDA_VISIBLE_DEVICES=$i \
      LAMBDA=$LAM \
      SUFFIX="$RECOV_SUFFIX" \
      N_TRAJ=$N_TRAJ \
      IMPORT_CFG="$CHROMA_CFG" \
      INTEGRATOR=MinimumNorm2 \
      LAMBDA_MN2=0.1789 \
      MDSTEPS=30 \
      TRAJL=0.353553390593274 \
      GAUGE_MULT=4 \
      GAUGE_INNER_MULT=4 \
      AUX_MULT=4 \
      HASEN_DM=0 \
      NO_METROP=0 \
      WEAK_FIELD_SCALE=0.1 \
      QUDA_FORCE=1 \
      QUDA_FORCE_KERNEL=1 \
      TXQCD_QUDA_HYBRID=1 \
      TXQCD_QUDA_FULL=1 \
      TXQCD_PRECOMPUTE_GPU=1 \
      TXQCD_MOOEEINV_CUBLAS=1 \
      TXQCD_MOOEE_CUBLAS=1 \
      QUDA_ENABLE_MPS=1 \
      mpirun -np 1 --map-by ppr:1:socket:PE=16 \
          ./gen_txqcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
      >"$logfile" 2>&1 &
  sleep 2
done

# --- QCD comparison stream on GPU 3 (extends qcd_s702_nf2p1_mdscan_mds30) ---
qcdlog="slurm-logs/qcd_s702_nf2p1_mdscan_mds30.${SLURM_JOB_ID}.out"
echo "[stream 3] GPU=3  QCD STREAM_ID=702 SUFFIX=_nf2p1_mdscan_mds30  log=$qcdlog"
CUDA_VISIBLE_DEVICES=3 \
    STREAM_ID=702 \
    SUFFIX="_nf2p1_mdscan_mds30" \
    N_TRAJ=$N_TRAJ \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=30 \
    TRAJL=0.353553390593274 \
    HASEN_DM=0 \
    NO_METROP=0 \
    WEAK_FIELD_SCALE=0.1 \
    QUDA_FORCE=1 \
    QUDA_FORCE_KERNEL=1 \
    QUDA_ENABLE_MPS=1 \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_qcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$qcdlog" 2>&1 &

wait

echo "=== All recovery streams exited ==="
date
