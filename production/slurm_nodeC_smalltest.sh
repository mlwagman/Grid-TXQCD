#!/bin/bash
#SBATCH --job-name=nodeC_smt
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/nodeC_smt.%j.out

# Node C: mix of (a) the lone λ=9 fresh-from-chroma MDS=10 thermalization
# test that's still running, (b) two NEW from-λ=7-cfg.420 fork experiments
# at λ=6 and λ=6.5 (BASIN_LO seed via a well-thermalized λ=7 cfg), and (c)
# continuing the λ=7 nf2p1_mds10_fork donor stream past cfg.420 so we get
# fresher donor candidates if these forks succeed.
#
#  GPU 0: λ=9   fresh from chroma cfg_11100.lime, MDS=10
#         dir: cfgs/txqcd_lam9.0000_fromchroma_md10
#  GPU 1: λ=6   forked from txqcd_lam7.0000_nf2p1_mds10_fork/ckpoint_lat.420
#         dir: cfgs/txqcd_lam6.0000_from_lam7_mds10
#  GPU 2: λ=6.5 forked from same λ=7 cfg.420 source
#         dir: cfgs/txqcd_lam6.5000_from_lam7_mds10
#  GPU 3: λ=7   nf2p1_mds10_fork continue (the donor stream)
#         dir: cfgs/txqcd_lam7.0000_nf2p1_mds10_fork
#
# The from-λ=7 forks (b) test: can a BASIN_LO-thermalized λ=7 gauge cfg
# drive λ=6 and λ=6.5 chains into BASIN_LO at MDS=10?  σ per-comp goes from
# 0.026 (λ=7) → 0.042 (λ=6 equilibrium), milder perturbation than chroma
# import.  λ=6, 6.5 _fromchroma_md20 streams at MDS=30 remain the safe
# fallback if MDS=10 here proves unstable on the perturbed start.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

N_TRAJ=${N_TRAJ:-1200}
CHROMA_CFG="/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime"

# --------------------------------------------------------------------------
# Setup: seed the from_lam7_mds10 dirs from λ=7 nf2p1_mds10_fork/ckpoint_lat.420
# (latest available).  Restart-safe.  Uses 420 by default; advance by
# editing DONOR_TRAJ if the donor stream has moved forward.
# --------------------------------------------------------------------------
DONOR_DIR="cfgs/txqcd_lam7.0000_nf2p1_mds10_fork"
DONOR_TRAJ=${DONOR_TRAJ:-420}
echo "=== Seeding from_lam7 forks from $DONOR_DIR/ckpoint_lat.${DONOR_TRAJ} ==="
for L in 6.0000 6.5000; do
  fork="cfgs/txqcd_lam${L}_from_lam7_mds10"
  mkdir -p "$fork"
  for f in ckpoint_lat ckpoint_lat_aux ckpoint_rng; do
    if [ -f "$DONOR_DIR/${f}.${DONOR_TRAJ}" ] && [ ! -f "$fork/${f}.${DONOR_TRAJ}" ]; then
      cp "$DONOR_DIR/${f}.${DONOR_TRAJ}" "$fork/${f}.${DONOR_TRAJ}"
      echo "  copied $DONOR_DIR/${f}.${DONOR_TRAJ} → $fork/"
    fi
  done
done

echo "=== Launching 4 streams on node C ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

# -----------------------------------------------------------------
# GPU 0: λ=7 chroma-pedigree fork_t50_mds10 CONTINUE (extend past cfg.320).
# Was λ=9 fresh-from-chroma — that stream paused at cfg.240 to free this
# GPU for chroma-pedigree continuation of the metastability-region donor.
# -----------------------------------------------------------------
logfile="slurm-logs/nodeC_lam7_chroma_fork_t50.${SLURM_JOB_ID}.out"
echo "[gpu 0] λ=7 _fromchroma_md20_fork_t50_mds10 continue log=$logfile"
CUDA_VISIBLE_DEVICES=0 \
    LAMBDA=7 \
    SUFFIX="_fromchroma_md20_fork_t50_mds10" \
    N_TRAJ="$N_TRAJ" \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=4 \
    AUX_MULT=4 \
    HASEN_DM=0 \
    NO_METROP=0 \
    QUDA_FORCE=1 \
    QUDA_FORCE_KERNEL=1 \
    TXQCD_QUDA_HYBRID=1 \
    TXQCD_QUDA_FULL=1 \
    TXQCD_PRECOMPUTE_GPU=1 \
    TXQCD_MOOEEINV_CUBLAS=1 \
    TXQCD_MOOEE_CUBLAS=1 \
    EIG_DIAG=1 \
    QUDA_ENABLE_MPS=1 \
    mpirun -np 1 --bind-to none --map-by ppr:1:socket:PE=16 \
        ./gen_txqcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$logfile" 2>&1 &
sleep 2

# -----------------------------------------------------------------
# GPU 1 + 2: λ=6, 6.5 forked from λ=7 nf2p1_mds10_fork/cfg.420 at MDS=10.
# NEW basin-recovery strategy: the donor cfg sits in BASIN_LO (plaq=0.515),
# and σ at λ=7 is close enough to λ=6, 6.5 equilibrium that HMC should
# relax without crossing into BASIN_HI.  If unstable, fall back to
# slurm_nodeE_mixed.sh's MDS=30 _fromchroma_md20 streams.
# -----------------------------------------------------------------
declare -a TGPU=(1 2)
declare -a TLAM=(6 6.5)
for i in 0 1; do
  gpu="${TGPU[$i]}"
  LAM="${TLAM[$i]}"
  logfile="slurm-logs/nodeC_lam${LAM}_from_lam7.${SLURM_JOB_ID}.out"
  echo "[gpu $gpu] λ=$LAM from λ=7 cfg.${DONOR_TRAJ} MDS=10 log=$logfile"
  CUDA_VISIBLE_DEVICES=$gpu \
      LAMBDA="$LAM" \
      SUFFIX="_from_lam7_mds10" \
      N_TRAJ="$N_TRAJ" \
      INTEGRATOR=MinimumNorm2 \
      LAMBDA_MN2=0.1789 \
      MDSTEPS=10 \
      TRAJL=0.353553390593274 \
      GAUGE_MULT=4 \
      GAUGE_INNER_MULT=4 \
      AUX_MULT=4 \
      HASEN_DM=0 \
      NO_METROP=0 \
      QUDA_FORCE=1 \
      QUDA_FORCE_KERNEL=1 \
      TXQCD_QUDA_HYBRID=1 \
      TXQCD_QUDA_FULL=1 \
      TXQCD_PRECOMPUTE_GPU=1 \
      TXQCD_MOOEEINV_CUBLAS=1 \
      TXQCD_MOOEE_CUBLAS=1 \
      EIG_DIAG=1 \
      QUDA_ENABLE_MPS=1 \
      mpirun -np 1 --bind-to none --map-by ppr:1:socket:PE=16 \
          ./gen_txqcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
      >"$logfile" 2>&1 &
  sleep 2
done

# -----------------------------------------------------------------
# GPU 3: λ=8 chroma-pedigree fork_t50_mds10 CONTINUE (extend past cfg.290).
# Was λ=7 _nf2p1_mds10_fork (donor for from_lam7), but that's already
# being run by nodeA_v2 — directory collision.  This slot now extends
# the λ=8 chroma-pedigree stream which had no active runner.
# -----------------------------------------------------------------
logfile="slurm-logs/nodeC_lam8_chroma_fork_t50.${SLURM_JOB_ID}.out"
echo "[gpu 3] λ=8 _fromchroma_md20_fork_t50_mds10 continue log=$logfile"
CUDA_VISIBLE_DEVICES=3 \
    LAMBDA=8 \
    SUFFIX="_fromchroma_md20_fork_t50_mds10" \
    N_TRAJ="$N_TRAJ" \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=4 \
    AUX_MULT=4 \
    HASEN_DM=0 \
    NO_METROP=0 \
    QUDA_FORCE=1 \
    QUDA_FORCE_KERNEL=1 \
    TXQCD_QUDA_HYBRID=1 \
    TXQCD_QUDA_FULL=1 \
    TXQCD_PRECOMPUTE_GPU=1 \
    TXQCD_MOOEEINV_CUBLAS=1 \
    TXQCD_MOOEE_CUBLAS=1 \
    EIG_DIAG=1 \
    QUDA_ENABLE_MPS=1 \
    mpirun -np 1 --bind-to none --map-by ppr:1:socket:PE=16 \
        ./gen_txqcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$logfile" 2>&1 &
sleep 2

wait
echo "=== node C done ==="
date
