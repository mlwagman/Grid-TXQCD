#!/bin/bash
#SBATCH --job-name=lam6_md10
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/lam6_md10.%j.out

# Sibling to slurm_lam6_6p5_md30_lam7_5_md10.sh but MDS=10 instead of MDS=20.
# Speculative cheaper variant — if MDS=10 makes it through NO_METROP=100 burn-in
# with acceptable dH, we get ~2× more cfgs/day at λ=6 vs MDS=20.
#
#   GPU0  λ=6  cfgs/txqcd_lam6.0000_weakfield_md10           MDS=10  Nf=2+1
#                FRESH weak-field cold start.  Light = Nf=2 TXQCD; strange = Nf=1 QCD action.
#   GPU1  λ=6  cfgs/txqcd_lam6.0000_fromchroma_md10          MDS=10  Nf=2+1
#                FRESH from chroma cl3_16_48_b6p1 cfg_11100.
#   GPU2  λ=6  cfgs/txqcd_lam6.0000_nostrange_weakfield_md10 MDS=10  Nf=2 (NO strange)
#                FRESH weak-field cold start (gen_txqcd_cfgs_nf2). Diagnostic:
#                strange-quark role.  Must be weak-field (NOT chroma import) —
#                chroma cfgs were generated WITH the strange quark, so a clean
#                no-strange history requires a strange-free start.
#   GPU3  λ=7  cfgs/txqcd_lam7.0000_nf3_weakfield_md10       MDS=10  Nf=3 (QUDA hybrid)
#                FRESH weak-field cold start (gen_txqcd_cfgs, compiled at
#                TXQCD_Nf=3 with common mass m_l = m_s — three rational PFs
#                sharing the σ background, strange embedded IN the TXQCD operator).
#                λ=7 (separate from GPU0/1/2 at λ=6) to compare against the
#                existing λ=7 Nf=2+1 weak-field stream.  QUDA HYBRID + FULL force
#                kernels validated bit-exact at Nf=3 (4⁴, dH match to all digits).
#                cuBLAS Mooee paths NOT enabled at Nf=3 (latent Nf=2 assumption).
#
# 4 co-resident streams on a 64-core node ⇒ OMP=16 each (4×16=64) with
# --bind-to none, to avoid the CPU oversubscription / socket-collision that
# 4 independent ppr:1:socket:PE=N mpiruns would cause.
#
# NO_METROP defaults to 0 (Metropolis ON).  Submit pattern (same as md20 sibling):
#   sbatch --export=ALL,NO_METROP=100  slurm_lam6_md10.sh   # head: 100-traj burn-in
#   sbatch --dependency=afterany:$J    slurm_lam6_md10.sh   # dependents: NO_METROP=0
# GPU2 needs a 100-traj burn-in matching the GPU0 weak-field stream, but it
# lives only in the dependent jobs (no head with --export NO_METROP=100).
# Grid's NoMetropolisUntil is RELATIVE to StartTrajectory — the Metropolis test
# runs when traj >= StartTrajectory + NoMetropolisUntil (HMC.h:273) — so a
# constant NO_METROP=100 would re-burn-in 100 trajs on EVERY resume.  Instead
# GPU2 computes NO_METROP = max(0, 100 - latest_ckpt) from its checkpoint, so the
# no-Metropolis window always ends at ABSOLUTE traj 100 (mirrors the driver's own
# max(0,10-start_traj) default, just at 100).  One-time burn-in, same as GPU0.

source /lustre2/nplqcd/Grid-TXQCD/env_lq2_grid.sh
cd "$PRODUCTION_DIR"
mkdir -p slurm-logs
export OMP_NUM_THREADS=16
N_TRAJ=${N_TRAJ:-2000}
NO_METROP=${NO_METROP:-0}
echo "[setup] NO_METROP=$NO_METROP (override at submit via --export=ALL,NO_METROP=...)"
CHROMA_CFG="/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime"

LAMS=("6"                       "6")
SUFS=("_weakfield_md10"         "_fromchroma_md10")
MDSS=("10"                      "10")
IMPS=(""                        "$CHROMA_CFG")

echo "=== 4-stream MDS=10 (λ=6 Nf2+1 weak + λ=6 Nf2+1 chroma + λ=6 Nf2 no-strange weak + λ=7 Nf3 weak QUDA-hybrid) — N_TRAJ=$N_TRAJ ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

for i in 0 1; do
  LAM="${LAMS[$i]}"; SUF="${SUFS[$i]}"; MDS="${MDSS[$i]}"; IMP="${IMPS[$i]}"
  logfile="slurm-logs/lam${LAM}${SUF}.${SLURM_JOB_ID}.out"
  echo "[gpu $i] λ=$LAM MDS=$MDS dir=cfgs/txqcd_lam$(printf '%.4f' "$LAM")${SUF}  log=$logfile  IMPORT=${IMP:-<weak-field>}"
  CUDA_VISIBLE_DEVICES=$i \
      LAMBDA=$LAM \
      SUFFIX="$SUF" \
      N_TRAJ=$N_TRAJ \
      IMPORT_CFG="$IMP" \
      INTEGRATOR=MinimumNorm2 \
      LAMBDA_MN2=0.1789 \
      MDSTEPS=$MDS \
      TRAJL=0.353553390593274 \
      GAUGE_MULT=4 \
      GAUGE_INNER_MULT=2 \
      AUX_MULT=1 \
      HASEN_DM=0 \
      NO_METROP=$NO_METROP \
      WEAK_FIELD_SCALE=0.1 \
      QUDA_FORCE=1 \
      QUDA_FORCE_KERNEL=1 \
      TXQCD_QUDA_HYBRID=1 \
      TXQCD_QUDA_FULL=1 \
      TXQCD_PRECOMPUTE_GPU=1 \
      TXQCD_MOOEEINV_CUBLAS=1 \
      TXQCD_MOOEE_CUBLAS=1 \
      EIG_DIAG=1 \
      QUDA_ENABLE_MPS=1 \
      mpirun -np 1 --bind-to none \
          ./gen_txqcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
      >"$logfile" 2>&1 &
  sleep 2
done

# GPU2  λ=6  Nf=2 no-strange diagnostic — RETIRED 2026-06-01.
#   Cut after the stream equilibrated at plaq=0.4737 (vs Nf=2+1 fromchroma 0.5143)
#   with m_pi=1.08 (vs Nf=2+1 fromchroma 0.59) — confirmed as a distinct heavy-pion
#   basin, not a small perturbation of Nf=2+1.  Diagnostic question answered;
#   freeing GPU2 from future jobs.  Existing cfgs preserved under
#   cfgs/txqcd_lam6.0000_nostrange_weakfield_md10/ for reference.
if false; then
NS_DIR="cfgs/txqcd_lam6.0000_nostrange_weakfield_md10"
NS_LATEST=$(ls "$NS_DIR"/ckpoint_lat.* 2>/dev/null | grep -oE 'ckpoint_lat\.[0-9]+$' | sed 's/ckpoint_lat\.//' | sort -n | tail -1)
NS_LATEST=${NS_LATEST:-0}
NS_NOMETROP=$(( 100 - NS_LATEST ))
[ "$NS_NOMETROP" -lt 0 ] && NS_NOMETROP=0
logfile_ns="slurm-logs/lam6_nostrange_weakfield_md10.${SLURM_JOB_ID}.out"
echo "[gpu 2] λ=6 NO-STRANGE Nf=2 weak-field dir=$NS_DIR  latest_ckpt=$NS_LATEST  NO_METROP=$NS_NOMETROP  log=$logfile_ns"
CUDA_VISIBLE_DEVICES=2 \
    LAMBDA=6 \
    SUFFIX="_nostrange_weakfield_md10" \
    N_TRAJ=$N_TRAJ \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=2 \
    AUX_MULT=1 \
    HASEN_DM=0 \
    NO_METROP=$NS_NOMETROP \
    WEAK_FIELD_SCALE=0.1 \
    QUDA_FORCE=1 \
    QUDA_FORCE_KERNEL=1 \
    TXQCD_QUDA_HYBRID=1 \
    TXQCD_QUDA_FULL=1 \
    TXQCD_PRECOMPUTE_GPU=1 \
    TXQCD_MOOEEINV_CUBLAS=1 \
    TXQCD_MOOEE_CUBLAS=1 \
    EIG_DIAG=1 \
    QUDA_ENABLE_MPS=1 \
    mpirun -np 1 --bind-to none \
        ./gen_txqcd_cfgs_nf2 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$logfile_ns" 2>&1 &
sleep 2
fi

# GPU2  λ=7  Nf=3 fromchroma — companion to GPU3 (Nf=3 weak-field) to isolate
#   the basin-of-attraction axis at fixed Nf=3, λ=7.  Same env stack as GPU3.
#   No NO_METROP burn-in: chroma cfg is QCD-thermalized so Metropolis handles
#   the σ thermalization transient.
#   MDS=10 — initial MDS=15 bump (vs the σ period-2 ringing at Nf=3) reverted
#   2026-06-02 after observing the ringing damps naturally at MDS=10 (λ=7 1283916
#   midpoint stable at 0.094 = Σ_eq/(2λ²) with steady drift toward equilibrium
#   0.187 = Σ_eq/λ²).  MDS=15 was unnecessary; revert to save 33% wallclock.
N3C_DIR="cfgs/txqcd_lam7.0000_nf3_fromchroma_md10"
logfile_n3c="slurm-logs/lam7_nf3_fromchroma_md10.${SLURM_JOB_ID}.out"
echo "[gpu 2] λ=7 Nf=3 fromchroma dir=$N3C_DIR  IMPORT=$CHROMA_CFG  log=$logfile_n3c"
CUDA_VISIBLE_DEVICES=2 \
    LAMBDA=7 \
    SUFFIX="_nf3_fromchroma_md10" \
    N_TRAJ=$N_TRAJ \
    IMPORT_CFG="$CHROMA_CFG" \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=2 \
    AUX_MULT=1 \
    HASEN_DM=0 \
    NO_METROP=0 \
    TXQCD_QUDA_HYBRID=1 \
    TXQCD_QUDA_FULL=1 \
    TXQCD_PRECOMPUTE_BUILD_GPU=1 \
    TXQCD_LOGDET_BUILD_GPU=1 \
    TXQCD_LOGDET_S_GPU=1 \
    EIG_DIAG=1 \
    QUDA_ENABLE_MPS=1 \
    mpirun -np 1 --bind-to none \
        ./gen_txqcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$logfile_n3c" 2>&1 &
sleep 2

# GPU3  λ=7  Nf=3 TXQCD diagnostic (gen_txqcd_cfgs, TXQCD_Nf=3)  MDS=15
#   FRESH weak-field cold start.  Strange embedded IN the TXQCD operator (three
#   rational PFs at common mass m_l = m_s sharing aux background) instead of as
#   a separate Nf=1 QCD action.  λ=7 (decoupled from GPU0/1/2 which are at λ=6)
#   for comparison against the existing λ=7 weak-field Nf=2+1 stream
#   (_nf2p1_mds10_fork) — isolates the operator-structure axis at a fixed λ.
#   QUDA HYBRID + FULL force kernels (validated bit-exact at Nf=3 on 4⁴, dH
#   matches Grid-only to all printed digits).  Do NOT enable
#   TXQCD_PRECOMPUTE_GPU / TXQCD_MOOEEINV_CUBLAS / TXQCD_MOOEE_CUBLAS at Nf=3 —
#   those cuBLAS Mooee paths have a latent Nf=2 assumption that produces silent
#   corruption (~1e17× force) when compiled at TXQCD_Nf=3.  Real bug to fix
#   later; for now the V2 config (Wilson+σ QUDA only, Grid Mooee) is the
#   correct Nf=3 hybrid setting.  Same absolute-traj-100 burn-in formula as GPU2.
#   Nf=3 perf knobs (validated bit-exact 2026-05-30 at 16³×48 single-rank):
#   J.4 (TXQCD_PRECOMPUTE_BUILD_GPU=1) dominant win −13.9% per traj; J.3
#   (TXQCD_LOGDET_BUILD_GPU=1) marginal on top; TXQCD_LOGDET_S_GPU=1 helps
#   action-S evals under Metropolis.  Combined ≈ −14.6% per traj vs HYBRID+FULL
#   alone.  WCF_LOGDET_GPU / WCF_LOGDET_DERIV_GPU are QCD-strange paths that
#   don't apply at Nf=3 (strange embedded in TXQCD); not set.
#   MDS=10 — initial MDS=15 bump (vs the σ period-2 ringing in λ=6 reference
#   1283899) was reverted 2026-06-02 after the λ=7 1283916 chain showed natural
#   damping at MDS=10: midpoint=Σ_eq/(2λ²) stable, amplitude halving every ~6
#   trajs.  Continues 1283916's _md10 chain from latest ckpt (cfg 10 saved).
N3_DIR="cfgs/txqcd_lam7.0000_nf3_weakfield_md10"
N3_LATEST=$(ls "$N3_DIR"/ckpoint_lat.* 2>/dev/null | grep -oE 'ckpoint_lat\.[0-9]+$' | sed 's/ckpoint_lat\.//' | sort -n | tail -1)
N3_LATEST=${N3_LATEST:-0}
N3_NOMETROP=$(( 100 - N3_LATEST ))
[ "$N3_NOMETROP" -lt 0 ] && N3_NOMETROP=0
logfile_n3="slurm-logs/lam7_nf3_weakfield_md10.${SLURM_JOB_ID}.out"
echo "[gpu 3] λ=7 Nf=3 weak-field dir=$N3_DIR  latest_ckpt=$N3_LATEST  NO_METROP=$N3_NOMETROP  log=$logfile_n3"
CUDA_VISIBLE_DEVICES=3 \
    LAMBDA=7 \
    SUFFIX="_nf3_weakfield_md10" \
    N_TRAJ=$N_TRAJ \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=2 \
    AUX_MULT=1 \
    HASEN_DM=0 \
    NO_METROP=$N3_NOMETROP \
    WEAK_FIELD_SCALE=0.1 \
    TXQCD_QUDA_HYBRID=1 \
    TXQCD_QUDA_FULL=1 \
    TXQCD_PRECOMPUTE_BUILD_GPU=1 \
    TXQCD_LOGDET_BUILD_GPU=1 \
    TXQCD_LOGDET_S_GPU=1 \
    EIG_DIAG=1 \
    QUDA_ENABLE_MPS=1 \
    mpirun -np 1 --bind-to none \
        ./gen_txqcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$logfile_n3" 2>&1 &
sleep 2

wait
echo "=== all four streams exited ==="
date
