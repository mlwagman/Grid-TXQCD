#!/bin/bash
#SBATCH --job-name=lam6_z10_md10
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/lam6_z10_md10.%j.out

# λ=6 Z=10 production chains — TXQCD with the Fierz-correct aux kinetic term
# (Z·(∂X)² on all 5 fields) and matching FFT init filter so cfg 0 starts at
# the (λ=6, Z=10) equilibrium.  Mean-field-on-QCD-gauge scan at b6.1 16³×48
# (test_aux_init_kinetic.cc-validated FFT filter + on-chroma m_pi probe on
# cfg 11610) identified this as the target candidate for a +0.20 connected
# pion mass lift: Δπ ≈ +0.194, ΔN ≈ +0.235 vs vanilla-QCD baseline at this
# lattice spacing.  CG iterations ~220 (Z=10) vs ~540 (Z=0) — ~2.5× faster
# measurement solver as a bonus.
#
# Two streams co-resident on a single node:
#   GPU0  λ=6 Z=10 weakfield   cfgs/txqcd_lam6.0000_z10_weakfield_md10
#   GPU1  λ=6 Z=10 fromchroma  cfgs/txqcd_lam6.0000_z10_fromchroma_md10
#                              IMPORT_CFG cl3_16_48_b6p1_m0p2450 cfg_11100
#
# All other parameters mirror slurm_lam6_md10.sh exactly so this is a clean
# add-Z=10 sibling to the existing λ=6 production chain.  The FFT init filter
# (TXQCDKineticFilter::ApplyFromEnv, fired on every FillAuxFields path in
# gen_txqcd_cfgs_2plus1.cc) makes the weak-field start jump directly to the
# (λ, Z) equilibrium aux distribution — no extra thermalization vs Z=0.
#
# Submit pattern (same as slurm_lam6_md10.sh):
#   sbatch --export=ALL,NO_METROP=100 slurm_lam6_z10_md10.sh   # head: burn-in
#   sbatch --dependency=afterany:$J   slurm_lam6_z10_md10.sh   # dependents
# Per the same Grid HMC.h:273 NoMetropolisUntil-is-relative-to-StartTrajectory
# convention, NO_METROP=0 on dependents lets Metropolis fire from traj 1 of
# the resume (no re-burn-in).

source /lustre2/nplqcd/Grid-TXQCD/env_lq2_grid.sh
cd "$PRODUCTION_DIR"
mkdir -p slurm-logs
export OMP_NUM_THREADS=16
N_TRAJ=${N_TRAJ:-2000}
NO_METROP=${NO_METROP:-0}
echo "[setup] NO_METROP=$NO_METROP (override at submit via --export=ALL,NO_METROP=...)"
CHROMA_CFG="/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime"

# Single λ, single Z, two starts.  Add more entries here to fan out streams.
LAMS=("6"                       "6")
SUFS=("_z10_weakfield_md10"     "_z10_fromchroma_md10")
MDSS=("10"                      "10")
IMPS=(""                        "$CHROMA_CFG")

echo "=== λ=6 Z=10 2-stream MDS=10 (weak + fromchroma) — N_TRAJ=$N_TRAJ ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

for i in 0 1; do
  LAM="${LAMS[$i]}"; SUF="${SUFS[$i]}"; MDS="${MDSS[$i]}"; IMP="${IMPS[$i]}"
  logfile="slurm-logs/lam${LAM}${SUF}.${SLURM_JOB_ID}.out"
  echo "[gpu $i] λ=$LAM Z=10 MDS=$MDS dir=cfgs/txqcd_lam$(printf '%.4f' "$LAM")${SUF}  log=$logfile  IMPORT=${IMP:-<weak-field>}"
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
      SIGMA_KINETIC_Z=10 \
      PI_KINETIC_Z=10 \
      S_KINETIC_Z=10 \
      P_KINETIC_Z=10 \
      T_KINETIC_Z=10 \
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

wait
echo "=== both streams exited ==="
date
