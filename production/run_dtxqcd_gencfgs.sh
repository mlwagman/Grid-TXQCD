#!/bin/bash
# ============================================================================
# Canonical DTXQCD production gauge-generation launcher.
#
# All the GPU / autoscale / saddle-init / stout / integrator knobs that make a
# DTXQCD stream production-grade are ON BY DEFAULT here -- a bare invocation
# inside a 4-GPU allocation runs the chroma-matched 16^3x48 point.  Every knob
# is env-overridable (${VAR:-default}); set only what you want to change.
#
# Usage (inside an interactive 4-GPU allocation on lq2):
#   ./run_dtxqcd_gencfgs.sh                       # resume/continue lam10 stream
#   IMPORT_CFG=/path/cfg.lime DTXQCD_SUFFIX=_test ./run_dtxqcd_gencfgs.sh
#   TRAJ=10 NO_METROP=1 ./run_dtxqcd_gencfgs.sh   # 10-traj smoke, accept traj 1
#
# It is also called by the sbatch wrappers (slurm_dtxqcd_*.sh).  Pure-Grid
# multi-GPU -> srun_gpu_wrapper.sh binds one GPU per rank (NOT a QUDA run, so no
# CUDA_VISIBLE_DEVICES=0,1,2,3 / MPS).  Always --shm-mpi 1.
# ============================================================================
set -u
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SELF_DIR"
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh

# ---- physics (chroma-matched 16^3x48 b6.1) -------------------------------
LATT="${LATT:-16.16.16.48}"
MPI="${MPI:-1.1.1.4}"
LAMBDA_DTXQCD="${LAMBDA_DTXQCD:-10.0}"
MASS_LIGHT_DTXQCD="${MASS_LIGHT_DTXQCD:--0.245}"
CSW="${CSW:-1.24930970916466}"
STOUT_NSMEAR="${STOUT_NSMEAR:-1}"
STOUT_RHO="${STOUT_RHO:-0.125}"
# Nf=2+1: ADD_STRANGE=1 adds a plain-QCD Nf=1 spectator strange (mass
# MASS_STRANGE) so plaq/VEV match the Nf=2+1 chroma reference (DTXQCD itself is
# the Nf=2 light sector).  Default OFF (pure Nf=2).
ADD_STRANGE="${ADD_STRANGE:-0}"
MASS_STRANGE="${MASS_STRANGE:--0.245}"

# ---- integrator (canonical: MN2 MDS=10 trajL=sqrt2/4; multi-rate) ----
#   L1 fermion x1 (coarsest), L2 gauge x8, L3 aux x1 (innermost).
#   Gauge x8 matches TXQCD's effective gauge substeps (4 x INNER 2) -> smooth
#   weak-field thermalization (gauge x2 overshot the large plaq~1 start force).
#   aux x1 with gauge x8 keeps the aux step size identical to the old
#   gauge x2 / aux x4 (trajL/(10*8*1) = trajL/(10*2*4) = trajL/80), so aux
#   stability and overhead are unchanged; only the gauge is 4x finer.
INTEGRATOR="${INTEGRATOR:-MinimumNorm2}"
MDSTEPS="${MDSTEPS:-10}"
TRAJL="${TRAJL:-0.353553390593274}"     # sqrt(2)/4
GAUGE_MULT="${GAUGE_MULT:-8}"
AUX_MULT="${AUX_MULT:-1}"

# ---- saddle init + rational autoscale (the "fancy init") -----------------
AUX_INIT_AUTO="${AUX_INIT_AUTO:-1}"
RAT_AUTO_HI="${RAT_AUTO_HI:-1}"

# ---- GPU acceleration (cuBLAS batched 48x48; all default ON under CUDA) ---
DTXQCD_PRECOMPUTE_GPU="${DTXQCD_PRECOMPUTE_GPU:-1}"
DTXQCD_LOGDET_S_GPU="${DTXQCD_LOGDET_S_GPU:-1}"
DTXQCD_LOGDET_GPU="${DTXQCD_LOGDET_GPU:-1}"
DTXQCD_MOOEEINV_CUBLAS="${DTXQCD_MOOEEINV_CUBLAS:-1}"
DTXQCD_RATFORCE_GPU="${DTXQCD_RATFORCE_GPU:-1}"

# ---- run control ---------------------------------------------------------
TRAJ="${TRAJ:-2000}"                 # TARGET total (not an increment)
N_SKIP="${N_SKIP:-10}"               # checkpoint save interval
CG_TOL="${CG_TOL:-1e-8}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DEVICE_MEM="${DEVICE_MEM:-38000}"
SHM="${SHM:-1024}"
DTXQCD_SUFFIX="${DTXQCD_SUFFIX:-}"
IMPORT_CFG="${IMPORT_CFG:-}"
NO_METROP_ARG=""
if [ -n "${NO_METROP:-}" ]; then NO_METROP_ARG="NO_METROP=$NO_METROP"; fi

# Binary selection (override with BIN=... to point at an alternate build).
BIN="${BIN:-./gen_dtxqcd_cfgs}"

NTASKS=$(echo "$MPI" | awk -F. '{print $1*$2*$3*$4}')

echo "=== run_dtxqcd_gencfgs  $(date) ==="
echo "  lattice=$LATT  mpi=$MPI ($NTASKS ranks)  lambda=$LAMBDA_DTXQCD  m=$MASS_LIGHT_DTXQCD  csw=$CSW"
echo "  integrator=$INTEGRATOR MDS=$MDSTEPS trajL=$TRAJL  (gauge x$GAUGE_MULT, aux x$AUX_MULT)"
echo "  stout=$STOUT_NSMEAR(rho $STOUT_RHO)  AUX_INIT_AUTO=$AUX_INIT_AUTO  RAT_AUTO_HI=$RAT_AUTO_HI"
echo "  GPU: precompute=$DTXQCD_PRECOMPUTE_GPU logdetS=$DTXQCD_LOGDET_S_GPU logdet=$DTXQCD_LOGDET_GPU mooeeinv=$DTXQCD_MOOEEINV_CUBLAS ratforce=$DTXQCD_RATFORCE_GPU"
echo "  TRAJ(target)=$TRAJ  N_SKIP=$N_SKIP  CG_TOL=$CG_TOL  suffix='${DTXQCD_SUFFIX}'  import='${IMPORT_CFG:-<resume-or-weakfield>}'  ${NO_METROP_ARG:+$NO_METROP_ARG}"

srun --overlap --mpi=pmix -N 1 -n "$NTASKS" --cpu-bind=none --gres=gpu:"$NTASKS" \
  env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
      LATT="$LATT" \
      LAMBDA_DTXQCD="$LAMBDA_DTXQCD" MASS_LIGHT_DTXQCD="$MASS_LIGHT_DTXQCD" CSW="$CSW" \
      STOUT_NSMEAR="$STOUT_NSMEAR" STOUT_RHO="$STOUT_RHO" \
      ADD_STRANGE="$ADD_STRANGE" MASS_STRANGE="$MASS_STRANGE" \
      AUX_INIT_AUTO="$AUX_INIT_AUTO" RAT_AUTO_HI="$RAT_AUTO_HI" \
      DTXQCD_PRECOMPUTE_GPU="$DTXQCD_PRECOMPUTE_GPU" DTXQCD_LOGDET_S_GPU="$DTXQCD_LOGDET_S_GPU" \
      DTXQCD_LOGDET_GPU="$DTXQCD_LOGDET_GPU" DTXQCD_MOOEEINV_CUBLAS="$DTXQCD_MOOEEINV_CUBLAS" \
      DTXQCD_RATFORCE_GPU="$DTXQCD_RATFORCE_GPU" \
      INTEGRATOR="$INTEGRATOR" MDSTEPS="$MDSTEPS" TRAJL="$TRAJL" \
      GAUGE_MULT="$GAUGE_MULT" AUX_MULT="$AUX_MULT" \
      TRAJ="$TRAJ" N_SKIP="$N_SKIP" CG_TOL="$CG_TOL" \
      ${DTXQCD_SUFFIX:+DTXQCD_SUFFIX="$DTXQCD_SUFFIX"} \
      ${IMPORT_CFG:+IMPORT_CFG="$IMPORT_CFG"} \
      ${USE_FULL_PF:+USE_FULL_PF="$USE_FULL_PF"} \
      $NO_METROP_ARG \
  ./srun_gpu_wrapper.sh "$BIN" --grid "$LATT" --mpi "$MPI" \
      --shm "$SHM" --shm-mpi 1 --device-mem "$DEVICE_MEM"
echo "=== run_dtxqcd_gencfgs done rc=$? $(date) ==="
