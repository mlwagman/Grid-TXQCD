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
# AUX_INIT=<Sigma> gives an explicit saddle and SKIPS the EO-based auto-solve
# (takes precedence over AUX_INIT_AUTO).  Needed at small lambda where the
# now-correct (larger) init aux makes the EO saddle solve near-singular / stall.
AUX_INIT="${AUX_INIT:-}"
RAT_AUTO_HI="${RAT_AUTO_HI:-1}"
# RHMC_DEG overrides the light (DTXQCD/full) rational degree (generator default
# 12).  Fewer poles are adequate where the spectrum is well-gapped (small lambda,
# huge aux regularize -> lambda_min elevated) -> cheaper multishift.
RHMC_DEG="${RHMC_DEG:-}"

# ---- GPU acceleration (cuBLAS batched 48x48; all default ON under CUDA) ---
DTXQCD_PRECOMPUTE_GPU="${DTXQCD_PRECOMPUTE_GPU:-1}"
DTXQCD_LOGDET_S_GPU="${DTXQCD_LOGDET_S_GPU:-1}"
DTXQCD_LOGDET_GPU="${DTXQCD_LOGDET_GPU:-1}"
DTXQCD_MOOEEINV_CUBLAS="${DTXQCD_MOOEEINV_CUBLAS:-1}"
DTXQCD_MOOEE_CUBLAS="${DTXQCD_MOOEE_CUBLAS:-1}"
DTXQCD_MOOEE_FWDCACHE="${DTXQCD_MOOEE_FWDCACHE:-1}"
DTXQCD_RATFORCE_GPU="${DTXQCD_RATFORCE_GPU:-1}"
# Mixed-precision multishift CG for the light rational PF (validated 2026-06-23,
# dH bit-equivalent at cfg.10000 lam=10; per-traj ~17% faster on top of fwdcache).
DTXQCD_MP_CG="${DTXQCD_MP_CG:-1}"
# Pure-SP cleanup inside the MP-CG (no mid-CG DP reliable update during the
# per-shift cleanup pass; one DP HermOp verify at the end with fall-back to
# legacy MP cleanup on per-shift failure).  Default OFF; turn on after the
# DTXQCD_MP_CG_CLEANUP_RESULTS dH+wallclock evidence is in.
DTXQCD_MP_CG_CLEANUP="${DTXQCD_MP_CG_CLEANUP:-0}"

# ---- QUDA fermion-force acceleration (default ON; validated multi-rank) ----
# Set QUDA_FORCE=0 / DTXQCD_QUDA_HYBRID=0 to fall back to the cuBLAS force path
# (bit-comparable; used for parity checks).  Validated at 1.1.1.4 16^3x48-class:
# parity machine-precision vs cuBLAS, gates green, ~8.7x on the Wilson-hop force.
# QUDA_FORCE=1          : strange Nf=1 force via QUDA (Nf=2+1 runs).
# DTXQCD_QUDA_HYBRID=1  : light Nf=2 Wilson-hopping force via QUDA (EO).
# DTXQCD_QUDA_FULL=1    : (reserved) additionally route the clover-sigma force.
# Multi-rank QUDA needs QUDA_ENABLE_MPS=1 so QUDA's per-host gpuid (= local
# rank) maps onto the one GPU the wrapper exposes (gpuid % device_count(1) = 0
# = the rank's physical GPU); without it QUDA aborts "Too few GPUs".  Grid and
# QUDA agree on that device because QudaInit now binds QUDA to Grid's current
# device (cudaGetDevice()), so the wrapper's one-GPU-per-rank binding holds for
# both.  MPS is auto-enabled here when any QUDA force path is on; override with
# QUDA_ENABLE_MPS=0 to force off.
QUDA_FORCE="${QUDA_FORCE:-1}"
# QUDA_FORCE_KERNEL=1 selects Path B (fused QUDA force kernel) inside
# OneFlavourSchurCloverQudaForceRationalActionMP.  Without it, production
# was hitting Path A (Grid-side per-pole loop) where strange's deriv()
# was ~17 s/call due to 80x MeeDeriv/MooDeriv calls at 156 ms each.
# TXQCD already sets this; matching here brings strange [0][3] from
# ~17 s -> ~2 s (measured 2026-06-23).
QUDA_FORCE_KERNEL="${QUDA_FORCE_KERNEL:-1}"
DTXQCD_QUDA_HYBRID="${DTXQCD_QUDA_HYBRID:-1}"
DTXQCD_QUDA_FULL="${DTXQCD_QUDA_FULL:-}"
_QUDA_ON=0
[ "${QUDA_FORCE:-0}" != "0" ] && [ -n "${QUDA_FORCE}" ] && _QUDA_ON=1
[ "${DTXQCD_QUDA_HYBRID:-0}" != "0" ] && [ -n "${DTXQCD_QUDA_HYBRID}" ] && _QUDA_ON=1
[ "${DTXQCD_QUDA_FULL:-0}" != "0" ] && [ -n "${DTXQCD_QUDA_FULL}" ] && _QUDA_ON=1
if [ "$_QUDA_ON" = 1 ]; then
  QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
  QUDA_ENABLE_DEVICE_MEMORY_POOL="${QUDA_ENABLE_DEVICE_MEMORY_POOL:-0}"
  QUDA_ENABLE_MANAGED_MEMORY="${QUDA_ENABLE_MANAGED_MEMORY:-1}"
fi

# ---- run control ---------------------------------------------------------
TRAJ="${TRAJ:-2000}"                 # TARGET total (not an increment)
N_SKIP="${N_SKIP:-10}"               # checkpoint save interval
# Sub-sample the expensive Tr M^-1 (Hutchinson <qbar q>) diagnostic.  signPf +
# extremal M^dag M spectrum stay every traj; only the slow multi-source CG is
# throttled.  Default 1 (every traj) = code default; production sets >1.
DIAG_TRMINV_INTERVAL="${DIAG_TRMINV_INTERVAL:-1}"
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
echo "  GPU: precompute=$DTXQCD_PRECOMPUTE_GPU logdetS=$DTXQCD_LOGDET_S_GPU logdet=$DTXQCD_LOGDET_GPU mooeeinv=$DTXQCD_MOOEEINV_CUBLAS mooee=$DTXQCD_MOOEE_CUBLAS mooee_fwdcache=$DTXQCD_MOOEE_FWDCACHE ratforce=$DTXQCD_RATFORCE_GPU mp_cg=$DTXQCD_MP_CG mp_cg_cleanup=$DTXQCD_MP_CG_CLEANUP"
echo "  TRAJ(target)=$TRAJ  N_SKIP=$N_SKIP  CG_TOL=$CG_TOL  diag_trminv_interval=$DIAG_TRMINV_INTERVAL  suffix='${DTXQCD_SUFFIX}'  import='${IMPORT_CFG:-<resume-or-weakfield>}'  ${NO_METROP_ARG:+$NO_METROP_ARG}"

srun --overlap --mpi=pmix -N 1 -n "$NTASKS" --cpu-bind=none --gres=gpu:"$NTASKS" \
  env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
      LATT="$LATT" \
      LAMBDA_DTXQCD="$LAMBDA_DTXQCD" MASS_LIGHT_DTXQCD="$MASS_LIGHT_DTXQCD" CSW="$CSW" \
      STOUT_NSMEAR="$STOUT_NSMEAR" STOUT_RHO="$STOUT_RHO" \
      ADD_STRANGE="$ADD_STRANGE" MASS_STRANGE="$MASS_STRANGE" \
      AUX_INIT_AUTO="$AUX_INIT_AUTO" RAT_AUTO_HI="$RAT_AUTO_HI" \
      ${AUX_INIT:+AUX_INIT="$AUX_INIT"} \
      ${RHMC_DEG:+RHMC_DEG="$RHMC_DEG"} \
      DTXQCD_PRECOMPUTE_GPU="$DTXQCD_PRECOMPUTE_GPU" DTXQCD_LOGDET_S_GPU="$DTXQCD_LOGDET_S_GPU" \
      DTXQCD_LOGDET_GPU="$DTXQCD_LOGDET_GPU" DTXQCD_MOOEEINV_CUBLAS="$DTXQCD_MOOEEINV_CUBLAS" \
      DTXQCD_MOOEE_CUBLAS="$DTXQCD_MOOEE_CUBLAS" \
      DTXQCD_MOOEE_FWDCACHE="$DTXQCD_MOOEE_FWDCACHE" \
      DTXQCD_MP_CG="$DTXQCD_MP_CG" \
      DTXQCD_MP_CG_CLEANUP="$DTXQCD_MP_CG_CLEANUP" \
      DTXQCD_RATFORCE_GPU="$DTXQCD_RATFORCE_GPU" \
      INTEGRATOR="$INTEGRATOR" MDSTEPS="$MDSTEPS" TRAJL="$TRAJL" \
      GAUGE_MULT="$GAUGE_MULT" AUX_MULT="$AUX_MULT" \
      TRAJ="$TRAJ" N_SKIP="$N_SKIP" CG_TOL="$CG_TOL" \
      DIAG_TRMINV_INTERVAL="$DIAG_TRMINV_INTERVAL" \
      ${DTXQCD_SUFFIX:+DTXQCD_SUFFIX="$DTXQCD_SUFFIX"} \
      ${IMPORT_CFG:+IMPORT_CFG="$IMPORT_CFG"} \
      ${USE_FULL_PF:+USE_FULL_PF="$USE_FULL_PF"} \
      ${QUDA_FORCE:+QUDA_FORCE="$QUDA_FORCE"} \
      ${QUDA_FORCE_KERNEL:+QUDA_FORCE_KERNEL="$QUDA_FORCE_KERNEL"} \
      ${DTXQCD_QUDA_HYBRID:+DTXQCD_QUDA_HYBRID="$DTXQCD_QUDA_HYBRID"} \
      ${DTXQCD_QUDA_FULL:+DTXQCD_QUDA_FULL="$DTXQCD_QUDA_FULL"} \
      ${QUDA_ENABLE_MPS:+QUDA_ENABLE_MPS="$QUDA_ENABLE_MPS"} \
      ${QUDA_ENABLE_DEVICE_MEMORY_POOL:+QUDA_ENABLE_DEVICE_MEMORY_POOL="$QUDA_ENABLE_DEVICE_MEMORY_POOL"} \
      ${QUDA_ENABLE_MANAGED_MEMORY:+QUDA_ENABLE_MANAGED_MEMORY="$QUDA_ENABLE_MANAGED_MEMORY"} \
      $NO_METROP_ARG \
  ./srun_gpu_wrapper.sh "$BIN" --grid "$LATT" --mpi "$MPI" \
      --shm "$SHM" --shm-mpi 1 --device-mem "$DEVICE_MEM"
echo "=== run_dtxqcd_gencfgs done rc=$? $(date) ==="
