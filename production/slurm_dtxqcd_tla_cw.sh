#!/bin/bash
#SBATCH --job-name=dtx_tla_cw
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/slurm-logs/dtx_tla_cw.%j.out
# Clean-window streams RESTART under TRACELESS_AUX (commit 96a6d1ef: traceless
# sigma/pi + (3/7) lambda^2 s,p prior — fermion-blind trace mode integrated
# out; dynamical equivalence to the legacy action verified 2026-07-03).
# Fresh fromchroma fork, NEW suffix _tla (old-action checkpoints are not
# compatible: different s distribution + nonzero traces).  EO Style C fast
# path for lambda>=5; pass BACKEND=noneo for the lambda=4 point (EO MDS=10
# M_ee pathology).  Parametrized via --export: LAM, optional BACKEND.
#
# NO_METROP defaults to 0 (2026-07-14): Grid's NoMetropolisUntil is RELATIVE
# to StartTrajectory, so the old hardcoded 5 force-accepted the first 5
# trajectories of EVERY chained restart (one unfiltered 5-traj window per
# job).  Fresh chroma forks that want the import-equilibration window must
# now pass NO_METROP=5 explicitly.
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader

CHROMA=/lustre2/nplqcd/cfgs/cl3_16_48_b6p1_m0p2450/a/cl3_16_48_b6p1_m0p2450_a_cfg_10000.lime
P=/lustre2/nplqcd/Grid-DTXQCD/production

# Pre-flight: the binary MUST contain the TRACELESS_AUX knob.  Guards against
# the 2026-07-03 race (afterany dependency already satisfied -> job exec'd the
# pre-rebuild binary 2 min before make finished; Lustre let the relink
# overwrite the file under the running process).
if ! strings $P/gen_dtxqcd_cfgs | grep -q "TRACELESS_AUX"; then
  echo "FATAL: gen_dtxqcd_cfgs lacks TRACELESS_AUX support — stale binary" >&2
  exit 1
fi

# CLEAN=1: wipe the target cfg dir first (recovery from a wrong-binary run
# that left incompatible checkpoints under the _tla suffix).
if [ "$CLEAN" = "1" ]; then
  rm -rf $P/cfgs/dtxqcd_lam${LAM}000_tla_fromchroma_cw
  echo "=== CLEAN=1: wiped cfgs/dtxqcd_lam${LAM}000_tla_fromchroma_cw ==="
fi

EXTRA=""
if [ "$BACKEND" = "noneo" ]; then
  EXTRA="USE_FULL_PF=1"
fi

env $EXTRA \
LAMBDA_DTXQCD=$LAM MDSTEPS=10 TRAJL=0.353553390593274 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 \
IMPORT_CFG=$CHROMA DTXQCD_SUFFIX="_tla_fromchroma_cw" \
DTXQCD_TRACELESS_AUX=1 \
TRAJ=${TRAJ:-500} NO_METROP=${NO_METROP:-0} N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== tla clean-window lam=$LAM backend=${BACKEND:-eo} exited rc=$? $(date) ==="
