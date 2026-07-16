#!/bin/bash
#SBATCH --job-name=qdef_wcg_l4
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_wcg_l4.%j.out
# ============================================================================
# lambda=4 both-walls width scan (eta=0.8): is lambda=4's pathological
# conditioning (rate-probe kappa_eff 5.5e10 at sharp tent) the SAME
# mu-domain-wall physics amplified — in which case tent_rounded2 rescues
# it like lambda=5 (59,931 -> 8,307 at w=8) — or something else (the
# eta=0.4 11-mode cluster suggests lambda=4 has extra structure)?
# w=0 sharp anchor runs first with the 200k cap: NONCONVERGENCE + final
# relres is itself the baseline datum (expected: cap out).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG4=$P/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"

for W in 0 8 12 16; do
  if [ "$W" = "0" ]; then PROF=tent; else PROF=tent_rounded2; fi
  mkdir -p $P/meas_wscan_l4_w$W
  srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=4 CFG_DIR=$CFG4 \
        PROFILE=$PROF PROFILE_W=$W ETA=0.8 \
        CG_TOL=1e-7 CG_MAX=200000 DTXQCD_DEFORM_QUDA_CHECK=0 \
        MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=1 \
        DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_EO_DEFLATE=0 \
        DATA_DIR=$P/meas_wscan_l4_w$W \
    $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/wcg_l4_w$W.log 2>&1
  echo "== w=$W =="
  grep -aE "deform-CG.*iters=|NOT CONVERGED" $LOGD/wcg_l4_w$W.log | tail -2
done
touch $LOGD/qdef_wcg_l4.DONE
