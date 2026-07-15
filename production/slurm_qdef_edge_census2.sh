#!/bin/bash
#SBATCH --job-name=qdef_edge
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_edge.%j.out
# ============================================================================
# Onset-edge census at lambda=5: extend the eta ladder to
# {0.85,0.9,0.95,1.0,1.05} with the SAME instrument/settings as the
# {0..0.8} census (census_lam5.log) so sigma_min(eta)=sqrt(ev[0]) drops
# straight into the onset fit.  Physics: the deformation-induced mode is
# the near-Goldstone of baryon onset at eta ~ aM_N ~ 1.2 (working point
# eta <= 0.9 ~ M_N - m_pi/2); the 0.6->0.8 linear tail extrapolates
# sigma_min->0 at eta ~ 1.0-1.1, while the loose-EO eta=0.9 point hints
# at exponential softening — this ladder discriminates.  ord=501 (vs the
# census 301): the band deepens toward the edge.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 \
      CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
      ETAS=0.85,0.9,0.95,1.0,1.05 NEV=24 IRL_LO_FRAC=3e-5 IRL_ORD=1001 \
  $P/probe_deform_spectrum 300 $GRID > $LOGD/deform_spec_lam5_edge2.log 2>&1
grep -aE "spectrum. eta.*(lambda_max|converged|ev\[0\]|ev\[1\])" \
  $LOGD/deform_spec_lam5_edge2.log
touch $LOGD/qdef_edge.DONE
