#!/bin/bash
#SBATCH --job-name=qdef_lamsweep
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=10
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_lamsweep.%j.out
# ============================================================================
# lambda x eta spectrum interplay: the eta ladder {0,0.2,0.4,0.6,0.8} of
# M~^dag M~ low modes at lambda=3/5/6 (tla streams, traj 300 — the same
# family/traj as the lambda=4 anchor).  One lambda per GPU, etas sequential.
#
# lambda=4 findings this maps against: undeformed band [3.2e-4,1.2e-3+]
# (kappa 2.7e5), eta=0.2 a ~15% shift, eta=0.4 an 11-mode cluster with a
# x220 gap to the bulk at 6.7e-2.  Physics questions: does lambda=3's DENSE
# undeformed band shed a similar small cluster (deformation affordable at
# the sign frontier?) and do the gapped lambda=5/6 spectra deform at all?
#
# IRL windows per lambda (undeformed conditioning: lam3 kappa~6e7 fine-op;
# lam5/6 mild): lam3 lo_frac 1e-8 + ord 501; lam5/6 lo_frac 3e-5 + ord 301.
# NEV=24 census depth everywhere.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
ETAS=0.0,0.2,0.4,0.6,0.8

srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env CUDA_VISIBLE_DEVICES=0 LATT=16.16.16.48 \
      CFG_DIR=$P/cfgs/dtxqcd_lam3.0000_tla_fromchroma_cw \
      ETAS=$ETAS NEV=24 IRL_LO_FRAC=1e-8 IRL_ORD=501 \
  $P/probe_deform_spectrum 300 $GRID > $LOGD/deform_spec_lam3.log 2>&1 &

srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env CUDA_VISIBLE_DEVICES=1 LATT=16.16.16.48 \
      CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
      ETAS=$ETAS NEV=24 IRL_LO_FRAC=3e-5 IRL_ORD=301 \
  $P/probe_deform_spectrum 300 $GRID > $LOGD/deform_spec_lam5.log 2>&1 &

srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env CUDA_VISIBLE_DEVICES=2 LATT=16.16.16.48 \
      CFG_DIR=$P/cfgs/dtxqcd_lam6.0000_tla_fromchroma_cw \
      ETAS=$ETAS NEV=24 IRL_LO_FRAC=3e-5 IRL_ORD=301 \
  $P/probe_deform_spectrum 300 $GRID > $LOGD/deform_spec_lam6.log 2>&1 &
wait

for l in 3 5 6; do
  echo "===== lambda=$l spectrum ladder ====="
  grep -aE "spectrum. eta.*(lambda_max|converged|ev\[)" $LOGD/deform_spec_lam$l.log
done
touch $LOGD/qdef_lamsweep.DONE
