#!/bin/bash
#SBATCH --job-name=qdef_conj
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_conj.%j.out
# APBC-signed-translation gate: rerun the D1 deflated leg (eta=0.6,
# NTSRC=2 -> the t_src=24 solve goes through the conjugation) and compare
# against the EXISTING undeflated reference h5 (meas_defl_b, job 1301281).
# The naive-Cshift bug localized to row 24 at 4.4e-3; with the sign flip
# on the wrapped slab every row must match at CG tolerance.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG5=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
mkdir -p $P/meas_defl_a3

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 PROFILE=tent LAMBDA_DTXQCD=5 CFG_DIR=$CFG5 \
      CG_TOL=1e-7 DTXQCD_DEFORM_QUDA_CHECK=0 ETA=0.6 \
      MEAS_BARYON_NTSRC=2 MEAS_BARYON_MAXSOLVES=5 CG_MAX=20000 \
      DTXQCD_DEFORM_EO_DEFLATE=16 DTXQCD_DEFORM_EO_DEFLATE_PROJ=500 \
      DATA_DIR=$P/meas_defl_a3 \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/conj_gate.log 2>&1
grep -aE "deform-defl|deform-CG" $LOGD/conj_gate.log | tail -7
echo "== conjugation gate: vs undeflated reference =="
python3 $P/_qdef_cmp2.py $P/meas_defl_b/baryon_deform_dtxqcd_300.h5 \
  $P/meas_defl_a3/baryon_deform_dtxqcd_300.h5 1e-4 | tail -1
touch $LOGD/qdef_conj.DONE
