#!/bin/bash
#SBATCH --job-name=qdef_defl
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_defl.%j.out
# ============================================================================
# EO deflation gates + the eta=0.8/0.9 unlock (lambda=5, tla traj 300).
#  D1  eta=0.6 A/B: deflated (nev=16, eigenbasis at t_src=0, second t_src
#      exercises the solve-frame CONJUGATION) vs undeflated — h5 to 1e-4 +
#      iteration counts.  NTSRC=2 (t_srcs {0,24}: even spacing), 5 solves.
#  D2  eta=0.8 deflated: the census-predicted 60k -> ~1k.
#  D3  eta=0.9 deflated: FIRST numbers at the target corner (also prints
#      the eigensolve's ev[0..15] = the first eta=0.9 spectrum census).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG5=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
COMMON="LATT=16.16.16.48 TRAJ=300 PROFILE=tent LAMBDA_DTXQCD=5 CFG_DIR=$CFG5 CG_TOL=1e-7 DTXQCD_DEFORM_QUDA_CHECK=0"
mkdir -p $P/meas_defl_a $P/meas_defl_b $P/meas_defl_e08 $P/meas_defl_e09

# D1a: eta=0.6 deflated (basis at first t_src; t_src=24 solve conjugated)
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON ETA=0.6 MEAS_BARYON_NTSRC=2 MEAS_BARYON_MAXSOLVES=5 \
      CG_MAX=20000 DTXQCD_DEFORM_EO_DEFLATE=16 DATA_DIR=$P/meas_defl_a \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/defl_e06_on.log 2>&1
echo "D1a (deflated): $(grep -ac 'deform-CG' $LOGD/defl_e06_on.log) solves"
grep -aE "deform-defl|deform-CG" $LOGD/defl_e06_on.log | tail -8

# D1b: eta=0.6 undeflated reference
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON ETA=0.6 MEAS_BARYON_NTSRC=2 MEAS_BARYON_MAXSOLVES=5 \
      CG_MAX=20000 DATA_DIR=$P/meas_defl_b \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/defl_e06_off.log 2>&1
grep -a "deform-CG" $LOGD/defl_e06_off.log | tail -5
echo "== D1 h5 compare =="
python3 $P/_qdef_cmp2.py $P/meas_defl_b/baryon_deform_dtxqcd_300.h5 \
  $P/meas_defl_a/baryon_deform_dtxqcd_300.h5 1e-4 | tail -1

# D2: eta=0.8 deflated
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON ETA=0.8 MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=2 \
      CG_MAX=80000 DTXQCD_DEFORM_EO_DEFLATE=16 DATA_DIR=$P/meas_defl_e08 \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/defl_e08.log 2>&1
echo "== D2 eta=0.8 deflated =="
grep -aE "deform-defl|deform-CG" $LOGD/defl_e08.log | tail -5

# D3: eta=0.9 deflated (FIRST eta=0.9 data)
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON ETA=0.9 MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=2 \
      CG_MAX=80000 DTXQCD_DEFORM_EO_DEFLATE=16 DATA_DIR=$P/meas_defl_e09 \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/defl_e09.log 2>&1
echo "== D3 eta=0.9 deflated =="
grep -aE "deform-defl|deform-CG" $LOGD/defl_e09.log | tail -5
touch $LOGD/qdef_defl.DONE
