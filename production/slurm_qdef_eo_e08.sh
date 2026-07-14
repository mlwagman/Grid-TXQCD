#!/bin/bash
#SBATCH --job-name=qdef_eo_e08
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_eo_e08.%j.out
# eta=0.8 EO-vs-full pair at lambda=5, MEAS_BARYON_MAXSOLVES=1 (one solve
# per leg: EO ~24 min + full up-to-cap ~31 min fits the wall with room;
# the 4-solve legs at 2h could never write their h5s — the gate 1301184
# covers gates + eta=0.4/0.6).  G-sol h5 compare closes it.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG5=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
COMMON="LATT=16.16.16.48 TRAJ=300 PROFILE=tent CFG_DIR=$CFG5 CG_TOL=1e-7 ETA=0.8 MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=1 CG_MAX=80000 DTXQCD_DEFORM_QUDA_CHECK=0"
mkdir -p $P/meas_deform_e08_eo $P/meas_deform_e08_full

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON DTXQCD_DEFORM_EO=1 DATA_DIR=$P/meas_deform_e08_eo \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_e08_eo.log 2>&1
echo "EO   eta=0.8: $(grep -a 'deform-CG' $LOGD/qdef_e08_eo.log | head -1)"

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON DATA_DIR=$P/meas_deform_e08_full \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_e08_full.log 2>&1
echo "FULL eta=0.8: $(grep -a 'deform-CG' $LOGD/qdef_e08_full.log | head -1)"

echo "== G-sol eta=0.8 =="
python3 $P/_qdef_cmp2.py $P/meas_deform_e08_full/baryon_deform_dtxqcd_300.h5 \
  $P/meas_deform_e08_eo/baryon_deform_dtxqcd_300.h5 1e-4 | tail -1
touch $LOGD/qdef_eo_e08.DONE
