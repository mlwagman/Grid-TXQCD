#!/bin/bash
# MG=1 leg of the production A/B gate (plain leg + reference h5 banked in
# meas_mgab_0).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
mkdir -p $P/meas_mgab_1
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none --gres=gpu:2 \
  env LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
      PROFILE=cos ETA=0.6 PROF_T0=0 \
      CG_TOL=1e-7 CG_MAX=200000 DTXQCD_DEFORM_QUDA_CHECK=0 \
      MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=1 \
      DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_EO_DEFLATE=0 \
      DTXQCD_DEFORM_MG=1 DTXQCD_MG_SELFTEST=1 \
      DATA_DIR=$P/meas_mgab_1 \
  $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > $P/mtilde_runs/mgab_1c.log 2>&1
echo "== MG leg =="
grep -aE "deform-mg|deform-CG.*iters=|NOT CONVERGED" $P/mtilde_runs/mgab_1c.log | tail -3
echo "== h5 A/B =="
python3 $P/_qdef_cmp2.py $P/meas_mgab_0/baryon_deform_dtxqcd_300.h5 \
  $P/meas_mgab_1/baryon_deform_dtxqcd_300.h5 1e-4 | tail -1
