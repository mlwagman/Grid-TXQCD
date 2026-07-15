#!/bin/bash
# eta=0.8 adaptive M~-MG run (survives node turnover: script+log on lustre).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none --gres=gpu:2 \
  env CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw TRAJ=300 LAMBDA=5 \
      DEFORM_ETA=0.8 DEFORM_T0=0 NV_NEV=16 NV_TOL=1e-3 NV_MAXIT=50000 \
      BLOCK=4.4.4.2 SMOOTH_IT=8 COARSE_DIRECT=1 \
      NV_BOOTSTRAP=2 NV_BOOT_MAXIT=4 NV_BOOT_TOL=0.05 \
      CG_TOL=1e-7 MG_MAXIT=3000 RUN_BASELINE=0 \
  $P/Test_dtxqcd_deform_mg_mtilde --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1
