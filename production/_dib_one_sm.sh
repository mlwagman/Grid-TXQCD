#!/bin/bash
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh > /dev/null 2>&1
P=/lustre2/nplqcd/Grid-DTXQCD/production
OUT=$P/meas_2pt/dtxqcd_lam5_dibaryon_aux_sm
mkdir -p "$OUT"
time env LATT=16.16.16.48 \
    CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
    DATA_DIR=$OUT LAMBDA_DTXQCD=5 OMP_NUM_THREADS=16 \
  $P/meas_dibaryon_aux_dtxqcd 30 --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 2>&1 | grep -E "geometries|wrote|real"
