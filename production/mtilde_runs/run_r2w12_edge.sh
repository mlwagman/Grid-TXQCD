#!/bin/bash
# tent_rounded2 w=12 pushed AT and past the measured onset edge:
# ETA in {0.85, 0.9} at lambda=5.  With the walls rounded away, the
# straightaway slope-1 mu = eta itself crosses the critical mu_c ~ 0.85
# here — this probes whether the measured edge was WALL physics (rounded
# profile pushes through) or bulk criticality (caps out regardless).
# References: rounded2 w=12 eta=0.8 = 2,435 iters.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
GPU=0
for E in 0.85 0.9; do
  mkdir -p $P/meas_r2w12_e$E
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU \
        LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
        PROFILE=tent_rounded2 PROFILE_W=12 ETA=$E \
        CG_TOL=1e-7 CG_MAX=200000 DTXQCD_DEFORM_QUDA_CHECK=0 \
        MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=1 \
        DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_EO_DEFLATE=0 \
        DATA_DIR=$P/meas_r2w12_e$E \
    $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $P/mtilde_runs/r2w12_e$E.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for E in 0.85 0.9; do
  echo "== eta=$E =="
  grep -aE "deform-CG.*iters=|NOT CONVERGED" $P/mtilde_runs/r2w12_e$E.log | tail -1
done
