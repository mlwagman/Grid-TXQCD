#!/bin/bash
# M~-MG on a SMOOTH-profile operator (2 GPUs: fine stack + dense-LU).
# The sharp-tent MG failure was wall-state t-localization (block-invisible
# modes + bootstrap support collapse); smooth profiles' deep modes are
# WIDE supercritical-window states — the representable kind.  Targets:
# tent_rounded2 w=12 eta=1.2 (CGNR 49,639; beat sharp-eta-0.6's 4,479 at
# the same differential 14.4) or cos eta=0.7 (PROFILE/ETA env).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
NB=${NB:-16}
BIN=$P/Test_dtxqcd_deform_mg_mtilde
if [ "$NB" != "16" ]; then BIN=${BIN}_nb$NB; fi
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env CUDA_VISIBLE_DEVICES=${GPUS:-2,3} \
      CFG_DIR=$CFG TRAJ=300 LAMBDA=5 \
      DEFORM_ETA=${ETA:-1.2} DEFORM_T0=0 \
      PROFILE=${PROFILE:-tent_rounded2} PROFILE_W=${WA:-12} \
      PROFILE_W_SRC=${WS:-12} \
      NV_NEV=$NB NV_TOL=${NVTOL:-1e-3} NV_MAXIT=50000 \
      NV_RHS_FULL=${NV_RHS_FULL:-0} NV_LEFT_DAG=${NV_LEFT_DAG:-0} \
      BLOCK=${BLOCK:-4.4.4.2} SMOOTH_IT=8 COARSE_DIRECT=1 \
      NV_BOOTSTRAP=0 \
      CG_TOL=1e-7 MG_MAXIT=3000 RUN_BASELINE=0 \
      PGCR_MMAX=${MMAX:-16} PGCR_NSTEP=${MMAX:-16} \
  $BIN --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1
