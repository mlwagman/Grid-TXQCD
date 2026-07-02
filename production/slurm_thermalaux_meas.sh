#!/bin/bash
#SBATCH --job-name=dtx_thaux
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/slurm-logs/dtx_thaux.%j.out
# Thermal-aux-on-chroma-gauge measurement (2026-07-02): the cleanest
# measurement test point — chroma-thermalized gauge (plaq 0.5133) + saddle
# aux init at the given lambda, measured at traj 0 (no HMC evolution).
# Parametrized via --export: LAM (e.g. 10.0), TAG (e.g. lam10).
# Stage 1: gen_dtxqcd_cfgs as config factory (IMPORT_CFG + SAVE_INIT_CKPT).
# Stage 2: meas_conn (pion QCD-check + nucleon) + meas_baryon (aux baryon).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
cd $P

CHROMA=/lustre2/nplqcd/cfgs/cl3_16_48_b6p1_m0p2450/a/cl3_16_48_b6p1_m0p2450_a_cfg_10000.lime
SUF=_thermalaux
CFGD=$P/cfgs/dtxqcd_lam${LAM}000${SUF}
OUT=$P/meas_2pt/snr_sweep/thaux_$TAG
mkdir -p "$OUT"

echo "=== stage 1: traj-0 config factory  lambda=$LAM  $(date) ==="
srun --mpi=pmix -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 LAMBDA_DTXQCD=$LAM DTXQCD_SUFFIX=$SUF \
      IMPORT_CFG=$CHROMA AUX_INIT_AUTO=1 SAVE_INIT_CKPT=1 \
      TRAJ=0 NO_METROP=0 USE_FULL_PF=1 STOUT_NSMEAR=1 RAT_AUTO_HI=1 \
      CG_TOL=1e-8 \
  ./gen_dtxqcd_cfgs --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > "$OUT/factory.log" 2>&1
echo "factory rc=$? $(date)"
ls -la "$CFGD" | grep "\.0$"

echo "=== stage 2: measurements traj 0 $(date) ==="
srun --mpi=pmix -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 LAMBDA_DTXQCD=$LAM CFG_DIR=$CFGD DATA_DIR=$OUT \
      CG_TOL=1e-6 DTXQCD_MEAS_QUDA=1 MEAS_SPACE_SRC=1 MEAS_TIME_SRC=2 \
  ./meas_conn_dtxqcd 0 --grid 16.16.16.48 --mpi 1.1.1.1 --shm-mpi 1 \
  > "$OUT/conn.0.log" 2>&1
echo "conn rc=$? $(date)"
srun --mpi=pmix -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 LAMBDA_DTXQCD=$LAM CFG_DIR=$CFGD DATA_DIR=$OUT \
      CG_TOL=1e-6 DTXQCD_MEAS_QUDA=1 MEAS_BARYON_NTSRC=2 \
  ./meas_baryon_dtxqcd 0 --grid 16.16.16.48 --mpi 1.1.1.1 --shm-mpi 1 \
  > "$OUT/baryon.0.log" 2>&1
echo "baryon rc=$? $(date)"
echo "=== thermalaux $TAG done $(date) ==="
