#!/bin/bash
#SBATCH --job-name=dtx_thaux4r
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/slurm-logs/dtx_thaux4r.%j.out
# lambda=4 thermal-aux retry: the first job timed out at 6h mid-conn (solves
# cost ~13 min each at lambda=4 — the sign-wall conditioning).  Config
# ckpoint_lat.0 already exists from the factory stage; measure with ONE time
# source (12 solves ~2.6h) + baryon NTSRC=1, 12h margin.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
cd $P
CFGD=$P/cfgs/dtxqcd_lam4.0000_thermalaux
OUT=$P/meas_2pt/snr_sweep/thaux_lam4
mkdir -p "$OUT"

srun --mpi=pmix -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 LAMBDA_DTXQCD=4.0 CFG_DIR=$CFGD DATA_DIR=$OUT \
      CG_TOL=1e-6 DTXQCD_MEAS_QUDA=1 MEAS_SPACE_SRC=1 MEAS_TIME_SRC=1 \
  ./meas_conn_dtxqcd 0 --grid 16.16.16.48 --mpi 1.1.1.1 --shm-mpi 1 \
  > "$OUT/conn.0.retry.log" 2>&1
echo "conn rc=$? $(date)"
srun --mpi=pmix -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 LAMBDA_DTXQCD=4.0 CFG_DIR=$CFGD DATA_DIR=$OUT \
      CG_TOL=1e-6 DTXQCD_MEAS_QUDA=1 MEAS_BARYON_NTSRC=1 \
  ./meas_baryon_dtxqcd 0 --grid 16.16.16.48 --mpi 1.1.1.1 --shm-mpi 1 \
  > "$OUT/baryon.0.retry.log" 2>&1
echo "baryon rc=$? $(date)"
echo "=== thaux lam4 retry done $(date) ==="
