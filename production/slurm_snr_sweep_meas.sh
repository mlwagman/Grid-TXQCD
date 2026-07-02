#!/bin/bash
#SBATCH --job-name=dtx_snr_meas
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/slurm-logs/dtx_snr_meas.%j.out
# Aux-baryon SNR sweet-spot sweep (2026-07-01 overnight):
#   per config: meas_conn_dtxqcd (pion = thermalization gauge, m_pi_eq ~ 0.6;
#   nucleon = quark-baryon SNR reference) + meas_baryon_dtxqcd (diquark-sourced
#   aux baryon, interpolator ~ d ~ 1/lambda^2).
# Parametrized via --export: TAG, CFGD, TRAJS, LAM.
# CAVEAT (mwagman): chains are NOT thermalized except the lam10 chroma fork —
# rank results by pion meff before believing SNR numbers.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
cd /lustre2/nplqcd/Grid-DTXQCD/production

OUT=/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt/snr_sweep/$TAG
mkdir -p "$OUT"
echo "=== SNR sweep $TAG  cfgs [$TRAJS]  lambda=$LAM  $(date) ==="

# NB: payloads MUST go through srun --mpi=pmix — direct exec in the batch
# step fails MPI_Init on this cluster (learned 2026-07-01, jobs 1295868-72).
for t in $TRAJS; do
  echo "--- traj $t conn $(date) ---"
  srun --mpi=pmix -n 1 --cpu-bind=none \
    env LATT=16.16.16.48 LAMBDA_DTXQCD=$LAM CFG_DIR=$CFGD DATA_DIR=$OUT \
        CG_TOL=1e-6 DTXQCD_MEAS_QUDA=1 \
    ./meas_conn_dtxqcd $t --grid 16.16.16.48 --mpi 1.1.1.1 --shm-mpi 1 \
    > "$OUT/conn.$t.log" 2>&1
  echo "conn traj $t rc=$?"
  echo "--- traj $t baryon $(date) ---"
  srun --mpi=pmix -n 1 --cpu-bind=none \
    env LATT=16.16.16.48 LAMBDA_DTXQCD=$LAM CFG_DIR=$CFGD DATA_DIR=$OUT \
        CG_TOL=1e-6 DTXQCD_MEAS_QUDA=1 MEAS_BARYON_NTSRC=4 \
    ./meas_baryon_dtxqcd $t --grid 16.16.16.48 --mpi 1.1.1.1 --shm-mpi 1 \
    > "$OUT/baryon.$t.log" 2>&1
  echo "baryon traj $t rc=$?"
done
echo "=== $TAG done $(date) ==="
