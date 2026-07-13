#!/bin/bash
#SBATCH --job-name=dtx_deform_meas
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/dtx_deform_meas.%j.out
# Deformed aux baryon measurement validation:
#  (1) ETA=0 gate: must reproduce the standard baryon8 SB (traj 300, same
#      tsrc set) through the sandwich solver to CG tol,
#  (2) first deformed measurement ETA=0.2 tent, same config -- the paired
#      variance-reduction read (compare offline vs baryon8 + the reweighting
#      factor from the hop-form estimator).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
OUT=$P/meas_2pt/dtxqcd_lam4_deform_eta0p2
OUT0=$P/meas_2pt/dtxqcd_lam4_deform_eta0
mkdir -p "$OUT" "$OUT0"
echo "--- step 1: ETA=0 gate ---"
srun --mpi=pmix -N1 -n1 --cpu-bind=none \
  env LATT=16.16.16.48 ETA=0 PROFILE=tent MEAS_BARYON_NTSRC=2 CG_TOL=1e-7 \
      LAMBDA_DTXQCD=4 \
      CFG_DIR=$P/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw DATA_DIR=$OUT0 \
      OMP_NUM_THREADS=16 \
  $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1
echo "--- step 2: ETA=0.2 tent deformed measurement ---"
srun --mpi=pmix -N1 -n1 --cpu-bind=none \
  env LATT=16.16.16.48 ETA=0.2 PROFILE=tent MEAS_BARYON_NTSRC=2 CG_TOL=1e-7 \
      LAMBDA_DTXQCD=4 \
      CFG_DIR=$P/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw DATA_DIR=$OUT \
      OMP_NUM_THREADS=16 \
  $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1
echo "=== deform meas validation done rc=$? $(date) ==="
