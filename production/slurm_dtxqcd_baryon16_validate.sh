#!/bin/bash
#SBATCH --job-name=dtx_baryon16_val
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/%x.%j.out

# Validate the 16-component aux baryon (uu/ll x p/n x up/dn x fwd/bwd + avg).
# ONE stored lambda=3 config (tla stream, clean traj 180), NTSRC=4 (16 solves).
# DTXQCD_MEAS_EO=0: non-EO meas solve mandatory at lambda<=4 (M_ee cliff).
# Output to a SEPARATE dir so no production baryon data is touched.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
cd /lustre2/nplqcd/Grid-DTXQCD/production
export OMP_NUM_THREADS=16

CFG_DIR=/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/dtxqcd_lam3.0000_tla_fromchroma_cw
OUT_DIR=/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt/dtxqcd_lam3_baryon16_val
mkdir -p "$OUT_DIR"

srun --mpi=pmix -n 1 --gres=gpu:1 --cpu-bind=none \
  env LATT=16.16.16.48 LAMBDA_DTXQCD=3 MASS_LIGHT_DTXQCD=-0.245 \
      CFG_DIR="$CFG_DIR" DATA_DIR="$OUT_DIR" \
      MEAS_BARYON_NTSRC=4 CG_TOL=1e-6 \
      DTXQCD_MEAS_QUDA=1 DTXQCD_MEAS_EO=0 \
      QUDA_RESOURCE_PATH=/lustre2/nplqcd/cache/quda_resource_cuda12p2 \
  ./meas_baryon_dtxqcd 180 --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1
echo "=== baryon16 validate rc=$? $(date) ==="
