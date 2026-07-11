#!/bin/bash
#SBATCH --job-name=dtx_negpar_gate
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/dtx_negpar_gate.%j.out
# GAUGE_TWIST gate for the MEAS_BARYON_NEGPAR=1 (P- source projection) path:
# run the 4^4 lambda=4 signhop config untwisted + twisted; every correlator
# (P+ AND P-) must be identical to CG tol.  Compared offline.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
GATE=$P/_negpar_gate_out
mkdir -p "$GATE/t0" "$GATE/t1"
COMMON="LATT=4.4.4.4 LAMBDA_DTXQCD=4 MASS_LIGHT_DTXQCD=-0.245 CFG_DIR=$P/cfgs/dtxqcd_lam4.0000_4t4_signhop_lam4 MEAS_BARYON_NTSRC=2 MEAS_BARYON_NEGPAR=1 CG_TOL=1e-10 DTXQCD_MEAS_QUDA=0 DTXQCD_MEAS_EO=0 OMP_NUM_THREADS=16"
srun --mpi=pmix -N1 -n1 --cpu-bind=none \
  env $COMMON DATA_DIR=$GATE/t0 GAUGE_TWIST=0 \
  $P/meas_baryon_dtxqcd 250 --grid 4.4.4.4 --mpi 1.1.1.1
srun --mpi=pmix -N1 -n1 --cpu-bind=none \
  env $COMMON DATA_DIR=$GATE/t1 GAUGE_TWIST=1 \
  $P/meas_baryon_dtxqcd 250 --grid 4.4.4.4 --mpi 1.1.1.1
echo "=== negpar gate runs done $(date) ==="
