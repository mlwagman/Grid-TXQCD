#!/bin/bash
#SBATCH --job-name=qdef_defaults
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_defaults.%j.out
# Default-flip gate: a BARE-ENV run (no DTXQCD_DEFORM_* flags) must come up
# backend=QUDA-hopcorr + native gauge upload + qcheck PASSED.  CG_MAX=1
# self-terminates after the first (deliberately unconverged) solve.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
mkdir -p $P/meas_deform_qgate
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 ETA=0.2 PROFILE=tent MEAS_BARYON_NTSRC=1 \
      CG_MAX=1 CFG_DIR=$P/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw \
      DATA_DIR=$P/meas_deform_qgate \
  $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 \
  > $P/gridmg_setups/logs/qdef_defaults.log 2>&1
echo "=== bare-env defaults gate ==="
grep -aE "backend=|deform-native|deform-qcheck" \
  $P/gridmg_setups/logs/qdef_defaults.log | head -8
touch $P/gridmg_setups/logs/qdef_defaults.DONE
