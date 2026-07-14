#!/bin/bash
#SBATCH --job-name=qdef_prof
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_prof.%j.out
# Per-phase profile of the deformed QUDA leg (pack/base/unpack/corr) — the
# measured Amdahl table for the native-correction-kernel decision.
# CG_MAX=300 caps the run (~600 legs profiled, prints at 200/400/600, then
# the deliberate nonconvergence _Exit(3)).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
mkdir -p $P/meas_deform_qgate
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 ETA=0.2 PROFILE=tent MEAS_BARYON_NTSRC=1 \
      CG_MAX=300 CFG_DIR=$P/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw \
      DATA_DIR=$P/meas_deform_qgate \
      DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_QUDA_CHECK=0 DTXQCD_DEFORM_PROFILE=1 \
  $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 \
  > $P/gridmg_setups/logs/qdef_profile.log 2>&1
grep -a "deform-prof" $P/gridmg_setups/logs/qdef_profile.log | tail -3
touch $P/gridmg_setups/logs/qdef_profile.DONE
