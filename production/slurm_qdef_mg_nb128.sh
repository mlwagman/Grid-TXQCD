#!/bin/bash
#SBATCH --job-name=qdef_nb128
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_nb128.%j.out
# ============================================================================
# BRUTE-SPAN test of the Fermi-shell law: nb=128 vs the ~100-mode
# supercritical shell at cos eta=0.7.  BLOCK=8.8.8.4 keeps the dense
# coarse at N=12,288 (2.4 GB LU).  Known costs: 256 loose setup solves
# (~5-7 h), basis paging (~3-4x slower V-cycles).  Known risk: 8^3
# spatial blocks vs shell wavelength 2pi/p_F ~ 9 sites — if it stalls,
# the diagnosis is BLOCK RESOLUTION (fix = finer blocks + iterative
# coarse, a different project); if it converges, brute span beats the
# shell and the nb dial reopens.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
env PROFILE=cos ETA=0.7 GPUS=0,1 NB=128 BLOCK=8.8.8.4 MMAX=8 \
  bash $P/mtilde_runs/run_mg_smooth.sh > $P/mtilde_runs/mg_cos07_nb128.log 2>&1
grep -aE "two-level PGCR|PASSED|FAILED" $P/mtilde_runs/mg_cos07_nb128.log | tail -2
touch $P/gridmg_setups/logs/qdef_nb128.DONE
