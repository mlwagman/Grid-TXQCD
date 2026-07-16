#!/bin/bash
#SBATCH --job-name=qdef_nvdisc
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_nvdisc.%j.out
# ============================================================================
# NV-recipe DISCRIMINATOR (2026-07-16 production-MG stall hunt).  The
# production header stalls (relres 0.56, S2-consistent hierarchy) where the
# standalone converges (5,187 Ops) at cos eta=0.6; the ONLY structural
# differences are the two NV-generation semantics.  Three sequential runs
# flip them one at a time on the VALIDATED standalone harness:
#   A: baseline re-verify (also re-validates today's incremental-pack edit)
#   B: NV_RHS_FULL=1  (header's full-grid right-vector RHS)
#   C: NV_LEFT_DAG=1  (header's dagger-CGNR left vectors)
# Whichever arm stalls names the header bug.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
for ARM in A B C; do
  case $ARM in
    A) EX="" ;;
    B) EX="NV_RHS_FULL=1" ;;
    C) EX="NV_LEFT_DAG=1" ;;
  esac
  env PROFILE=cos ETA=0.6 GPUS=0,1 NB=16 $EX \
    bash $P/mtilde_runs/run_mg_smooth.sh > $P/mtilde_runs/mg_nvdisc_$ARM.log 2>&1
  echo "== arm $ARM =="
  grep -aE "two-level PGCR|FAILED" $P/mtilde_runs/mg_nvdisc_$ARM.log | tail -1
done
touch $P/gridmg_setups/logs/qdef_nvdisc.DONE
