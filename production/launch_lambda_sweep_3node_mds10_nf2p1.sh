#!/bin/bash
# Nf=2+1 (TXQCD light Nf=2 + Nf=1 QCD strange) variant of the MDs=10 sweep.
# Companion to launch_lambda_sweep_3node_mds10.sh; cfg dirs differentiated by
# SUFFIX=_mds10_nf2p1.  Runs alongside the Nf=3 MDs=10 sweep so the two
# action-structure choices can be compared at matched integrator settings on
# the same lattice/mass/lambda values.

set -e
cd "$(dirname "$0")"

COMMON_ENV="\
INTEGRATOR=MinimumNorm2,LAMBDA_MN2=0.1789,\
MDSTEPS=10,TRAJL=0.353553390593274,\
GAUGE_MULT=4,GAUGE_INNER_MULT=4,AUX_MULT=2,\
HASEN_DM=0,\
START_TYPE=thermal,WEAK_FIELD_SCALE=0.1,\
NO_METROP=0,N_TRAJ=200,\
SUFFIX=_mds10_nf2p1"

echo "=== Submitting Node A (Nf=2+1 TXQCD λ=3 3.5 4 5, MDs=10) ==="
sbatch --export=ALL,${COMMON_ENV},LAMBDAS="3 3.5 4 5" \
       slurm_gen_txqcd_4stream_2plus1.sh

echo "=== Submitting Node B (Nf=2+1 TXQCD λ=6 6.5 7 8, MDs=10) ==="
sbatch --export=ALL,${COMMON_ENV},LAMBDAS="6 6.5 7 8" \
       slurm_gen_txqcd_4stream_2plus1.sh

echo "=== Submitting Node C (Nf=2+1 TXQCD λ=9 12 18, MDs=10) ==="
sbatch --export=ALL,${COMMON_ENV},LAMBDAS="9 12 18" \
       slurm_gen_txqcd_4stream_2plus1.sh

echo
squeue -u "$USER" 2>&1 | head -20
