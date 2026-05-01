#!/bin/bash
# Variant of launch_lambda_sweep_3node.sh with MDSTEPS=10 (eps=0.0354 at
# trajL=√2/4=0.3536) instead of MDSTEPS=7 (eps=0.0505).  Smaller step → more
# stable integrator at the cost of ~40% more force evals per trajectory.
# Cfg dirs distinguished by SUFFIX=_mds10 (e.g. cfgs/txqcd_lam6_mds10/).

set -e
cd "$(dirname "$0")"

COMMON_ENV="\
INTEGRATOR=MinimumNorm2,LAMBDA_MN2=0.1789,\
MDSTEPS=10,TRAJL=0.353553390593274,\
GAUGE_MULT=4,GAUGE_INNER_MULT=4,AUX_MULT=2,\
HASEN_DM=0,\
START_TYPE=thermal,WEAK_FIELD_SCALE=0.1,\
NO_METROP=0,N_TRAJ=200,\
SUFFIX=_mds10,QCD_SUFFIX=_ref_mds10"

echo "=== Submitting Node A (TXQCD λ=3 3.5 4 5, MDs=10) ==="
sbatch --export=ALL,${COMMON_ENV},LAMBDAS="3 3.5 4 5" \
       slurm_gen_txqcd_4stream.sh

echo "=== Submitting Node B (TXQCD λ=6 6.5 7 8, MDs=10) ==="
sbatch --export=ALL,${COMMON_ENV},LAMBDAS="6 6.5 7 8" \
       slurm_gen_txqcd_4stream.sh

echo "=== Submitting Node C (TXQCD λ=9 12 18 + QCD ref, MDs=10) ==="
sbatch --export=ALL,${COMMON_ENV},LAMBDAS="9 12 18" \
       slurm_gen_mixed_3tx_1qcd.sh

echo
squeue -u "$USER" 2>&1 | head -15
