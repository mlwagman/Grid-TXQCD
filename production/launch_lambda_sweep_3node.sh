#!/bin/bash
# Launch the 3-node lambda sweep at chroma physical-match MD settings.
#
# 12 streams total (one per GPU on 3 lq2_gpu nodes):
#   Node A: TXQCD lambda = 3, 3.5, 4, 5
#   Node B: TXQCD lambda = 6, 6.5, 7, 8
#   Node C: TXQCD lambda = 9, 12, 18  + 1 QCD reference (no aux)
#
# All streams use:
#   INTEGRATOR=MinimumNorm2  LAMBDA_MN2=0.1789
#   MDSTEPS=10               TRAJL=sqrt(2)/4 = 0.353553391  (was 7 = chroma
#                                                            phys match, but
#                                                            MDs=7 + tepid
#                                                            locks in
#                                                            metastable
#                                                            plaq~0.534 basin)
#   GAUGE_MULT=4 (TXQCD) / GAUGE_INNER_MULT=4 (QCD)
#   AUX_MULT=2               (TXQCD only; chroma physical match)
#   START_TYPE=thermal       WEAK_FIELD_SCALE=0.1   (cold-ish start; was
#                                                    0.05 but several λ values
#                                                    locked into a metastable
#                                                    high-plaq basin from there
#                                                    — see project memory
#                                                    qcd_stuck_basin)
#   N_TRAJ=200               (will checkpoint, can extend later)

set -e
cd "$(dirname "$0")"

COMMON_ENV="\
INTEGRATOR=MinimumNorm2,LAMBDA_MN2=0.1789,\
MDSTEPS=10,TRAJL=0.353553390593274,\
GAUGE_MULT=4,GAUGE_INNER_MULT=4,AUX_MULT=2,\
HASEN_DM=0,\
START_TYPE=thermal,WEAK_FIELD_SCALE=0.1,\
NO_METROP=0,N_TRAJ=200"
# AUX_SIGMA_L deliberately NOT set → the binary auto-measures Σ from the
# weak-field gauge (AUX_INIT_AUTO mode).  MDSTEPS=7 (eps=0.0505) is chroma
# physical match for trajL=√2/4; the v11 instability we attributed to MDs=7
# was actually a σ-init issue (init at +0.135 vs eq at +0.0334).  With the
# correct AUX_INIT_AUTO σ=+0.0334 init and the FieldSquareNorm determinism
# fix landed 2026-04-26, MDs=7 + AUX_INIT_AUTO is stable (|dH| O(few k)).

echo "=== Submitting Node A (TXQCD λ=3 3.5 4 5) ==="
sbatch --export=ALL,${COMMON_ENV},LAMBDAS="3 3.5 4 5" \
       slurm_gen_txqcd_4stream.sh

echo "=== Submitting Node B (TXQCD λ=6 6.5 7 8) ==="
sbatch --export=ALL,${COMMON_ENV},LAMBDAS="6 6.5 7 8" \
       slurm_gen_txqcd_4stream.sh

echo "=== Submitting Node C (TXQCD λ=9 12 18 + QCD ref) ==="
sbatch --export=ALL,${COMMON_ENV},LAMBDAS="9 12 18" \
       slurm_gen_mixed_3tx_1qcd.sh

echo
echo "=== Submitted; check with squeue -u \$USER ==="
squeue -u "$USER" 2>&1 | head -10
