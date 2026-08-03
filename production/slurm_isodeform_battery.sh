#!/bin/bash
#SBATCH -A lqftml.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1 -n 1 --cpus-per-task=16
#SBATCH -t 3:00:00
#SBATCH -J iso_battery
#SBATCH -o /lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/iso_battery.%j.out
# U(1)_{u-d} isospin-contour battery on the lam=5 4^4 chain:
#   1. WARDCHECK (uniform profile, 2 configs): dlogPf == 0 gate.
#   2. tent eta scan 0.2/0.4/0.6/0.8 x 8 configs: sigma(log|w|), sigma(arg w)
#      (pion-currency phase + near-factorization reprieve test), conditioning
#      wall (expect ~ aM_pi = 0.5943 in this file's charge-2 units).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh > /dev/null 2>&1
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5
TRAJS=$(ls "$CFG" | grep -oP 'ckpoint_lat\.\K[0-9]+' | sort -n)
run () {  # run <traj> <eta> <ward> <derivcheck>
  srun --mpi=pmix -N 1 -n 1 --cpu-bind=none --gres=gpu:1 \
    env LATT=4.4.4.4 CFG_DIR=$CFG TRAJ=$1 LAMBDA_DTXQCD=5 \
        ETA=$2 PROFILE=tent WARDCHECK=$3 DERIVCHECK=$4 \
        N_GL=4 N_NOISE=16 NOISE_SEED=$((700 + $1)) OMP_NUM_THREADS=16 \
    $P/Test_dtxqcd_isodeform_pf --grid 4.4.4.4 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 2>&1 \
    | grep -aE "ISO-RESULT|ISO-WARDCHECK|ISO-DERIVCHECK|NOT CONV|ISO-NODE" \
    | sed "s/^/TRAJ=$1 ETA=$2 W=$3 /"
}
echo "== WARDCHECK (uniform profile, expect dlogPf == 0) =="
run 50 0.5 1 1
run 400 0.5 1 0
echo "== tent eta scan =="
for ETA in 0.2 0.4 0.6 0.8; do
  for t in $TRAJS; do
    run $t $ETA 0 0 | grep -av "ISO-NODE"
  done
done
