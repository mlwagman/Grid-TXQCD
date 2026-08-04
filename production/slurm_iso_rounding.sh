#!/bin/bash
#SBATCH -A lqftml.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1 -n 1 --cpus-per-task=16
#SBATCH -t 6:00:00
#SBATCH -J iso_rounding
#SBATCH -o /lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/iso_rounding.%j.out
# THE DECIDING SCAN for the isospin contour: O(eta) phase coefficients
# (ISO_LINEAR profile ladder: sharp tent vs kink-rounded at 1/2/3 pion
# Compton widths) per config.  Config-to-config spread of the im column =
# sigma_theta/eta per profile.  Runs the 4^4 chain (volume-scaling point at
# matched definitions) then the 8^3x16 scout chain (T=16 resolves the
# rounding widths).  Requires 8t16 checkpoints >= 150 (job 1308587).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh > /dev/null 2>&1
P=/lustre2/nplqcd/Grid-DTXQCD/production
run () {  # run <latt> <cfgdir> <traj> <nnoise>
  srun --mpi=pmix -N 1 -n 1 --cpu-bind=none --gres=gpu:1 \
    env LATT=$1 CFG_DIR=$2 TRAJ=$3 LAMBDA_DTXQCD=5 \
        ISO_LINEAR=1 DERIVCHECK=0 N_NOISE=$4 \
        NOISE_SEED=$((1200 + $3)) OMP_NUM_THREADS=16 \
    $P/Test_dtxqcd_isodeform_pf --grid $1 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 2>&1 \
    | grep -aE "ISO-LINEAR|NOT CONV" | sed "s/^/LATT=$1 TRAJ=$3 /"
}
echo "== 4^4 chain (volume-scaling reference) =="
C44=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5
for t in $(ls "$C44" | grep -oP 'ckpoint_lat\.\K[0-9]+' | sort -n); do
  run 4.4.4.4 "$C44" "$t" 64
done
echo "== 8^3x16 scout chain (rounding-resolved) =="
C816=$P/cfgs/dtxqcd_lam5.0000_8t16_isoscout
for t in $(ls "$C816" 2>/dev/null | grep -oP 'ckpoint_lat\.\K[0-9]+' | sort -n); do
  [ "$t" -ge 150 ] || continue
  run 8.8.8.16 "$C816" "$t" 64
done
