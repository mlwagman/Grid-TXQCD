#!/bin/bash
#SBATCH --job-name=dtx_deform_fleet
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/dtx_deform_fleet.%j.out
# Deformed aux baryon fleet: 5 lambda=4 configs, source-centered tent
# eta=0.2, NTSRC=4 (16 solves ~ 2h/config), one config per GPU, resumable.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
OUT=$P/meas_2pt/dtxqcd_lam4_deform_eta0p2
mkdir -p "$OUT"
TRAJS=(100 200 300 400 500)
declare -a SLOT_PID
launch() {
  local traj=$1 gpu=$2
  echo "  [launch] traj=$traj -> GPU $gpu"
  srun --overlap --mpi=pmix -N1 -n1 -c16 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$gpu \
        LATT=16.16.16.48 ETA=0.2 PROFILE=tent MEAS_BARYON_NTSRC=4 \
        CG_TOL=1e-7 LAMBDA_DTXQCD=4 \
        CFG_DIR=$P/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw DATA_DIR=$OUT \
        OMP_NUM_THREADS=16 \
    $P/meas_baryon_deform_dtxqcd "$traj" --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > "$OUT/deform.$traj.log" 2>&1 &
  SLOT_PID[$gpu]=$!
}
for traj in "${TRAJS[@]}"; do
  [ -f "$OUT/baryon_deform_dtxqcd_${traj}.h5" ] && continue
  placed=0
  while [ $placed -eq 0 ]; do
    for g in 0 1 2 3; do
      if [ -z "${SLOT_PID[$g]}" ] || ! kill -0 "${SLOT_PID[$g]}" 2>/dev/null; then
        launch "$traj" "$g"; placed=1; break
      fi
    done
    [ $placed -eq 0 ] && sleep 30
  done
done
wait
echo "=== deform fleet done $(date) ==="
