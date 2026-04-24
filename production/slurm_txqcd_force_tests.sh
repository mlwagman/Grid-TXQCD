#!/bin/bash
#SBATCH --job-name=txqcd_ft
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=slurm-logs/txqcd_ft.%j.out

# Rerun the 5 TXQCD force tests that ERRed on GPU on 2026-04-21 (before the
# sigma/pi flavor-force factor-of-2 fix in commit a4428e28).  Each test is a
# finite-difference check of an analytic force vs dS for a small lattice.
# One test per GPU so they run in parallel; each is single-rank.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh

export OMP_NUM_THREADS=16

TESTS_DIR=../build-gpu/tests/txqcd
TESTS=(
  Test_txqcd_pf_force
  Test_txqcd_stout_force
  Test_txqcd_rational_force
  Test_txqcd_rational_eo_force
  Test_txqcd_rational_clover_eo_force
)

echo "=== Running ${#TESTS[@]} TXQCD force tests on GPU ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

for i in "${!TESTS[@]}"; do
  gpu=$((i % 4))
  test="${TESTS[$i]}"
  logfile="slurm-logs/${test}.${SLURM_JOB_ID}.out"
  echo "[$i] GPU=$gpu  $test  -> $logfile"
  CUDA_VISIBLE_DEVICES=$gpu \
      mpirun -np 1 --map-by ppr:1:socket:PE=16 \
          "${TESTS_DIR}/${test}" --grid 4.4.4.8 --mpi 1.1.1.1 --shm 1024 --shm-mpi 0 \
      >"$logfile" 2>&1 &
  sleep 2
done

wait

echo "=== Force test summary ==="
for test in "${TESTS[@]}"; do
  log="slurm-logs/${test}.${SLURM_JOB_ID}.out"
  pass=$(grep -cE "^\s*PASS\b|\bPASS\(" "$log" 2>/dev/null)
  fail=$(grep -cE "^\s*FAIL\b|\bFAIL\(|assertion failed" "$log" 2>/dev/null)
  last=$(tail -5 "$log" 2>/dev/null | grep -E "PASS|FAIL|Error" | tail -1)
  printf "%-45s  PASS=%s FAIL=%s  %s\n" "$test" "${pass:-0}" "${fail:-0}" "${last:0:60}"
done

date
echo "=== Done ==="
