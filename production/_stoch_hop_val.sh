#!/bin/bash
#SBATCH --job-name=dtx_stoch_hop
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/dtx_stoch_hop.%j.out
# Hop-form insertion: (1) DERIVCHECK certification vs FD of the exact M~,
# (2) 4^4 stochastic vs dense ref (-0.191), (3) production smoke 16^3x48
# tent eta=0.2 with the low-variance insertion.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
echo "--- step 1: DERIVCHECK (tent eta=0.3 and 0.7) ---"
for E in 0.3 0.7; do
  srun --mpi=pmix -N1 -n1 --cpu-bind=none \
    env LATT=4.4.4.4 TRAJ=250 ETA=$E PROFILE=tent DERIVCHECK=1 \
        CFG_DIR=$P/cfgs/dtxqcd_lam3.0000_4t4_signhop_lam3 OMP_NUM_THREADS=16 \
    $P/Test_dtxqcd_eta_stoch_pf --grid 4.4.4.4 --mpi 1.1.1.1
done
echo "--- step 2: 4^4 stochastic, hop insertion (dense ref -0.191) ---"
srun --mpi=pmix -N1 -n1 --cpu-bind=none \
  env LATT=4.4.4.4 TRAJ=250 ETA=0.5 PROFILE=tent N_NOISE=8 N_GL=4 \
      CFG_DIR=$P/cfgs/dtxqcd_lam3.0000_4t4_signhop_lam3 OMP_NUM_THREADS=16 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 4.4.4.4 --mpi 1.1.1.1
echo "--- step 3: production smoke, hop insertion ---"
srun --mpi=pmix -N1 -n1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 ETA=0.2 PROFILE=tent N_NOISE=2 N_GL=2 \
      CG_TOL=1e-7 \
      CFG_DIR=$P/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw OMP_NUM_THREADS=16 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1
echo "=== stoch hop validation done rc=$? $(date) ==="
