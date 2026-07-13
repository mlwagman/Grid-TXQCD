#!/bin/bash
#SBATCH --job-name=dtx_pf_conv
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=22:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/dtx_pf_conv.%j.out
# Pf reweighting-factor CONVERGENCE STUDY on ONE config/offset (mwagman):
# what statistics converge dlogdet AND phase per config, and is the GL
# line-integral discretization under control?  Four concurrent arms:
#  GPU0: N_GL=4, N_NOISE=20  -- statistics reference (per-node errors +
#        integrated signal/noise; GLNODE lines give the integrand shape)
#  GPU1: N_GL=2, N_NOISE=10  -- coarse-quadrature comparison
#  GPU2: N_GL=8, N_NOISE=10  -- fine-quadrature comparison
#  GPU3: N_GL=4, N_NOISE=20, PROF_T0=24 -- second offset (offset dependence
#        + an independent statistics sample)
# lambda=4 traj 300, tent eta=0.2, hop insertion, chempot frame.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw
LOGD=$P/slurm-logs
run() {  # gpu ngl nnoise t0 tag
  srun --overlap --mpi=pmix -N1 -n1 -c16 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$1 LATT=16.16.16.48 TRAJ=300 ETA=0.2 \
        PROFILE=tent N_GL=$2 N_NOISE=$3 PROF_T0=$4 CG_TOL=1e-7 \
        NOISE_SEED=$((77 + $1)) \
        CFG_DIR=$CFG OMP_NUM_THREADS=16 \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > "$LOGD/pfconv.$5.log" 2>&1 &
}
run 0 4 20 0  gl4n20t0
run 1 2 10 0  gl2n10t0
run 2 8 10 0  gl8n10t0
run 3 4 20 24 gl4n20t24
wait
grep -h "GLNODE\|STOCHPF" "$LOGD"/pfconv.*.log
echo "=== pf convergence study done $(date) ==="
