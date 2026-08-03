#!/bin/bash
#SBATCH -A lqftml.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1 -n 1 --cpus-per-task=16
#SBATCH -t 6:00:00
#SBATCH -J iso_scout8t16
#SBATCH -o /lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/iso_scout8t16.%j.out
# 8^3x16 lam=5 scout chain for the isospin-contour rounded-kink scan:
# T=16 resolves the 1/aM_pi ~ 1.7-slice rounding width the 4^4 chain cannot,
# and V3 = 8x the 4^4 volume tests the sqrt(V3) kink-support phase scaling.
# Fresh weak-field start + AUX_INIT_AUTO; keep configs >= 150 for physics.
# (DIBARYON_INLINE default ON -> free every-traj dibaryon record at T=16.)
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh > /dev/null 2>&1
cd /lustre2/nplqcd/Grid-DTXQCD/production || exit 1   # cfg_dir is RELATIVE
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none --gres=gpu:1 \
  env LATT=8.8.8.16 LAMBDA_DTXQCD=5 DTXQCD_SUFFIX=_8t16_isoscout \
      TRAJ=310 NO_METROP=10 N_SKIP=10 \
      INTEGRATOR=MinimumNorm2 MDSTEPS=10 TRAJL=0.35355339 \
      GAUGE_MULT=8 AUX_MULT=1 AUX_INIT_AUTO=1 RAT_AUTO_HI=1 \
      OMP_NUM_THREADS=16 \
  ./gen_dtxqcd_cfgs --grid 8.8.8.16 --mpi 1.1.1.1 --shm 512 --shm-mpi 1
