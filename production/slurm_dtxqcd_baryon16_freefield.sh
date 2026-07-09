#!/bin/bash
#SBATCH --job-name=dtx_baryon16_ff
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/%x.%j.out

# FREE-FIELD sign/convention check for the 16-component aux baryon.
# FREE_FIELD=1: U=I (cold) + time-constant aux (broadcast t=0 slice) -> the
# correlator is clean+real and time-translation invariant, so:
#   - fwd MUST equal bwd (APBC index-map sign check),
#   - uu ~ ll (charge-conj dressing check),
#   - the 16 components combine COHERENTLY (|16avg| ~ mean|comp|) -- if they
#     cancel here too, the ~0.45 reinforce ratio on the real config is a
#     COMBINE-SIGN BUG, not physics; if they reinforce, the low single-cfg
#     m_eff is genuine baryon noise.
# Free field is well-conditioned -> EO default converges fast; NTSRC=8.
# Uses traj-300's aux for the field content; U is overwritten to I.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
cd /lustre2/nplqcd/Grid-DTXQCD/production
export OMP_NUM_THREADS=16

CFG_DIR=/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw
OUT_DIR=/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt/dtxqcd_lam4_baryon16_freefield
mkdir -p "$OUT_DIR"

# SELF-CONTAINED free field: cold U=I + drawn aux (no config load), SMALL
# lattice 8^3x24 + CPU path (DTXQCD_MEAS_QUDA=0) -> avoids QUDA's U=I gauge
# abort entirely and runs fast.  MASS=0.5 CSW=0 = well-conditioned free
# Wilson (away from critical mass ~0, no clover).  The sign relations are
# lattice/mass/csw-independent, so a small CPU run is a valid + fast check.
# LAMBDA=20: TINY aux (width 1/20=0.05 << bare mass 0.5) => M48 ~ free Wilson,
# well-conditioned (lambda=4's width-0.25 aux pushed modes near zero -> CG NaN).
# d stays nonzero (source ok); the sign relations are aux-magnitude-independent.
srun --mpi=pmix -n 1 --gres=gpu:1 --cpu-bind=none \
  env LATT=8.8.8.24 LAMBDA_DTXQCD=20 MASS_LIGHT_DTXQCD=0.5 CSW=0 \
      DATA_DIR="$OUT_DIR" \
      FREE_FIELD=1 MEAS_BARYON_NTSRC=8 CG_TOL=1e-10 CG_MAX=8000 \
      DTXQCD_MEAS_QUDA=0 \
  ./meas_baryon_dtxqcd 0 --grid 8.8.8.24 --mpi 1.1.1.1 --shm 512 --shm-mpi 1
echo "=== baryon16 freefield rc=$? $(date) ==="
