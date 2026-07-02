#!/bin/bash
#SBATCH --job-name=dtx_eo_scw
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/slurm-logs/dtx_eo_scw.%j.out
# Clean-window EO stream continuation on the FAST device path (EO Style C
# port, commit 0a943e1e): resumes dtxqcd_lam${LAM}000_eo_fromchroma_cw_mds10
# from its latest checkpoint with DTXQCD_MULTISHIFT_QUDA=1 (opt-in gate;
# gates 1-4 green, gate 5 stability run in flight — watch the first dH
# values against the pre-cycle stream envelope).
# Parametrized via --export: LAM (4.0|5.0|6.0).
# NO_METROP=0: this is a RESUME script — NoMetropolisUntil counts the first K
# trajectories of the PROCESS, so a nonzero value force-accepts post-resume
# proposals (learned 2026-07-02: 5 forced accepts at lam=4 let dH~1e3 kicks
# through and displaced the chain).  Only fresh starts want NO_METROP>0.
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader

CHROMA=/lustre2/nplqcd/cfgs/cl3_16_48_b6p1_m0p2450/a/cl3_16_48_b6p1_m0p2450_a_cfg_10000.lime

LAMBDA_DTXQCD=$LAM MDSTEPS=10 TRAJL=0.353553390593274 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 \
IMPORT_CFG=$CHROMA DTXQCD_SUFFIX="_eo_fromchroma_cw_mds10" \
DTXQCD_MULTISHIFT_QUDA=1 \
TRAJ=500 NO_METROP=0 N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== eo stylec clean-window lam=$LAM exited rc=$? $(date) ==="
