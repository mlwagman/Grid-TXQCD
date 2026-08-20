#!/bin/bash
#SBATCH --job-name=rwf_ts07mg
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_ts07mg_%j.out
# eta=0.7 TSRC MG-forced timing (the pass the 6h smoke wall cut): the full-
# period quadrature is ~96 nodes (not 12 -- the 1310893 lesson), NT=4.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
      ETA=0.7 N_NOISE=4 NOISE_SEED=2500 \
      PROFILE=tent PROF_T0=0 \
      DTXQCD_STOCHPF_TSRC_RATIO=12 \
      DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_MG=1 \
      DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 DTXQCD_MG_REUSE_NV=1 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > $LOGD/rwf_ts07mg.log 2>&1
echo "== TSRC eta=0.7 MG forced =="
grep -aE "TSRC-DERIVCHECK|deform-MG" $LOGD/rwf_ts07mg.log | head -2
grep -aE "TSRCNODE m=(0|11) " $LOGD/rwf_ts07mg.log | head -4
grep -aE "TSRCCLOSURE" $LOGD/rwf_ts07mg.log | tail -1
