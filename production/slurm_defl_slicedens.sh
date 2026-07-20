#!/bin/bash
#SBATCH --job-name=slicedens
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/slicedens.%j.out
# ============================================================================
# Wall-tower LOCALIZATION study: slice densities rho_k(t), grading
# fractions, and transverse IPR for every mode of the CERTIFIED
# (ord=1001/tol=1e-5) tent deflation caches written by the ladder
# (1303312).  Tests the bound-state picture (higher tower levels widen
# in t) and the Anderson question (transverse-localized => sigma_config
# self-averages; extended => coherent config noise) -- the mechanistic
# companion to the deep-segment sigma_config smokes.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/slicedens.DONE
for SPEC in "0.6:8" "0.7:8" "0.7:24" "0.8:8" "0.8:24" "0.85:24"; do
  ETA=${SPEC%%:*}
  NEV=${SPEC##*:}
  ETAF=$(python3 -c "print('%.4f' % float('$ETA'))")
  CF=$(ls $P/defl_cache/*eta${ETAF}_t00_nev${NEV}_tol1e-05_ord1001.evc 2>/dev/null | head -1)
  if [ -z "$CF" ]; then
    echo "== eta=$ETA nev=$NEV: cache not present, skipping =="
    continue
  fi
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=16.16.16.48 TRAJ=300 ETA=$ETA PROFILE=tent PROF_T0=0 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        DEFL_CACHE_FILE=$CF \
    $P/Test_dtxqcd_defl_slicedens --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/slicedens_e${ETA}_nev${NEV}.log 2>&1
  echo "== eta=$ETA nev=$NEV done =="
  grep -ac "SLICEDENS" $LOGD/slicedens_e${ETA}_nev${NEV}.log
done
touch $LOGD/slicedens.DONE
