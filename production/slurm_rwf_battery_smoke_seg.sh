#!/bin/bash
#SBATCH --job-name=rwf_smokeseg
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_smokeseg_%j.out
# ============================================================================
# SEGMENT battery smokes (companions to the eta=0.2 pilot 1302596; same
# 4 configs traj 300/400/500/600 so per-config target sums assemble):
#   ELO/EHI: the segment; NA: anchor N_NOISE (x4 comb solves);
#   NT: TSRC-ratio N_NOISE (x1) at fixed eta=EHI.
# sbatch --export=ALL,ELO=0.2,EHI=0.4,NA=625,NT=500  (~24 h)
# sbatch --export=ALL,ELO=0.4,EHI=0.6,NA=312,NT=250  (~20 h; half stats
#   -- the smoke's job is the measured sigma/wallclock table that sets
#   the real stats adjustment, not full precision)
# Full locked stack: comb-4 + Neyman + tol 1e-4 + MG-lite L1 + REUSE_NV.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
ELO=${ELO:-0.2}
EHI=${EHI:-0.4}
NA=${NA:-625}
NT=${NT:-500}          # 0 = skip the TSRC half (anchor-only smoke)
PROF=${PROF:-cos}      # profile A/B smokes (e.g. tent: subcritical at all
                       # production eta -- the ESS/config-variance
                       # head-to-head vs cos's supercritical segments)
SEEDBASE=${SEEDBASE:-400}   # distinct per submission (multi-job averaging)
DEFL=${DEFL:-0}             # oblique-projector deflation nev (0 = off);
                            # when >0 the certified eigensolve settings
                            # (ord=1001 tol=1e-5 lo=3e-5) are forced.
RUNTAG=${RUNTAG:-}          # optional log-name suffix: REQUIRED for any
                            # rerun of an existing (segment, profile) --
                            # identical tags OVERWRITE prior logs.
SLATT=${SLATT:-16.16.16.48} # lattice (4.4.4.4 for gate-scale arms)
SCFG=${SCFG:-}              # cfg dir override (default = lam5 production)
TAG=$(echo "seg${ELO}_${EHI}_${PROF}${RUNTAG}" | tr -d .)
rm -f $LOGD/rwf_smoke_$TAG.DONE
STRAJS=${STRAJS:-"300 400 500 600"}
GPU=0
for TRAJ in $STRAJS; do
  (
    srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
      env CUDA_VISIBLE_DEVICES=$GPU LATT=$SLATT TRAJ=$TRAJ \
          LAMBDA_DTXQCD=5 \
          CFG_DIR=${SCFG:-$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw} \
          ETA=$EHI ETA_LO=$ELO N_GL=2 N_NOISE=$NA \
          NOISE_SEED=$((SEEDBASE + TRAJ)) \
          PROFILE=$PROF PROF_T0=0 DTXQCD_STOCHPF_ANTI=0 \
          DTXQCD_STOCHPF_DEFLATE=$DEFL \
          DTXQCD_DEFORM_EO_DEFLATE_ORD=1001 \
          DTXQCD_DEFORM_EO_DEFLATE_TOL=1e-5 \
          DTXQCD_DEFORM_EO_DEFLATE_LO=${SLO:-3e-5} \
          DTXQCD_DEFORM_EO_DEFLATE_CACHE=$P/defl_cache \
          DTXQCD_STOCHPF_DEFLATE_RANDOM=${SRAND:-0} \
          DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
          DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 DTXQCD_MG_REUSE_NV=1 \
      $P/Test_dtxqcd_eta_stoch_pf --grid $SLATT --mpi 1.1.1.1 \
      --shm 512 --shm-mpi 1 > $LOGD/rwf_smoke_${TAG}_tr${TRAJ}_anchor.log 2>&1
    if [ "$NT" -gt 0 ]; then
    srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
      env CUDA_VISIBLE_DEVICES=$GPU LATT=$SLATT TRAJ=$TRAJ \
          LAMBDA_DTXQCD=5 \
          CFG_DIR=${SCFG:-$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw} \
          ETA=$EHI N_NOISE=$NT NOISE_SEED=$((SEEDBASE + 100 + TRAJ)) \
          PROFILE=$PROF PROF_T0=0 \
          DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
          DTXQCD_STOCHPF_TSRC_RATIO=12 \
          DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 DTXQCD_MG_REUSE_NV=1 \
      $P/Test_dtxqcd_eta_stoch_pf --grid $SLATT --mpi 1.1.1.1 \
      --shm 512 --shm-mpi 1 > $LOGD/rwf_smoke_${TAG}_tr${TRAJ}_tsrc.log 2>&1
    fi
  ) &
  GPU=$((GPU+1))
done
wait
for TRAJ in $STRAJS; do
  echo "== traj $TRAJ =="
  grep -aE "STOCHPF" $LOGD/rwf_smoke_${TAG}_tr${TRAJ}_anchor.log | tail -1
  grep -aE "TSRCCLOSURE" $LOGD/rwf_smoke_${TAG}_tr${TRAJ}_tsrc.log | tail -1
done
touch $LOGD/rwf_smoke_$TAG.DONE
