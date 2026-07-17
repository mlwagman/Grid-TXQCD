#!/bin/bash
#SBATCH --job-name=stochpf_schur
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/stochpf_schur.%j.out
# ============================================================================
# 4^4 gate for the SCHUR/oblique-projector/dilution estimator upgrade.
# eta=0.8 with the 1-point GL rule puts the single node at ep=0.4, where
# the dense study banked the EXACT trace:
#     Tr[S^-1 dS](ep=0.4) = -0.0356616 - 0.110153 i   (job 1302016/30)
# Arms (same NOISE_SEED, N_NOISE=64):
#   R: legacy full-grid estimator        (bias + baseline sigma)
#   S: SCHUR=1                            (form change alone)
#   D: SCHUR=1 DEFLATE=16                 (oblique projector, bias-free?)
#   T: SCHUR=1 DEFLATE=16 DILUTE_EOT=1    (the full stack)
# PASS = all four GLNODE means consistent with the dense value within
# errors; the sigma column is the first measured 4^4 win table.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/stochpf_schur.DONE
for ARM in R S D T; do
  case $ARM in
    R) EX="DTXQCD_STOCHPF_SCHUR=0" ;;
    S) EX="DTXQCD_STOCHPF_SCHUR=1" ;;
    # 4^4 Chebyshev-window override: the production LO=3e-4 fraction sits
    # BELOW this volume's bandless spectrum (lambda_min ~ 0.55, lmax ~ 51)
    # -> IRL 0/16 (the known window-edge trap).  0.02*51 ~ 1.0 sits above
    # the 16 wanted modes.
    D) EX="DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_DEFLATE=16 DTXQCD_DEFORM_EO_DEFLATE_LO=0.02" ;;
    T) EX="DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_DEFLATE=16 DTXQCD_DEFORM_EO_DEFLATE_LO=0.02 DTXQCD_STOCHPF_DILUTE_EOT=1" ;;
  esac
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=4.4.4.4 TRAJ=300 LAMBDA_DTXQCD=5 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5 \
        ETA=0.8 N_GL=1 N_NOISE=64 NOISE_SEED=77 \
        PROFILE=cos PROF_T0=0 DTXQCD_DEFORM_MG=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 $EX \
    $P/Test_dtxqcd_eta_stoch_pf --grid 4.4.4.4 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/stochpf_schur_$ARM.log 2>&1
  echo "== arm $ARM =="
  grep -aE "stochpf-defl|GLNODE" $LOGD/stochpf_schur_$ARM.log | tail -2
done
touch $LOGD/stochpf_schur.DONE
