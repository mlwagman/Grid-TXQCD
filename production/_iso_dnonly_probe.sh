#!/bin/bash
# d/n-ONLY isospin contour probe (mwagman): grade only the diquark
# cross-blocks, leave mesons undeformed.  Compare darg (Pf phase) vs the
# FULL isospin contour to see if the meson off-diagonal spurions were the
# phase source.  Runs on the interactive alloc (jobid 1326245, lq2gpu06).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh > /dev/null 2>&1
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5
JID=${JID:-1326245}
run () {  # run <dn_only> <eta> <derivcheck> <ward> <tag>
  srun --overlap --jobid=$JID --mpi=pmix -N 1 -n 1 --cpu-bind=none --gres=gpu:1 \
    env LATT=4.4.4.4 CFG_DIR=$CFG TRAJ=400 LAMBDA_DTXQCD=5 \
        ISO_DN_ONLY=$1 ETA=$2 PROFILE=tent DERIVCHECK=$3 WARDCHECK=$4 \
        N_GL=4 N_NOISE=32 NOISE_SEED=808 OMP_NUM_THREADS=16 \
    $P/Test_dtxqcd_isodeform_pf --grid 4.4.4.4 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 2>&1 \
    | grep -aE "ISO-RESULT|ISO-WARDCHECK|ISO-DERIVCHECK|NOT CONV" \
    | sed "s/^/[dn_only=$1 eta=$2 ward=$4] /"
}
echo "=== d/n-only contour: DERIVCHECK + eta scan ==="
run 1 0.4 1 0 dn
run 1 0.6 0 0 dn
echo "=== d/n-only UNIFORM (Ward gate -- NOT expected 0 for partial contour) ==="
run 1 0.5 0 1 dn_ward
echo "=== FULL isospin contour (reference, same seed) ==="
run 0 0.4 0 0 full
run 0 0.6 0 0 full
