#!/bin/bash
# Deeper smearing ladder: N=30 (continuity) / 60 / 100 / 200.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh > /dev/null 2>&1
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
OUT=$P/meas_2pt/dtxqcd_lam5_dibaryon_aux_sm2
mkdir -p "$OUT"
n=0
for t in $(ls -v "$CFG"/ckpoint_lat.* 2>/dev/null | grep -v daux | grep -v rng | sed 's/.*\.//'); do
  [ "$t" -gt 20 ] 2>/dev/null || continue
  [ -f "$OUT/dibaryon_aux_dtxqcd_${t}.h5" ] && continue
  env LATT=16.16.16.48 CFG_DIR=$CFG DATA_DIR=$OUT LAMBDA_DTXQCD=5 \
      SMEARS="30,60,100,200" OMP_NUM_THREADS=16 \
    $P/meas_dibaryon_aux_dtxqcd "$t" --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > "$OUT/dib.$t.log" 2>&1
  n=$((n+1))
done
echo "measured $n new configs; total h5: $(ls $OUT/*.h5 2>/dev/null | wc -l)"
