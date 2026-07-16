#!/bin/bash
# Wait for the w12/eta=1.2 MG run (GPUs 2/3) to finish, then fire the
# cosine eta=0.8 MG probe on the same GPUs.
L=/lustre2/nplqcd/Grid-DTXQCD/production/mtilde_runs/mg_smooth_w12_e12.log
until grep -aqE "PASSED|FAILED|Aborted" "$L" 2>/dev/null; do sleep 30; done
sleep 10
env PROFILE=cos ETA=0.8 GPUS=2,3 \
  bash /lustre2/nplqcd/Grid-DTXQCD/production/mtilde_runs/run_mg_smooth.sh
