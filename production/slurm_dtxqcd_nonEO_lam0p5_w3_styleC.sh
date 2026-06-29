#!/bin/bash
# Phase W.3 default-flip 16³×48 1-trajectory smoke at λ=0.5 production
# settings.  Apples-to-apples vs the Style C baseline.
#
# Baseline (slurm_dtxqcd_nonEO_lam0p5_msshift_styleC.sh, Style C alone):
#   λ=0.5, MDS=10, trajL=√2/8, MN2, gauge×8 aux×1, mpi=1.1.1.4
#   Style C steady-state ≈ 87.4 s/substep → ≈ 29.1 min/traj
#
# W.3 (USE_FULL_PF_QUDA_WILSON=1, now DEFAULT) replaces Grid DhopDeriv with
# QUDA's native computeCloverWilsonForceWithSchurFields for the Wilson hop.
# Force-side validated bit-exact at 4⁴, 8⁴, 16³×48 4-GPU (cos=1.0, |B|/|A|=1.0).
#
# Suffix _w3_styleC_lam0p5_deg8 keeps W.3 cfgs separate from the Style C
# baseline directory to avoid clobbering.

mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
mkdir -p /lustre2/nplqcd/cache/quda_resource_cuda12p2

export QUDA_RESOURCE_PATH=/lustre2/nplqcd/cache/quda_resource_cuda12p2
export LD_LIBRARY_PATH=/lustre2/nplqcd/install/quda_cuda12p2/lib:${LD_LIBRARY_PATH:-}

LAMBDA_DTXQCD=0.5 USE_FULL_PF=1 MDSTEPS=10 TRAJL=0.176776695296637 \
GAUGE_MULT=8 AUX_MULT=1 \
RHMC_DEG=8 CG_TOL=1e-8 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 DTXQCD_SUFFIX="_w3_styleC_lam0p5_deg8" \
TRAJ=1 NO_METROP=1 N_SKIP=1 \
DTXQCD_MULTISHIFT_QUDA=1 DTXQCD_MULTISHIFT_QUDA_STAGE=B \
DTXQCD_STAGEB_STYLE=C \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== W.3 default-flip lam0p5 1-traj exited rc=$? $(date) ==="
