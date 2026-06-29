#!/bin/bash
# Style B Phase 2.5 Style C 16³×48 1-trajectory smoke.
#
# Stage B inner loop using DTXQCDMultiShiftCGQUDA_StyleC: all CG state lives on
# native ColorSpinorField (FLOAT2+UKQCD), inner mat-vec via M_device_csf
# (FloatNOrder accessors + persistent Dirac), inner blas via quda::blas::*
# (proper device tree reductions).  Layout convert ONLY at CG entry/exit.
#
# Aux uses the same Option β device path automatically inside M_device_csf
# (no separate AUX_OPTION env needed — Style C unconditionally uses the
# native FloatNOrder aux kernel).
#
# Comparison:
#   Production (1289218 substep-1):    80.27 s
#   Stage A (M-wrap.6):                79.73 s
#   Stage B Option β (2026-06-27):    228 s   (per-call MatQuda + host aux)
#   Stage B Style B Phase 2 (Sun):    ~460 s  (per-call csf.copy overhead)
#   Style C target:                   ≤ 65 s  (stretch ≤ 50 s)
#                                            All inner kernels on native CSF;
#                                            csf.copy only at entry/exit.
#
# Runtime env:
#   QUDA_RESOURCE_PATH=/lustre2/nplqcd/cache/quda_resource_cuda12p2
#       (version-keyed tunecache for the new CUDA-12.2 QUDA install)
#   LD_LIBRARY_PATH includes /lustre2/nplqcd/install/quda_cuda12p2/lib/
#       (binary's rpath should pick it up — exported here as a safety belt)

mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
mkdir -p /lustre2/nplqcd/cache/quda_resource_cuda12p2

export QUDA_RESOURCE_PATH=/lustre2/nplqcd/cache/quda_resource_cuda12p2
export LD_LIBRARY_PATH=/lustre2/nplqcd/install/quda_cuda12p2/lib:${LD_LIBRARY_PATH:-}

LAMBDA_DTXQCD=3.0 USE_FULL_PF=1 MDSTEPS=30 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 DTXQCD_SUFFIX="_msshift_styleC" \
TRAJ=1 NO_METROP=1 N_SKIP=1 \
DTXQCD_MULTISHIFT_QUDA=1 DTXQCD_MULTISHIFT_QUDA_STAGE=B \
DTXQCD_STAGEB_STYLE=C \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== Style C Phase 2.5 lam3 1-traj exited rc=$? $(date) ==="
