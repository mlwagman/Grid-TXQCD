# Grid-TXQCD environment — NERSC Perlmutter (A100 GPU stack), 2026 toolchain.
#
# Replaces systems/Perlmutter/sourceme.sh, whose `cudatoolkit/11.4` is stale
# (current default is 12.9). Sourced at build time AND by the HMC sbatch on
# compute nodes (smoke.sbatch / tune.sbatch source $GRID_TXQCD/sourceme.sh).
#
# GMP: complete in the system default paths (/usr/include/gmp.h,
#      /usr/lib64/libgmp.so) — found automatically, no configure flag.
# MPFR: Grid's RHMC Remez requires it, but /usr ships no dev header. The Cray PE
#      gcc tree has a complete copy (header + libmpfr.so.4); point configure at it
#      with --with-mpfr=$GRID_MPFR_PREFIX, and make its lib dir resolvable at
#      runtime (the HMC binary links libmpfr.so.4 dynamically) via LD_LIBRARY_PATH.

export CRAY_ACCEL_TARGET=nvidia80

module load PrgEnv-gnu
module load cudatoolkit/12.9
module load craype-accel-nvidia80

export GRID_MPFR_PREFIX=/opt/cray/pe/gcc/mpfr/3.1.4
export LD_LIBRARY_PATH="${GRID_MPFR_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
