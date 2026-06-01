#!/bin/bash
# Unified environment for the Grid-TXQCD GPU build and production runs on lq2.
# Source once per shell or at the top of a SLURM job script:
#   source env_lq2_grid.sh
# Loads: GCC 12.3, OpenMPI (via gompi), UCX/UCC for CUDA, CUDA toolkit, HDF5
# (required by all Grid-TXQCD measurement + checkpoint I/O).

export https_proxy=http://squid.fnal.gov:3128
export http_proxy=http://squid.fnal.gov:3128

# Ensure `module` is available inside non-interactive shells (SLURM jobs).
if ! command -v module >/dev/null 2>&1; then
  if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
  fi
fi

module load cmake gompi ucx_cuda ucc_cuda gcc/12.3.0
module load hdf5/1.14.2_gompi_2023a

CUDATOOLKIT_HOME=/srv/software/el8/x86_64/hpc/nvhpc/Linux_x86_64/23.7/cuda/12.2

export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_MANAGED_MEMORY=1

export LD_LIBRARY_PATH=/srv/software/el8/x86_64/hpc/nvhpc/Linux_x86_64/23.7/compilers/lib/:/lustre2/nplqcd/install/Python-3.12.2/lib/:/srv/software/el8/x86_64/eb/GCCcore/12.3.0/lib64/${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=/lustre2/nplqcd/install/Python-3.12.2/bin/:/lustre1/nplqcd/install/Python-3.12.2/include/:$PATH

# Canonical paths.  Set ONCE here so production SLURM scripts can do
#   source <abs-path-to>/env_lq2_grid.sh
#   cd "$PRODUCTION_DIR"
# and not care about sbatch cwd, SLURM_SUBMIT_DIR quirks, or BASH_SOURCE
# under SLURM script staging. See production/slurm_b6p5_lam8_mds10_2node_noquda.sh
# for the canonical template.
export GRID_TXQCD_ROOT=/lustre2/nplqcd/Grid-TXQCD
export PRODUCTION_DIR=$GRID_TXQCD_ROOT/production

export CUDA_CACHE_PATH=/lustre2/nplqcd/cache
# QUDA persists kernel autotuning to disk when this is set — saves ~10-30s
# on first force/inverter calls per run. Critical for MG solvers (lighter
# quarks where MG setup is slow). Cache is per-(architecture, gauge geometry,
# kappa, csw, etc.) and re-tunes only when those change.
export QUDA_RESOURCE_PATH=/lustre2/nplqcd/cache/quda_resource

# Emit module list and paths when sourced interactively, keep quiet in jobs.
if [ -t 1 ]; then
  module list 2>&1
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
fi
