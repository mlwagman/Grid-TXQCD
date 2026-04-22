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

export LD_LIBRARY_PATH=/srv/software/el8/x86_64/hpc/nvhpc/Linux_x86_64/23.7/compilers/lib/:/lustre2/nplqcd/install/Python-3.12.2/lib/:/srv/software/el8/x86_64/eb/GCCcore/12.3.0/lib64/${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=/lustre2/nplqcd/install/Python-3.12.2/bin/:/lustre1/nplqcd/install/Python-3.12.2/include/:$PATH

export CUDA_CACHE_PATH=/lustre2/nplqcd/cache

# Emit module list and paths when sourced interactively, keep quiet in jobs.
if [ -t 1 ]; then
  module list 2>&1
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
fi
