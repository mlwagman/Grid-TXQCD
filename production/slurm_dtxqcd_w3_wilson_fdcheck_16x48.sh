#!/bin/bash
#SBATCH --job-name=dtx_w3_fd
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --output=slurm-logs/dtx_w3_fd.%j.out
# Phase W.3: full QUDA Wilson hop force convention closed at 4⁴ and 8⁴
# (cos=1, |B|/|A|=1, off_scale_wilson=-2.0).  This script validates the
# convention scales to production 16³×48 4-GPU geometry.
#
# Path A: parent DTXQCDWilsonCloverRationalFullAction (Grid DhopDeriv)
# Path B: child DTXQCDWilsonCloverRationalFullActionQudaPrimitive
#         (off_scale_wilson=-2.0 default via the W.3-closed header).
#
# Verdict line: [QUDA-WILSON-W.1 PASS] iff all per-μ cos=1 and ratio=1.
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
mkdir -p /lustre2/nplqcd/cache/quda_resource_cuda12p2
nvidia-smi --query-gpu=index,name --format=csv,noheader

export QUDA_RESOURCE_PATH=/lustre2/nplqcd/cache/quda_resource_cuda12p2
export LD_LIBRARY_PATH=/lustre2/nplqcd/install/quda_cuda12p2/lib:${LD_LIBRARY_PATH:-}
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_MANAGED_MEMORY=1

# rat_degree=3 keeps the multishift CG cost modest at 16³×48 random gauge.
DTXQCD_TEST_RAT_DEGREE=3 \
  srun --mpi=pmix -N 1 -n 4 --cpu-bind=none \
    /lustre2/nplqcd/Grid-DTXQCD/production/Test_dtxqcd_wilson_hop_full_quda \
      --grid 16.16.16.48 --mpi 1.1.1.4 --shm 1024 --shm-mpi 1
echo "=== W.3 FD check 16³×48 4-GPU exited rc=$? $(date) ==="
