#!/bin/bash
# QUDA-on-setdevice=no wrapper.  Keeps Grid's PROVEN per-rank comms model
# ("device 0" == the rank's GPU, nvlink-shm via IPC) but exposes ALL the gres
# GPUs to every rank -- the rank's GPU FIRST -- so:
#   - Grid (--enable-setdevice=no) uses the default device 0 == first entry ==
#     the rank's physical GPU (identical binding to srun_gpu_wrapper.sh);
#   - QUDA's comm_init sees cudaGetDeviceCount() == ngpu >= ranks_per_node
#     (no "Too few GPUs" abort), and QudaInit binds QUDA to cudaGetDevice()
#     (== device 0 == the rank's GPU), consistent with Grid.
# Reorders the SLURM-provided CUDA_VISIBLE_DEVICES (e.g. "0,1,2,3") to put
# G[LOCAL_RANK] first.
LOCAL_RANK=${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
ORIG=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
IFS=',' read -ra G <<< "$ORIG"
N=${#G[@]}
SEL=${G[$LOCAL_RANK]}
LIST=$SEL
for ((i=0; i<N; i++)); do
  [ "$i" != "$LOCAL_RANK" ] && LIST="$LIST,${G[$i]}"
done
export CUDA_VISIBLE_DEVICES=$LIST
exec "$@"
