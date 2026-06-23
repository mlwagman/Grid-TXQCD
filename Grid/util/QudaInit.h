#pragma once
// Grid + QUDA integration: minimal init/finalize wrapper.
//
// Compiled in only when configure was run with --with-quda=PATH (the
// configure flag sets -DGRID_HAVE_QUDA).  When absent, the Quda namespace
// stubs return without doing anything so callers can use unconditional
// invocations.

#include <Grid/GridCore.h>

#ifdef GRID_HAVE_QUDA
#  include <quda.h>
#  include <qmp.h>
#  ifdef GRID_CUDA
#    include <cuda_runtime_api.h>  // cudaSetDevice/cudaGetDevice/cudaGetDeviceCount
#  endif
#  include <cstdlib>
#endif

NAMESPACE_BEGIN(Grid);
namespace Quda {

// Idempotent: safe to call multiple times.  Initializes QUDA on the GPU
// associated with the current MPI rank.  In a multi-rank setup, callers
// must already have CUDA_VISIBLE_DEVICES configured per-rank (which our
// production slurm scripts do).
//
// Order matters per QUDA docs: QMP_init_msg_passing →
// QMP_declare_logical_topology_map → initCommsGridQuda → initQuda.
// QMP's MPI backend tolerates a pre-existing MPI_Init (Grid has already
// called MPI_Init by the time Grid_init returns) and just queries the
// thread level instead of re-initializing.
inline void initialize(int device = -1, const int *mpi_dims = nullptr) {
#ifdef GRID_HAVE_QUDA
  static bool initialized = false;
  if (initialized) return;
  // If caller didn't tell us the layout, read it from Grid (Grid_init must
  // have run already; GridDefaultMpi() returns the --mpi=Lx.Ly.Lz.Lt grid).
  int detected_dims[4] = {1, 1, 1, 1};
  if (!mpi_dims) {
    Coordinate g = GridDefaultMpi();
    for (int d = 0; d < 4 && d < (int)g.size(); ++d) detected_dims[d] = g[d];
  }
  const int *dims = mpi_dims ? mpi_dims : detected_dims;

  // QMP init — must precede initCommsGridQuda when QUDA is built with
  // QMP_COMMS (libquda links libqmp).  Coexists with Grid's MPI_Init.
  int dummy_argc = 0;
  char **dummy_argv = nullptr;
  QMP_thread_level_t tl_provided;
  QMP_init_msg_passing(&dummy_argc, &dummy_argv, QMP_THREAD_SINGLE, &tl_provided);
  // QUDA's reference uses map={3,2,1,0} for column-major (t fastest) ordering.
  int map[4] = {3, 2, 1, 0};
  // QMP_declare_logical_topology_map takes a non-const dims pointer.
  int dims_mut[4] = {dims[0], dims[1], dims[2], dims[3]};
  QMP_declare_logical_topology_map(dims_mut, 4, map, 4);

  initCommsGridQuda(4, dims, /*rank_from_coords=*/nullptr, /*fdata=*/nullptr);

  // --- Device binding (multi-rank crux) -----------------------------------
  // Grid (configured --enable-setdevice=no, GRID_DEFAULT_GPU) calls
  // cudaSetDevice(0) on every rank, so Grid's streams + cuBLAS handles + all
  // lattice data live on whatever device is "device 0" for that rank.  QUDA's
  // initQuda(-1) then does its OWN default allocation (comm_gpuid = rank%ngpu),
  // so when all GPUs are visible (the QUDA strange/HMC launch, which needs all
  // GPUs visible for initCommsGridQuda) rank>0 binds QUDA to a DIFFERENT device
  // than Grid is on.  Control returns to Grid with the current device changed,
  // and Grid's next stream op faults: "Cuda error invalid resource handle".
  //
  // Fix: bind QUDA to the device GRID IS ALREADY ON (cudaGetDevice()), not
  // QUDA's rank%ngpu guess.  Then Grid and QUDA always agree, regardless of the
  // --enable-setdevice build flag.  Per-GPU work distribution is the LAUNCH
  // WRAPPER's job (srun_gpu_wrapper.sh sets CUDA_VISIBLE_DEVICES=$LOCAL_RANK, so
  // each rank's "device 0" is a distinct physical GPU — Grid AND QUDA both land
  // on it).  This is exactly TXQCD's --enable-setdevice=yes behaviour (rank N on
  // device N) reproduced for the setdevice=no DTXQCD build via the wrapper, with
  // QUDA no longer fighting Grid for the device.  Explicit override: QUDA_DEVICE.
  //   * 1 GPU/rank visible (srun_gpu_wrapper.sh): cudaGetDevice()==0 == the
  //     rank's only/physical GPU; QUDA binds there too. Correct multi-GPU.
  //   * all GPUs visible, no wrapper: every rank's cudaGetDevice()==0; QUDA also
  //     binds device 0 — consistent (no fault), though all ranks share GPU 0.
#ifdef GRID_CUDA
  if (device < 0) {
    if (const char *e = std::getenv("QUDA_DEVICE"); e && *e) {
      device = std::atoi(e);
      cudaSetDevice(device);
    } else {
      // Adopt Grid's current device so QUDA and Grid never disagree.
      cudaGetDevice(&device);
    }
    int cur = -1;
    cudaGetDevice(&cur);
    std::cout << GridLogMessage << "[Grid::Quda] binding rank "
              << GlobalSharedMemory::WorldRank << " (shmRank "
              << GlobalSharedMemory::WorldShmRank << ") to CUDA device "
              << device << " (== Grid's current device " << cur << ")"
              << std::endl;
  }
#endif

  initQuda(device);
  initialized = true;
  std::cout << GridLogMessage << "[Grid::Quda] initialized (device=" << device
            << ", grid=" << dims[0] << "." << dims[1] << "."
            << dims[2] << "." << dims[3] << ")" << std::endl;
#else
  std::cout << GridLogMessage << "[Grid::Quda] STUB — built without QUDA "
            << "(rebuild with --with-quda=PATH)" << std::endl;
#endif
}

inline void finalize() {
#ifdef GRID_HAVE_QUDA
  endQuda();
  // Don't call QMP_finalize_msg_passing — that calls MPI_Finalize, which
  // Grid_finalize will also do.  Let Grid own MPI shutdown.
  std::cout << GridLogMessage << "[Grid::Quda] finalized" << std::endl;
#endif
}

inline bool available() {
#ifdef GRID_HAVE_QUDA
  return true;
#else
  return false;
#endif
}

}  // namespace Quda
NAMESPACE_END(Grid);
