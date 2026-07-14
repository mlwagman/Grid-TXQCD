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
#  ifdef GRID_QUDA_USE_QMP
#    include <qmp.h>   // only the legacy QMP-comms build needs the QMP headers
#  endif
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
#ifdef GRID_HAVE_QUDA
// QUDA process-coordinate -> Grid MPI rank, so QUDA's multi-GPU halo exchange uses
// the SAME rank layout as Grid.  fdata = the CartesianCommunicator*.  Without this
// (initCommsGridQuda(...,nullptr,...)), QUDA picks its own coord->rank order, which
// disagrees with Grid's (MPI_Cart_create + OptimalCommunicator remap) for >=2
// partitioned directions -> wrong-neighbor halos -> few-% operator error.
inline int grid_rank_from_coords_(const int *coords, void *fdata) {
  auto *comm = static_cast<CartesianCommunicator *>(fdata);
  Coordinate c(4);
  bool rev = std::getenv("QUDA_GRID_RANKMAP_REV") != nullptr;  // coord-order toggle (test)
  for (int d = 0; d < 4; ++d) c[d] = rev ? coords[3 - d] : coords[d];
  return comm->RankFromProcessorCoor(c);
}
#endif

inline void initialize(int device = -1, const int *mpi_dims = nullptr,
                       CartesianCommunicator *comm = nullptr) {
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

#ifdef GRID_QUDA_USE_QMP
  // ===== Legacy QMP-comms build =====
  // QMP's declared logical topology governs QUDA's rank layout.  It CANNOT
  // reproduce Grid's NUMA-remapped MPI rank order (OptimalCommunicator), so it is
  // only exact for <=1 partitioned direction.  The MPI branch below is the correct
  // multi-direction path; this is kept only for the old QMP install.
  //
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

  // (QUDA_GRID_RANKMAP callback is inert under QMP — the QMP topology wins — but
  // kept here so the env knob is harmless on this build.)
  bool use_grid_map = (comm != nullptr) && (std::getenv("QUDA_GRID_RANKMAP") != nullptr);
  QudaCommsMap rfc = use_grid_map ? &grid_rank_from_coords_ : nullptr;
  void *rfc_data   = use_grid_map ? (void *)comm : nullptr;
  initCommsGridQuda(4, dims, rfc, rfc_data);
  if (use_grid_map)
    std::cout << GridLogMessage << "[Grid::Quda] using Grid rank map (QUDA_GRID_RANKMAP"
              << (std::getenv("QUDA_GRID_RANKMAP_REV") ? "+REV" : "") << ")" << std::endl;
#else
  // ===== MPI-comms build (default; the rank-map fix) =====
  // Hand QUDA *Grid's own* MPI communicator and coord->rank map so QUDA's
  // multi-GPU halo exchange uses EXACTLY Grid's NUMA-remapped rank layout.  No QMP
  // topology in the loop => >=2 partitioned directions are correct.
  if (comm != nullptr) {
    // Share Grid's communicator: QUDA rank space == Grid rank space.
    setMPICommHandleQuda((void *)&comm->communicator);
    // Per-coord rank via Grid's RankFromProcessorCoor (carries the remap).
    initCommsGridQuda(4, dims, &grid_rank_from_coords_, (void *)comm);
    std::cout << GridLogMessage
              << "[Grid::Quda] MPI comms: using Grid communicator + rank map"
              << (std::getenv("QUDA_GRID_RANKMAP_REV") ? " (coord REV)" : "")
              << std::endl;
  } else {
    // No comm passed.  QUDA's default lexicographic map is correct only for <=1
    // partitioned direction; for >=2 it disagrees with Grid's NUMA-remapped ranks
    // (few-% halo error).  Warn loudly so a multi-GPU caller that forgot to pass
    // the gauge grid is never *silently* wrong.
    int npart = 0;
    for (int d = 0; d < 4; ++d) if (dims[d] > 1) ++npart;
    if (npart >= 2)
      std::cout << GridLogWarning
                << "[Grid::Quda] initialize() called with NO communicator but "
                << npart << " partitioned directions -> QUDA rank map will disagree "
                   "with Grid (few-% operator/force error).  Pass the gauge grid: "
                   "Quda::initialize(-1, nullptr, U.Grid())." << std::endl;
    initCommsGridQuda(4, dims, nullptr, nullptr);
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
