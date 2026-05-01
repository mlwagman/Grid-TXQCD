// Phase 1 smoke test: confirm Grid links against QUDA and QUDA initializes.
//
// Build (after configuring Grid with --with-quda=$QUDA_PREFIX):
//   make test_quda_init
// Run:
//   CUDA_VISIBLE_DEVICES=0 ./test_quda_init --grid 4.4.4.8 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/util/QudaInit.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::cout << GridLogMessage << "QUDA available at compile time: "
            << (Quda::available() ? "YES" : "NO") << std::endl;

  Quda::initialize(/*device=*/-1);

#ifdef GRID_HAVE_QUDA
  // QUDA version (from quda_constants.h: MAJOR.MINOR.SUBMINOR).
  std::cout << GridLogMessage << "QUDA version = "
            << QUDA_VERSION_MAJOR << "." << QUDA_VERSION_MINOR
            << "." << QUDA_VERSION_SUBMINOR << std::endl;
#endif

  Quda::finalize();
  Grid_finalize();
  return 0;
}
