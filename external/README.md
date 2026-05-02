# external/ — third-party builds for Grid-TXQCD

Tracked: `cpm/` (CMake CPM module cache).
Gitignored: `quda-build/`, `quda-install/`, build logs, `eigen/` (cmake-fetched).

## QUDA from chroma's source

The default Grid build links against `/home/agrebe/install/quda-5/lib/libquda.so`,
which is a stock-ish upstream QUDA. That QUDA has a bug in
`computeCloverForceQuda`: `qParam.x[0] /= 2` is mutated *inside* the loop over
`nvector`, so iteration k wraps the host X buffer at x[0]/2^(k+1). For Phase 6
Path B (QUDA-computed gauge force) we need a fork without that bug.

`external/quda-install/` is built from
`/lustre2/nplqcd/chroma/source_jul_2025/quda` — chroma's QUDA fork, which
constructs a fresh `ColorSpinorParam` per loop iteration and is therefore
unaffected by the upstream bug.

To rebuild (takes ~25 min on lq2):

```bash
source ./env_lq2_grid.sh
mkdir -p external/quda-build external/quda-install
export CPM_SOURCE_CACHE=/lustre2/nplqcd/Grid-TXQCD/external
cd external/quda-build
cmake \
  -DCMAKE_INSTALL_PREFIX=$PWD/../quda-install \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DQUDA_GPU_ARCH=sm_80 \
  -DQUDA_BUILD_SHAREDLIB=ON \
  -DQUDA_QMP=ON \
  -DQUDA_MPI=OFF \
  -DQUDA_QDPJIT=OFF \
  -DQUDA_INTERFACE_QDP=ON \
  -DQUDA_INTERFACE_QDPJIT=OFF \
  -DQUDA_INTERFACE_MILC=ON \
  -DQUDA_DIRAC_WILSON=ON \
  -DQUDA_DIRAC_CLOVER=ON \
  -DQUDA_MULTIGRID=ON \
  -DQUDA_PRECISION=14 \
  -DQUDA_RECONSTRUCT=7 \
  -DQMP_DIR=/lustre2/nplqcd/chroma/install_oct_2025/qmp-cmake/lib/cmake/QMP \
  /lustre2/nplqcd/chroma/source_jul_2025/quda
make -j 16 install
```

To link Grid against this QUDA instead of agrebe's:

```bash
cd production/
QUDA_PREFIX=/lustre2/nplqcd/Grid-TXQCD/external/quda-install \
QMP_PREFIX=/lustre2/nplqcd/chroma/install_oct_2025/qmp-cmake \
make -B gen_qcd_cfgs
```

## Status (2026-05-02)

The chroma-source QUDA links and runs successfully for the multishift CG
(`solve_rb_even` produces correct X), but its `computeCloverForceQuda` and
`computeTMCloverForceQuda` both produce all-NaN mom output in our calling
context. The chroma test suite passes the same routines, so something in our
setup differs that we have not yet isolated. Working theories include:
- gauge_param differences between our setup and chroma test
- QUDA's verbose log shows "Refining shift K: L2 residual inf" during the
  QUDA-side multishift cleanup pass (even though final X comes out correct),
  hinting at a numerical issue we may be papering over with the Grid-side
  cleanup CG.

Production currently uses agrebe's QUDA + the per-pole `nvector=1` workaround
for Path B; this is a real upstream bug and should be filed against QUDA.
