# external/ — third-party builds for Grid-TXQCD

Tracked: `cpm/` (CMake CPM module cache).
Gitignored: `quda-build/`, `quda-install/`, build logs, `eigen/` (cmake-fetched).

## QUDA from chroma's source

The default Grid build links against `/home/agrebe/install/quda-5/lib/libquda.so`,
which is a stock-ish upstream QUDA. That QUDA has a bug in
`computeCloverForceQuda`: `qParam.x[0] /= 2` is mutated *inside* the loop over
`nvector`, so iteration k wraps the host X buffer at x[0]/2^(k+1). For Phase 6
Path B (QUDA-computed gauge force) we need a fork without that bug.

`external/quda-install/` is built from `external/quda-src/`, a clone of
the latest upstream `lattice/quda` `develop` branch (commit `9d987974`,
2026-04-27). The clone is gitignored — to re-create:

```bash
git clone https://github.com/lattice/quda.git external/quda-src
```

The fix landed via PR #1339 / #1338 in late 2023, so any QUDA newer than
~Dec 2023 has the right `ColorSpinorParam` per-iteration pattern. The
fact that we hit the bug means we were linking against
`/home/agrebe/install/quda-5/lib/libquda.so`, which was built from a
December 2023 snapshot (pre-fix). Note: `lattice/quda/source_jul_2025` in
chroma's tree is just an upstream snapshot, not a chroma-specific fork.

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

Latest upstream QUDA links and runs cleanly. `computeCloverForceQuda` returns
finite mom values (no NaN, no `1e+274` garbage). The earlier all-NaN we saw
when building from `chroma/source_jul_2025/quda` was a transient state from
that snapshot — current upstream is fine.

Single fused `nvector=Npole` call works (no need for the per-pole loop
workaround needed against agrebe's old build). The action class still
provides `QUDA_FORCE_DBG_NVEC1=1` to opt back into the per-pole loop for
older QUDA installs.

**Open issue:** PathB and PathA forces are not parallel — cos θ ≈ -0.5,
empirical projection factor ≈ -4.36 (close to but not exactly `-1/(8κ²) =
-7.05`). This is a structural mismatch in our unpacking / sign convention,
not an upstream bug. See memory `project_quda_path_b_blocker.md` for the
diagnostic toggles available (`QUDA_FORCE_KERNEL_COMPARE`,
`QUDA_FORCE_DBG_XZERO`, `QUDA_FORCE_DBG_NOTRACE`).
