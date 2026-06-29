# Default to build-gpu (production cluster); local laptop builds typically
# live in ../build instead.  Override on the command line:
#   make GRID_BUILD_DIR=../build gen_dtxqcd_cfgs
GRID_BUILD_DIR ?= ../build-gpu
GRID_CONFIG = $(GRID_BUILD_DIR)/grid-config
CXX      = $(shell $(GRID_CONFIG) --cxx)
# Laptop build (homebrew OpenSSL/HDF5/etc.) needs the brew include/lib paths
# that the system configure missed but Grid/util/Sha.h needs.  Harmless on
# Linux production builds (the directories don't exist there).
BREW_PREFIX ?= /opt/homebrew/opt
CXXFLAGS = $(shell $(GRID_CONFIG) --cxxflags) -I.. -I$(GRID_BUILD_DIR)/Grid -I$(BREW_PREFIX)/openssl@3/include
LDFLAGS  = $(shell $(GRID_CONFIG) --ldflags) -L$(GRID_BUILD_DIR)/Grid -L$(BREW_PREFIX)/openssl@3/lib
LIBS     = $(shell $(GRID_CONFIG) --libs)

# QUDA backend.  Enabled by default — all HMC and measurement binaries link
# against QUDA and pick it up at runtime when the per-binary QUDA env vars
# (QUDA_FORCE, QUDA_SOLVER, …) are set.  To build without QUDA:  QUDA_PREFIX= make
# (Once configure-driven build lands, GRID_HAVE_QUDA is defined automatically.)
# QMP_PREFIX defaults to QUDA_PREFIX (which is true for /home/agrebe/install/quda-5),
# but chroma's install puts QMP in a separate directory so override as needed.
# QUDA_STATIC=1 forces -l:libquda.a + -l:libqmp.a (chroma's archives are static).
#
# 2026-06-29: switched default QUDA_PREFIX to the CUDA-12.2-compatible install
# at /lustre2/nplqcd/install/quda_cuda12p2/ (built from /lustre2/nplqcd/src/quda
# during Path B2).  The previous external/quda-install was built against
# CUDA-12.3+ headers (cuda/ptx + libcudacxx prefetch_L1/L2 that aren't in
# 12.2) and could not be re-used by anyone needing the FloatNOrder accessor
# in external code.  The new install ships the older API our code uses:
#   - QUDA_FLOAT2_FIELD_ORDER (no NATIVE alias)
#   - FloatNOrder<Float, Ns, Nc, N_=2> (explicit 4th template arg)
# Extra include paths -I targets/cuda + targets/cuda/externals are required to
# pick up target_device.h and trove/ptr.h from external code.  Runtime needs
# QUDA_RESOURCE_PATH=/lustre2/nplqcd/cache/quda_resource_cuda12p2 (version-keyed
# tunecache).
QUDA_PREFIX ?= /lustre2/nplqcd/install/quda_cuda12p2
# quda_cuda12p2 (locally built from src/quda for CUDA 12.2) does not ship QMP;
# pull QMP from the older install which has both.
QMP_PREFIX ?= /lustre2/nplqcd/install/quda
QUDA_STATIC ?=
ifneq ($(QUDA_PREFIX),)
  CXXFLAGS += -DGRID_HAVE_QUDA -I$(QUDA_PREFIX)/include -I$(QUDA_PREFIX)/include/targets/cuda -I$(QUDA_PREFIX)/include/targets/cuda/externals -I$(QMP_PREFIX)/include
  LDFLAGS  += -L$(QUDA_PREFIX)/lib -L$(QMP_PREFIX)/lib -Xlinker -rpath -Xlinker $(QUDA_PREFIX)/lib -Xlinker -rpath -Xlinker $(QMP_PREFIX)/lib
  ifneq ($(QUDA_STATIC),)
    # nvcc treats positional .a paths as input files; use -l:libname syntax
    # so the linker picks up the static archives from -L paths.
    LIBS   := -l:libquda.a -l:libqmp.a $(LIBS)
  else
    LIBS   := -lquda -lqmp $(LIBS)
  endif
endif

PROGRAMS = gen_dtxqcd_cfgs gen_txqcd_cfgs gen_txqcd_cfgs_2plus1 gen_txqcd_cfgs_nf2 gen_qcd_cfgs gen_qcd_cfgs_2plus1 \
           meas_conn_txqcd meas_conn_qcd meas_conn_dtxqcd meas_conn_weakfield meas_conn_weakfield_nf2 \
           meas_disco_txqcd meas_disco_qcd \
           meas_aux_txqcd meas_aux_dtxqcd meas_baryon_dtxqcd compare compute_plaq compute_vev \
           eigspec_diag eigspec_diag_nf2 \
           test_quda_init test_quda_roundtrip test_quda_inverter \
           test_quda_clover_mg \
           test_quda_multishift test_quda_propagator \
           bench_dslash bench_clover_oprod bench_qudahermop_dslash \
           test_quda_txqcd_solver

all: $(PROGRAMS)

# gen_txqcd_cfgs is compiled at TXQCD_Nf=3 with diag mass {m_l, m_l, m_s}.
# Other production binaries default to TXQCD_Nf=2 (or are Nf-agnostic).
gen_txqcd_cfgs: gen_txqcd_cfgs.cc params.h eig_diag.h
	$(CXX) $(CXXFLAGS) -DTXQCD_Nf=3 -o $@ $< $(LDFLAGS) $(LIBS)

# gen_txqcd_cfgs_2plus1 is the previous Nf=2 TXQCD light + Nf=1 QCD strange
# action structure (pre-b04af352), kept for direct comparison against the
# Nf=3 form on the same chroma-imported config.
gen_txqcd_cfgs_2plus1: gen_txqcd_cfgs_2plus1.cc params.h eig_diag.h
	$(CXX) $(CXXFLAGS) -DTXQCD_Nf=2 -o $@ $< $(LDFLAGS) $(LIBS)

# gen_txqcd_cfgs_nf2: Nf=2 light-only diagnostic — gen_txqcd_cfgs_2plus1 with the
# strange Nf=1 monomials removed.  Isolates the strange quark's role.  Nf=2.
gen_txqcd_cfgs_nf2: gen_txqcd_cfgs_nf2.cc params.h eig_diag.h
	$(CXX) $(CXXFLAGS) -DTXQCD_Nf=2 -o $@ $< $(LDFLAGS) $(LIBS)

# meas_conn_weakfield uses TXQCDFermionNf — also Nf=3.
meas_conn_weakfield: meas_conn_weakfield.cc params.h
	$(CXX) $(CXXFLAGS) -DTXQCD_Nf=3 -o $@ $< $(LDFLAGS) $(LIBS)

# Nf=2 build of the same source for the Nf=2+1 connected-pion comparison.
meas_conn_weakfield_nf2: meas_conn_weakfield.cc params.h
	$(CXX) $(CXXFLAGS) -DTXQCD_Nf=2 -o $@ $< $(LDFLAGS) $(LIBS)

# eigspec_diag uses TXQCDFermionNf — Nf=3 default.
eigspec_diag: eigspec_diag.cc params.h
	$(CXX) $(CXXFLAGS) -DTXQCD_Nf=3 -o $@ $< $(LDFLAGS) $(LIBS)

# Nf=2 build for the Nf=2+1 vs Nf=3 spectrum comparison.
eigspec_diag_nf2: eigspec_diag.cc params.h
	$(CXX) $(CXXFLAGS) -DTXQCD_Nf=2 -o $@ $< $(LDFLAGS) $(LIBS)

# Nf=3 variants of the production meas binaries — built for measurement on
# the gen_txqcd_cfgs (Nf=3) cfgs (e.g. _nf3_weakfield_md10).  Default catch-all
# below builds the Nf=2 versions of the same sources.
meas_conn_txqcd_nf3: meas_conn_txqcd.cc params.h meas_helper.h
	$(CXX) $(CXXFLAGS) -DTXQCD_Nf=3 -o $@ $< $(LDFLAGS) $(LIBS)

meas_disco_txqcd_nf3: meas_disco_txqcd.cc params.h meas_helper.h
	$(CXX) $(CXXFLAGS) -DTXQCD_Nf=3 -o $@ $< $(LDFLAGS) $(LIBS)

# Style B Phase 2.5 Session 3d — DTXQCD native kernel impl objects.
# These are compiled with the trove externals visible (no `using namespace
# Grid;` in their TUs).  gen_dtxqcd_cfgs needs them linked in for the Style C
# CG path; other binaries link them too as a safety since they appear in
# DTXQCDMpcOpQUDA.h forward decls.  Each impl .cc lives in
# Grid/qcd/action/dtxqcd/.
DTXQCD_NATIVE_IMPL_SRCS = \
  ../Grid/qcd/action/dtxqcd/dtxqcd_quda_gamma5_native_impl.cc \
  ../Grid/qcd/action/dtxqcd/dtxqcd_quda_pre_mat_lower_native_impl.cc \
  ../Grid/qcd/action/dtxqcd/dtxqcd_quda_post_mat_lower_native_impl.cc \
  ../Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_native_v2_impl.cc

DTXQCD_NATIVE_IMPL_OBJS = \
  dtxqcd_quda_gamma5_native_impl.o \
  dtxqcd_quda_pre_mat_lower_native_impl.o \
  dtxqcd_quda_post_mat_lower_native_impl.o \
  dtxqcd_quda_aux_kernel_native_v2_impl.o

dtxqcd_quda_gamma5_native_impl.o: \
  ../Grid/qcd/action/dtxqcd/dtxqcd_quda_gamma5_native_impl.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

dtxqcd_quda_pre_mat_lower_native_impl.o: \
  ../Grid/qcd/action/dtxqcd/dtxqcd_quda_pre_mat_lower_native_impl.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

dtxqcd_quda_post_mat_lower_native_impl.o: \
  ../Grid/qcd/action/dtxqcd/dtxqcd_quda_post_mat_lower_native_impl.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

dtxqcd_quda_aux_kernel_native_v2_impl.o: \
  ../Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_native_v2_impl.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Specific rule for gen_dtxqcd_cfgs: split compile + link so nvcc receives only
# .cu/.cc sources during compile and only .o objects during link (nvcc gets
# confused if .o appears alongside .cc on the same invocation).
gen_dtxqcd_cfgs.o: gen_dtxqcd_cfgs.cc params.h eig_diag.h meas_helper.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Use nvcc-link by stripping -x cu from CXX (the nvcc wrapper otherwise tries
# to compile .o inputs as CUDA source — they're already ELF objects).
LINKER = $(subst -x cu,,$(CXX))

gen_dtxqcd_cfgs: gen_dtxqcd_cfgs.o $(DTXQCD_NATIVE_IMPL_OBJS)
	$(LINKER) $(CXXFLAGS) -o $@ gen_dtxqcd_cfgs.o $(DTXQCD_NATIVE_IMPL_OBJS) \
	  $(LDFLAGS) $(LIBS)

# Catch-all.  eig_diag.h + meas_helper.h listed as prereqs so header-only
# edits (e.g. EIG_DIAG tuning) force a rebuild of gen_qcd_cfgs* and the
# meas_* binaries that include them.  (Over-broad for binaries that don't
# include these headers, but a header edit is rare and a spurious rebuild
# is harmless.)
%: %.cc params.h eig_diag.h meas_helper.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -f $(PROGRAMS)

.PHONY: all clean
