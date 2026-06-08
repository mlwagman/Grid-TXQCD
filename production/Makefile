# Default to build-gpu (production cluster); local laptop builds typically
# live in ../build instead.  Override on the command line:
#   make GRID_BUILD_DIR=../build gen_dtxqcd_cfgs
GRID_BUILD_DIR ?= ../build-gpu
GRID_CONFIG = $(GRID_BUILD_DIR)/grid-config
CXX      = $(shell $(GRID_CONFIG) --cxx)
CXXFLAGS = $(shell $(GRID_CONFIG) --cxxflags) -I.. -I$(GRID_BUILD_DIR)/Grid
LDFLAGS  = $(shell $(GRID_CONFIG) --ldflags) -L$(GRID_BUILD_DIR)/Grid
LIBS     = $(shell $(GRID_CONFIG) --libs)

# QUDA backend.  Enabled by default — all HMC and measurement binaries link
# against QUDA and pick it up at runtime when the per-binary QUDA env vars
# (QUDA_FORCE, QUDA_SOLVER, …) are set.  To build without QUDA:  QUDA_PREFIX= make
# (Once configure-driven build lands, GRID_HAVE_QUDA is defined automatically.)
# QMP_PREFIX defaults to QUDA_PREFIX (which is true for /home/agrebe/install/quda-5),
# but chroma's install puts QMP in a separate directory so override as needed.
# QUDA_STATIC=1 forces -l:libquda.a + -l:libqmp.a (chroma's archives are static).
QUDA_PREFIX ?= /lustre2/nplqcd/Grid-TXQCD/external/quda-install
# external/quda-install (locally built, matches the QUDA API our code uses)
# does not ship QMP; pull QMP from the older install which has both.
QMP_PREFIX ?= /lustre2/nplqcd/install/quda
QUDA_STATIC ?=
ifneq ($(QUDA_PREFIX),)
  CXXFLAGS += -DGRID_HAVE_QUDA -I$(QUDA_PREFIX)/include -I$(QMP_PREFIX)/include
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
           meas_conn_txqcd meas_conn_qcd meas_conn_weakfield meas_conn_weakfield_nf2 \
           meas_disco_txqcd meas_disco_qcd \
           meas_aux_txqcd compare compute_plaq compute_vev \
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
