GRID_CONFIG = ../build-gpu/grid-config
CXX      = $(shell $(GRID_CONFIG) --cxx)
CXXFLAGS = $(shell $(GRID_CONFIG) --cxxflags) -I.. -I../build-gpu/Grid
LDFLAGS  = $(shell $(GRID_CONFIG) --ldflags) -L../build-gpu/Grid
LIBS     = $(shell $(GRID_CONFIG) --libs)

# Optional QUDA backend.  Set QUDA_PREFIX to enable.
# (Once configure-driven build lands, GRID_HAVE_QUDA is defined automatically.)
QUDA_PREFIX ?=
ifneq ($(QUDA_PREFIX),)
  CXXFLAGS += -DGRID_HAVE_QUDA -I$(QUDA_PREFIX)/include
  LDFLAGS  += -L$(QUDA_PREFIX)/lib -Xlinker -rpath -Xlinker $(QUDA_PREFIX)/lib
  LIBS     := -lquda -lqmp $(LIBS)
endif

PROGRAMS = gen_txqcd_cfgs gen_txqcd_cfgs_2plus1 gen_qcd_cfgs gen_qcd_cfgs_2plus1 \
           meas_conn_txqcd meas_conn_qcd meas_conn_weakfield meas_conn_weakfield_nf2 \
           meas_disco_txqcd meas_disco_qcd \
           meas_aux_txqcd compare compute_plaq compute_vev \
           eigspec_diag eigspec_diag_nf2 \
           test_quda_init test_quda_roundtrip test_quda_inverter

all: $(PROGRAMS)

# gen_txqcd_cfgs is compiled at TXQCD_Nf=3 with diag mass {m_l, m_l, m_s}.
# Other production binaries default to TXQCD_Nf=2 (or are Nf-agnostic).
gen_txqcd_cfgs: gen_txqcd_cfgs.cc params.h
	$(CXX) $(CXXFLAGS) -DTXQCD_Nf=3 -o $@ $< $(LDFLAGS) $(LIBS)

# gen_txqcd_cfgs_2plus1 is the previous Nf=2 TXQCD light + Nf=1 QCD strange
# action structure (pre-b04af352), kept for direct comparison against the
# Nf=3 form on the same chroma-imported config.
gen_txqcd_cfgs_2plus1: gen_txqcd_cfgs_2plus1.cc params.h
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

%: %.cc params.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -f $(PROGRAMS)

.PHONY: all clean
