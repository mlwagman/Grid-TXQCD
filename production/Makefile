GRID_CONFIG = ../build-gpu/grid-config
CXX      = $(shell $(GRID_CONFIG) --cxx)
CXXFLAGS = $(shell $(GRID_CONFIG) --cxxflags) -I.. -I../build-gpu/Grid
LDFLAGS  = $(shell $(GRID_CONFIG) --ldflags) -L../build-gpu/Grid
LIBS     = $(shell $(GRID_CONFIG) --libs)

PROGRAMS = gen_txqcd_cfgs gen_qcd_cfgs \
           meas_conn_txqcd meas_conn_qcd \
           meas_disco_txqcd meas_disco_qcd \
           meas_aux_txqcd compare compute_plaq compute_vev

all: $(PROGRAMS)

# gen_txqcd_cfgs is compiled at TXQCD_Nf=3 with diag mass {m_l, m_l, m_s}.
# Other production binaries default to TXQCD_Nf=2 (or are Nf-agnostic).
gen_txqcd_cfgs: gen_txqcd_cfgs.cc params.h
	$(CXX) $(CXXFLAGS) -DTXQCD_Nf=3 -o $@ $< $(LDFLAGS) $(LIBS)

%: %.cc params.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -f $(PROGRAMS)

.PHONY: all clean
