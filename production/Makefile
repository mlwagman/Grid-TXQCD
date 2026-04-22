GRID_CONFIG = ../build-gpu/grid-config
CXX      = $(shell $(GRID_CONFIG) --cxx)
CXXFLAGS = $(shell $(GRID_CONFIG) --cxxflags) -I.. -I../build-gpu/Grid
LDFLAGS  = $(shell $(GRID_CONFIG) --ldflags) -L../build-gpu/Grid
LIBS     = $(shell $(GRID_CONFIG) --libs)

PROGRAMS = gen_txqcd_cfgs gen_qcd_cfgs \
           meas_conn_txqcd meas_conn_qcd \
           meas_disco_txqcd meas_disco_qcd \
           meas_aux_txqcd compare

all: $(PROGRAMS)

%: %.cc params.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -f $(PROGRAMS)

.PHONY: all clean
