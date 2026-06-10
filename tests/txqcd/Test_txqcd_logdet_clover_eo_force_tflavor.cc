// Mode-B (flavor-t) wrapper for Test_txqcd_logdet_clover_eo_force.
// Build with -DTXQCD_T_FLAVOR=1 (see Make.inc).  Verifies that the analytic
// log-det force matches finite-difference perturbations of sigma, pi, s, p, t
// (and U at csw != 0) when t is reinterpreted as a flavor matrix.  The same
// test source is compiled in both modes, providing a regression-equivalent
// gate for mode B.
#include "Test_txqcd_logdet_clover_eo_force.cc"
