// Mode-B (flavor-t) wrapper for Test_aux_gaussian.
// Build with -DTXQCD_T_FLAVOR=1 (see Make.inc).  Compiles the same test source
// against the flavor-t variant of TXQCD; verifies that aux-field Gaussian
// initialization moments and the AuxGaussianAction force pass FD checks in
// mode B.  Inactive color slots of t must be bit-exact zero by invariant.
#include "Test_aux_gaussian.cc"
