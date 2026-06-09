// Mode-B (flavor-t) wrapper for Test_txqcd_fierz_sigma_saddle.
// Build with -DTXQCD_T_FLAVOR=1 (see Make.inc).  At the sigma saddle (t=0,
// others=0), the action identity S_TXQCD(sigma=c·I) == 2·S_QCD(m+c) must
// still hold in mode B because the t channel is unused and the sigma/pi
// rescale (1/sqrt(2) factor) cancels in the sigma=c·I substitution.
#include "Test_txqcd_fierz_sigma_saddle.cc"
