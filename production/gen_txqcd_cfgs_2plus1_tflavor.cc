// Flavor-t (mode B) variant of gen_txqcd_cfgs_2plus1.  Reuses the entire
// driver source via include; the build target sets TXQCD_T_FLAVOR=1 in
// production/Makefile, which switches:
//   - t_{mu,nu} index structure (color -> flavor),
//   - sigma/pi vs s/p pre-factors (1 vs 1/sqrt(2) swap),
//   - clover gauge-force kernel (t channel orthogonal to color, so a separate
//     color-sigma extraction kernel is used).
//
// All other action structure (sigma, pi, s, p Gaussian priors and quark
// monomials) is unchanged.  Used to probe whether a flavor-t reformulation
// has a lower lambda_crit than the color-t variant; see
// memory/project_t_condensate_low_lambda.md.
#include "gen_txqcd_cfgs_2plus1.cc"
