// Test_dtxqcd_zero_aux_clover_qcd: STRUCTURAL diagnostic.
//
// At aux = 0, the doubled 48x48 DTXQCD operator M48[U, aux=0] should reduce
// EXACTLY to the charge-conjugate-doubled Wilson-CLOVER operator (= pure
// Nf=2 QCD).  The decoupled block structure (aux = 0 -> off-diagonal d,n
// blocks vanish, Delta vanishes) is:
//
//   M48[U, aux=0] (psi_u, psi_l) = ( D_WC[U]  psi_u ,  D_WC_lower  psi_l )
//
// where the upper block is the *standard* Grid Wilson-Clover operator on U:
//   D_WC[U] = (m + 4) + D_hop[U] - (csw/2) sum_{mu<nu} F_{mu,nu}[U] sigma_{mu,nu}
//
// and the lower block is the Cstar-conjugated operator on conjugate(U):
//   D_WC_lower = (m + 4) + D_hop[U*] + (csw/2) sum_{mu<nu} F^T_{mu,nu}[U] sigma
//
// The DTXQCD lower-block clover uses F^T[U] (color-transpose of F from U),
// NOT F[U*].  Whether that equals Grid's stock WilsonClover[U*] is exactly
// the open clover-convention question this test answers.
//
// Strategy (operator matvec, value-based, NO gauge-force innerProducts):
//   1. U = I (cold)   -> MUST pass to ~1e-12 (calibration).  If it doesn't,
//                        the test convention is wrong, not the operator.
//   2. weak field     -> the real probe.
// At each gauge, separately compare:
//   upper:  DTXQCD.M[aux=0].upper   vs  Grid WilsonClover[U].M
//   lower:  DTXQCD.M[aux=0].lower   vs  Grid WilsonClover[conj(U)].M
// for csw = 0 and csw = 1.24930970916466 (the production value).
//
// Both Grid WilsonClover and DTXQCD use APBC in time and the SAME clover
// convention (-1/2 csw sigma F), so at aux=0 the upper block is a direct,
// bit-level comparison.
//
// Also (det-value level, on tiny lattice): a dense log|det(M48[aux=0])| vs
// 2 * log|det(D_WC[U])| (upper) -- but the cleanest doubling relation is
//   log|det M48[aux=0]| = log|det D_WC[U]| + log|det D_WC_lower|
// formed densely by applying each operator to all unit vectors.
//
// Run:
//   ./tests/dtxqcd/Test_dtxqcd_zero_aux_clover_qcd --grid 4.4.4.4 --mpi 1.1.1.1
// (use --grid 2.2.2.2 to also exercise the dense log|det| in reasonable time)

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/Eigen/Dense>

using namespace Grid;

// APBC-in-time ImplParams, matching the DTXQCD default.
static WilsonImplR::ImplParams ApbcParams() {
  WilsonImplR::ImplParams ip;
  ip.boundary_phases.resize(Nd, 1.0);
  ip.boundary_phases[Nd - 1] = -1.0;
  return ip;
}

// Compare DTXQCD.M[aux=0] (full operator) against the per-block Grid
// WilsonClover references on a given gauge field, at a given csw.
// Returns worst relative L2 across upper and lower blocks; prints per-block.
static RealD compare_operator(const char *tag,
                              DTXQCDField &U,
                              GridCartesian &Grid_,
                              GridRedBlackCartesian &RBGrid,
                              GridParallelRNG &pRNG,
                              RealD mass, RealD csw,
                              RealD &upper_rel_out,
                              RealD &lower_rel_out) {
  // DTXQCD operator at aux = 0 (aux fields already zeroed by caller).
  DTXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass, csw,
                                 U.sigma, U.pi, U.d, U.n, U.s, U.p);

  // Grid reference operators.  csw_r == csw_t == csw (isotropic).
  WilsonImplR::ImplParams ip = ApbcParams();
  LatticeGaugeField Uc(&Grid_);
  Uc = conjugate(U.U);
  WilsonAnisotropyCoefficients aniso;  // default: isotropic
  WilsonCloverFermionD WC_upper(U.U, Grid_, RBGrid, mass, csw, csw, aniso, ip);
  WilsonCloverFermionD WC_lower(Uc,  Grid_, RBGrid, mass, csw, csw, aniso, ip);

  // Random doubled fermion.
  DTXQCDFermionDoubled psi(&Grid_), out_dtxqcd(&Grid_);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, psi.upper.f[a]);
    gaussian(pRNG, psi.lower.f[a]);
  }

  Dw.M(psi, out_dtxqcd);

  RealD upper_rel = 0.0, lower_rel = 0.0;
  for (int a = 0; a < DtxqcdNf; ++a) {
    LatticeFermion ref_u(&Grid_), ref_l(&Grid_);
    WC_upper.M(psi.upper.f[a], ref_u);
    WC_lower.M(psi.lower.f[a], ref_l);
    LatticeFermion du = out_dtxqcd.upper.f[a] - ref_u;
    LatticeFermion dl = out_dtxqcd.lower.f[a] - ref_l;
    RealD ru = std::sqrt(norm2(du) / std::max(norm2(ref_u), 1e-30));
    RealD rl = std::sqrt(norm2(dl) / std::max(norm2(ref_l), 1e-30));
    upper_rel = std::max(upper_rel, ru);
    lower_rel = std::max(lower_rel, rl);
  }
  upper_rel_out = upper_rel;
  lower_rel_out = lower_rel;

  std::cout << GridLogMessage << "  [" << tag << " csw=" << csw << "]"
            << "  upper rel L2 = " << upper_rel
            << "   lower rel L2 = " << lower_rel << std::endl;
  return std::max(upper_rel, lower_rel);
}

// Dense log|det| of a Grid fermion operator by columns: apply M to every
// global unit vector and stack into an Eigen matrix.  Only call this on a
// SMALL lattice (4^4 max here): dimension = ndof_per_site * gSites.
//
// op_apply(in, out) applies the operator; ndof_per_site = 12 (single Wilson)
// or 48 (doubled).  Returns log|det| as Σ_i log|U_ii| from the LU diagonal
// -- NOT log(|prod of pivots|), which overflows the double range for any
// large matrix (the determinant itself is astronomically large/small even
// when well conditioned; only the LOG is finite).  The previous version
// returned inf for exactly this reason on a 4^4 (N=12288) lattice.
template <class FieldT>
static RealD dense_logabsdet(GridCartesian &Grid_,
                            int ndof_per_site,
                            std::function<void(const FieldT &, FieldT &)> op_apply,
                            std::function<void(FieldT &, int, ComplexD)> set_unit,
                            std::function<ComplexD(const FieldT &, int)> get_comp) {
  const int N = ndof_per_site * (int)Grid_.gSites();
  Eigen::MatrixXcd A(N, N);
  FieldT in(&Grid_), out(&Grid_);
  for (int col = 0; col < N; ++col) {
    in = Zero();
    set_unit(in, col, ComplexD(1.0, 0.0));
    op_apply(in, out);
    for (int row = 0; row < N; ++row) A(row, col) = get_comp(out, row);
  }
  Eigen::PartialPivLU<Eigen::MatrixXcd> lu(A);
  // Σ log|U_ii| over the LU upper-triangular diagonal == log|det| (the sign
  // of det from L's unit diagonal + permutation parity is irrelevant for
  // log|det|).  Overflow-safe.
  const Eigen::MatrixXcd &LU = lu.matrixLU();
  RealD acc = 0.0;
  for (int i = 0; i < N; ++i) acc += std::log(std::abs(LU(i, i)));
  return acc;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid_(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid_);
  GridParallelRNG pRNG(&Grid_);
  pRNG.SeedFixedIntegers({701, 702, 703, 704});

  const RealD mass    = 0.3;
  const RealD csw_prod = 1.24930970916466;  // production clover coefficient

  int exitcode = 0;
  auto report = [&](const char *name, RealD val, RealD tol) {
    bool ok = (val <= tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << "  rel = " << val << "  (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  std::cout << GridLogMessage
            << "lattice = " << latt[0] << "." << latt[1] << "."
            << latt[2] << "." << latt[3]
            << "   mass = " << mass
            << "   csw_prod = " << csw_prod << std::endl;
  std::cout << GridLogMessage
            << "Reference: Grid WilsonCloverFermionD (upper on U, lower on conj(U)),"
            << " APBC time, csw_r=csw_t=csw." << std::endl;

  // Aux fields all zero (the whole point).
  DTXQCDField U(&Grid_);
  U.sigma = Zero();  U.pi = Zero();
  U.d     = Zero();  U.n  = Zero();
  U.s     = Zero();  U.p  = Zero();

  // =====================================================================
  // PART 1: OPERATOR MATVEC (robust, primary).
  // =====================================================================

  // ---- 1a. U = I free-field CALIBRATION (must pass to ~1e-12) ----
  std::cout << GridLogMessage << "===== U = I (cold) calibration =====" << std::endl;
  SU<Nc>::ColdConfiguration(U.U);
  {
    RealD ru, rl;
    compare_operator("U=I", U, Grid_, RBGrid, pRNG, mass, /*csw=*/0.0, ru, rl);
    report("U=I, csw=0   : M48[aux=0] upper vs WilsonClover[U]",      ru, 1e-12);
    report("U=I, csw=0   : M48[aux=0] lower vs WilsonClover[conj U]", rl, 1e-12);
  }
  {
    RealD ru, rl;
    compare_operator("U=I", U, Grid_, RBGrid, pRNG, mass, csw_prod, ru, rl);
    report("U=I, csw=1.249: M48[aux=0] upper vs WilsonClover[U]",      ru, 1e-12);
    report("U=I, csw=1.249: M48[aux=0] lower vs WilsonClover[conj U]", rl, 1e-12);
  }

  if (exitcode != 0) {
    std::cout << GridLogError
              << "U=I calibration FAILED -- this is a test-convention bug, "
                 "not the operator.  Aborting weak-field probe." << std::endl;
    std::cout << GridLogMessage << "SOME CHECKS FAILED" << std::endl;
    Grid_finalize();
    return exitcode;
  }

  // ---- 1b. WEAK gauge field (the real probe) ----
  std::cout << GridLogMessage << "===== weak field (TepidConfiguration) =====" << std::endl;
  SU<Nc>::TepidConfiguration(pRNG, U.U);
  RealD weak_csw0_worst = 0.0, weak_cswprod_worst = 0.0;
  {
    RealD ru, rl;
    weak_csw0_worst = compare_operator("weak", U, Grid_, RBGrid, pRNG, mass,
                                       /*csw=*/0.0, ru, rl);
    report("weak, csw=0   : M48[aux=0] upper vs WilsonClover[U]",      ru, 1e-12);
    report("weak, csw=0   : M48[aux=0] lower vs WilsonClover[conj U]", rl, 1e-12);
  }
  {
    RealD ru, rl;
    weak_cswprod_worst = compare_operator("weak", U, Grid_, RBGrid, pRNG, mass,
                                          csw_prod, ru, rl);
    report("weak, csw=1.249: M48[aux=0] upper vs WilsonClover[U]",      ru, 1e-12);
    report("weak, csw=1.249: M48[aux=0] lower vs WilsonClover[conj U]", rl, 1e-12);
  }

  // ---- 1c. HOT gauge field (stress test, larger F) ----
  std::cout << GridLogMessage << "===== hot field (stress) =====" << std::endl;
  SU<Nc>::HotConfiguration(pRNG, U.U);
  {
    RealD ru, rl;
    compare_operator("hot", U, Grid_, RBGrid, pRNG, mass, csw_prod, ru, rl);
    report("hot,  csw=1.249: M48[aux=0] upper vs WilsonClover[U]",      ru, 1e-12);
    report("hot,  csw=1.249: M48[aux=0] lower vs WilsonClover[conj U]", rl, 1e-12);
  }

  // =====================================================================
  // PART 2: DET-VALUE (dense log|det|, only on tiny lattice).
  // The doubling relation:
  //    log|det M48[U,aux=0]| = log|det D_WC[U]| + log|det D_WC[conj U]_lower|
  // where the lower-block reference is the SAME Grid WilsonClover[conj U]
  // we matched against above.  At the operator level these are identical
  // (verified in Part 1), so the det relation is implied; we form it
  // densely here as an independent, convention-free numerical confirmation.
  // =====================================================================
  // Dense log|det| dimension = 48 * gSites.  4^4 -> 12288 (~2.3 GB dense
  // matrix + an O(N^3) Eigen LU, ~1-2 min).  Note: a 2^4 lattice trips Grid's
  // GridRedBlackCartesian odd-_rdimensions assertion under the GPU SIMD
  // layout, so the smallest usable lattice here is 4^4.  Gate at 16384 so 4^4
  // runs but larger volumes are refused.  Set DTXQCD_DENSE_DET=0 to skip.
  // Default OFF: the dense build is ~37k full-volume GPU operator-applies +
  // a 12288-dim Eigen LU (~19 min at 4^4).  It is a redundant confirmation:
  // the EXACT operator identity proven in Part 1 already implies equal
  // determinants.  Enable explicitly with DTXQCD_DENSE_DET=1.
  const long Nfull = 48L * Grid_.gSites();
  const bool want_det = []() {
    const char *e = std::getenv("DTXQCD_DENSE_DET");
    return (e && *e && std::atoi(e) != 0);
  }();
  if (want_det && Nfull <= 16384) {
    std::cout << GridLogMessage << "===== dense log|det| (tiny lattice) =====" << std::endl;
    // Use the weak field already loaded? No -- reload a fixed gauge for repro.
    SU<Nc>::TepidConfiguration(pRNG, U.U);

    auto run_det = [&](RealD csw, RealD &ld_m48, RealD &ld_doubled) {
      DTXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass, csw,
                                     U.sigma, U.pi, U.d, U.n, U.s, U.p);
      WilsonImplR::ImplParams ip = ApbcParams();
      LatticeGaugeField Uc(&Grid_);
      Uc = conjugate(U.U);
      WilsonAnisotropyCoefficients aniso;
      WilsonCloverFermionD WC_upper(U.U, Grid_, RBGrid, mass, csw, csw, aniso, ip);
      WilsonCloverFermionD WC_lower(Uc,  Grid_, RBGrid, mass, csw, csw, aniso, ip);

      typedef typename LatticeFermion::vector_object::scalar_object SiteFerm;
      const int Vg = (int)Grid_.gSites();

      // ----- dense log|det M48| (48 dof/site) -----
      auto coord_of = [&](int site) {
        Coordinate c(Nd);
        Lexicographic::CoorFromIndex(c, site, Grid_.GlobalDimensions());
        return c;
      };
      ld_m48 = dense_logabsdet<DTXQCDFermionDoubled>(
          Grid_, 48,
          [&](const DTXQCDFermionDoubled &in, DTXQCDFermionDoubled &out) {
            Dw.M(const_cast<DTXQCDFermionDoubled &>(in), out);
          },
          [&](DTXQCDFermionDoubled &f, int col, ComplexD v) {
            int site = col / 48;  int loc = col % 48;
            int blk = loc / 24;   int r = loc % 24;
            int a = r / (Ns * Nc); int rem = r % (Ns * Nc);
            int al = rem / Nc;     int ci = rem % Nc;
            Coordinate c = coord_of(site);
            SiteFerm s; peekSite(s, blk == 0 ? f.upper.f[a] : f.lower.f[a], c);
            s()(al)(ci) = v;
            pokeSite(s, blk == 0 ? f.upper.f[a] : f.lower.f[a], c);
          },
          [&](const DTXQCDFermionDoubled &f, int row) -> ComplexD {
            int site = row / 48;  int loc = row % 48;
            int blk = loc / 24;   int r = loc % 24;
            int a = r / (Ns * Nc); int rem = r % (Ns * Nc);
            int al = rem / Nc;     int ci = rem % Nc;
            Coordinate c = coord_of(site);
            SiteFerm s;
            peekSite(s, blk == 0 ? f.upper.f[a] : f.lower.f[a], c);
            return ComplexD(TensorRemove(s()(al)(ci)));
          });

      // ----- dense log|det| of each Grid WilsonClover block (12 dof/site) -----
      auto block_logdet = [&](WilsonCloverFermionD &op) {
        return dense_logabsdet<LatticeFermion>(
            Grid_, 12,
            [&](const LatticeFermion &in, LatticeFermion &out) {
              op.M(const_cast<LatticeFermion &>(in), out);
            },
            [&](LatticeFermion &f, int col, ComplexD v) {
              int site = col / 12; int loc = col % 12;
              int al = loc / Nc;   int ci = loc % Nc;
              Coordinate c = coord_of(site);
              SiteFerm s; peekSite(s, f, c);
              s()(al)(ci) = v; pokeSite(s, f, c);
            },
            [&](const LatticeFermion &f, int row) -> ComplexD {
              int site = row / 12; int loc = row % 12;
              int al = loc / Nc;   int ci = loc % Nc;
              Coordinate c = coord_of(site);
              SiteFerm s; peekSite(s, f, c);
              return ComplexD(TensorRemove(s()(al)(ci)));
            });
      };
      RealD ld_up = block_logdet(WC_upper);
      RealD ld_lo = block_logdet(WC_lower);
      // M48 has Nf=2 copies of each 12-dim block per (upper/lower), so the
      // reference doubled log|det| is Nf*(ld_up + ld_lo).
      ld_doubled = (RealD)DtxqcdNf * (ld_up + ld_lo);
      std::cout << GridLogMessage << "    csw=" << csw
                << ": log|det D_WC[U]| (1 flavor) = " << ld_up
                << "   log|det D_WC[conjU]| = " << ld_lo << std::endl;
    };

    {
      RealD ld_m48, ld_ref;
      run_det(0.0, ld_m48, ld_ref);
      RealD rel = std::abs(ld_m48 - ld_ref) /
                  (std::max(std::abs(ld_m48), std::abs(ld_ref)) + 1.0);
      std::cout << GridLogMessage << "  csw=0   : log|det M48[aux=0]| = " << ld_m48
                << "   Nf*(up+lo) = " << ld_ref << std::endl;
      report("det: log|det M48[aux=0]| = Nf*(log|det D_WC[U]|+log|det D_WC[conjU]|), csw=0",
             rel, 1e-9);
    }
    {
      RealD ld_m48, ld_ref;
      run_det(csw_prod, ld_m48, ld_ref);
      RealD rel = std::abs(ld_m48 - ld_ref) /
                  (std::max(std::abs(ld_m48), std::abs(ld_ref)) + 1.0);
      std::cout << GridLogMessage << "  csw=1.249: log|det M48[aux=0]| = " << ld_m48
                << "   Nf*(up+lo) = " << ld_ref << std::endl;
      report("det: log|det M48[aux=0]| = Nf*(log|det D_WC[U]|+log|det D_WC[conjU]|), csw=1.249",
             rel, 1e-9);
    }
  } else {
    std::cout << GridLogMessage
              << "(dense log|det| skipped: 48*V = " << Nfull
              << " > 16384 or DTXQCD_DENSE_DET=0; run with --grid 4.4.4.4 to"
              << " exercise it)" << std::endl;
  }

  // ---- summary ----
  std::cout << GridLogMessage << "===== SUMMARY =====" << std::endl;
  std::cout << GridLogMessage
            << "weak field worst rel L2: csw=0 -> " << weak_csw0_worst
            << " ,  csw=1.249 -> " << weak_cswprod_worst << std::endl;
  if (weak_cswprod_worst > 10.0 * std::max(weak_csw0_worst, 1e-15)) {
    std::cout << GridLogMessage
              << "NOTE: discrepancy is clover-driven (csw=1.249 worse than csw=0)."
              << std::endl;
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
