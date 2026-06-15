// Test_dtxqcd_parity_mirror: apply the lattice parity transform to a saved
// DTXQCD configuration (gauge + aux fields) and write it out as a new cfg.
//
// Parity P: (t, x⃗) → (t, −x⃗) = (t, (L − x⃗) mod L).  Equivalent on the
// operator side to γ4 M48 γ4 + spatial link reorientation.  Aux fields
// transform by their parity:
//
//   parity-even (no γ5 in coupling):  σ, n, s   →  field(P(x))
//   parity-odd  (γ5 in coupling):     π, d, p   → −field(P(x))
//
// Gauge transforms:
//   U_0(t, x⃗) → U_0(t, P(x⃗))
//   U_i(t, x⃗) → U_i(t, P(x⃗) − î)†   for i ∈ {1,2,3}
//
// The † on the spatial link reflects orientation reversal: under x⃗→−x⃗,
// the forward link from x⃗ to x⃗+î becomes the backward link from −x⃗ to
// −x⃗−î, i.e. the dagger of the forward link at the source site −x⃗−î.
//
// Action S_gauge + S_aux + |Pf(M48)| is invariant under this transform —
// verifiable by computing γ5·M48 spectrum on both cfgs (should match).
//
// Usage:
//   MASS=0.3 CSW=0.0 ./Test_dtxqcd_parity_mirror <src_dir> <src_traj> <dst_dir> <dst_traj> --grid ... --mpi ...

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>

using namespace Grid;

// Reflect a single spatial coordinate component about origin under
// periodic BC: c → (L − c) mod L.
inline int reflect(int c, int L) { return (L - c) % L; }

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  if (argc < 5) {
    std::cerr << "Usage: <src_dir> <src_traj> <dst_dir> <dst_traj>\n";
    return 1;
  }
  std::string src_dir = argv[1];
  int src_traj = std::atoi(argv[2]);
  std::string dst_dir = argv[3];
  int dst_traj = std::atoi(argv[4]);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid_(latt, simd, mpi);
  GridSerialRNG sRNG;  sRNG.SeedFixedIntegers({1});
  GridParallelRNG pRNG(&Grid_);
  pRNG.SeedFixedIntegers({1234, 5678, 91011, 121314, 151617});

  DTXQCDField Uold(&Grid_);
  DTXQCDCheckpointer::ReadConfig(Uold, sRNG, pRNG,
                                  src_dir + "/ckpoint_lat",
                                  src_dir + "/ckpoint_rng", src_traj);

  int Lx = latt[Xdir], Ly = latt[Ydir], Lz = latt[Zdir], Lt = latt[Tdir];
  std::cout << GridLogMessage << "Lattice = " << Lx << "." << Ly << "."
            << Lz << "." << Lt << std::endl;

  DTXQCDField Unew(&Grid_);
  Unew = Zero();

  // Site-by-site peek/poke.  4^3 × 8 = 512 sites: simple and fast.
  // For each new site x⃗ in the dest cfg, find the parity-image P(x⃗)
  // in the source cfg and copy the appropriately-transformed data.
  // The temporal component is unchanged; spatial components reflect.
  //
  // Gauge:
  //   U_0_new(t, x⃗) = U_0_old(t, P(x⃗))
  //   U_i_new(t, x⃗) = U_i_old(t, P(x⃗) − î)†
  //
  // The î subtraction uses periodic BC.

  typedef typename LatticeGaugeField::vector_object::scalar_object SiteLink;
  typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SiteCF;
  typedef typename LatticeDtxqcdS::vector_object::scalar_object SiteSc;

  for (int t = 0; t < Lt; ++t) {
    for (int x3 = 0; x3 < Lz; ++x3) {
      for (int x2 = 0; x2 < Ly; ++x2) {
        for (int x1 = 0; x1 < Lx; ++x1) {
          Coordinate dst(4);
          dst[0] = x1; dst[1] = x2; dst[2] = x3; dst[3] = t;
          Coordinate src(4);
          src[0] = reflect(x1, Lx); src[1] = reflect(x2, Ly);
          src[2] = reflect(x3, Lz); src[3] = t;

          // ---- Gauge ----
          SiteLink Lold;
          peekSite(Lold, Uold.U, src);  // 4 SU(3) at src
          SiteLink Lnew;
          // Time link: copy from src directly.
          Lnew(Tdir) = Lold(Tdir);
          // Spatial links: need link at (src - î) in direction i, daggered.
          for (int mu = 0; mu < 3; ++mu) {
            Coordinate src_minus_mu = src;
            int Lmu = latt[mu];
            src_minus_mu[mu] = (src_minus_mu[mu] + Lmu - 1) % Lmu;
            SiteLink Lold_mm;
            peekSite(Lold_mm, Uold.U, src_minus_mu);
            Lnew(mu) = adj(Lold_mm(mu));
          }
          pokeSite(Lnew, Unew.U, dst);

          // ---- Aux: parity-even (σ, n, s) ----
          SiteCF sig_o; peekSite(sig_o, Uold.sigma, src);
          pokeSite(sig_o, Unew.sigma, dst);
          SiteCF n_o;   peekSite(n_o, Uold.n, src);
          pokeSite(n_o, Unew.n, dst);
          SiteSc s_o;   peekSite(s_o, Uold.s, src);
          pokeSite(s_o, Unew.s, dst);

          // ---- Aux: parity-odd (π, d, p) — flip sign ----
          SiteCF pi_o; peekSite(pi_o, Uold.pi, src);
          pi_o = (-1.0) * pi_o;
          pokeSite(pi_o, Unew.pi, dst);
          SiteCF d_o;  peekSite(d_o, Uold.d, src);
          d_o = (-1.0) * d_o;
          pokeSite(d_o, Unew.d, dst);
          SiteSc p_o;  peekSite(p_o, Uold.p, src);
          p_o = (-1.0) * p_o;
          pokeSite(p_o, Unew.p, dst);
        }
      }
    }
  }

  // Sanity report
  std::cout << GridLogMessage << "norms (src/dst should match for even fields, "
            << "and ||odd|| identical, since x²=x²):" << std::endl;
  std::cout << GridLogMessage << "  plaq: src=" << WilsonLoops<PeriodicGimplR>::avgPlaquette(Uold.U)
            << "  dst=" << WilsonLoops<PeriodicGimplR>::avgPlaquette(Unew.U) << std::endl;
  RealD V = Grid_.gSites();
  std::cout << GridLogMessage << "  ||σ||²/V src=" << norm2(Uold.sigma)/V
            << "  dst=" << norm2(Unew.sigma)/V << std::endl;
  std::cout << GridLogMessage << "  ||π||²/V src=" << norm2(Uold.pi)/V
            << "  dst=" << norm2(Unew.pi)/V << std::endl;
  std::cout << GridLogMessage << "  ||d||²/V src=" << norm2(Uold.d)/V
            << "  dst=" << norm2(Unew.d)/V << std::endl;
  std::cout << GridLogMessage << "  ||n||²/V src=" << norm2(Uold.n)/V
            << "  dst=" << norm2(Unew.n)/V << std::endl;
  std::cout << GridLogMessage << "  ⟨s⟩ src=" << sum(Uold.s)()()()/V
            << "  dst=" << sum(Unew.s)()()()/V << std::endl;
  std::cout << GridLogMessage << "  ⟨p⟩ src=" << sum(Uold.p)()()()/V
            << "  dst=" << sum(Unew.p)()()()/V
            << "   (should be opposite sign)" << std::endl;

  // Write out: instantiate a checkpointer pointed at dst_dir and let it
  // do the work.  saveInterval=1 so any traj number writes.
  CheckpointerParameters cpp;
  cpp.config_prefix  = dst_dir + "/ckpoint_lat";
  cpp.smeared_prefix = dst_dir + "/ckpoint_lat_smr";
  cpp.rng_prefix     = dst_dir + "/ckpoint_rng";
  cpp.saveInterval   = 1;
  cpp.saveSmeared    = false;
  cpp.format         = "IEEE64BIG";
  DTXQCDCheckpointer ckpt(cpp);
  ckpt.initialize(cpp);
  ckpt.TrajectoryComplete(dst_traj, Unew, sRNG, pRNG);
  std::cout << GridLogMessage
            << "Parity-mirrored cfg written to " << dst_dir
            << "/ckpoint_lat." << dst_traj << std::endl;

  Grid_finalize();
  return 0;
}
