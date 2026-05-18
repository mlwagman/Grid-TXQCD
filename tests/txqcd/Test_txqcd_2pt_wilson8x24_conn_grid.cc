// Source-grid version of Test_txqcd_2pt_wilson8x24_conn.
//
// For each cfg, computes connected pion and nucleon correlators from a grid
// of equally-spaced point sources (default 2x2x2x8 on the 8^3x24 lattice =
// stride 4,4,4,3 = 64 sources / cfg / quark).  Each source's correlator is
// time-shifted so the source sits at t=0, then averaged across sources.
// 64x more inversions per cfg, but ~sqrt(64)=8x noise reduction on the mean.
//
// SRC_GRID env var ("Nx.Ny.Nz.Nt") overrides the grid; default "2.2.2.8".
// Each Ni must divide latt[i].
//
// Output: meas_2pt_wilson8x24*/meas_{qcd,txqcd}_conn_grid.h5 — same datasets
// as the single-source variant (pion_conn[ncfg, T], nucleon[ncfg, T], plaq).
// Existing meas_{qcd,txqcd}_conn.h5 (single source) are not touched.

#include "Test_txqcd_2pt_wilson8x24_utils.h"
#include <Grid/serialisation/Hdf5IO.h>
#include <Grid/qcd/utils/BaryonUtils.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <algorithm>
#include <cstdio>
#include <numeric>

using namespace TxqcdTest2ptWilson8x24;

static void PointSource(const Coordinate &site, LatticePropagator &src) {
  src = Zero();
  SpinColourMatrix kron;
  kron = 1.0;
  pokeSite(kron, src, site);
}

static std::vector<RealD> PionCorrelator(const LatticePropagator &S_d,
                                         const LatticePropagator &S_u) {
  LatticeComplex corr(S_d.Grid());
  corr = trace(S_d * adj(S_u));
  std::vector<TComplex> Csl;
  sliceSum(corr, Csl, Nd - 1);
  std::vector<RealD> out(Csl.size());
  for (size_t t = 0; t < Csl.size(); ++t)
    out[t] = TensorRemove(Csl[t]).real();
  return out;
}

static std::vector<ComplexD> NucleonCorrelator(const LatticePropagator &S_u,
                                               const LatticePropagator &S_d) {
  Gamma G_A(Gamma::Algebra::Identity);
  Gamma G_B(Gamma::Algebra::SigmaXZ);
  int wick = 0;
  BaryonUtils<WilsonImplR>::WickContractions("uud", "uud", wick);
  LatticeComplex Cn(S_u.Grid());
  BaryonUtils<WilsonImplR>::ContractBaryons(S_u, S_u, S_d, G_A, G_B, G_A, G_B,
                                            wick, +1, Cn);
  std::vector<TComplex> sl;
  sliceSum(Cn, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

// Hand-rolled CG on M^dag M for TXQCDFermionNf.
static void TxqcdCG(TXQCDWilsonOp &Mop, const TXQCDFermionNf &b,
                     TXQCDFermionNf &x, RealD tol, int maxit) {
  GridBase *g = b.Grid();
  TXQCDFermionNf r(g), p(g), Mp(g), MdMp(g);
  x = Zero();
  r = b;
  p = r;
  RealD rsq = norm2(r);
  RealD bsq = std::max(norm2(b), 1e-30);
  RealD tol2 = tol * tol * bsq;
  int it;
  for (it = 0; it < maxit; ++it) {
    Mop.M(p, Mp);
    Mop.Mdag(Mp, MdMp);
    ComplexD pAp = innerProduct(p, MdMp);
    ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
    axpy(x, alpha, p);
    axpy(r, -alpha, MdMp);
    RealD rsq_new = norm2(r);
    if (rsq_new < tol2) { rsq = rsq_new; break; }
    RealD beta_cg = rsq_new / rsq;
    for (int a = 0; a < TxqcdNf; ++a) p.f[a] = r.f[a] + beta_cg * p.f[a];
    rsq = rsq_new;
  }
  std::cout << GridLogMessage << "[conn CG] iter=" << it
            << " rsq/bsq=" << rsq / bsq << std::endl;
}

static void TxqcdPointProp(LatticePropagator &S_u, LatticePropagator &S_d,
                           TXQCDWilsonOp &Mop, GridBase *g,
                           const Coordinate &src, RealD tol, int maxit) {
  LatticePropagator srcP(g);
  PointSource(src, srcP);
  S_u = Zero();
  S_d = Zero();

  for (int flavor = 0; flavor < TxqcdNf; ++flavor) {
    LatticePropagator &Sout = (flavor == 0) ? S_u : S_d;
    for (int spin = 0; spin < Ns; ++spin) {
      for (int col = 0; col < Nc; ++col) {
        LatticeFermion sf(g);
        PropToFerm<WilsonImplR>(sf, srcP, spin, col);
        TXQCDFermionNf snf(g), b(g), x(g);
        snf.f[0] = Zero();
        snf.f[1] = Zero();
        snf.f[flavor] = sf;
        Mop.Mdag(snf, b);
        TxqcdCG(Mop, b, x, tol, maxit);
        FermToProp<WilsonImplR>(Sout, x.f[flavor], spin, col);
      }
    }
  }
}

static void QcdPointProp(LatticePropagator &S, WilsonFermionD &Dw,
                         MdagMLinearOperator<WilsonFermionD, LatticeFermion> &HermOp,
                         GridCartesian &Grid,
                         const Coordinate &src, RealD tol, int maxit) {
  ConjugateGradient<LatticeFermion> CG(tol, maxit);
  LatticePropagator srcP(&Grid);
  PointSource(src, srcP);
  S = Zero();
  for (int spin = 0; spin < Ns; ++spin) {
    for (int col = 0; col < Nc; ++col) {
      LatticeFermion sf(&Grid), b(&Grid), x(&Grid);
      PropToFerm<WilsonImplR>(sf, srcP, spin, col);
      Dw.Mdag(sf, b);
      x = Zero();
      CG(HermOp, b, x);
      FermToProp<WilsonImplR>(S, x, spin, col);
    }
  }
}

// Parse "Nx.Ny.Nz.Nt" -> {Nx, Ny, Nz, Nt}.
static std::vector<int> parse_src_grid(const std::string &s) {
  std::vector<int> v;
  size_t pos = 0;
  while (pos < s.size()) {
    size_t dot = s.find('.', pos);
    std::string tok = s.substr(pos, dot == std::string::npos ? std::string::npos : dot - pos);
    if (!tok.empty()) v.push_back(std::atoi(tok.c_str()));
    if (dot == std::string::npos) break;
    pos = dot + 1;
  }
  return v;
}

// Enumerate source locations on (Nx, Ny, Nz, Nt) sub-grid of (Lx, Ly, Lz, Lt).
// Origin offset by src_site() so it inherits the existing single-source convention
// for the (0,0,0,0) corner.
static std::vector<Coordinate> source_locations(const Coordinate &latt,
                                                const std::vector<int> &grid,
                                                const Coordinate &origin) {
  std::vector<Coordinate> out;
  Coordinate stride(4);
  for (int d = 0; d < 4; ++d) {
    if (grid[d] <= 0 || latt[d] % grid[d] != 0) {
      std::cerr << "[conn_grid] grid dim " << d << " (" << grid[d]
                << ") must divide latt dim " << latt[d] << std::endl;
      std::exit(1);
    }
    stride[d] = latt[d] / grid[d];
  }
  for (int it = 0; it < grid[3]; ++it) {
    for (int iz = 0; iz < grid[2]; ++iz) {
      for (int iy = 0; iy < grid[1]; ++iy) {
        for (int ix = 0; ix < grid[0]; ++ix) {
          Coordinate c(4);
          c[0] = (origin[0] + ix * stride[0]) % latt[0];
          c[1] = (origin[1] + iy * stride[1]) % latt[1];
          c[2] = (origin[2] + iz * stride[2]) % latt[2];
          c[3] = (origin[3] + it * stride[3]) % latt[3];
          out.push_back(c);
        }
      }
    }
  }
  return out;
}

// In-place: out += C[(t + t_src) mod T] / norm
template <class T>
static void accum_shifted(std::vector<T> &out, const std::vector<T> &C,
                          int t_src, int Tlen, double norm) {
  for (int t = 0; t < Tlen; ++t) {
    out[t] = out[t] + C[(t + t_src) % Tlen] * norm;
  }
}

// Per-cfg incremental h5 storage so a killed run can resume.  We rewrite the
// file each cfg (cheap; ~10kb/cfg).  A `traj` int dataset records which trajs
// are present so resume can skip them.  Existing analysis scripts that read
// pion_conn/nucleon/plaq are unchanged.
struct ConnGridStore {
  std::vector<std::vector<RealD>>    pion;
  std::vector<std::vector<ComplexD>> nucl;
  std::vector<RealD>                 plaq;
  std::vector<int>                   traj;
};

static void load_existing(const std::string &path, ConnGridStore &s) {
  if (!TxqcdTest2pt::file_exists(path)) return;
  // HDF5 throws H5::Exception (not std::exception) when datasets are missing;
  // suppress its stderr noise and use a catch-all.
  H5::Exception::dontPrint();
  try {
    Hdf5Reader rd(path);
    read(rd, "pion_conn", s.pion);
    read(rd, "nucleon",   s.nucl);
    read(rd, "plaq",      s.plaq);
    read(rd, "traj",      s.traj);
  } catch (...) {
    std::cout << GridLogMessage << "[conn_grid] could not resume from " << path
              << " (missing 'traj' field?); recomputing all" << std::endl;
    s = ConnGridStore{};
    return;
  }
  if (s.pion.size() != s.traj.size() || s.nucl.size() != s.traj.size() ||
      s.plaq.size() != s.traj.size()) {
    std::cout << GridLogMessage << "[conn_grid] " << path
              << " has inconsistent dataset sizes; recomputing all" << std::endl;
    s = ConnGridStore{};
    return;
  }
  std::cout << GridLogMessage << "[conn_grid] resumed " << s.traj.size()
            << " cfgs from " << path << std::endl;
}

static void write_store(const std::string &path, ConnGridStore s,
                        const std::string &src_grid_str, int N_src) {
  // sort by traj
  std::vector<size_t> idx(s.traj.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](size_t a, size_t b) { return s.traj[a] < s.traj[b]; });
  ConnGridStore o;
  o.pion.reserve(idx.size()); o.nucl.reserve(idx.size());
  o.plaq.reserve(idx.size()); o.traj.reserve(idx.size());
  for (size_t i : idx) {
    o.pion.push_back(s.pion[i]);
    o.nucl.push_back(s.nucl[i]);
    o.plaq.push_back(s.plaq[i]);
    o.traj.push_back(s.traj[i]);
  }
  std::string tmp = path + ".tmp";
  {
    Hdf5Writer wr(tmp);
    write(wr, "pion_conn", o.pion);
    write(wr, "nucleon",   o.nucl);
    write(wr, "plaq",      o.plaq);
    write(wr, "traj",      o.traj);
    write(wr, "src_grid",  src_grid_str);
    write(wr, "n_src",     N_src);
  }
  // atomic replace so a kill mid-write doesn't corrupt the file
  std::rename(tmp.c_str(), path.c_str());
}

static bool traj_in(const std::vector<int> &v, int t) {
  return std::find(v.begin(), v.end(), t) != v.end();
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  int T = latt[Nd - 1];
  auto trajs = meas_trajs();
  Coordinate origin = src_site();
  mkdir_p(meas_dir());

  std::string SRC_GRID = "2.2.2.8";
  if (const char *v = std::getenv("SRC_GRID"); v && *v) SRC_GRID = v;
  auto grid = parse_src_grid(SRC_GRID);
  if ((int)grid.size() != 4) {
    std::cerr << "[conn_grid] SRC_GRID must have 4 dot-separated ints, got '"
              << SRC_GRID << "'" << std::endl;
    return 1;
  }
  auto src_locs = source_locations(latt, grid, origin);
  const int N_src = (int)src_locs.size();
  const double inv_N = 1.0 / N_src;
  std::cout << GridLogMessage << "[conn_grid] SRC_GRID=" << SRC_GRID
            << " N_src=" << N_src << std::endl;

  std::string WHICH = "both";
  if (const char *v = std::getenv("WHICH"); v && *v) WHICH = v;
  bool do_qcd   = (WHICH == "qcd"   || WHICH == "both");
  bool do_txqcd = (WHICH == "txqcd" || WHICH == "both");
  if (!txqcd_configs_exist()) do_txqcd = false;
  if (!qcd_configs_exist())   do_qcd   = false;
  std::cout << GridLogMessage << "[conn_grid] WHICH=" << WHICH
            << " do_qcd=" << do_qcd << " do_txqcd=" << do_txqcd << std::endl;

  const std::string txqcd_h5 = meas_dir() + "/meas_txqcd_conn_grid.h5";
  const std::string qcd_h5   = meas_dir() + "/meas_qcd_conn_grid.h5";

  // TXQCD
  if (do_txqcd) {
    ConnGridStore s;
    load_existing(txqcd_h5, s);
    TXQCDField U(&Grid);
    for (int traj : trajs) {
      if (traj_in(s.traj, traj)) {
        std::cout << GridLogMessage << "[conn_grid] TXQCD traj=" << traj
                  << " already done, skipping" << std::endl;
        continue;
      }
      std::cout << GridLogMessage << "[conn_grid] TXQCD traj=" << traj << std::endl;
      LoadTxqcdConfig(U, sRNG, pRNG, traj);
      RealD plaq_val = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
      TXQCDWilsonOp Mop(U.U, Grid, RBGrid, mass_runtime(),
                        U.sigma, U.pi, U.s, U.p, U.t);
      std::vector<RealD>    Cpi(T, 0.0);
      std::vector<ComplexD> CN(T, ComplexD(0.0, 0.0));
      for (int isrc = 0; isrc < N_src; ++isrc) {
        const Coordinate &src = src_locs[isrc];
        std::cout << GridLogMessage << "[conn_grid]   src " << (isrc + 1)
                  << "/" << N_src << " = (" << src[0] << "," << src[1]
                  << "," << src[2] << "," << src[3] << ")" << std::endl;
        LatticePropagator Su(&Grid), Sd(&Grid);
        TxqcdPointProp(Su, Sd, Mop, &Grid, src, meas_tol, cg_max);
        auto Cpi_s = PionCorrelator(Sd, Su);
        auto CN_s  = NucleonCorrelator(Su, Sd);
        accum_shifted(Cpi, Cpi_s, src[Nd - 1], T, inv_N);
        accum_shifted(CN,  CN_s,  src[Nd - 1], T, inv_N);
      }
      s.pion.push_back(Cpi);
      s.nucl.push_back(CN);
      s.plaq.push_back(plaq_val);
      s.traj.push_back(traj);
      write_store(txqcd_h5, s, SRC_GRID, N_src);
      std::cout << GridLogMessage << "[conn_grid] TXQCD wrote traj=" << traj
                << " (" << s.traj.size() << " cfgs total in h5)" << std::endl;
    }
  }

  // QCD
  if (do_qcd) {
    ConnGridStore s;
    load_existing(qcd_h5, s);
    LatticeGaugeField Umu(&Grid);
    for (int traj : trajs) {
      if (traj_in(s.traj, traj)) {
        std::cout << GridLogMessage << "[conn_grid] QCD traj=" << traj
                  << " already done, skipping" << std::endl;
        continue;
      }
      std::cout << GridLogMessage << "[conn_grid] QCD traj=" << traj << std::endl;
      LoadQcdConfig(Umu, sRNG, pRNG, traj);
      RealD plaq_val = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);
      WilsonFermionD Dw(Umu, Grid, RBGrid, mass_runtime());
      MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
      std::vector<RealD>    Cpi(T, 0.0);
      std::vector<ComplexD> CN(T, ComplexD(0.0, 0.0));
      for (int isrc = 0; isrc < N_src; ++isrc) {
        const Coordinate &src = src_locs[isrc];
        std::cout << GridLogMessage << "[conn_grid]   src " << (isrc + 1)
                  << "/" << N_src << " = (" << src[0] << "," << src[1]
                  << "," << src[2] << "," << src[3] << ")" << std::endl;
        LatticePropagator S(&Grid);
        QcdPointProp(S, Dw, HermOp, Grid, src, meas_tol, cg_max);
        auto Cpi_s = PionCorrelator(S, S);
        auto CN_s  = NucleonCorrelator(S, S);
        accum_shifted(Cpi, Cpi_s, src[Nd - 1], T, inv_N);
        accum_shifted(CN,  CN_s,  src[Nd - 1], T, inv_N);
      }
      s.pion.push_back(Cpi);
      s.nucl.push_back(CN);
      s.plaq.push_back(plaq_val);
      s.traj.push_back(traj);
      write_store(qcd_h5, s, SRC_GRID, N_src);
      std::cout << GridLogMessage << "[conn_grid] QCD wrote traj=" << traj
                << " (" << s.traj.size() << " cfgs total in h5)" << std::endl;
    }
  }

  std::cout << GridLogMessage << "[conn_grid] Connected 2pt (grid src) written to "
            << meas_dir() << "/*conn_grid.h5" << std::endl;
  Grid_finalize();
  return 0;
}
