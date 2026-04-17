// End-to-end TXQCD vs QCD two-point function test (Phase 4e).
//
// Clean Nf=2 comparison (enabled by the rational PF):
//   TXQCD  ->  ONE rational PF on single-aux TXQCDField.
//              S = phi^dag (M_TX^dag M_TX)^{-1/2} phi -> weight |det M_TX|.
//              M_TX has internal Nf_tx=2 flavor block, so at aux=0
//              det M_TX = (det M_W)^2, giving Nf=2 Wilson in the zero-aux
//              limit. Integrating out the Gaussian aux brings this back to
//              Nf=2 Wilson with the TXQCD-specific quartic quark-bilinear
//              structure (whose Fierz cancellation is the content of the
//              notes).
//   QCD    ->  ONE TwoFlavourPseudoFermionAction<WilsonImplR>, |det M_W|^2
//              = Nf=2. Matches the TXQCD zero-aux limit directly; the TXQCD
//              aux integration must reproduce this ensemble at the level of
//              observables (Fierz test target).
//
// The aux-field diagnostics (volume-averaged and connected aux-pi, aux-p) and
// the per-flavor VEV/SD/Fierz blocks below are aux-count-agnostic and are kept
// as the validation targets.

#include <Grid/Grid.h>
#include <Grid/qcd/utils/BaryonUtils.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonRationalPseudoFermionAction.h>
#include <Grid/qcd/action/txqcd/TXQCDCheckpointer.h>
#include <sys/stat.h>

using namespace Grid;

// -------------------------- helpers ------------------------------------

static bool t2pt_file_exists(const std::string &f) {
  struct stat st;
  return stat(f.c_str(), &st) == 0;
}
static void t2pt_mkdir_p(const std::string &d) {
  if (!d.empty()) mkdir(d.c_str(), 0755);
}
static bool txqcd_configs_cached(const std::string &cfg_prefix,
                                 const std::string &rng_prefix,
                                 int first, int last) {
  for (int t = first; t <= last; ++t) {
    std::ostringstream c, a, r;
    c << cfg_prefix << "." << t;
    a << cfg_prefix << "_aux." << t;
    r << rng_prefix << "." << t;
    if (!t2pt_file_exists(c.str()) || !t2pt_file_exists(a.str()) ||
        !t2pt_file_exists(r.str()))
      return false;
  }
  return true;
}
static bool qcd_configs_cached(const std::string &cfg_prefix,
                               const std::string &rng_prefix, int first,
                               int last) {
  for (int t = first; t <= last; ++t) {
    std::ostringstream c, r;
    c << cfg_prefix << "." << t;
    r << rng_prefix << "." << t;
    if (!t2pt_file_exists(c.str()) || !t2pt_file_exists(r.str())) return false;
  }
  return true;
}

static void PointSource(const Coordinate &site, LatticePropagator &src) {
  src = Zero();
  SpinColourMatrix kron; kron = 1.0;
  pokeSite(kron, src, site);
}

// Pion correlator from propagators: C_pi(t) = sum_x tr[S_d(x;0) S_u(x;0)^dag].
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

// Proton "uud"/"uud" via BaryonUtils. Grid convention: G_A=Identity,
// G_B=SigmaXZ, parity +1.
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

// Hand-rolled CG on M^dag M for TXQCDFermionNf (same as
// TXQCDWilsonPseudoFermionAction's internal solver).
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
    RealD beta = rsq_new / rsq;
    for (int aa = 0; aa < TxqcdNf; ++aa) p.f[aa] = r.f[aa] + beta * p.f[aa];
    rsq = rsq_new;
  }
  std::cout << GridLogMessage << "[2pt TXQCD CG] iter=" << it
            << " rsq=" << rsq << " tol2=" << tol2 << std::endl;
}

// TXQCD point-source propagators on a single-aux TXQCDField.
// Outputs: S_u, S_d (flavor-diagonal), L_du_src = Tr[G_{du}(src,src) γ₅]
// for the disconnected pion source loop.
static void TxqcdPointProp(LatticePropagator &S_u, LatticePropagator &S_d,
                           ComplexD &L_du_src,
                           TXQCDField &U, RealD mass,
                           const Coordinate &src_site, RealD tol, int maxit) {
  GridBase *g = U.Grid();
  GridCartesian *Ug = dynamic_cast<GridCartesian *>(g);
  GridRedBlackCartesian RB(Ug);
  TXQCDWilsonOp Mop(U.U, *Ug, RB, mass,
                    U.sigma, U.pi, U.s, U.p, U.t);

  LatticePropagator src(g);
  PointSource(src_site, src);
  S_u = Zero();
  S_d = Zero();

  LatticePropagator S_du(g);
  S_du = Zero();

  for (int flavor = 0; flavor < TxqcdNf; ++flavor) {
    LatticePropagator &Sout = (flavor == 0) ? S_u : S_d;
    for (int spin = 0; spin < Ns; ++spin) {
      for (int col = 0; col < Nc; ++col) {
        LatticeFermion src_ferm(g);
        PropToFerm<WilsonImplR>(src_ferm, src, spin, col);

        TXQCDFermionNf src_nf(g), b(g), x(g);
        src_nf.f[0] = Zero();
        src_nf.f[1] = Zero();
        src_nf.f[flavor] = src_ferm;
        Mop.Mdag(src_nf, b);

        TxqcdCG(Mop, b, x, tol, maxit);
        FermToProp<WilsonImplR>(Sout, x.f[flavor], spin, col);

        if (flavor == 0) {
          FermToProp<WilsonImplR>(S_du, x.f[1], spin, col);
        }
      }
    }
  }

  Gamma g5(Gamma::Algebra::Gamma5);
  LatticeComplex tr_g5_Sdu(g);
  tr_g5_Sdu = trace(g5 * S_du);
  TComplex tval;
  peekSite(tval, tr_g5_Sdu, src_site);
  L_du_src = TensorRemove(tval);
}

// Stochastic estimate of per-timeslice flavor-off-diagonal loop
//   L_ud(t) = Σ_{x∈t} Tr_sc[G_{ud}(x,x) γ₅]
// using n_noise Gaussian volume sources on flavor d (index 1).
// Grid's gaussian gives E[|η|²]=2 per complex DOF; the 1/2 corrects this.
static std::vector<ComplexD>
TxqcdStochasticLoop_ud(TXQCDField &U, RealD mass,
                       GridBase *grid, GridParallelRNG &pRNG,
                       int n_noise, RealD tol, int maxit) {
  GridCartesian *Ug = dynamic_cast<GridCartesian *>(grid);
  GridRedBlackCartesian RB(Ug);
  TXQCDWilsonOp Mop(U.U, *Ug, RB, mass, U.sigma, U.pi, U.s, U.p, U.t);

  Coordinate latt = grid->GlobalDimensions();
  int T = latt[Nd - 1];
  Gamma g5(Gamma::Algebra::Gamma5);

  std::vector<ComplexD> L(T, ComplexD(0, 0));

  for (int h = 0; h < n_noise; ++h) {
    TXQCDFermionNf src(grid), b(grid), x(grid);
    src.f[0] = Zero();
    gaussian(pRNG, src.f[1]);

    Mop.Mdag(src, b);
    TxqcdCG(Mop, b, x, tol, maxit);

    LatticeFermion g5x(grid);
    g5x = g5 * x.f[0];
    LatticeComplex loop_field(grid);
    loop_field = localInnerProduct(src.f[1], g5x);

    std::vector<TComplex> sl;
    sliceSum(loop_field, sl, Nd - 1);
    for (int t = 0; t < T; ++t)
      L[t] += TensorRemove(sl[t]) / (2.0 * n_noise);
  }
  return L;
}

// Plain-Wilson QCD point-source propagator.
static void QcdPointProp(LatticePropagator &S, LatticeGaugeField &Umu,
                         RealD mass, GridCartesian &Grid,
                         GridRedBlackCartesian &RBGrid,
                         const Coordinate &src_site, RealD tol, int maxit) {
  WilsonFermionD Dw(Umu, Grid, RBGrid, mass);
  MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(tol, maxit);

  LatticePropagator src(&Grid);
  PointSource(src_site, src);
  S = Zero();

  for (int spin = 0; spin < Ns; ++spin) {
    for (int col = 0; col < Nc; ++col) {
      LatticeFermion src_ferm(&Grid), b(&Grid), x(&Grid);
      PropToFerm<WilsonImplR>(src_ferm, src, spin, col);
      Dw.Mdag(src_ferm, b);
      x = Zero();
      CG(HermOp, b, x);
      FermToProp<WilsonImplR>(S, x, spin, col);
    }
  }
}

// ------------- aux-field correlators (TXQCD only) ---------------------

// Per-element zero-momentum slice sums for an Nf×Nf flavor matrix field.
// Returns Nf*Nf vectors of length T: out[a*Nf+b][t] = Σ_{x∈t} π_{ab}(x).
static std::vector<std::vector<ComplexD>>
SliceSumPiAll(const LatticePiField &piF) {
  GridBase *g = piF.Grid();
  Coordinate latt = g->GlobalDimensions();
  int T = latt[Nd - 1];
  const int Nf2 = TxqcdNf * TxqcdNf;
  std::vector<std::vector<ComplexD>> out(Nf2, std::vector<ComplexD>(T));
  for (int a = 0; a < TxqcdNf; ++a) {
    for (int b = 0; b < TxqcdNf; ++b) {
      LatticeComplex piab(g);
      piab = PeekIndex<2>(piF, a, b);
      std::vector<TComplex> sl;
      sliceSum(piab, sl, Nd - 1);
      for (int t = 0; t < T; ++t)
        out[a * TxqcdNf + b][t] = TensorRemove(sl[t]);
    }
  }
  return out;
}

// Flavor trace slice sum: Tr_f[π](t) = Σ_a Σ_{x∈t} π_{aa}(x).
static std::vector<ComplexD> SliceSumPiTrace(const LatticePiField &piF) {
  GridBase *g = piF.Grid();
  LatticeComplex tr_pi(g);
  tr_pi = trace(piF);
  std::vector<TComplex> sl;
  sliceSum(tr_pi, sl, Nd - 1);
  int T = (int)sl.size();
  std::vector<ComplexD> out(T);
  for (int t = 0; t < T; ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

// Volume-averaged correlator from per-timeslice data:
//   C(Dt) = (1/V) sum_{t0} f(t0 + Dt) * conj(f(t0)).
static std::vector<ComplexD>
CorrelatorFromSlice(const std::vector<ComplexD> &pi_s, RealD V) {
  int T = (int)pi_s.size();
  std::vector<ComplexD> C(T, ComplexD(0, 0));
  for (int Dt = 0; Dt < T; ++Dt) {
    for (int t0 = 0; t0 < T; ++t0) {
      int t1 = (t0 + Dt) % T;
      C[Dt] += pi_s[t1] * std::conj(pi_s[t0]);
    }
    C[Dt] /= V;
  }
  return C;
}

// Zero-momentum slice sum of color trace: Tr_c[p](t) = sum_{x in t} sum_i p^{ii}(x).
static std::vector<ComplexD> SliceSumPTrace(const LatticePFieldC &pF) {
  GridBase *g = pF.Grid();
  LatticeComplex tr_p(g);
  tr_p = trace(pF);
  std::vector<TComplex> sl;
  sliceSum(tr_p, sl, Nd - 1);
  int T = (int)sl.size();
  std::vector<ComplexD> out(T);
  for (int t = 0; t < T; ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

// Volume-averaged color-p 2pt using color trace (singlet projection):
//   C(Dt) = (1/V) sum_{t0} Tr_c[p](t0+Dt) * conj(Tr_c[p](t0)).
static std::vector<ComplexD>
AuxPCorrelatorColorVol(const LatticePFieldC &pF) {
  RealD V = 1.0;
  Coordinate latt = pF.Grid()->GlobalDimensions();
  for (int mu = 0; mu < Nd; ++mu) V *= latt[mu];
  return CorrelatorFromSlice(SliceSumPTrace(pF), V);
}

// ------------- stochastic Tr(M^{-1}) estimators -----------------------

// Stochastic estimate of (Re Tr M^{-1}) / V using complex Gaussian noise on
// TXQCDFermionNf: E[eta^dag eta] = 2 * (Ns * Nc * Nf * V), so
//   E[eta^dag M^{-1} eta] = 2 Re Tr M^{-1}.
// Solve M^dag M x = M^dag eta, giving x = M^{-1} eta; inner product with eta.
static RealD TxqcdStochasticTrMinvV(TXQCDWilsonOp &Mop, GridBase *grid,
                                    GridParallelRNG &pRNG, int n_noise,
                                    RealD tol, int maxit) {
  RealD V = (RealD)grid->gSites();
  RealD acc = 0.0;
  for (int h = 0; h < n_noise; ++h) {
    TXQCDFermionNf eta(grid), Mdeta(grid), x(grid);
    for (int a = 0; a < TxqcdNf; ++a) gaussian(pRNG, eta.f[a]);
    Mop.Mdag(eta, Mdeta);
    x = Zero();
    TXQCDFermionNf r(grid), p(grid), Mp(grid), MdMp(grid);
    r = Mdeta; p = r;
    RealD rsq = norm2(r);
    RealD bsq = std::max(norm2(Mdeta), 1e-30);
    RealD tol2 = tol * tol * bsq;
    for (int it = 0; it < maxit; ++it) {
      Mop.M(p, Mp); Mop.Mdag(Mp, MdMp);
      ComplexD pAp = innerProduct(p, MdMp);
      ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
      axpy(x,  alpha, p);
      axpy(r, -alpha, MdMp);
      RealD rsq_new = norm2(r);
      if (rsq_new < tol2) { rsq = rsq_new; break; }
      RealD beta = rsq_new / rsq;
      for (int aa = 0; aa < TxqcdNf; ++aa) p.f[aa] = r.f[aa] + beta * p.f[aa];
      rsq = rsq_new;
    }
    ComplexD dot = innerProduct(eta, x);
    acc += dot.real() / (2.0 * V);
  }
  return acc / n_noise;
}

// Stochastic estimate of (Re Tr M_Wilson^{-1}) / V using a single-flavor
// WilsonFermionD. The complex-Gaussian noise normalization gives
// E[eta^dag M^{-1} eta] / (2V) = Re Tr M_W^{-1} / V summed over spin and color
// of the single flavor.
static RealD QcdStochasticTrMinvV(WilsonFermionD &Dw, GridBase *grid,
                                  GridParallelRNG &pRNG, int n_noise,
                                  RealD tol, int maxit) {
  RealD V = (RealD)grid->gSites();
  MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(tol, maxit);
  RealD acc = 0.0;
  for (int h = 0; h < n_noise; ++h) {
    LatticeFermion eta(grid), b(grid), x(grid);
    gaussian(pRNG, eta);
    Dw.Mdag(eta, b);
    x = Zero();
    CG(HermOp, b, x);
    ComplexD dot = innerProduct(eta, x);
    acc += dot.real() / (2.0 * V);
  }
  return acc / n_noise;
}

// ------------------- TXQCD per-traj observer --------------------------

struct TxqcdCorrObs : public HmcObservable<TXQCDField> {
  int n_therm, meas_skip;
  RealD mass, tol, lambda;
  int maxit, n_noise;
  Coordinate src_site;
  std::vector<std::vector<RealD>> pion;        // quark-prop pion
  std::vector<std::vector<ComplexD>> nucl;     // quark-prop nucleon
  std::vector<std::vector<ComplexD>> pi_I0_aux; // aux-pi I=0 sigma: Tr_f[π] Tr_f[π]*
  std::vector<std::vector<ComplexD>> pi_I1_aux; // aux-pi I=1 pion: Tr_f[π π†]
  std::vector<std::vector<ComplexD>> p_aux;     // color-p 2pt (trace-trace)
  // Raw slice sums per config for mean-subtracted (connected) correlators.
  std::vector<std::vector<ComplexD>> pi_trace_slice;       // [cfg][t] = Tr_f[π](t)
  std::vector<std::vector<ComplexD>> p_trace_slice;        // [cfg][t] = Tr_c[p](t)
  std::vector<RealD> tr_sigma;   // <Tr sigma>/V
  std::vector<RealD> tr_s;       // <Tr s>/V
  std::vector<RealD> tr_minv;    // stochastic Re Tr M_TXQCD^{-1}/V
  std::vector<std::vector<RealD>> pion_disc; // disconnected pion per config

  TxqcdCorrObs(int ntherm, int mskip, RealD m, RealD lam, RealD t, int it,
               int nn, Coordinate s)
      : n_therm(ntherm), meas_skip(mskip), mass(m), tol(t), lambda(lam),
        maxit(it), n_noise(nn), src_site(s) {}

  void TrajectoryComplete(int traj, TXQCDField &U, GridSerialRNG &,
                          GridParallelRNG &pRNG) override {
    if (traj <= n_therm) return;
    if ((traj - n_therm) % meas_skip != 0) return;
    GridBase *g = U.Grid();
    RealD V = (RealD)g->gSites();

    // Quark-prop correlators (also extracts source-point off-diagonal loop).
    LatticePropagator S_u(g), S_d(g);
    ComplexD L_du_src;
    TxqcdPointProp(S_u, S_d, L_du_src, U, mass, src_site, tol, maxit);
    pion.push_back(PionCorrelator(S_d, S_u));
    nucl.push_back(NucleonCorrelator(S_u, S_d));

    // Disconnected pion: Disc(t) = L_ud(t) × L_du(src)
    // where L_ud(t) = Σ_{x∈t} Tr[G_{ud}(x,x) γ₅] estimated stochastically,
    // and L_du(src) = Tr[G_{du}(src,src) γ₅] from the point-source solve.
    // Full pion (Wick) = Disc - Connected, so
    // PionCorrelator_full = PionCorrelator_connected - Disc.
    auto L_ud = TxqcdStochasticLoop_ud(U, mass, g, pRNG, n_noise, tol, maxit);
    Coordinate latt_dim = g->GlobalDimensions();
    int Tdisc = latt_dim[Nd - 1];
    std::vector<RealD> disc(Tdisc);
    for (int t = 0; t < Tdisc; ++t)
      disc[t] = (L_ud[t] * L_du_src).real();
    pion_disc.push_back(disc);

    std::cout << GridLogMessage << "[2pt TXQCD] traj=" << traj
              << " L_du_src=" << L_du_src << " |L_ud(0)|=" << std::abs(L_ud[0])
              << std::endl;

    // Volume-averaged aux correlators + raw slice sums for connected build.
    auto pi_all = SliceSumPiAll(U.pi);
    auto pi_tr  = SliceSumPiTrace(U.pi);
    int T = (int)pi_tr.size();
    pi_trace_slice.push_back(pi_tr);

    const RealD lam4 = lambda * lambda * lambda * lambda;

    // I=0 sigma: λ⁴ <Tr_f[π] Tr_f[π]*>
    auto c_I0 = CorrelatorFromSlice(pi_tr, V);
    std::vector<ComplexD> ca0(T);
    for (int t = 0; t < T; ++t) ca0[t] = lam4 * c_I0[t];
    pi_I0_aux.push_back(ca0);

    // I=1 pion: λ⁴ [Tr_f(π π†) - Tr_f(π) Tr_f(π†)]  (traceless projection)
    std::vector<ComplexD> ca1(T, ComplexD(0, 0));
    for (const auto &ps : pi_all) {
      auto Cab = CorrelatorFromSlice(ps, V);
      for (int t = 0; t < T; ++t) ca1[t] += lam4 * Cab[t];
    }
    for (int t = 0; t < T; ++t) ca1[t] -= ca0[t];
    pi_I1_aux.push_back(ca1);

    auto ptr = SliceSumPTrace(U.p);
    p_trace_slice.push_back(ptr);
    auto c_p = CorrelatorFromSlice(ptr, V);
    std::vector<ComplexD> pa(T);
    for (int t = 0; t < T; ++t) pa[t] = lam4 * c_p[t];
    p_aux.push_back(pa);

    // Aux-field VEVs.
    tr_sigma.push_back(TensorRemove(sum(trace(U.sigma))).real() / V);
    tr_s.push_back(TensorRemove(sum(trace(U.s))).real() / V);

    // Stochastic Re Tr M_TXQCD^{-1} / V.
    GridCartesian *Ug = dynamic_cast<GridCartesian *>(g);
    GridRedBlackCartesian RB(Ug);
    TXQCDWilsonOp Mop(U.U, *Ug, RB, mass,
                      U.sigma, U.pi, U.s, U.p, U.t);
    tr_minv.push_back(
        TxqcdStochasticTrMinvV(Mop, g, pRNG, n_noise, tol, maxit));
  }
};

// ------------------------- stats helpers --------------------------------

static RealD vmean(const std::vector<RealD> &v) {
  RealD s = 0; for (auto x : v) s += x; return s / v.size();
}
static RealD vstderr(const std::vector<RealD> &v) {
  RealD m = vmean(v), s2 = 0;
  for (auto x : v) s2 += (x - m) * (x - m);
  return std::sqrt(s2 / (v.size() * (v.size() - 1)));
}

// ----------------------------------- main -------------------------------

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 8});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  RealD beta   = 5.6;
  RealD lambda = 3.0;
  RealD mass   = 0.3;
  RealD cg_tol = 1e-8;
  int   cg_max = 10000;
  int   n_therm = 100;
  int   n_meas  = 100;
  int   meas_skip = 10;  // measure every meas_skip trajectories
  RealD meas_tol = 1e-10;

  Coordinate src_site(std::vector<int>{0, 0, 0, 0});

  int first_traj = n_therm + 1;
  int last_traj  = n_therm + n_meas;

  // ================ TXQCD (single aux, ONE rational PF) ==========
  //
  // Clean Nf=2 setup. TXQCDWilsonRationalPseudoFermionAction gives weight
  // |det M_TX|^1. With M_TX's internal Nf_tx=2 flavor block, at aux=0 this
  // is (det M_W)^2 = Nf=2 Wilson effective, exactly matched by the
  // single-TwoFlavour-PF QCD block below.

  // Rational approx bracket: lo/hi must contain the spectrum of M^dag M on
  // typical gauge configs at this mass. At mass=0.3, beta=5.6 the smallest
  // eigenvalue is well above ~1e-3 on a 4^3x8 volume; hi=64 brackets the
  // operator norm of the Wilson kernel. Degree 12 gives Remez error <1e-9.
  OneFlavourRationalParams rat_params(/*lo=*/1e-4, /*hi=*/64.0,
                                      /*maxit=*/cg_max,
                                      /*tol=*/cg_tol,
                                      /*degree=*/12,
                                      /*precision=*/64,
                                      /*BoundsCheckFreq=*/100,
                                      /*mdtol=*/1e-6,
                                      /*BoundsCheckTol=*/1e-4);

  std::vector<std::vector<RealD>>    pion_txqcd;
  std::vector<std::vector<RealD>>    pion_disc_txqcd;
  std::vector<std::vector<ComplexD>> nucl_txqcd;
  std::vector<std::vector<ComplexD>> pi_I0_aux_txqcd;
  std::vector<std::vector<ComplexD>> pi_I1_aux_txqcd;
  std::vector<std::vector<ComplexD>> p_aux_txqcd;
  std::vector<std::vector<ComplexD>> pi_trace_slice_txqcd;
  std::vector<std::vector<ComplexD>> p_trace_slice_txqcd;
  std::vector<RealD> tr_sigma_txqcd, tr_s_txqcd, tr_minv_txqcd;
  {
    std::string dir = "configs_2pt_txqcd";
    std::string cfg = dir + "/ckpoint_lat";
    std::string rng = dir + "/ckpoint_rng";
    t2pt_mkdir_p(dir);

    GaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta);
    AuxiliaryFieldGaussianAction           AuxAction(lambda);
    TXQCDWilsonRationalPseudoFermionAction PF(Grid, RBGrid, mass, rat_params);

    typedef Representations<EmptyRep<TXQCDField>> Reps;
    ActionLevel<TXQCDField, Reps> L1(1);
    L1.push_back(&PF); L1.push_back(&AuxAction);
    ActionLevel<TXQCDField, Reps> L2(4);
    L2.push_back(&GaugeAction);
    ActionSet<TXQCDField, Reps> Aset;
    Aset.push_back(L1); Aset.push_back(L2);

    IntegratorParameters MD;
    MD.name = "LeapFrog"; MD.MDsteps = 80; MD.trajL = 0.5;
    HMCparameters HMCp;
    HMCp.StartTrajectory = 0;
    HMCp.Trajectories    = n_meas;
    HMCp.NoMetropolisUntil = n_therm;
    HMCp.MetropolisTest = true;
    HMCp.PerformRandomShift = false;
    HMCp.StartingType = "ColdStart";
    HMCp.MD = MD;

    NoSmearing<TXQCDCompositeImpl> Smear;
    typedef LeapFrog<TXQCDCompositeImpl,
                     NoSmearing<TXQCDCompositeImpl>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);

    TXQCDField U(&Grid);
    TXQCDCompositeImpl::ColdConfiguration(pRNG, U);
    Smear.set_Field(U);

    int n_noise = 4;
    TxqcdCorrObs obs(n_therm, meas_skip, mass, lambda, meas_tol, cg_max,
                     n_noise, src_site);

    CheckpointerParameters CPp;
    CPp.config_prefix = cfg; CPp.rng_prefix = rng;
    CPp.saveInterval = 1; CPp.format = "IEEE64BIG";
    TXQCDCheckpointer ckpt(CPp);

    bool cached = txqcd_configs_cached(cfg, rng, first_traj, last_traj);
    if (cached) {
      std::cout << GridLogMessage << "2pt test: reusing cached TXQCD configs "
                << "from " << dir << std::endl;
      for (int t = first_traj; t <= last_traj; ++t) {
        ckpt.CheckpointRestore(t, U, sRNG, pRNG);
        obs.TrajectoryComplete(t, U, sRNG, pRNG);
      }
    } else {
      std::cout << GridLogMessage << "2pt test: generating TXQCD configs in "
                << dir << std::endl;
      std::vector<HmcObservable<TXQCDField> *> Obs = {&obs, &ckpt};
      HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
      HMC.evolve();
    }
    pion_txqcd      = std::move(obs.pion);
    pion_disc_txqcd = std::move(obs.pion_disc);
    nucl_txqcd      = std::move(obs.nucl);
    pi_I0_aux_txqcd = std::move(obs.pi_I0_aux);
    pi_I1_aux_txqcd = std::move(obs.pi_I1_aux);
    p_aux_txqcd    = std::move(obs.p_aux);
    pi_trace_slice_txqcd = std::move(obs.pi_trace_slice);
    p_trace_slice_txqcd = std::move(obs.p_trace_slice);
    tr_sigma_txqcd = std::move(obs.tr_sigma);
    tr_s_txqcd     = std::move(obs.tr_s);
    tr_minv_txqcd  = std::move(obs.tr_minv);
  }

  // ================== QCD (Nf=2, one TwoFlavour PF) ====================
  //
  // Matches TXQCD block's Nf=2 zero-aux weight with one TwoFlavour action
  // -> |det M_W|^2. The Fierz test compares observables between these two
  // ensembles: disagreement is a bug; agreement (within stats) confirms the
  // TXQCD aux-integration reproduces the Wilson determinant.

  std::vector<std::vector<RealD>>    pion_qcd;
  std::vector<std::vector<ComplexD>> nucl_qcd;
  std::vector<RealD> tr_minv_qcd;
  {
    std::string dir = "configs_2pt_qcd_nf2";
    std::string cfg = dir + "/ckpoint_lat";
    std::string rng = dir + "/ckpoint_rng";
    t2pt_mkdir_p(dir);

    GridSerialRNG   sRNG_q;
    GridParallelRNG pRNG_q(&Grid);
    sRNG_q.SeedFixedIntegers({11, 12, 13, 14, 15});
    pRNG_q.SeedFixedIntegers({16, 17, 18, 19, 20});

    LatticeGaugeField Umu(&Grid);
    SU<Nc>::ColdConfiguration(Umu);

    WilsonFermionD FermOp(Umu, Grid, RBGrid, mass);
    ConjugateGradient<LatticeFermion> CG(cg_tol, cg_max);
    TwoFlavourPseudoFermionAction<WilsonImplR> Nf2(FermOp, CG, CG);
    Nf2.is_smeared = false;

    WilsonGaugeActionR GaugeAction(beta);

    typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
    ActionLevel<LatticeGaugeField, Reps> L1(1);
    L1.push_back(&Nf2);
    ActionLevel<LatticeGaugeField, Reps> L2(4);
    L2.push_back(&GaugeAction);
    ActionSet<LatticeGaugeField, Reps> Aset;
    Aset.push_back(L1); Aset.push_back(L2);

    IntegratorParameters MD;
    MD.name = "LeapFrog"; MD.MDsteps = 80; MD.trajL = 0.5;
    HMCparameters HMCp;
    HMCp.StartTrajectory = 0;
    HMCp.Trajectories    = n_meas;
    HMCp.NoMetropolisUntil = n_therm;
    HMCp.MetropolisTest = true;
    HMCp.PerformRandomShift = false;
    HMCp.StartingType = "ColdStart";
    HMCp.MD = MD;

    NoSmearing<PeriodicGimplR> Smear;
    typedef LeapFrog<PeriodicGimplR, NoSmearing<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);

    Smear.set_Field(Umu);

    struct QcdObs : public HmcObservable<LatticeGaugeField> {
      int n_therm, meas_skip;
      RealD mass; RealD tol; int maxit; int n_noise;
      Coordinate src_site;
      GridCartesian *Ug; GridRedBlackCartesian *RBg;
      std::vector<std::vector<RealD>> pion;
      std::vector<std::vector<ComplexD>> nucl;
      std::vector<RealD> tr_minv;
      void TrajectoryComplete(int t, LatticeGaugeField &U, GridSerialRNG &,
                              GridParallelRNG &pRNG) override {
        if (t <= n_therm) return;
        if ((t - n_therm) % meas_skip != 0) return;
        LatticePropagator S(U.Grid());
        QcdPointProp(S, U, mass, *Ug, *RBg, src_site, tol, maxit);
        pion.push_back(PionCorrelator(S, S));
        nucl.push_back(NucleonCorrelator(S, S));
        WilsonFermionD Dw(U, *Ug, *RBg, mass);
        tr_minv.push_back(
            QcdStochasticTrMinvV(Dw, U.Grid(), pRNG, n_noise, tol, maxit));
      }
    };
    QcdObs obs;
    obs.n_therm = n_therm; obs.meas_skip = meas_skip;
    obs.mass = mass; obs.tol = meas_tol;
    obs.maxit = cg_max; obs.n_noise = 4; obs.src_site = src_site;
    obs.Ug = &Grid; obs.RBg = &RBGrid;

    struct GaugeCkpt : public HmcObservable<LatticeGaugeField> {
      std::string cfg_prefix, rng_prefix;
      typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
      void TrajectoryComplete(int t, LatticeGaugeField &U, GridSerialRNG &sR,
                              GridParallelRNG &pR) override {
        std::ostringstream cf, rf;
        cf << cfg_prefix << "." << t;
        rf << rng_prefix << "." << t;
        NerscIO::writeRNGState(sR, pR, rf.str());
        NerscIO::writeConfiguration<GaugeStats>(U, cf.str(), 0, 1);
      }
    };
    GaugeCkpt ckpt;
    ckpt.cfg_prefix = cfg; ckpt.rng_prefix = rng;

    bool cached = qcd_configs_cached(cfg, rng, first_traj, last_traj);
    if (cached) {
      std::cout << GridLogMessage << "2pt test: reusing cached QCD configs "
                << "from " << dir << std::endl;
      for (int t = first_traj; t <= last_traj; ++t) {
        std::ostringstream cf, rf;
        cf << cfg << "." << t;
        rf << rng << "." << t;
        FieldMetaData header;
        NerscIO::readRNGState(sRNG_q, pRNG_q, header, rf.str());
        NerscIO::readConfiguration<typename GaugeCkpt::GaugeStats>(Umu, header,
                                                                   cf.str());
        obs.TrajectoryComplete(t, Umu, sRNG_q, pRNG_q);
      }
    } else {
      std::cout << GridLogMessage << "2pt test: generating QCD Nf=2 configs in "
                << dir << std::endl;
      std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&obs, &ckpt};
      HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG_q, pRNG_q, Obs, Umu);
      HMC.evolve();
    }
    pion_qcd     = std::move(obs.pion);
    nucl_qcd     = std::move(obs.nucl);
    tr_minv_qcd  = std::move(obs.tr_minv);
  }

  // ========================= compare ===================================

  int T = latt[Nd - 1];
  int N = (int)pion_txqcd.size();
  int M = (int)pion_qcd.size();
  std::cout << GridLogMessage << "TXQCD samples: " << N
            << "   QCD samples: " << M << std::endl;

  auto per_t_mean = [&](const std::vector<std::vector<RealD>> &X, int t) {
    std::vector<RealD> col; col.reserve(X.size());
    for (auto &row : X) col.push_back(row[t]);
    return std::make_pair(vmean(col), vstderr(col));
  };
  auto per_t_mean_c = [&](const std::vector<std::vector<ComplexD>> &X, int t) {
    std::vector<RealD> col; col.reserve(X.size());
    for (auto &row : X) col.push_back(row[t].real());
    return std::make_pair(vmean(col), vstderr(col));
  };

  int exitcode = 0;

  // Build full pion = connected - disconnected per config.
  std::vector<std::vector<RealD>> pion_full_txqcd(N);
  for (int c = 0; c < N; ++c) {
    pion_full_txqcd[c].resize(T);
    for (int t = 0; t < T; ++t)
      pion_full_txqcd[c][t] = pion_txqcd[c][t] - pion_disc_txqcd[c][t];
  }

  std::cout << GridLogMessage
            << "----- Pion correlator (connected only) -----" << std::endl;
  std::cout << GridLogMessage
            << "t    TXQCD C_pi_conn(t)       QCD C_pi(t)             nsigma"
            << std::endl;
  for (int t = 0; t < T; ++t) {
    auto [tm, te] = per_t_mean(pion_txqcd, t);
    auto [qm, qe] = per_t_mean(pion_qcd, t);
    RealD diff = tm - qm;
    RealD derr = std::sqrt(te * te + qe * qe);
    RealD ns = (derr > 0) ? std::abs(diff) / derr : 0.0;
    std::cout << GridLogMessage
              << t << "    " << tm << " +/- " << te
              << "    " << qm << " +/- " << qe
              << "    " << ns << std::endl;
  }

  std::cout << GridLogMessage
            << "----- Pion disconnected (TXQCD) -----" << std::endl;
  std::cout << GridLogMessage << "t    Disc(t)" << std::endl;
  for (int t = 0; t < T; ++t) {
    auto [dm, de] = per_t_mean(pion_disc_txqcd, t);
    std::cout << GridLogMessage
              << t << "    " << dm << " +/- " << de << std::endl;
  }

  std::cout << GridLogMessage
            << "----- Pion correlator (conn - disc, Fierz test) -----"
            << std::endl;
  std::cout << GridLogMessage
            << "t    TXQCD C_pi_full(t)       QCD C_pi(t)             nsigma"
            << std::endl;
  for (int t = 0; t < T; ++t) {
    auto [tm, te] = per_t_mean(pion_full_txqcd, t);
    auto [qm, qe] = per_t_mean(pion_qcd, t);
    RealD diff = tm - qm;
    RealD derr = std::sqrt(te * te + qe * qe);
    RealD ns = (derr > 0) ? std::abs(diff) / derr : 0.0;
    bool pass = ns < 3.0;
    std::cout << GridLogMessage
              << t << "    " << tm << " +/- " << te
              << "    " << qm << " +/- " << qe
              << "    " << ns << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  std::cout << GridLogMessage << "----- Proton correlator (Re) -----"
            << std::endl;
  std::cout << GridLogMessage
            << "t    TXQCD C_N(t)             QCD C_N(t)              nsigma"
            << std::endl;
  for (int t = 0; t < T; ++t) {
    auto [tm, te] = per_t_mean_c(nucl_txqcd, t);
    auto [qm, qe] = per_t_mean_c(nucl_qcd, t);
    RealD diff = tm - qm;
    RealD derr = std::sqrt(te * te + qe * qe);
    RealD ns = (derr > 0) ? std::abs(diff) / derr : 0.0;
    bool pass = ns < 3.0;
    std::cout << GridLogMessage
              << t << "    " << tm << " +/- " << te
              << "    " << qm << " +/- " << qe
              << "    " << ns << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // Aux-pi I=0 "sigma" channel: λ⁴ <Tr_f[π](t) Tr_f[π]*(0)>
  std::cout << GridLogMessage
            << "----- Aux pi I=0 sigma correlator (TXQCD diagnostic) -----"
            << std::endl;
  std::cout << GridLogMessage << "t    C_pi_I0(t)" << std::endl;
  for (int t = 0; t < T; ++t) {
    auto [am, ae] = per_t_mean_c(pi_I0_aux_txqcd, t);
    std::cout << GridLogMessage
              << t << "    " << am << " +/- " << ae << std::endl;
  }

  // Aux-pi I=1 pion channel: λ⁴ Tr_f[π(t) π†(0)] vs quark C_pi(t).
  std::cout << GridLogMessage
            << "----- Aux pi I=1 pion correlator (TXQCD, Fierz diagnostic) -----"
            << std::endl;
  std::cout << GridLogMessage
            << "t    C_pi_I1(t)               C_pi_quark(t)           nsigma"
            << std::endl;
  for (int t = 0; t < T; ++t) {
    auto [am, ae] = per_t_mean_c(pi_I1_aux_txqcd, t);
    auto [qm, qe] = per_t_mean(pion_txqcd, t);
    RealD diff = am - qm;
    RealD derr = std::sqrt(ae * ae + qe * qe);
    RealD ns = (derr > 0) ? std::abs(diff) / derr : 0.0;
    std::cout << GridLogMessage
              << t << "    " << am << " +/- " << ae
              << "    " << qm << " +/- " << qe
              << "    " << ns << std::endl;
  }

  std::cout << GridLogMessage
            << "----- Aux color-p 2pt (TXQCD diagnostic) -----" << std::endl;
  std::cout << GridLogMessage << "t    C_p_aux(t)" << std::endl;
  for (int t = 0; t < T; ++t) {
    auto [am, ae] = per_t_mean_c(p_aux_txqcd, t);
    std::cout << GridLogMessage
              << t << "    " << am << " +/- " << ae << std::endl;
  }

  // --------- Connected (mean-subtracted) aux correlators ---------------
  // C_conn(Dt) = C_total(Dt) - C_disc(Dt) where
  //   C_disc(Dt) = (1/V) sum_{t0} <pi_s(t0+Dt)> conj(<pi_s(t0)>)
  // using ensemble-averaged slice means. Per-config C_conn computed as
  //   (pi_s - <pi_s>)(pi_s - <pi_s>)*, averaged across configs.
  {
    RealD V4 = 1.0;
    for (int mu = 0; mu < Nd; ++mu) V4 *= latt[mu];
    int Ncfg = (int)pi_trace_slice_txqcd.size();
    const RealD lam4 = lambda * lambda * lambda * lambda;

    // I=0 connected (mean-subtracted Tr_f[π]):
    std::vector<ComplexD> pi_tr_mean(T, ComplexD(0, 0));
    for (int c = 0; c < Ncfg; ++c)
      for (int t = 0; t < T; ++t) pi_tr_mean[t] += pi_trace_slice_txqcd[c][t];
    for (int t = 0; t < T; ++t) pi_tr_mean[t] /= RealD(Ncfg);

    std::vector<std::vector<ComplexD>> pi_I0_conn(Ncfg, std::vector<ComplexD>(T));
    for (int c = 0; c < Ncfg; ++c) {
      std::vector<ComplexD> dpi(T);
      for (int t = 0; t < T; ++t) dpi[t] = pi_trace_slice_txqcd[c][t] - pi_tr_mean[t];
      auto Cc = CorrelatorFromSlice(dpi, V4);
      for (int t = 0; t < T; ++t) pi_I0_conn[c][t] = lam4 * Cc[t];
    }

    std::cout << GridLogMessage
              << "----- Aux pi I=0 CONNECTED (mean-subtracted) -----"
              << std::endl;
    std::cout << GridLogMessage << "t    C_pi_I0_conn(t)" << std::endl;
    for (int t = 0; t < T; ++t) {
      auto [am, ae] = per_t_mean_c(pi_I0_conn, t);
      std::cout << GridLogMessage
                << t << "    " << am << " +/- " << ae << std::endl;
    }

    // Connected aux-p (trace-trace, mean-subtracted):
    //   Tr_c[p] at source and sink gives the color-singlet projection.
    std::vector<ComplexD> ptr_mean(T, ComplexD(0, 0));
    for (int c = 0; c < Ncfg; ++c)
      for (int t = 0; t < T; ++t) ptr_mean[t] += p_trace_slice_txqcd[c][t];
    for (int t = 0; t < T; ++t) ptr_mean[t] /= RealD(Ncfg);

    std::vector<std::vector<ComplexD>> p_aux_conn(Ncfg,
                                                  std::vector<ComplexD>(T));
    for (int c = 0; c < Ncfg; ++c) {
      std::vector<ComplexD> dp(T);
      for (int t = 0; t < T; ++t)
        dp[t] = p_trace_slice_txqcd[c][t] - ptr_mean[t];
      auto Cc = CorrelatorFromSlice(dp, V4);
      for (int t = 0; t < T; ++t) p_aux_conn[c][t] = lam4 * Cc[t];
    }

    std::cout << GridLogMessage
              << "----- Aux color-p 2pt, CONNECTED (mean-subtracted) -----"
              << std::endl;
    std::cout << GridLogMessage
              << "t    C_p_aux_total(t)         C_p_aux_conn(t)"
              << std::endl;
    for (int t = 0; t < T; ++t) {
      auto [tm, te] = per_t_mean_c(p_aux_txqcd, t);
      auto [cm, ce] = per_t_mean_c(p_aux_conn, t);
      std::cout << GridLogMessage
                << t << "    " << tm << " +/- " << te
                << "    " << cm << " +/- " << ce << std::endl;
    }

    std::cout << GridLogMessage
              << "<Tr_c p>_slice-avg (V_3 units) = " << ptr_mean[0]
              << std::endl;
  }

  // --------------------------- VEV checks ------------------------------
  // Identities being checked:
  //   (SD)    lambda^2 <Tr sigma>/V = <Re Tr M_TXQCD^{-1}>/V
  //           The rational PF gives weight |det M_TX|^1, so
  //             d ln|det M_TX|/dsigma_{aa} = Re Tr(M_TX^{-1} * dM/dsigma_{aa}),
  //           a single factor (NOT 2, which only applies to |det M_TX|^2
  //           TwoFlavour PFs). The complex-Gaussian factor 2 from
  //           E[eta^dag A eta] = 2 Tr A is absorbed inside
  //           TxqcdStochasticTrMinvV's /(2V) normalization so mtx_m is already
  //           <Re Tr M_TXQCD^{-1}>/V.
  //   (Fierz) per-flavor TXQCD vs QCD <Re Tr M^{-1}>: matched when both
  //           ensembles have the same effective Nf Wilson dressing.

  std::cout << GridLogMessage << "----- VEV checks -----" << std::endl;

  auto sm_r = [&](const std::vector<RealD> &v) {
    return std::make_pair(vmean(v), vstderr(v));
  };

  auto [sig_m, sig_e] = sm_r(tr_sigma_txqcd);
  auto [s_m,   s_e]   = sm_r(tr_s_txqcd);
  auto [mtx_m, mtx_e] = sm_r(tr_minv_txqcd);
  auto [mqc_m, mqc_e] = sm_r(tr_minv_qcd);

  // TxqcdStochasticTrMinvV sums over TxqcdNf internal flavors of M_TXQCD;
  // QcdStochasticTrMinvV runs a single-flavor WilsonFermionD.
  const int Nf_tx = TxqcdNf;        // TXQCD internal flavors (=2)
  const int Nf_qcd = 1;             // single-flavor Wilson in the QCD estimator
  const RealD sig_pf = sig_m / Nf_tx, sig_pf_e = sig_e / Nf_tx;
  const RealD s_pf   = s_m   / Nf_tx, s_pf_e   = s_e   / Nf_tx;
  const RealD mtx_pf = mtx_m / Nf_tx, mtx_pf_e = mtx_e / Nf_tx;
  const RealD mqc_pf = mqc_m / Nf_qcd, mqc_pf_e = mqc_e / Nf_qcd;

  std::cout << GridLogMessage << "<Tr sigma>/V_TXQCD = " << sig_m << " +/- "
            << sig_e << "  (per flavor: " << sig_pf << " +/- " << sig_pf_e
            << ")" << std::endl;
  std::cout << GridLogMessage << "<Tr s>/V_TXQCD     = " << s_m   << " +/- "
            << s_e   << "  (per flavor: " << s_pf << " +/- " << s_pf_e
            << ")" << std::endl;
  std::cout << GridLogMessage
            << "<Re Tr M_TXQCD^{-1}>/V = " << mtx_m << " +/- "
            << mtx_e << "  (per flavor: " << mtx_pf << " +/- " << mtx_pf_e
            << ")" << std::endl;
  std::cout << GridLogMessage
            << "<Re Tr M_W^{-1}>/V_QCD         = " << mqc_m << " +/- "
            << mqc_e << "  (per flavor: " << mqc_pf << " +/- " << mqc_pf_e
            << ")" << std::endl;

  // (SD) lambda^2 <Tr sigma>/V vs <Re Tr M_TXQCD^{-1}>/V.
  {
    RealD lhs   = lambda * lambda * sig_m;
    RealD lhs_e = lambda * lambda * sig_e;
    RealD rhs   = mtx_m;
    RealD rhs_e = mtx_e;
    RealD diff = lhs - rhs;
    RealD de   = std::sqrt(lhs_e * lhs_e + rhs_e * rhs_e);
    RealD ns   = (de > 0) ? std::abs(diff) / de : 0.0;
    bool  pass = ns < 3.0;
    std::cout << GridLogMessage
              << "[TXQCD SD] lambda^2 <Tr sigma>/V = " << lhs << " +/- "
              << lhs_e << "   <Re Tr M_TXQCD^{-1}>/V = " << rhs << " +/- "
              << rhs_e << "   (" << ns << " sigma)"
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // (Fierz) per-flavor: <Re Tr M_TXQCD^{-1}>/(V Nf_tx) vs QCD per-flavor.
  // TXQCD side (1 rational PF, |det M_TX|^1 with Nf_tx=2 internal -> (det M_W)^2)
  // and QCD side (1 TwoFlavour PF, |det M_W|^2) both give Nf=2 gauge weight.
  {
    RealD diff = mtx_pf - mqc_pf;
    RealD de   = std::sqrt(mtx_pf_e * mtx_pf_e + mqc_pf_e * mqc_pf_e);
    RealD ns   = (de > 0) ? std::abs(diff) / de : 0.0;
    bool  pass = ns < 3.0;
    std::cout << GridLogMessage
              << "[Fierz VEV per-flavor] TXQCD/Nf vs QCD/Nf: "
              << mtx_pf << " +/- " << mtx_pf_e << " vs "
              << mqc_pf << " +/- " << mqc_pf_e
              << "  diff = " << diff << " +/- " << de
              << " (" << ns << " sigma)"
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // Chained (per-flavor): lambda^2 <Tr sigma>/(V Nf_tx) vs QCD per-flavor.
  {
    RealD lhs   = lambda * lambda * sig_pf;
    RealD lhs_e = lambda * lambda * sig_pf_e;
    RealD diff  = lhs - mqc_pf;
    RealD de    = std::sqrt(lhs_e * lhs_e + mqc_pf_e * mqc_pf_e);
    RealD ns    = (de > 0) ? std::abs(diff) / de : 0.0;
    bool  pass  = ns < 3.0;
    std::cout << GridLogMessage
              << "[TXQCD aux vs QCD per-flavor] lambda^2<Tr sigma>/(V Nf_tx) = "
              << lhs << " +/- " << lhs_e << "   QCD/Nf = " << mqc_pf
              << " +/- " << mqc_pf_e << "   (" << ns << " sigma)"
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // <Tr sigma> / <Tr s> = sqrt(2) ratio test (independent of Fierz match).
  {
    RealD ratio    = sig_m / s_m;
    RealD ratio_e  = std::abs(ratio) * std::sqrt(
        (sig_e / sig_m) * (sig_e / sig_m) + (s_e / s_m) * (s_e / s_m));
    RealD expected = std::sqrt(2.0);
    RealD ns       = std::abs(ratio - expected) / ratio_e;
    bool  pass     = ns < 3.0;
    std::cout << GridLogMessage
              << "[ratio] <Tr sigma>/<Tr s> = " << ratio << " +/- " << ratio_e
              << " expected " << expected << " (" << ns << " sigma)"
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME 2pt CHECKS FAILED" : "ALL 2pt CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
