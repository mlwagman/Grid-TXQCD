// dtxqcd_dibaryon_core.h: shared measurement core for the PURE-AUX dibaryon
// (triple-diquark) two-point functions.  Factored verbatim out of
// meas_dibaryon_aux_dtxqcd.cc so the offline binary and the inline HMC
// observer (dtxqcd_dibaryon_inline.h) run BIT-IDENTICAL code paths.
//
// Physics/convention documentation lives in meas_dibaryon_aux_dtxqcd.cc
// (I=1/I=0 operators, contact-freedom, B=2 periodicity, offline grading).
//
// NOTE: WriteH5 takes the plaquette as an ARGUMENT -- avgPlaquette is a
// collective and must be called by all ranks, outside any IsBoss() guard.

#pragma once

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/utils/WilsonLoops.h>

namespace DtxqcdDibaryon {

using namespace Grid;

// isosinglet diquark colour matrix (post-Elitzur-fix convention, as in
// meas_baryon_dtxqcd)
inline LatticeColourMatrix IsoSingletDiquark(const LatticeDtxqcdD &dF) {
  GridBase *g = dF.Grid();
  LatticeColourMatrix out(g);
  out = Zero();
  static_assert(DtxqcdNf == 2, "isosinglet diquark assumes Nf=2");
  const ComplexD inv_sqrt2 = ComplexD(1.0 / std::sqrt(2.0), 0.0);
  autoView(dv, dF, CpuRead);
  autoView(ov, out, CpuWrite);
  thread_for(ss, g->oSites(), {
    for (int i = 0; i < Nc; ++i)
      for (int j = 0; j < Nc; ++j) {
        auto v = dv[ss]()(0, 1)(i, j) - dv[ss]()(1, 0)(i, j);
        ov[ss]()()(i, j) = inv_sqrt2 * v;
      }
  });
  return out;
}

// eps-reduced 3bar vector w^a = eps^{abc} d_iso(b,c) (full field, no mask)
inline LatticeColourVector EpsReduce(const LatticeColourMatrix &d_iso) {
  GridBase *g = d_iso.Grid();
  LatticeColourVector w(g);
  w = Zero();
  autoView(dv, d_iso, CpuRead);
  autoView(wv, w, CpuWrite);
  thread_for(ss, g->oSites(), {
    auto d = dv[ss]()();
    wv[ss]()()(0) = d(1, 2) - d(2, 1);
    wv[ss]()()(1) = d(2, 0) - d(0, 2);
    wv[ss]()()(2) = d(0, 1) - d(1, 0);
  });
  return w;
}

class MeasCore {
 public:
  struct Geo { Coordinate r1, r2; };

  MeasCore(GridCartesian &Grid_, int nstout, RealD rho, int LMAX,
           std::vector<int> smears, RealD salpha)
      : grid_(Grid_), nstout_(nstout), rho_(rho), LMAX_(LMAX),
        smears_(std::move(smears)), salpha_(salpha) {
    std::sort(smears_.begin(), smears_.end());
    T_ = grid_.FullDimensions()[Nd - 1];
    V3_ = 1.0;
    for (int mu = 0; mu < Nd - 1; ++mu) V3_ *= (RealD)grid_.FullDimensions()[mu];

    // ---- geometry lists ---------------------------------------------------
    // I=1: unordered displacement pairs (r1, r3) for the two w's; d6 at origin.
    // I=0: (r2, r3) with the first w at origin; all three sites distinct.
    auto vec = [&](int mu, int s, int L) {
      Coordinate r(4, 0);
      r[mu] = s * L;
      return r;
    };
    Coordinate zero(4, 0);
    geoI1.push_back({zero, zero});                       // single site
    for (int L = 1; L <= LMAX_; ++L) {
      for (int mu = 0; mu < 3; ++mu)
        for (int s : {+1, -1})
          geoI1.push_back({vec(mu, s, L), zero});        // (r, 0)
      for (int mu = 0; mu < 3; ++mu) {
        geoI1.push_back({vec(mu, +1, L), vec(mu, -1, L)});   // collinear
        geoI0.push_back({vec(mu, +1, L), vec(mu, -1, L)});
      }
      for (int mu = 0; mu < 3; ++mu)
        for (int s : {+1, -1})
          geoI1.push_back({vec(mu, s, L), vec(mu, s, L)});   // both same side
      for (int mu = 0; mu < 3; ++mu)
        for (int nu = mu + 1; nu < 3; ++nu)
          for (int s1 : {+1, -1})
            for (int s2 : {+1, -1}) {
              geoI1.push_back({vec(mu, s1, L), vec(nu, s2, L)});  // planar
              geoI0.push_back({vec(mu, s1, L), vec(nu, s2, L)});
            }
    }
  }

  int NG1() const { return (int)geoI1.size(); }
  int NG0() const { return (int)geoI0.size(); }
  int NLEV() const { return (int)smears_.size(); }
  int T() const { return T_; }

  // S1[level][geo][t], S0[level][geo][t]
  void ComputeAll(const DTXQCDField &F,
                  std::vector<std::vector<std::vector<ComplexD>>> &S1,
                  std::vector<std::vector<std::vector<ComplexD>>> &S0) const {
    GridCartesian &Grid = grid_;
    const int T = T_;
    const RealD V3 = V3_;
    const RealD salpha = salpha_;
    const int nlev = NLEV();

    // stout smear (after any twist: covariant)
    LatticeGaugeField Usm(F.U);
    if (nstout_ > 0) {
      Smear_Stout<PeriodicGimplR> stout(rho_);
      LatticeGaugeField tmp(&Grid);
      for (int i = 0; i < nstout_; ++i) { stout.smear(tmp, Usm); Usm = tmp; }
    }
    std::array<LatticeColourMatrix, 3> Umu{
        LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid),
        LatticeColourMatrix(&Grid)};
    std::array<LatticeColourMatrix, 3> cU{
        LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid),
        LatticeColourMatrix(&Grid)};
    std::array<LatticeColourMatrix, 3> tU{
        LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid),
        LatticeColourMatrix(&Grid)};
    std::array<LatticeColourMatrix, 3> aU{
        LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid),
        LatticeColourMatrix(&Grid)};
    for (int mu = 0; mu < 3; ++mu) {
      Umu[mu] = PeekIndex<LorentzIndex>(Usm, mu);
      cU[mu] = conjugate(Umu[mu]);
      tU[mu] = transpose(Umu[mu]);
      aU[mu] = adj(Umu[mu]);
    }

    // base diquark fields
    LatticeColourMatrix d_iso = IsoSingletDiquark(F.d);
    LatticeColourVector w = EpsReduce(d_iso);
    LatticeColourMatrix d6(&Grid);
    {
      LatticeColourMatrix m00(PeekIndex<1>(F.d, 0, 0));
      LatticeColourMatrix m11(PeekIndex<1>(F.d, 1, 1));
      d6 = m00 + m11;                    // colour-symmetric (6), I3=+-1 mix
    }

    // one covariant Gaussian smearing step, spatial only (w: 3bar transport;
    // d6: fwd U d U^T, bwd U^dag d U^*)
    auto SmearStep = [&](LatticeColourVector &wv, LatticeColourMatrix &dm) {
      LatticeColourVector wa(&Grid); wa = Zero();
      LatticeColourMatrix da(&Grid); da = Zero();
      LatticeColourVector tv(&Grid);
      LatticeColourMatrix tm(&Grid);
      for (int mu = 0; mu < 3; ++mu) {
        tv = Cshift(wv, mu, +1);
        wa = wa + cU[mu] * tv;
        tv = tU[mu] * wv;
        wa = wa + Cshift(tv, mu, -1);
        tm = Cshift(dm, mu, +1);
        da = da + Umu[mu] * tm * tU[mu];
        tm = aU[mu] * dm * cU[mu];
        da = da + Cshift(tm, mu, -1);
      }
      wv = (1.0 - salpha) * wv + (salpha / 6.0) * wa;
      dm = (1.0 - salpha) * dm + (salpha / 6.0) * da;
    };

    auto slice = [&](const LatticeComplex &D, std::vector<ComplexD> &out) {
      std::vector<typename LatticeComplex::scalar_object> sl;
      sliceSum(D, sl, Nd - 1);
      out.resize(T);
      for (int t = 0; t < T; ++t) out[t] = TensorRemove(sl[t]) / V3;
    };

    // evaluate the full geometry set on the current (w, d6)
    auto EvalGeoms = [&](const LatticeColourVector &wc,
                         const LatticeColourMatrix &d6c,
                         std::vector<std::vector<ComplexD>> &S1l,
                         std::vector<std::vector<ComplexD>> &S0l) {
      // cached 3bar transports of wc
      std::map<std::tuple<int, int, int>, LatticeColourVector> V;
      for (int mu = 0; mu < 3; ++mu)
        for (int s : {+1, -1}) {
          LatticeColourVector v = wc;
          for (int L = 1; L <= LMAX_; ++L) {
            if (s > 0) {
              LatticeColourVector sh = Cshift(v, mu, +1);
              v = cU[mu] * sh;
            } else {
              LatticeColourVector tvv = tU[mu] * v;
              v = Cshift(tvv, mu, -1);
            }
            V.emplace(std::make_tuple(mu, s, L), v);
          }
        }
      auto getW = [&](const Coordinate &r) -> const LatticeColourVector & {
        for (int mu = 0; mu < 3; ++mu)
          if (r[mu] != 0)
            return V.at(
                std::make_tuple(mu, r[mu] > 0 ? +1 : -1, std::abs(r[mu])));
        return wc;
      };

      S1l.assign(NG1(), {});
      for (int g = 0; g < NG1(); ++g) {
        const LatticeColourVector &w1 = getW(geoI1[g].r1);
        const LatticeColourVector &w3 = getW(geoI1[g].r2);
        LatticeColourVector u = d6c * w3;
        LatticeComplex D(&Grid);
        D = Zero();
        for (int k = 0; k < Nc; ++k)
          D = D + peekColour(w1, k) * peekColour(u, k);
        slice(D, S1l[g]);
      }
      S0l.assign(NG0(), {});
      for (int g = 0; g < NG0(); ++g) {
        const LatticeColourVector &w2 = getW(geoI0[g].r1);
        const LatticeColourVector &w3 = getW(geoI0[g].r2);
        LatticeComplex b0 = peekColour(w2, 0), b1 = peekColour(w2, 1),
                       b2 = peekColour(w2, 2);
        LatticeComplex c0 = peekColour(w3, 0), c1 = peekColour(w3, 1),
                       c2 = peekColour(w3, 2);
        LatticeComplex D = peekColour(wc, 0) * (b1 * c2 - b2 * c1) +
                           peekColour(wc, 1) * (b2 * c0 - b0 * c2) +
                           peekColour(wc, 2) * (b0 * c1 - b1 * c0);
        slice(D, S0l[g]);
      }
    };

    S1.assign(nlev, {});
    S0.assign(nlev, {});
    int cur = 0;
    for (int lev = 0; lev < nlev; ++lev) {
      while (cur < smears_[lev]) {
        SmearStep(w, d6);
        ++cur;
      }
      EvalGeoms(w, d6, S1[lev], S0[lev]);
    }
  }

  // Boss-only h5 write.  plaq is an argument (collective, compute outside).
  void WriteH5(const std::string &outfile, RealD lam, RealD plaq,
               const std::vector<std::vector<std::vector<ComplexD>>> &S1,
               const std::vector<std::vector<std::vector<ComplexD>>> &S0) const {
    Hdf5Writer wr(outfile);
    write(wr, "T", T_);
    write(wr, "V3", V3_);
    write(wr, "lambda", lam);
    write(wr, "plaq", plaq);
    write(wr, "stout_nsmear", nstout_);
    write(wr, "n_geo_I1", NG1());
    write(wr, "n_geo_I0", NG0());
    std::vector<int> g1meta, g0meta;
    for (auto &g : geoI1)
      for (int c = 0; c < 3; ++c) g1meta.push_back(g.r1[c]);
    for (auto &g : geoI1)
      for (int c = 0; c < 3; ++c) g1meta.push_back(g.r2[c]);
    for (auto &g : geoI0)
      for (int c = 0; c < 3; ++c) g0meta.push_back(g.r1[c]);
    for (auto &g : geoI0)
      for (int c = 0; c < 3; ++c) g0meta.push_back(g.r2[c]);
    write(wr, "geo_I1", g1meta);
    write(wr, "geo_I0", g0meta);
    write(wr, "smears", smears_);
    write(wr, "smear_alpha", salpha_);
    for (int l = 0; l < NLEV(); ++l) {
      const std::string sm = "sm" + std::to_string(smears_[l]) + "_";
      for (int g = 0; g < NG1(); ++g)
        write(wr, "dib_I1_" + sm + "S_" + std::to_string(g), S1[l][g]);
      for (int g = 0; g < NG0(); ++g)
        write(wr, "dib_I0_" + sm + "S_" + std::to_string(g), S0[l][g]);
    }
  }

  std::vector<Geo> geoI1, geoI0;

 private:
  GridCartesian &grid_;
  int nstout_;
  RealD rho_;
  int LMAX_;
  std::vector<int> smears_;
  RealD salpha_;
  int T_;
  RealD V3_;
};

}  // namespace DtxqcdDibaryon
