#pragma once
// Analytic per-4D-site LU of the TXQCD Möbius Mooee block.
//
// Mooee couples only within a single 4D site (across the Ls 5th-dim slices),
// so it is block-diagonal over 4D sites; per site the block is a dense
// (Ls·Nf·Ns·Nc)² matrix.  Each per-site block is obtained by *probing the
// already-FD-validated Mooee* — applying it to the Ls·Nf·Ns·Nc unit fields
// and reading the columns at every site — then Eigen-LU-factored once per
// ImportAux().  MooeeInv is then a per-site dense solve, replacing inner PCG.
//
// All rb<->scalar conversions go through unvectorizeToLexOrdArray /
// vectorizeFromLexOrdArray (the proven checkerboard-correct idiom; the same
// one TXQCDWilsonCloverFermionEO uses).  The (lex-index)->(site4, s) map is
// derived from LatticeCoordinate fields, so it is fully layout-independent.
// The PCG MooeeInv remains an exact validation oracle.

#include <Grid/qcd/action/txqcd/TXQCDDeltaOp.h>
#include <Grid/Eigen/Dense>

NAMESPACE_BEGIN(Grid);

class TXQCDMobiusSiteLU {
 public:
  static constexpr int kInt = TxqcdNf * Ns * Nc;   // 24 per (site, slice)

  TXQCDMobiusSiteLU(GridRedBlackCartesian *FrbGrid, int Ls)
      : FrbGrid_(FrbGrid), Ls_(Ls), kBlk_(Ls * kInt), built_(false) {}

  bool built() const { return built_; }
  int  Ls()    const { return Ls_; }

  static inline int idx(int s, int a, int sp, int col) {
    return ((s * TxqcdNf + a) * Ns + sp) * Nc + col;
  }

  // (lex-index) -> (s, dense site4 id), from LatticeCoordinate fields.
  void BuildIndexMap(int cb) {
    typedef typename LatticeComplex::vector_object::scalar_object CSobj;
    LatticeComplex co(FrbGrid_);
    int nd = FrbGrid_->_ndimension;                 // 5; dim 0 = Ls
    std::vector<std::vector<CSobj>> c(nd);
    for (int mu = 0; mu < nd; ++mu) {
      co.Checkerboard() = cb;
      LatticeCoordinate(co, mu);
      unvectorizeToLexOrdArray(c[mu], co);
    }
    const int64_t nlex = (int64_t)c[0].size();
    s_of_.assign(nlex, 0);
    site4_of_.assign(nlex, 0);
    std::map<uint64_t, int64_t> key2site;
    for (int64_t x = 0; x < nlex; ++x) {
      auto rd = [&](int mu) {
        return (int)std::llround(TensorRemove(c[mu][x]()()()).real());
      };
      s_of_[x] = rd(0);
      uint64_t key = 0;
      for (int mu = 1; mu < nd; ++mu) key = key * 100003ull + (uint64_t)rd(mu);
      auto it = key2site.find(key);
      int64_t site4;
      if (it == key2site.end()) { site4 = (int64_t)key2site.size();
                                  key2site[key] = site4; }
      else                       site4 = it->second;
      site4_of_[x] = site4;
    }
    nlex_   = nlex;
    nloc4_  = (int64_t)key2site.size();
    cb_     = cb;
    // Reverse map: site4 -> array of (lex index) for its Ls slices, indexed
    // by s.  Lets Build/Apply parallelize over 4D sites (fused gather/solve/
    // scatter) instead of serial loops over the full lex volume.
    lex_of_.assign(nloc4_, std::vector<int64_t>(Ls_, -1));
    for (int64_t x = 0; x < nlex; ++x)
      lex_of_[site4_of_[x]][s_of_[x]] = x;
  }

  template <class MooeeFn>
  void Build(MooeeFn &&applyMooee, int cb) {
    typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
        DMat;
    typedef typename LatticeFermion::vector_object::scalar_object FSobj;
    if (s_of_.empty() || cb_ != cb) BuildIndexMap(cb);
    blocks_.assign(nloc4_, DMat::Zero(kBlk_, kBlk_));

    TXQCDFermionNf e(FrbGrid_), Me(FrbGrid_);
    for (int a = 0; a < TxqcdNf; ++a) {
      e.f[a].Checkerboard()  = cb;
      Me.f[a].Checkerboard() = cb;
    }
    std::array<std::vector<FSobj>, TxqcdNf> es, ms;
    for (int a = 0; a < TxqcdNf; ++a) es[a].resize(nlex_);

    for (int j = 0; j < kBlk_; ++j) {
      int cj  = j % Nc;
      int spj = (j / Nc) % Ns;
      int aj  = (j / (Nc * Ns)) % TxqcdNf;
      int sj  = j / kInt;
      for (int a = 0; a < TxqcdNf; ++a)
        for (int64_t x = 0; x < nlex_; ++x) es[a][x] = Zero();
      for (int64_t x = 0; x < nlex_; ++x)
        if (s_of_[x] == sj) es[aj][x]()(spj)(cj) = std::complex<double>(1, 0);
      for (int a = 0; a < TxqcdNf; ++a) {
        vectorizeFromLexOrdArray(es[a], e.f[a]);
        e.f[a].Checkerboard() = cb;
      }
      applyMooee(e, Me);
      for (int a = 0; a < TxqcdNf; ++a) unvectorizeToLexOrdArray(ms[a], Me.f[a]);
      for (int a = 0; a < TxqcdNf; ++a) {
        for (int64_t x = 0; x < nlex_; ++x) {
          int s = s_of_[x]; int64_t site4 = site4_of_[x];
          for (int sp = 0; sp < Ns; ++sp)
            for (int col = 0; col < Nc; ++col) {
              auto z = ms[a][x]()(sp)(col);
              blocks_[site4](idx(s, a, sp, col), j) =
                  std::complex<double>(z.real(), z.imag());
            }
        }
      }
    }
    lu_.clear();    lu_.reserve(nloc4_);
    ludag_.clear(); ludag_.reserve(nloc4_);
    for (int64_t x = 0; x < nloc4_; ++x) {
      lu_.emplace_back(blocks_[x].partialPivLu());
      DMat adj = blocks_[x].adjoint();           // materialize before factoring
      ludag_.emplace_back(adj.partialPivLu());
    }
    blocks_.clear(); blocks_.shrink_to_fit();   // factored; raw blocks no longer needed
    built_ = true;
  }

  void Apply(const TXQCDFermionNf &in, TXQCDFermionNf &out, bool dag) const {
    typedef typename LatticeFermion::vector_object::scalar_object FSobj;
    typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> DVec;
    std::array<std::vector<FSobj>, TxqcdNf> is, os;
    for (int a = 0; a < TxqcdNf; ++a) {
      unvectorizeToLexOrdArray(is[a], in.f[a]);
      os[a].resize(is[a].size());
    }
    // Fused: parallel over 4D sites — gather this site's Ls·24 vector from
    // the lex arrays, solve, scatter back.  Replaces the serial nlex loops.
    thread_for(site4, nloc4_, {
      DVec b = DVec::Zero(kBlk_);
      const auto &lex = lex_of_[site4];
      for (int s = 0; s < Ls_; ++s) {
        int64_t x = lex[s];
        for (int a = 0; a < TxqcdNf; ++a)
          for (int sp = 0; sp < Ns; ++sp)
            for (int col = 0; col < Nc; ++col) {
              auto z = is[a][x]()(sp)(col);
              b(idx(s, a, sp, col)) = std::complex<double>(z.real(), z.imag());
            }
      }
      DVec y = dag ? ludag_[site4].solve(b) : lu_[site4].solve(b);
      for (int s = 0; s < Ls_; ++s) {
        int64_t x = lex[s];
        for (int a = 0; a < TxqcdNf; ++a) {
          FSobj v; v = Zero();
          for (int sp = 0; sp < Ns; ++sp)
            for (int col = 0; col < Nc; ++col)
              v()(sp)(col) = y(idx(s, a, sp, col));
          os[a][x] = v;
        }
      }
    });
    for (int a = 0; a < TxqcdNf; ++a) {
      vectorizeFromLexOrdArray(os[a], out.f[a]);
      out.f[a].Checkerboard() = cb_;
    }
  }

 private:
  GridRedBlackCartesian *FrbGrid_;
  int Ls_, kBlk_, cb_ = -1;
  bool built_;
  int64_t nlex_ = 0, nloc4_ = 0;
  std::vector<int>     s_of_;
  std::vector<int64_t> site4_of_;
  std::vector<std::vector<int64_t>> lex_of_;   // site4 -> [lex idx per s]
  std::vector<Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                            Eigen::Dynamic>> blocks_;
  std::vector<Eigen::PartialPivLU<
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>>> lu_,
      ludag_;
};

NAMESPACE_END(Grid);
