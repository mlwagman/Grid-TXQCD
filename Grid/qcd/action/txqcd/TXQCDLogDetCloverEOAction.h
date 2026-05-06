#pragma once
// Log-determinant action for the even-site diagonal block in the
// EO-preconditioned TXQCD Wilson operator.
//
// det(M) = det(Mee) * det(Mpc), so when using Mpc for the pseudofermion
// action we need S_logdet = -ln det(Mee) as a separate action term.
//
// Since Mee = (4+m)I + Δ_even is site-diagonal with a 24×24 matrix per site,
// the determinant factorises:
//   ln det(Mee) = Σ_{x∈even} ln det(M_site(x))
//
// The force is:
//   dS/daux_even(x) = -Tr(M_site(x)^{-1} dM_site/daux(x))
//
// Only even-site aux fields contribute; odd-site force is zero.
// Gauge force is zero for csw=0; for csw!=0 the clover term couples
// Mee to the gauge links through F_{μν}, producing a gauge force via Cmunu.

#include <Grid/qcd/action/txqcd/TXQCDSiteMatrix.h>
#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetGpuKernel.h>
#include <Grid/qcd/action/fermion/WilsonCloverHelpers.h>
#include <Grid/util/QudaPackGpu.h>
#include <Grid/algorithms/blas/BatchedBlas.h>
#ifdef GRID_CUDA
#include <cublas_v2.h>
#endif

NAMESPACE_BEGIN(Grid);

class TXQCDLogDetCloverEOAction : public Action<TXQCDField> {
 public:
  typedef TXQCDSiteMatrixUtil SMU;
  static constexpr int kDim = SMU::kDim;

  // Per-flavor mass constructor (for non-degenerate Nf>2).
  TXQCDLogDetCloverEOAction(GridCartesian &grid, GridRedBlackCartesian &rbgrid,
                      const std::array<RealD, TxqcdNf> &mass, RealD csw = 0.0)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), csw_(csw) {
    for (int a = 0; a < TxqcdNf; ++a) diag_mass_[a] = 4.0 + mass_[a];
  }

  // Backward-compat: degenerate scalar mass.
  TXQCDLogDetCloverEOAction(GridCartesian &grid, GridRedBlackCartesian &rbgrid,
                      RealD mass, RealD csw = 0.0)
      : TXQCDLogDetCloverEOAction(grid, rbgrid, SMU::MassArray(mass), csw) {}

  std::string action_name() override { return "TXQCDLogDetCloverEOAction"; }
  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=";
    for (int a = 0; a < TxqcdNf; ++a)
      os << (a ? "," : "{") << mass_[a];
    os << "}" << std::endl;
    return os.str();
  }

  void refresh(const TXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {}

  RealD S(const TXQCDField &U) override {
    auto aux = GetEvenAux(U);
    auto cl = GetEvenClover(U);
    uint64_t nsites = aux.sig.size();

    // Per-site terms are independent → thread_for over sites; M is per-thread.
    std::vector<RealD> partial(thread_max(0), 0.0);
    thread_for(x, nsites, {
      SMU::SiteMatrix M;
      std::array<SMU::FmnSobj, 6> fmn_site;
      const std::array<SMU::FmnSobj, 6> *fmn_ptr = nullptr;
      if (csw_ != 0.0) {
        for (int k = 0; k < 6; ++k) fmn_site[k] = cl.fs[k][x];
        fmn_ptr = &fmn_site;
      }
      SMU::BuildSiteMatrix(sm_, diag_mass_, aux.sig[x], aux.pi[x],
                          aux.s[x], aux.p[x], aux.t[x],
                          csw_, fmn_ptr, M);
      auto lu = M.partialPivLu();
      auto d = lu.determinant();
      partial[thread_num(0)] += std::log(std::abs(d));
    });
    RealD logdet = 0.0;
    for (auto &p : partial) logdet += p;
    grid_.GlobalSum(logdet);
    RealD action = -logdet;
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    static int use_gpu = []() {
      const char *e = std::getenv("TXQCD_LOGDET_GPU");
      // Default ON: env var must be set to "0" to opt out.
      if (!e || !*e) return 1;
      return std::atoi(e);
    }();
    if (use_gpu) { deriv_gpu(U, dSdU); return; }
    deriv_cpu(U, dSdU);
  }

  // -------------------- GPU path (Phase J) --------------------
  // Per-deriv cost on 16³×48 production: ~1.5 s vs CPU 2.8 s.
  //   1) Construct a TXQCDWilsonCloverFermionEO over U; ImportFields() runs
  //      PrecomputeInverses on EVEN+ODD parities (TXQCD_PRECOMPUTE_GPU=1
  //      path: ~1.3 s for both — even-only single-parity refactor possible).
  //   2) ExtractTraces(): single accelerator_for over RB-Even oSites reads
  //      M^{-1} from EOp.MdevForCb(Even), computes the 5 force traces,
  //      writes via putlane to F_*_e RB Lattices.
  //   3) dSdU.* = Zero(); Quda::AccumulateRbScaledToFull(dSdU.*, 1, F_*_e).
  //   4) For csw≠0: derive 6 clover_sigma RB ColourMatrix from F_t_e and
  //      run the existing Cmunu chain on the full grid.
  void deriv_gpu(const TXQCDField &U, TXQCDField &dSdU) {
    auto t_total0 = usecond();
    // 1) CPU build M_e^{-1} on EVEN parity only (half the work of constructing
    //    a full TXQCDWilsonCloverFermionEO which would do both parities + many
    //    extra Lattice allocs for Mooee/Wilson scratch we never use here).
    auto t_pickcb0 = usecond();
    auto aux = GetEvenAux(U);
    auto cl  = GetEvenClover(U);
    uint64_t nsites = aux.sig.size();
    t_pickcb_us_ += usecond() - t_pickcb0;

    auto t_build0 = usecond();
    // Phase J.2: BUILD forward M (not inverse) on CPU, then invert on GPU
    // via cublasZgetrfBatched + cublasZgetriBatched.  Saves ~half the
    // per-site CPU time (Eigen invert, ~190 ms → ~50 ms cuBLAS).
    if (m_fwd_host_.size() < nsites) m_fwd_host_.resize(nsites);
    thread_for(x, nsites, {
      SMU::SiteMatrix M;
      std::array<SMU::FmnSobj, 6> fmn_site;
      const std::array<SMU::FmnSobj, 6> *fmn_ptr = nullptr;
      if (csw_ != 0.0) {
        for (int k = 0; k < 6; ++k) fmn_site[k] = cl.fs[k][x];
        fmn_ptr = &fmn_site;
      }
      SMU::BuildSiteMatrix(sm_, diag_mass_, aux.sig[x], aux.pi[x],
                          aux.s[x], aux.p[x], aux.t[x],
                          csw_, fmn_ptr, M);
      m_fwd_host_[x] = M;  // forward M; cuBLAS will LU-decompose + invert
    });
    t_inv_us_ += usecond() - t_build0;  // re-purposed: now CPU build only

    auto t_upload0 = usecond();
    constexpr int N  = SMU::kDim;        // 24
    constexpr int N2 = N * N;            // 576

    // 2) Upload forward M to device buffer M_fwd_dev_ (will be overwritten by
    //    cuBLAS getrf with the LU decomposition).  M_inv_dev_ receives the
    //    inverse.  Both are sized lazily.
    if (M_fwd_dev_.size() < nsites * N2) M_fwd_dev_.resize(nsites * N2);
    if (M_inv_dev_.size() < nsites * N2) M_inv_dev_.resize(nsites * N2);
    static_assert(sizeof(std::complex<double>) == sizeof(ComplexD),
                  "std::complex<double> and ComplexD must share layout");
    acceleratorCopyToDevice(
        reinterpret_cast<void *>(const_cast<std::complex<double> *>(
            m_fwd_host_.data()->data())),
        &M_fwd_dev_[0],
        nsites * N2 * sizeof(ComplexD));

    // 3) Build lex-index table for the RB-Even grid once and cache.
    if (!lex_built_) {
      Quda::BuildLexTable(&rbgrid_, lex_table_dev_);
      lex_built_ = true;
    }

    // 4) Build cuBLAS pointer arrays + pivot/info buffers once and cache.
    //    pivots: nsites × N ints, info: nsites ints.
    if (!cublas_built_ || cublas_nsites_ != nsites) {
      Amk_ptrs_.resize(nsites);
      Cmk_ptrs_.resize(nsites);
      pivots_dev_.resize(nsites * N);
      info_dev_.resize(nsites);
      ComplexD *Mfwd = &M_fwd_dev_[0];
      ComplexD *Minv = &M_inv_dev_[0];
      ComplexD **Amk = &Amk_ptrs_[0];
      ComplexD **Cmk = &Cmk_ptrs_[0];
      accelerator_for(i, nsites, 1, {
        Amk[i] = &Mfwd[i * N2];
        Cmk[i] = &Minv[i * N2];
      });
      cublas_built_ = true;
      cublas_nsites_ = nsites;
    }
    t_upload_us_ += usecond() - t_upload0;

    auto t_cublas0 = usecond();
    // 5) cuBLAS getrfBatched + getriBatched: invert all 24×24 matrices on GPU.
    //    Eigen layout is column-major, matching cuBLAS expectations directly.
#ifdef GRID_CUDA
    cublasHandle_t handle = GridBLAS::gridblasHandle;
    // pointer mode device — we passed pivot/info as device pointers.
    cublasZgetrfBatched(handle, N,
                        reinterpret_cast<cuDoubleComplex **>(&Amk_ptrs_[0]),
                        N, &pivots_dev_[0], &info_dev_[0],
                        nsites);
    cublasZgetriBatched(handle, N,
                        reinterpret_cast<cuDoubleComplex **>(&Amk_ptrs_[0]),
                        N, &pivots_dev_[0],
                        reinterpret_cast<cuDoubleComplex **>(&Cmk_ptrs_[0]),
                        N, &info_dev_[0],
                        nsites);
#else
#  error "Phase J.2 GPU LogDet requires GRID_CUDA"
#endif
    t_cublas_us_ += usecond() - t_cublas0;

    auto t_traces0 = usecond();
    // 4) GPU kernel: extract 5 force traces from M^{-1}.
    LatticeSigmaField F_sig_e(&rbgrid_);
    LatticePiField   F_pi_e(&rbgrid_);
    LatticeSFieldC   F_s_e(&rbgrid_);
    LatticePFieldC   F_p_e(&rbgrid_);
    LatticeTField    F_t_e(&rbgrid_);

    TxqcdLogDet::ExtractTracesFromBuffers(
        &M_inv_dev_[0], &lex_table_dev_[0],
        F_sig_e, F_pi_e, F_s_e, F_p_e, F_t_e);
    t_traces_us_ += usecond() - t_traces0;

    dSdU.sigma = Zero();
    dSdU.pi    = Zero();
    dSdU.s     = Zero();
    dSdU.p     = Zero();
    dSdU.t     = Zero();
    dSdU.U     = Zero();

    Quda::AccumulateRbScaledToFull(dSdU.sigma, 1.0, F_sig_e);
    Quda::AccumulateRbScaledToFull(dSdU.pi,    1.0, F_pi_e);
    Quda::AccumulateRbScaledToFull(dSdU.s,     1.0, F_s_e);
    Quda::AccumulateRbScaledToFull(dSdU.p,     1.0, F_p_e);
    Quda::AccumulateRbScaledToFull(dSdU.t,     1.0, F_t_e);

    if (csw_ != 0.0) {
      // Derive 6 clover_sigma RB ColourMatrix from F_t_e on GPU.
      std::array<LatticeColourMatrix, 6> clover_sigma_e_arr = {
          LatticeColourMatrix(&rbgrid_), LatticeColourMatrix(&rbgrid_),
          LatticeColourMatrix(&rbgrid_), LatticeColourMatrix(&rbgrid_),
          LatticeColourMatrix(&rbgrid_), LatticeColourMatrix(&rbgrid_)};
      TxqcdLogDet::DeriveCloverSigma(F_t_e, csw_, clover_sigma_e_arr);

      // Push each clover_sigma_e[k] to full grid via fused acc helper, then
      // run the existing Cmunu chain (already GPU-resident via Grid exprs).
      typedef WilsonImplR Impl;
      std::vector<LatticeColourMatrix> Sigma_full;
      for (int k = 0; k < 6; ++k) {
        Sigma_full.emplace_back(&grid_);
        Sigma_full.back() = Zero();
        Quda::AccumulateRbScaledToFull(Sigma_full[k], 1.0, clover_sigma_e_arr[k]);
      }

      std::vector<LatticeColourMatrix> Ulinks(Nd, &grid_);
      for (int mu = 0; mu < Nd; ++mu)
        Ulinks[mu] = PeekIndex<LorentzIndex>(U.U, mu);

      LatticeGaugeField clover_force(&grid_);
      clover_force = Zero();

      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix force_mu(&grid_);
        force_mu = Zero();
        for (int nu = 0; nu < Nd; ++nu) {
          if (mu == nu) continue;
          int mn = (mu < nu) ? SMU::FmnIndex(mu, nu) : SMU::FmnIndex(nu, mu);
          LatticeColourMatrix lambda = (mu < nu) ? Sigma_full[mn]
                                                 : (-1.0) * Sigma_full[mn];
          force_mu += 0.25 *
              WilsonCloverHelpers<Impl>::Cmunu(Ulinks, lambda, mu, nu);
        }
        pokeLorentz(clover_force, Ulinks[mu] * force_mu, mu);
      }
      // Convention-A (-1/2) factor — see CPU path notes.
      dSdU.U = (-0.5) * clover_force;
    }
    t_total_us_ += usecond() - t_total0;
    n_deriv_++;
  }

  // Per-component timer dump (called from destructor or on demand).
  void PrintGpuTimers(const char *tag = "") const {
    if (n_deriv_ == 0) return;
    std::cout << GridLogMessage
              << "[TXQCDLogDet.gpu/" << tag << "] " << n_deriv_
              << " deriv_gpu calls (ms/call):"
              << "  pickCB=" << double(t_pickcb_us_) * 1e-3 / n_deriv_
              << "  cpu_build=" << double(t_inv_us_) * 1e-3 / n_deriv_
              << "  upload=" << double(t_upload_us_) * 1e-3 / n_deriv_
              << "  cublas_inv=" << double(t_cublas_us_) * 1e-3 / n_deriv_
              << "  traces=" << double(t_traces_us_) * 1e-3 / n_deriv_
              << "  total=" << double(t_total_us_) * 1e-3 / n_deriv_
              << std::endl;
  }
  ~TXQCDLogDetCloverEOAction() { PrintGpuTimers("dtor"); }

  // -------------------- CPU path (legacy) --------------------
  void deriv_cpu(const TXQCDField &U, TXQCDField &dSdU) {
    auto aux = GetEvenAux(U);
    auto cl = GetEvenClover(U);
    uint64_t nsites = aux.sig.size();

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const double neg_csw_half = -0.5 * csw_;

    std::vector<SMU::SigSobj> sig_force(nsites);
    std::vector<SMU::PiSobj> pi_force(nsites);
    std::vector<SMU::SSobj> s_force(nsites);
    std::vector<SMU::PSobj> p_force(nsites);
    std::vector<SMU::TSobj> t_force(nsites);

    typedef typename LatticeColourMatrix::vector_object::scalar_object CMsobj;
    std::array<std::vector<CMsobj>, 6> clover_sigma;
    if (csw_ != 0.0)
      for (int k = 0; k < 6; ++k) clover_sigma[k].resize(nsites);

    // Per-site loop: each iteration reads aux/cl[x], computes M, M.inverse(),
    // then writes per-site outputs sig_force[x], ... — independent across x,
    // perfect for thread_for.  M, Inv, fmn_site are per-thread scratch.
    thread_for(x, nsites, {
      SMU::SiteMatrix M, Inv;
      std::array<SMU::FmnSobj, 6> fmn_site;
      const std::array<SMU::FmnSobj, 6> *fmn_ptr = nullptr;
      if (csw_ != 0.0) {
        for (int k = 0; k < 6; ++k) fmn_site[k] = cl.fs[k][x];
        fmn_ptr = &fmn_site;
      }
      SMU::BuildSiteMatrix(sm_, diag_mass_, aux.sig[x], aux.pi[x],
                          aux.s[x], aux.p[x], aux.t[x],
                          csw_, fmn_ptr, M);
      Inv = M.inverse();

      // sigma force: F_{ab} = -Σ_{α,i} Inv_{(a,α,i),(b,α,i)}
      for (int a = 0; a < TxqcdNf; ++a) {
        for (int b = 0; b < TxqcdNf; ++b) {
          std::complex<double> val(0, 0);
          for (int alpha = 0; alpha < Ns; ++alpha)
            for (int i = 0; i < Nc; ++i) {
              int ra = a * Ns * Nc + alpha * Nc + i;
              int rb = b * Ns * Nc + alpha * Nc + i;
              val += Inv(ra, rb);
            }
          sig_force[x]()()(a, b) = ComplexD(-val.real(), -val.imag());
        }
      }

      // pi force: F_{ab} = -Σ_{α,β,i} γ₅(β,α) Inv_{(a,α,i),(b,β,i)}
      for (int a = 0; a < TxqcdNf; ++a) {
        for (int b = 0; b < TxqcdNf; ++b) {
          std::complex<double> val(0, 0);
          for (int alpha = 0; alpha < Ns; ++alpha)
            for (int beta = 0; beta < Ns; ++beta) {
              auto g5 = sm_.gamma5(beta, alpha);
              if (g5 == std::complex<double>(0, 0)) continue;
              for (int i = 0; i < Nc; ++i) {
                int ra = a * Ns * Nc + alpha * Nc + i;
                int rb = b * Ns * Nc + beta * Nc + i;
                val += g5 * Inv(ra, rb);
              }
            }
          pi_force[x]()()(a, b) = ComplexD(-val.real(), -val.imag());
        }
      }

      // s force: F_{ij} = -(1/√2) Σ_{a,α} Inv_{(a,α,i),(a,α,j)}
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          std::complex<double> val(0, 0);
          for (int a = 0; a < TxqcdNf; ++a)
            for (int alpha = 0; alpha < Ns; ++alpha) {
              int ri = a * Ns * Nc + alpha * Nc + i;
              int rj = a * Ns * Nc + alpha * Nc + j;
              val += Inv(ri, rj);
            }
          s_force[x]()()(i, j) =
              ComplexD(-inv_sqrt2 * val.real(), -inv_sqrt2 * val.imag());
        }
      }

      // p force: F_{ij} = -(1/√2) Σ_{a,α,β} γ₅(β,α) Inv_{(a,α,i),(a,β,j)}
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          std::complex<double> val(0, 0);
          for (int a = 0; a < TxqcdNf; ++a)
            for (int alpha = 0; alpha < Ns; ++alpha)
              for (int beta = 0; beta < Ns; ++beta) {
                auto g5 = sm_.gamma5(beta, alpha);
                if (g5 == std::complex<double>(0, 0)) continue;
                int ri = a * Ns * Nc + alpha * Nc + i;
                int rj = a * Ns * Nc + beta * Nc + j;
                val += g5 * Inv(ri, rj);
              }
          p_force[x]()()(i, j) =
              ComplexD(-inv_sqrt2 * val.real(), -inv_sqrt2 * val.imag());
        }
      }

      // t force: F_{μν,ij} = -Σ_{a,α,β} (iσ_{μν})(β,α) Inv_{(a,α,i),(a,β,j)}
      for (int mu = 0; mu < Nd; ++mu)
        for (int nu = 0; nu < Nd; ++nu)
          for (int i = 0; i < Nc; ++i)
            for (int j = 0; j < Nc; ++j)
              t_force[x]()(mu, nu)(i, j) = ComplexD(0, 0);

      for (int mu = 0; mu < Nd; ++mu) {
        for (int nu = mu + 1; nu < Nd; ++nu) {
          for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
              std::complex<double> val(0, 0);
              for (int a = 0; a < TxqcdNf; ++a)
                for (int alpha = 0; alpha < Ns; ++alpha)
                  for (int beta = 0; beta < Ns; ++beta) {
                    auto isig = sm_.isigma[mu][nu](beta, alpha);
                    if (isig == std::complex<double>(0, 0)) continue;
                    int ri = a * Ns * Nc + alpha * Nc + i;
                    int rj = a * Ns * Nc + beta * Nc + j;
                    val += isig * Inv(ri, rj);
                  }
              t_force[x]()(mu, nu)(i, j) =
                  ComplexD(-val.real(), -val.imag());
              t_force[x]()(nu, mu)(i, j) =
                  ComplexD(val.real(), val.imag());

              if (csw_ != 0.0) {
                // dS/dF_{mu,nu}^{ij} = -Tr(M^{-1} dM/dF)
                // dM/dF = i*(csw/2) * isigma => dS/dF = -i*(csw/2) * val
                int k = SMU::FmnIndex(mu, nu);
                std::complex<double> cv(0.0, -0.5 * csw_);
                std::complex<double> cval = cv * val;
                clover_sigma[k][x]()()(i, j) =
                    ComplexD(cval.real(), cval.imag());
              }
            }
          }
        }
      }
    });

    LatticeSigmaField F_sig_e(&rbgrid_);
    vectorizeFromLexOrdArray(sig_force, F_sig_e);
    F_sig_e.Checkerboard() = Even;

    LatticePiField F_pi_e(&rbgrid_);
    vectorizeFromLexOrdArray(pi_force, F_pi_e);
    F_pi_e.Checkerboard() = Even;

    LatticeSFieldC F_s_e(&rbgrid_);
    vectorizeFromLexOrdArray(s_force, F_s_e);
    F_s_e.Checkerboard() = Even;

    LatticePFieldC F_p_e(&rbgrid_);
    vectorizeFromLexOrdArray(p_force, F_p_e);
    F_p_e.Checkerboard() = Even;

    LatticeTField F_t_e(&rbgrid_);
    vectorizeFromLexOrdArray(t_force, F_t_e);
    F_t_e.Checkerboard() = Even;

    dSdU.sigma = Zero();
    dSdU.pi = Zero();
    dSdU.s = Zero();
    dSdU.p = Zero();
    dSdU.t = Zero();
    dSdU.U = Zero();

    setCheckerboard(dSdU.sigma, F_sig_e);
    setCheckerboard(dSdU.pi, F_pi_e);
    setCheckerboard(dSdU.s, F_s_e);
    setCheckerboard(dSdU.p, F_p_e);
    setCheckerboard(dSdU.t, F_t_e);

    if (csw_ != 0.0) {
      typedef WilsonImplR Impl;
      std::vector<LatticeColourMatrix> Sigma_full;
      for (int k = 0; k < 6; ++k) {
        LatticeColourMatrix Sigma_e(&rbgrid_);
        vectorizeFromLexOrdArray(clover_sigma[k], Sigma_e);
        Sigma_e.Checkerboard() = Even;
        Sigma_full.emplace_back(&grid_);
        Sigma_full.back() = Zero();
        setCheckerboard(Sigma_full[k], Sigma_e);
      }

      std::vector<LatticeColourMatrix> Ulinks(Nd, &grid_);
      for (int mu = 0; mu < Nd; ++mu)
        Ulinks[mu] = PeekIndex<LorentzIndex>(U.U, mu);

      LatticeGaugeField clover_force(&grid_);
      clover_force = Zero();

      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix force_mu(&grid_);
        force_mu = Zero();
        for (int nu = 0; nu < Nd; ++nu) {
          if (mu == nu) continue;
          int mn = (mu < nu) ? SMU::FmnIndex(mu, nu) : SMU::FmnIndex(nu, mu);
          LatticeColourMatrix lambda = (mu < nu) ? Sigma_full[mn]
                                                 : (-1.0) * Sigma_full[mn];
          force_mu += 0.25 *
              WilsonCloverHelpers<Impl>::Cmunu(Ulinks, lambda, mu, nu);
        }
        pokeLorentz(clover_force, Ulinks[mu] * force_mu, mu);
      }
      // Cmunu-based force is "Convention B" (full gradient: dS/dh = Tr(E*F)).
      // The HMC integrator multiplies gauge forces by HMC_MOMENTUM_DENOMINATOR=2,
      // so deriv() must return "Convention A" (half gradient): multiply by -1/2.
      dSdU.U = (-0.5) * clover_force;
    }
  }

 private:
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  std::array<RealD, TxqcdNf> mass_;
  std::array<RealD, TxqcdNf> diag_mass_;
  RealD csw_;
  SMU::SpinMatrices sm_;
  std::vector<LatticeColourMatrix> FS_;
  // Phase J GPU LogDet scratch (TXQCD_LOGDET_GPU=1 default ON).
  //   J.1: M^{-1} extraction kernel (CPU build+invert, GPU traces).
  //   J.2: cuBLAS GPU invert.  m_fwd_host_ holds CPU-built forward M;
  //   M_fwd_dev_ is its device mirror (overwritten by getrf with LU);
  //   M_inv_dev_ holds the inverse from getriBatched.
  std::vector<SMU::SiteMatrix> m_fwd_host_;
  deviceVector<ComplexD>       M_fwd_dev_;
  deviceVector<ComplexD>       M_inv_dev_;
  deviceVector<int>            lex_table_dev_;
  bool                         lex_built_{false};
  // cuBLAS getrf/getri pointer arrays + pivot/info scratch.
  deviceVector<ComplexD *>     Amk_ptrs_, Cmk_ptrs_;
  deviceVector<int>            pivots_dev_, info_dev_;
  bool                         cublas_built_{false};
  uint64_t                     cublas_nsites_{0};
  mutable uint64_t t_pickcb_us_{0}, t_inv_us_{0}, t_upload_us_{0};
  mutable uint64_t t_cublas_us_{0}, t_traces_us_{0}, t_total_us_{0};
  mutable uint64_t n_deriv_{0};

  SMU::AuxSiteArrays GetEvenAux(const TXQCDField &U) {
    LatticeSigmaField sigma_e(&rbgrid_);
    LatticePiField pi_e(&rbgrid_);
    LatticeSFieldC s_e(&rbgrid_);
    LatticePFieldC p_e(&rbgrid_);
    LatticeTField t_e(&rbgrid_);
    pickCheckerboard(Even, sigma_e, U.sigma);
    pickCheckerboard(Even, pi_e, U.pi);
    pickCheckerboard(Even, s_e, U.s);
    pickCheckerboard(Even, p_e, U.p);
    pickCheckerboard(Even, t_e, U.t);
    return SMU::UnvectorizeAux(sigma_e, pi_e, s_e, p_e, t_e);
  }

  SMU::CloverSiteArrays GetEvenClover(const TXQCDField &U) {
    if (csw_ == 0.0) return SMU::CloverSiteArrays();
    FS_.clear();
    std::vector<LatticeColourMatrix> FS_e;
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu) {
        FS_.emplace_back(&grid_);
        WilsonLoops<WilsonImplR>::FieldStrength(FS_.back(), U.U, mu, nu);
        FS_e.emplace_back(&rbgrid_);
        pickCheckerboard(Even, FS_e.back(), FS_.back());
      }
    return SMU::UnvectorizeClover(FS_e);
  }
};

NAMESPACE_END(Grid);
