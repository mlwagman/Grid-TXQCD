#pragma once
// Log-determinant action for the even-site diagonal block of the
// EO-preconditioned DTXQCD doubled Wilson-Clover operator.
//
// For the Pfaffian weight  |Pf(D_doubled)| = |det(D_doubled)|^{1/2}
// (paper Eq. 321: Pf(D) = det(D^2)^{1/4}), the total fermion action is
//   S = -log|Pf(D)| = -(1/2) log|det(D)|
//                  = -(1/2) [log|det(M_ee)| + log|det(Mpc)|]
// so this LogDet action contributes
//   S_LD = -(1/2) sum_{x even} log|det(M_ee_48(x))|
// The complementary 1/2 factor goes into the RHMC pseudofermion action on
// Mpc via the x^{-1/4} (action) and x^{+1/8} (heatbath) rational exponents,
// so each block contributes |det|^{1/2} to the path-integral weight.
//
// Each per-site M_ee is the 48x48 doubled site matrix from DTXQCDSiteMatrix:
//   upper block:  m I + Delta_diag (sigma^A, pi^A, t^A) + (optional) -(csw/2) F sigma
//   lower block:  m I + Delta_diag_lower (tensor sign-flipped) + (optional) +(csw/2) F^T sigma
//   off-diag:     2 d gamma_5 + 2 n
//
// Force formula at each even site x:
//   dS/d(X)(x) = -Tr(M_ee^{-1}(x) * dM_ee/dX(x))
// where X ranges over the aux fields (sigma^A, pi^A, t^A_{mu,nu}, d^{ij}, n^{ij})
// and (csw != 0 case) the gauge field via the clover term.
//
// v1 implementation:
//   - CPU-only (no GPU acceleration; TXQCD's GPU path can port later).
//   - Aux-field forces:    full analytic deriv (this file).
//   - Gauge clover force:  via dF/dU chain rule, WilsonCloverHelpers::Cmunu,
//                          mirroring TXQCDLogDetCloverEOAction.  Includes
//                          contributions from BOTH upper (-(csw/2) F sigma)
//                          and lower (+(csw/2) F^T sigma) clover terms.
//                          For the lower block, F^T means F's (j, i) entry,
//                          so the M_inv_ll trace formula has swapped (i, j)
//                          indices on the M_inv access.

#include <Grid/qcd/action/dtxqcd/DTXQCDField.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDBatchedInverse48.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteForceKernel.h>
#include <cstring>
#include <Grid/qcd/action/fermion/WilsonCloverHelpers.h>
#include <Grid/qcd/action/fermion/WilsonImpl.h>
#include <Grid/qcd/utils/WilsonLoops.h>

NAMESPACE_BEGIN(Grid);

class DTXQCDLogDetCloverEOAction : public Action<DTXQCDField> {
 public:
  static constexpr int kDim24 = kDtxqcdSiteDim24;
  static constexpr int kDim48 = kDtxqcdSiteDim48;

  DTXQCDLogDetCloverEOAction(GridCartesian &grid,
                             GridRedBlackCartesian &rbgrid,
                             RealD mass, RealD csw = 0.0)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), csw_(csw), spin_(grid) {}

  std::string action_name() override { return "DTXQCDLogDetCloverEOAction"; }
  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=" << mass_
       << " csw=" << csw_ << std::endl;
    return os.str();
  }

  void refresh(const DTXQCDField &U, GridSerialRNG &, GridParallelRNG &) override {}

  // ------------------------------------------------------------------
  //  S(U) = -(1/2) sum_{x in EVEN} log |det(M_ee_48(x))|
  //  (1/2 factor for the Pfaffian weight; see header.)
  //
  //  Dispatch: GPU path (batched cuBLAS LU + device log-diagonal reduction)
  //  when DTXQCD_LOGDET_S_GPU is on (default ON under CUDA, OFF otherwise).
  //  The dispatch itself is GRID_CUDA-gated so non-CUDA builds always take the
  //  Eigen CPU path (cf. TXQCD commit 8c0f721f).
  // ------------------------------------------------------------------
  RealD S(const DTXQCDField &U) override {
    static int use_gpu = []() {
#ifndef GRID_CUDA
      const char *e = std::getenv("DTXQCD_LOGDET_S_GPU");
      return (e && *e) ? std::atoi(e) : 0;
#else
      const char *e = std::getenv("DTXQCD_LOGDET_S_GPU");
      if (!e || !*e) return 1;
      return std::atoi(e);
#endif
    }();
    if (use_gpu) return S_gpu(U);
    return S_cpu(U);
  }

  RealD S_cpu(const DTXQCDField &U) {
    EvenLocalAux L;
    ExtractEvenLocalAux(U, L);
    RealD logdet = 0.0;
    RealD sum_arg = 0.0;
    RealD min_log_absdet = 1e300;
    RealD n_small = 0.0;
    const RealD small_threshold = -10.0;  // log|det| < -10 ~ |det| < 5e-5
    for (uint64_t idx = 0; idx < L.Nsite; ++idx) {  // LOCAL even sites
      Eigen::MatrixXcd M48;
      BuildSiteMatrix48Local(L, idx, M48);
      std::complex<double> det = M48.partialPivLu().determinant();
      RealD logabs = std::log(std::abs(det));
      logdet += logabs;
      sum_arg += std::arg(det);
      if (logabs < min_log_absdet) min_log_absdet = logabs;
      if (logabs < small_threshold) n_small += 1.0;
    }
    grid_.GlobalSum(logdet);   // per-rank partial sums -> correct global total
    grid_.GlobalSum(sum_arg);
    grid_.GlobalSum(n_small);
    RealD action = -0.5 * logdet;
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << "  sum_arg(det) = " << sum_arg
              << "  min_log|det| = " << min_log_absdet
              << "  n_small = " << n_small << std::endl;
    return action;
  }

  // GPU LogDet action: one batched cuBLAS getrf over the even-site 48x48 M_ee,
  // then a device reduction of sum_k log|U_kk| = log|det|.  Pfaffian factor
  // -1/2.  Matches S_cpu to roundoff (LU |det| = prod |U_kk|).
  RealD S_gpu(const DTXQCDField &U) {
#ifdef GRID_CUDA
    EvenLocalAux L;
    ExtractEvenLocalAux(U, L);
    const uint64_t nsites = L.Nsite;
    constexpr int N = kDim48;
    constexpr int N2 = N * N;

    // Forward M_ee per LOCAL even site (no peekSite -> parallel thread_for).
    std::vector<std::complex<double>> h_fwd((size_t)nsites * N2);
    thread_for(idx, nsites, {
      Eigen::MatrixXcd M48;
      BuildSiteMatrix48Local(L, idx, M48);  // column-major storage
      std::memcpy(&h_fwd[(size_t)idx * N2], M48.data(),
                  (size_t)N2 * sizeof(std::complex<double>));
    });

    DtxqcdBlas::BatchedInverse48::Ensure(nsites, /*need_inv=*/false);
    acceleratorCopyToDevice((void *)h_fwd.data(),
                            (void *)&DtxqcdBlas::BatchedInverse48::M_fwd[0],
                            (size_t)nsites * N2 * sizeof(ComplexD));
    DtxqcdBlas::BatchedInverse48::FactorLU(nsites);  // M_fwd <- LU in place

    static deviceVector<RealD> ld_dev;
    if (ld_dev.size() < nsites) ld_dev.resize(nsites);
    ComplexD *MA = &DtxqcdBlas::BatchedInverse48::M_fwd[0];
    RealD    *LD = &ld_dev[0];
    accelerator_for(i, nsites, 1, {
      RealD acc = 0.0;
      for (int k = 0; k < N; ++k) {
        ComplexD u = MA[(uint64_t)i * N2 + k + (uint64_t)k * N];  // U_kk
        RealD m2 = u.real() * u.real() + u.imag() * u.imag();
        acc += 0.5 * ::log(m2);
      }
      LD[i] = acc;
    });

    std::vector<RealD> ld_host(nsites);
    acceleratorCopyFromDevice((void *)&ld_dev[0], (void *)ld_host.data(),
                              nsites * sizeof(RealD));
    RealD logdet = 0.0;
    for (uint64_t i = 0; i < nsites; ++i) logdet += ld_host[i];
    grid_.GlobalSum(logdet);
    RealD action = -0.5 * logdet;
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << " (GPU)" << std::endl;
    return action;
#else
    return S_cpu(U);  // never reached: dispatch forces CPU on non-CUDA builds
#endif
  }

  // ------------------------------------------------------------------
  //  deriv(U, dSdU): aux-field forces + gauge clover force (csw != 0).
  //  Aux: per-site -Tr(M_ee^{-1} dM_ee/dX) via DTXQCDSiteForceKernel.
  //  Gauge: per-site clover_sigma fed to WilsonCloverHelpers::Cmunu, both
  //  upper (-(csw/2) F sigma) and lower (+(csw/2) F^T sigma) contributions.
  //  Dispatch: GPU offloads the per-site 48x48 inverse to batched cuBLAS and
  //  reuses the CPU force kernel (DTXQCD_LOGDET_GPU, default ON under CUDA).
  //  Dispatch is GRID_CUDA-gated (cf. TXQCD commit 8c0f721f).
  // ------------------------------------------------------------------
  void deriv(const DTXQCDField &U, DTXQCDField &dSdU) override {
    static int use_gpu = []() {
      const char *e = std::getenv("DTXQCD_LOGDET_GPU");
      if (!e || !*e) return 1;
      return std::atoi(e);
    }();
#ifdef GRID_CUDA
    if (use_gpu) { deriv_gpu(U, dSdU); return; }
#else
    (void)use_gpu;
#endif
    deriv_cpu(U, dSdU);
  }

  void deriv_cpu(const DTXQCDField &U, DTXQCDField &dSdU) {
    dSdU = Zero();
    EvenLocalAux L;
    ExtractEvenLocalAux(U, L);
    const uint64_t Nsite = L.Nsite;

    using DtxqcdSiteForceKernel::SigSobj;
    using DtxqcdSiteForceKernel::PiSobj;
    using DtxqcdSiteForceKernel::DSobj;
    using DtxqcdSiteForceKernel::NSobj;
    using DtxqcdSiteForceKernel::SSobj;
    using DtxqcdSiteForceKernel::PSobj;

    // Per-local-even-site force outputs (lex order, even CB), scattered into
    // the full-grid dSdU after the loop.  Replaces the old global-coord
    // pokeSite loop (which OOM'd / over-counted at multi-rank).
    std::vector<SigSobj> fsig(Nsite);
    std::vector<PiSobj>  fpi(Nsite);
    std::vector<DSobj>   fd(Nsite);
    std::vector<NSobj>   fn(Nsite);
    std::vector<SSobj>   fss(Nsite);
    std::vector<PSobj>   fpp(Nsite);
    std::array<std::vector<CMSob>, 6> fcs;
    if (L.has_clover)
      for (int k = 0; k < 6; ++k) fcs[k].resize(Nsite);

    thread_for(idx, Nsite, {
      Eigen::MatrixXcd M48;
      BuildSiteMatrix48Local(L, idx, M48);
      Eigen::MatrixXcd Inv = M48.inverse();
      SigSobj sig_force; PiSobj pi_force; DSobj d_force;
      NSobj n_force; SSobj s_force; PSobj p_force;
      auto InvLookup = [&Inv](int r, int c) -> ComplexD { return Inv(r, c); };
      DtxqcdSiteForceKernel::AuxForceAt(InvLookup, spin_, sig_force, pi_force,
                                        d_force, n_force, s_force, p_force);
      if (L.has_clover) {
        std::array<DtxqcdSiteForceKernel::CMsobj, 6> cs_arr;
        DtxqcdSiteForceKernel::CloverSigmaAt(InvLookup, spin_, csw_, cs_arr);
        for (int k = 0; k < 6; ++k) fcs[k][idx] = cs_arr[k];
      }
      // Wirtinger -> physical-gradient: transpose (a<->b, i<->j) on the CF
      // force fields; singlets s,p direct.  (Was per-site TransposePoke.)
      auto T = [](const auto &fv, auto &ft) {
        for (int a = 0; a < DtxqcdNf; ++a)
          for (int b = 0; b < DtxqcdNf; ++b)
            for (int i = 0; i < Nc; ++i)
              for (int j = 0; j < Nc; ++j)
                ft()(a, b)(i, j) = fv()(b, a)(j, i);
      };
      T(sig_force, fsig[idx]);
      T(pi_force,  fpi[idx]);
      T(d_force,   fd[idx]);
      T(n_force,   fn[idx]);
      fss[idx] = s_force;
      fpp[idx] = p_force;
    });

    PackEvenToFull(fsig, dSdU.sigma);
    PackEvenToFull(fpi,  dSdU.pi);
    PackEvenToFull(fd,   dSdU.d);
    PackEvenToFull(fn,   dSdU.n);
    PackEvenToFull(fss,  dSdU.s);
    PackEvenToFull(fpp,  dSdU.p);

    std::vector<LatticeColourMatrix> clover_sigma_full;
    if (L.has_clover) {
      clover_sigma_full.reserve(6);
      for (int k = 0; k < 6; ++k) {
        clover_sigma_full.emplace_back(&grid_);
        clover_sigma_full.back() = Zero();
        PackEvenToFull(fcs[k], clover_sigma_full[k]);
      }
    }

    // ---- Gauge clover force via Cmunu chain rule ------------------------
    // For each Lorentz mu, sum over nu != mu of (sign * Cmunu(Ulinks,
    // clover_sigma_full[mn], mu, nu)), then poke Ulinks[mu] * force_mu into
    // the dSdU.U Lorentz mu slot.  Cmunu returns "Convention B" (full
    // gradient); apply -0.5 to convert to "Convention A" expected by
    // Grid's HMC integrator.  Mirrors TXQCDLogDetCloverEOAction.
    if (csw_ != 0.0) {
      typedef WilsonImplR Impl;
      std::vector<LatticeColourMatrix> Ulinks;
      Ulinks.reserve(Nd);
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix Umu(&grid_);
        Umu = PeekIndex<LorentzIndex>(U.U, mu);
        Ulinks.push_back(std::move(Umu));
      }

      LatticeGaugeField clover_force(&grid_);
      clover_force = Zero();

      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix force_mu(&grid_);
        force_mu = Zero();
        for (int nu = 0; nu < Nd; ++nu) {
          if (mu == nu) continue;
          // Canonical (m<n) index from (mu, nu) -- our 6-pair ordering.
          int m = std::min(mu, nu);
          int n = std::max(mu, nu);
          int mn = 0;
          {
            int k = 0;
            for (int mm = 0; mm < Nd; ++mm)
              for (int nn = mm + 1; nn < Nd; ++nn) {
                if (mm == m && nn == n) mn = k;
                ++k;
              }
          }
          // sigma_{nu, mu} = -sigma_{mu, nu}: fold antisymmetry sign into a
          // scalar rather than building a redundant -clover_sigma lattice.
          RealD sign = (mu < nu) ? 1.0 : -1.0;
          force_mu += (0.25 * sign) *
              WilsonCloverHelpers<Impl>::Cmunu(Ulinks, clover_sigma_full[mn],
                                               mu, nu);
        }
        pokeLorentz(clover_force, Ulinks[mu] * force_mu, mu);
      }
      dSdU.U = ComplexD(-0.5, 0.0) * clover_force;
    }

    // Overall Pfaffian factor: S = -(1/2) log|det(M_ee)|, so scale all
    // contributions (aux + gauge) by 1/2.  See header comment.
    dSdU.sigma = ComplexD(0.5, 0.0) * dSdU.sigma;
    dSdU.pi    = ComplexD(0.5, 0.0) * dSdU.pi;
    dSdU.d     = ComplexD(0.5, 0.0) * dSdU.d;
    dSdU.n     = ComplexD(0.5, 0.0) * dSdU.n;
    dSdU.s     = ComplexD(0.5, 0.0) * dSdU.s;
    dSdU.p     = ComplexD(0.5, 0.0) * dSdU.p;
    dSdU.U     = ComplexD(0.5, 0.0) * dSdU.U;
  }

  // GPU deriv: one batched cuBLAS getrf+getri over all even-site 48x48 M_ee,
  // then the SAME host analytic force kernel per site reading the device-
  // computed inverse.  Offloads the O(48^3) per-site inversion (dominant cost);
  // the O(48^2) force traces stay on host.  Matches deriv_cpu to roundoff.
  void deriv_gpu(const DTXQCDField &U, DTXQCDField &dSdU) {
#ifdef GRID_CUDA
    dSdU = Zero();
    EvenLocalAux L;
    ExtractEvenLocalAux(U, L);
    const uint64_t Nsite = L.Nsite;
    constexpr int N = kDim48;
    constexpr int N2 = N * N;

    using DtxqcdSiteForceKernel::SigSobj;
    using DtxqcdSiteForceKernel::PiSobj;
    using DtxqcdSiteForceKernel::DSobj;
    using DtxqcdSiteForceKernel::NSobj;
    using DtxqcdSiteForceKernel::SSobj;
    using DtxqcdSiteForceKernel::PSobj;

    // Forward M_ee per LOCAL even site -> one batched cuBLAS getrf+getri.
    std::vector<std::complex<double>> h_fwd((size_t)Nsite * N2);
    thread_for(idx, Nsite, {
      Eigen::MatrixXcd M48;
      BuildSiteMatrix48Local(L, idx, M48);  // column-major storage
      std::memcpy(&h_fwd[(size_t)idx * N2], M48.data(),
                  (size_t)N2 * sizeof(std::complex<double>));
    });
    DtxqcdBlas::BatchedInverse48::Ensure(Nsite, /*need_inv=*/true);
    acceleratorCopyToDevice((void *)h_fwd.data(),
                            (void *)&DtxqcdBlas::BatchedInverse48::M_fwd[0],
                            (size_t)Nsite * N2 * sizeof(ComplexD));
    DtxqcdBlas::BatchedInverse48::Invert(Nsite);
    std::vector<std::complex<double>> h_inv((size_t)Nsite * N2);
    acceleratorCopyFromDevice((void *)&DtxqcdBlas::BatchedInverse48::M_inv[0],
                              (void *)h_inv.data(),
                              (size_t)Nsite * N2 * sizeof(ComplexD));

    std::vector<SigSobj> fsig(Nsite);
    std::vector<PiSobj>  fpi(Nsite);
    std::vector<DSobj>   fd(Nsite);
    std::vector<NSobj>   fn(Nsite);
    std::vector<SSobj>   fss(Nsite);
    std::vector<PSobj>   fpp(Nsite);
    std::array<std::vector<CMSob>, 6> fcs;
    if (L.has_clover)
      for (int k = 0; k < 6; ++k) fcs[k].resize(Nsite);

    thread_for(idx, Nsite, {
      const std::complex<double> *Mi = &h_inv[(size_t)idx * N2];
      SigSobj sig_force; PiSobj pi_force; DSobj d_force;
      NSobj n_force; SSobj s_force; PSobj p_force;
      auto InvLookup = [Mi](int r, int c) -> ComplexD {
        return ComplexD(Mi[(size_t)c * kDim48 + r]);  // column-major -> (r,c)
      };
      DtxqcdSiteForceKernel::AuxForceAt(InvLookup, spin_, sig_force, pi_force,
                                        d_force, n_force, s_force, p_force);
      if (L.has_clover) {
        std::array<DtxqcdSiteForceKernel::CMsobj, 6> cs_arr;
        DtxqcdSiteForceKernel::CloverSigmaAt(InvLookup, spin_, csw_, cs_arr);
        for (int k = 0; k < 6; ++k) fcs[k][idx] = cs_arr[k];
      }
      auto T = [](const auto &fv, auto &ft) {
        for (int a = 0; a < DtxqcdNf; ++a)
          for (int b = 0; b < DtxqcdNf; ++b)
            for (int i = 0; i < Nc; ++i)
              for (int j = 0; j < Nc; ++j)
                ft()(a, b)(i, j) = fv()(b, a)(j, i);
      };
      T(sig_force, fsig[idx]);
      T(pi_force,  fpi[idx]);
      T(d_force,   fd[idx]);
      T(n_force,   fn[idx]);
      fss[idx] = s_force;
      fpp[idx] = p_force;
    });

    PackEvenToFull(fsig, dSdU.sigma);
    PackEvenToFull(fpi,  dSdU.pi);
    PackEvenToFull(fd,   dSdU.d);
    PackEvenToFull(fn,   dSdU.n);
    PackEvenToFull(fss,  dSdU.s);
    PackEvenToFull(fpp,  dSdU.p);

    std::vector<LatticeColourMatrix> clover_sigma_full;
    if (L.has_clover) {
      clover_sigma_full.reserve(6);
      for (int k = 0; k < 6; ++k) {
        clover_sigma_full.emplace_back(&grid_);
        clover_sigma_full.back() = Zero();
        PackEvenToFull(fcs[k], clover_sigma_full[k]);
      }
    }

    if (csw_ != 0.0) {
      typedef WilsonImplR Impl;
      std::vector<LatticeColourMatrix> Ulinks;
      Ulinks.reserve(Nd);
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix Umu(&grid_);
        Umu = PeekIndex<LorentzIndex>(U.U, mu);
        Ulinks.push_back(std::move(Umu));
      }
      LatticeGaugeField clover_force(&grid_);
      clover_force = Zero();
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix force_mu(&grid_);
        force_mu = Zero();
        for (int nu = 0; nu < Nd; ++nu) {
          if (mu == nu) continue;
          int m = std::min(mu, nu);
          int n = std::max(mu, nu);
          int mn = 0;
          {
            int k = 0;
            for (int mm = 0; mm < Nd; ++mm)
              for (int nn = mm + 1; nn < Nd; ++nn) {
                if (mm == m && nn == n) mn = k;
                ++k;
              }
          }
          RealD sign = (mu < nu) ? 1.0 : -1.0;
          force_mu += (0.25 * sign) *
              WilsonCloverHelpers<Impl>::Cmunu(Ulinks, clover_sigma_full[mn],
                                               mu, nu);
        }
        pokeLorentz(clover_force, Ulinks[mu] * force_mu, mu);
      }
      dSdU.U = ComplexD(-0.5, 0.0) * clover_force;
    }
    dSdU.sigma = ComplexD(0.5, 0.0) * dSdU.sigma;
    dSdU.pi    = ComplexD(0.5, 0.0) * dSdU.pi;
    dSdU.d     = ComplexD(0.5, 0.0) * dSdU.d;
    dSdU.n     = ComplexD(0.5, 0.0) * dSdU.n;
    dSdU.s     = ComplexD(0.5, 0.0) * dSdU.s;
    dSdU.p     = ComplexD(0.5, 0.0) * dSdU.p;
    dSdU.U     = ComplexD(0.5, 0.0) * dSdU.U;
#else
    deriv_cpu(U, dSdU);  // never reached: dispatch forces CPU on non-CUDA
#endif
  }

 private:
  // ---- LOCAL (per-rank) even-CB aux extraction -----------------------------
  // The S/deriv loops were originally over grid_.GlobalDimensions() with
  // peekSite -> every rank built the whole lattice (3.6 GB OOM at mpi!=1.1.1.1)
  // and the GlobalSum over-counted by N_ranks.  These helpers unvectorize the
  // even-CB aux (+FS) into THIS rank's lex-ordered host arrays so S/deriv loop
  // local sites only (multi-rank correct, no OOM, GPU-safe to thread_for since
  // there is no peekSite in the hot loop).  Mirrors BuildInverseCacheCB.
  typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SigSob;
  typedef typename LatticeDtxqcdPi::vector_object::scalar_object    PiSob;
  typedef typename LatticeDtxqcdD::vector_object::scalar_object     DSob;
  typedef typename LatticeDtxqcdN::vector_object::scalar_object     NSob;
  typedef typename LatticeDtxqcdS::vector_object::scalar_object     SSob;
  typedef typename LatticeDtxqcdP::vector_object::scalar_object     PSob;
  typedef typename LatticeColourMatrix::vector_object::scalar_object CMSob;

  struct EvenLocalAux {
    std::vector<SigSob> sig; std::vector<PiSob> pi;
    std::vector<DSob>   d;   std::vector<NSob>  n;
    std::vector<SSob>   s;   std::vector<PSob>  p;
    std::array<std::vector<CMSob>, 6> fs;  // populated iff csw != 0
    uint64_t Nsite{0};
    bool has_clover{false};
  };

  void ExtractEvenLocalAux(const DTXQCDField &U, EvenLocalAux &L) {
    LatticeDtxqcdSigma sig_e(&rbgrid_); pickCheckerboard(Even, sig_e, U.sigma);
    LatticeDtxqcdPi    pi_e(&rbgrid_);  pickCheckerboard(Even, pi_e,  U.pi);
    LatticeDtxqcdD     d_e(&rbgrid_);   pickCheckerboard(Even, d_e,   U.d);
    LatticeDtxqcdN     n_e(&rbgrid_);   pickCheckerboard(Even, n_e,   U.n);
    LatticeDtxqcdS     s_e(&rbgrid_);   pickCheckerboard(Even, s_e,   U.s);
    LatticeDtxqcdP     p_e(&rbgrid_);   pickCheckerboard(Even, p_e,   U.p);
    unvectorizeToLexOrdArray(L.sig, sig_e);
    unvectorizeToLexOrdArray(L.pi,  pi_e);
    unvectorizeToLexOrdArray(L.d,   d_e);
    unvectorizeToLexOrdArray(L.n,   n_e);
    unvectorizeToLexOrdArray(L.s,   s_e);
    unvectorizeToLexOrdArray(L.p,   p_e);
    L.Nsite = L.sig.size();
    if (csw_ != 0.0) {
      auto FS = BuildFS(U);
      for (int k = 0; k < 6; ++k) {
        LatticeColourMatrix fe(&rbgrid_); pickCheckerboard(Even, fe, FS[k]);
        unvectorizeToLexOrdArray(L.fs[k], fe);
      }
      L.has_clover = true;
    }
  }

  // Build the forward 48x48 M_ee at local lex even site idx (no peekSite).
  void BuildSiteMatrix48Local(const EvenLocalAux &L, uint64_t idx,
                              Eigen::MatrixXcd &M48) {
    DtxqcdSiteAux aux = DtxqcdSiteAux::FromSobjs(
        L.sig[idx], L.pi[idx], L.d[idx], L.n[idx], L.s[idx], L.p[idx]);
    Eigen::MatrixXcd M_upper, M_lower, M_off;
    if (L.has_clover) {
      std::array<CMSob, 6> fsite;
      for (int k = 0; k < 6; ++k) fsite[k] = L.fs[k][idx];
      DtxqcdSiteClover clover = DtxqcdSiteClover::FromSobjs(fsite);
      DtxqcdBuildUpperBlock24(mass_, aux, spin_, M_upper, csw_, &clover);
      DtxqcdBuildLowerBlock24(mass_, aux, spin_, M_lower, csw_, &clover);
    } else {
      DtxqcdBuildUpperBlock24(mass_, aux, spin_, M_upper);
      DtxqcdBuildLowerBlock24(mass_, aux, spin_, M_lower);
    }
    DtxqcdBuildOffDiagBlock24(aux, spin_, M_off);
    DtxqcdAssembleDoubled48(M_upper, M_lower, M_off, M48);
  }

  // Scatter a lex-ordered even-CB host array into the EVEN sites of a full-grid
  // lattice (odd sites untouched).  Inverse of unvectorize on the even CB.
  template <class Field, class Sobj>
  void PackEvenToFull(std::vector<Sobj> &arr, Field &full) {
    Field e(&rbgrid_);
    e.Checkerboard() = Even;
    vectorizeFromLexOrdArray(arr, e);
    setCheckerboard(full, e);
  }

  // Build the field-strength F_{mu,nu} (6 pairs) from U via Grid's WilsonLoops.
  std::vector<LatticeColourMatrix> BuildFS(const DTXQCDField &U) const {
    std::vector<LatticeColourMatrix> FS;
    if (csw_ == 0.0) return FS;
    FS.reserve(6);
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu) {
        LatticeColourMatrix F(U.U.Grid());
        WilsonLoops<WilsonImplR>::FieldStrength(F, U.U, mu, nu);
        FS.push_back(std::move(F));
      }
    return FS;
  }

  // Build the 48x48 per-site M_ee matrix.
  void BuildSiteMatrix48(const DTXQCDField &U,
                         const std::vector<LatticeColourMatrix> &FS,
                         const Coordinate &coord,
                         Eigen::MatrixXcd &M48) {
    DtxqcdSiteAux aux =
        DtxqcdSiteAux::Extract(U.sigma, U.pi, U.d, U.n, U.s, U.p, coord);
    Eigen::MatrixXcd M_upper, M_lower, M_off;
    if (csw_ != 0.0) {
      DtxqcdSiteClover clover = DtxqcdSiteClover::Extract(FS, coord);
      DtxqcdBuildUpperBlock24(mass_, aux, spin_, M_upper, csw_, &clover);
      DtxqcdBuildLowerBlock24(mass_, aux, spin_, M_lower, csw_, &clover);
    } else {
      DtxqcdBuildUpperBlock24(mass_, aux, spin_, M_upper);
      DtxqcdBuildLowerBlock24(mass_, aux, spin_, M_lower);
    }
    DtxqcdBuildOffDiagBlock24(aux, spin_, M_off);
    DtxqcdAssembleDoubled48(M_upper, M_lower, M_off, M48);
  }

  GridCartesian         &grid_;
  GridRedBlackCartesian &rbgrid_;
  RealD                  mass_;
  RealD                  csw_;
  DtxqcdSpinMatrices     spin_;
};

NAMESPACE_END(Grid);
