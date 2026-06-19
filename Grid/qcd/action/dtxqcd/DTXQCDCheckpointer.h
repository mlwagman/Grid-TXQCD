#pragma once
// Checkpointer for the v2 DTXQCD composite Field (gauge + 4 CF-Hermitian
// aux fields + 2 singlet scalars).
//
// Writes:
//   <config_prefix>.N         gauge, NERSC binary (interoperable with stock Grid)
//   <config_prefix>_daux.N    packed aux sidecar (DTXQCD v2 layout)
//   <rng_prefix>.N            RNG state (NERSC writer)
//
// Sidecar magic 'DTX3' (0x44545833) — bumped from 'DTX2' (2026-06-19) when
// d, n switched to truly complex-symmetric storage (42 reals each instead
// of 36) to preserve the imaginary-diagonal DOFs that the Hermitian pack
// silently discarded.
//
// Packed layout per site (158 doubles = 1264 bytes):
//   sigma:  36 reals  (6 real diag + 15 off-diag (re, im) pairs)   [Hermitian]
//   pi:     36 reals  [Hermitian]
//   d:      42 reals  (6 diag (re, im) pairs + 15 off-diag (re, im) pairs) [cplx-symm]
//   n:      42 reals  [cplx-symm]
//   s:       1 real
//   p:       1 real
//
// Hermitian and complex-symmetric pack share the same combined-index
// scheme k = a*Nc + i (k1 = row, k2 = col) but differ in how the
// diagonal is stored: Hermitian → real only (6 doubles), complex-symm →
// full complex (12 doubles).  On read the field is re-projected to its
// expected subspace via DTXQCDCompositeImpl::Project so small numerical
// drift in the stored values is absorbed.
//
// Multi-rank layout follows v1: each rank writes its rank-local
// lex-ordered slice at offset 16 + my_rank * local_bytes via MPI-IO
// collective.

#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>
#include <cstring>
#if defined(GRID_COMMS_MPI3)
#include <mpi.h>
#endif

NAMESPACE_BEGIN(Grid);

class DTXQCDCheckpointer : public BaseHmcCheckpointer<DTXQCDCompositeImpl> {
 private:
  CheckpointerParameters Params;

  static constexpr uint32_t kAuxMagic   = 0x44545833;  // 'DTX3'
  static constexpr uint32_t kAuxVersion = 3;

  static constexpr int kCFDim       = DtxqcdNfNc;                       // 6
  // Hermitian pack: 6 real diag + 15 off-diag (re, im) = 36 reals.
  static constexpr int kCFHermPacked = kCFDim
                                     + kCFDim * (kCFDim - 1);
  static_assert(kCFHermPacked == 36, "kCFHermPacked should be 36");
  // Complex-symm pack: 6 diag (re, im) + 15 off-diag (re, im) = 42 reals.
  static constexpr int kCFSymPacked  = 2 * kCFDim
                                     + kCFDim * (kCFDim - 1);
  static_assert(kCFSymPacked == 42, "kCFSymPacked should be 42");
  static constexpr int kScalarPacked = 1;
  static constexpr int kSiteDoubles =
      2 * kCFHermPacked   // sigma, pi
    + 2 * kCFSymPacked    // d, n
    + 2 * kScalarPacked;                                                // 158

  std::string aux_filename(int traj) const {
    std::ostringstream os;
    os << Params.config_prefix << "_daux." << traj;
    return os.str();
  }

  // ----- per-site CF Hermitian pack/unpack -----
  //
  // Combined index k = a*Nc + i for both row and column.  Diagonal entries
  // are stored as 6 reals first, then 15 off-diagonal upper-triangle
  // entries as (re, im) pairs.
  template <class CFSobj>
  static void PackCFHermitian(const CFSobj &M, double *buf) {
    int kk = 0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int i = 0; i < Nc; ++i) {
        buf[kk++] = M()(a, a)(i, i).real();
      }
    }
    for (int k1 = 0; k1 < kCFDim; ++k1) {
      int a1 = k1 / Nc, i1 = k1 % Nc;
      for (int k2 = k1 + 1; k2 < kCFDim; ++k2) {
        int a2 = k2 / Nc, i2 = k2 % Nc;
        auto z = M()(a1, a2)(i1, i2);
        buf[kk++] = z.real();
        buf[kk++] = z.imag();
      }
    }
  }
  template <class CFSobj>
  static void UnpackCFHermitian(const double *buf, CFSobj &M) {
    using SC = std::remove_reference_t<decltype(M()(0, 0)(0, 0))>;
    // Zero everything first; off-diag lower fills via conjugate below.
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int b = 0; b < DtxqcdNf; ++b)
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j)
            M()(a, b)(i, j) = SC(0.0, 0.0);
    int kk = 0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int i = 0; i < Nc; ++i) {
        M()(a, a)(i, i) = SC(buf[kk++], 0.0);
      }
    }
    for (int k1 = 0; k1 < kCFDim; ++k1) {
      int a1 = k1 / Nc, i1 = k1 % Nc;
      for (int k2 = k1 + 1; k2 < kCFDim; ++k2) {
        int a2 = k2 / Nc, i2 = k2 % Nc;
        double re = buf[kk++], im = buf[kk++];
        M()(a1, a2)(i1, i2) = SC(re, im);
        M()(a2, a1)(i2, i1) = SC(re, -im);
      }
    }
  }

  // ----- per-site CF complex-symmetric pack/unpack (truly complex-symm,
  // joint (color+flavor) transpose: M(k1,k2) = M(k2,k1) with complex
  // values throughout — used for d, n under DTXQCD_DN_COMPLEX_SYMMETRIC).
  template <class CFSobj>
  static void PackCFSymmetric(const CFSobj &M, double *buf) {
    int kk = 0;
    // Diagonal: 6 complex entries.
    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int i = 0; i < Nc; ++i) {
        auto z = M()(a, a)(i, i);
        buf[kk++] = z.real();
        buf[kk++] = z.imag();
      }
    }
    // Off-diagonal upper triangle: 15 complex entries.
    for (int k1 = 0; k1 < kCFDim; ++k1) {
      int a1 = k1 / Nc, i1 = k1 % Nc;
      for (int k2 = k1 + 1; k2 < kCFDim; ++k2) {
        int a2 = k2 / Nc, i2 = k2 % Nc;
        auto z = M()(a1, a2)(i1, i2);
        buf[kk++] = z.real();
        buf[kk++] = z.imag();
      }
    }
  }
  template <class CFSobj>
  static void UnpackCFSymmetric(const double *buf, CFSobj &M) {
    using SC = std::remove_reference_t<decltype(M()(0, 0)(0, 0))>;
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int b = 0; b < DtxqcdNf; ++b)
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j)
            M()(a, b)(i, j) = SC(0.0, 0.0);
    int kk = 0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int i = 0; i < Nc; ++i) {
        double re = buf[kk++], im = buf[kk++];
        M()(a, a)(i, i) = SC(re, im);
      }
    }
    for (int k1 = 0; k1 < kCFDim; ++k1) {
      int a1 = k1 / Nc, i1 = k1 % Nc;
      for (int k2 = k1 + 1; k2 < kCFDim; ++k2) {
        int a2 = k2 / Nc, i2 = k2 % Nc;
        double re = buf[kk++], im = buf[kk++];
        // M(k1,k2) = M(k2,k1) — joint transpose, NO conjugate.
        M()(a1, a2)(i1, i2) = SC(re, im);
        M()(a2, a1)(i2, i1) = SC(re, im);
      }
    }
  }

  template <class ScalarSobj>
  static void PackScalar(const ScalarSobj &S, double *buf) {
    buf[0] = S()()().real();
  }
  template <class ScalarSobj>
  static void UnpackScalar(const double *buf, ScalarSobj &S) {
    using SC = std::remove_reference_t<decltype(S()()())>;
    S()()() = SC(buf[0], 0.0);
  }

  template <class SigSobj, class PiSobj, class DSobj, class NSobj,
            class SSobj, class PSobj>
  static void PackSite(const SigSobj &sigma, const PiSobj &pi,
                       const DSobj &d, const NSobj &n,
                       const SSobj &s, const PSobj &p,
                       double *buf) {
    int off = 0;
    PackCFHermitian(sigma, buf + off); off += kCFHermPacked;
    PackCFHermitian(pi,    buf + off); off += kCFHermPacked;
    PackCFSymmetric(d,     buf + off); off += kCFSymPacked;
    PackCFSymmetric(n,     buf + off); off += kCFSymPacked;
    PackScalar(s,          buf + off); off += kScalarPacked;
    PackScalar(p,          buf + off); off += kScalarPacked;
  }

  template <class SigSobj, class PiSobj, class DSobj, class NSobj,
            class SSobj, class PSobj>
  static void UnpackSite(const double *buf,
                         SigSobj &sigma, PiSobj &pi,
                         DSobj &d, NSobj &n,
                         SSobj &s, PSobj &p) {
    int off = 0;
    UnpackCFHermitian(buf + off, sigma); off += kCFHermPacked;
    UnpackCFHermitian(buf + off, pi);    off += kCFHermPacked;
    UnpackCFSymmetric(buf + off, d);     off += kCFSymPacked;
    UnpackCFSymmetric(buf + off, n);     off += kCFSymPacked;
    UnpackScalar     (buf + off, s);     off += kScalarPacked;
    UnpackScalar     (buf + off, p);     off += kScalarPacked;
  }

 public:
  typedef GaugeStatistics<PeriodicGimplR> GaugeStats;

  DTXQCDCheckpointer(const CheckpointerParameters &Params_) {
    initialize(Params_);
  }

  void initialize(const CheckpointerParameters &Params_) override {
    Params = Params_;
    Params.format = "IEEE64BIG";
  }

  void TrajectoryComplete(int traj, DTXQCDField &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG) override {
    if ((traj % Params.saveInterval) != 0) return;

    std::string config, rng, smr;
    this->build_filenames(traj, Params, config, smr, rng);
    std::string auxfile = aux_filename(traj);

    int precision32 = 1;
    int tworow = 0;
    NerscIO::writeRNGState(sRNG, pRNG, rng);
    NerscIO::writeConfiguration<GaugeStats>(U.U, config, tworow, precision32);

    typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SigSobj;
    typedef typename LatticeDtxqcdPi::vector_object::scalar_object    PiSobj;
    typedef typename LatticeDtxqcdD::vector_object::scalar_object     DSobj;
    typedef typename LatticeDtxqcdN::vector_object::scalar_object     NSobj;
    typedef typename LatticeDtxqcdS::vector_object::scalar_object     SSobj;
    typedef typename LatticeDtxqcdP::vector_object::scalar_object     PSobj;

    std::vector<SigSobj> sig_s;  unvectorizeToLexOrdArray(sig_s, U.sigma);
    std::vector<PiSobj>  pi_s;   unvectorizeToLexOrdArray(pi_s,  U.pi);
    std::vector<DSobj>   d_s;    unvectorizeToLexOrdArray(d_s,   U.d);
    std::vector<NSobj>   n_s;    unvectorizeToLexOrdArray(n_s,   U.n);
    std::vector<SSobj>   s_s;    unvectorizeToLexOrdArray(s_s,   U.s);
    std::vector<PSobj>   p_s;    unvectorizeToLexOrdArray(p_s,   U.p);

    uint64_t nsites = sig_s.size();
    std::vector<double> buf(nsites * kSiteDoubles);
    thread_for(x, nsites, {
      PackSite(sig_s[x], pi_s[x], d_s[x], n_s[x], s_s[x], p_s[x],
               &buf[x * kSiteDoubles]);
    });
    BinaryIO::htobe64_v((void *)buf.data(), buf.size() * sizeof(double));

    GridBase *g = U.Grid();
    uint64_t local_bytes  = nsites * kSiteDoubles * sizeof(double);
    uint64_t global_nsites = g->gSites();
    uint64_t global_bytes  = global_nsites * kSiteDoubles * sizeof(double);
#if defined(GRID_COMMS_MPI3)
    int my_rank = g->ThisRank();
    MPI_File fh;
    int ierr = MPI_File_open(g->communicator,
                             const_cast<char *>(auxfile.c_str()),
                             MPI_MODE_WRONLY | MPI_MODE_CREATE,
                             MPI_INFO_NULL, &fh);
    GRID_ASSERT(ierr == MPI_SUCCESS);
    ierr = MPI_File_set_size(fh, (MPI_Offset)(16 + global_bytes));
    GRID_ASSERT(ierr == MPI_SUCCESS);
    if (my_rank == 0) {
      uint32_t magic = kAuxMagic, version = kAuxVersion;
      uint64_t pad = 0;
      char hdr[16];
      std::memcpy(hdr + 0,  &magic,   sizeof(magic));
      std::memcpy(hdr + 4,  &version, sizeof(version));
      std::memcpy(hdr + 8,  &pad,     sizeof(pad));
      ierr = MPI_File_write_at(fh, 0, hdr, 16, MPI_BYTE, MPI_STATUS_IGNORE);
      GRID_ASSERT(ierr == MPI_SUCCESS);
    }
    MPI_Offset offset = 16 + (MPI_Offset)my_rank * (MPI_Offset)local_bytes;
    ierr = MPI_File_write_at_all(fh, offset, buf.data(),
                                  local_bytes, MPI_BYTE, MPI_STATUS_IGNORE);
    GRID_ASSERT(ierr == MPI_SUCCESS);
    MPI_File_close(&fh);
#else
    if (g->IsBoss()) {
      std::ofstream ofs(auxfile, std::ios::binary | std::ios::trunc);
      uint32_t magic = kAuxMagic, version = kAuxVersion;
      uint64_t pad = 0;
      ofs.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char *>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char *>(&pad), sizeof(pad));
      ofs.write(reinterpret_cast<const char *>(buf.data()), local_bytes);
    }
#endif
    g->Barrier();
    std::cout << GridLogMessage << "DTXQCDCheckpointer: wrote packed daux "
              << auxfile << " (" << (16 + global_bytes) << " bytes, "
              << kSiteDoubles << " doubles/site, gSites=" << global_nsites
              << " across " << g->ProcessorCount() << " ranks)" << std::endl;
  }

  void CheckpointRestore(int traj, DTXQCDField &U, GridSerialRNG &sRNG,
                         GridParallelRNG &pRNG) override {
    std::string config, rng, smr;
    this->build_filenames(traj, Params, config, smr, rng);
    std::string auxfile = aux_filename(traj);
    ReadConfigFiles(U, sRNG, pRNG, config, rng, auxfile);
  }

  static void ReadConfig(DTXQCDField &U, GridSerialRNG &sRNG,
                         GridParallelRNG &pRNG,
                         const std::string &cfg_prefix,
                         const std::string &rng_prefix,
                         int traj) {
    std::string ts = std::to_string(traj);
    ReadConfigFiles(U, sRNG, pRNG,
                    cfg_prefix + "." + ts,
                    rng_prefix + "." + ts,
                    cfg_prefix + "_daux." + ts);
  }

 private:
  static void ReadConfigFiles(DTXQCDField &U, GridSerialRNG &sRNG,
                              GridParallelRNG &pRNG,
                              const std::string &config,
                              const std::string &rng,
                              const std::string &auxfile) {
    FieldMetaData header;
    NerscIO::readRNGState(sRNG, pRNG, header, rng);
    NerscIO::readConfiguration<GaugeStats>(U.U, header, config);

    GridBase *g = U.Grid();
    uint64_t local_nsites = g->lSites();
    uint64_t local_bytes  = local_nsites * kSiteDoubles * sizeof(double);
    uint32_t magic = 0, version = 0;

#if defined(GRID_COMMS_MPI3)
    int my_rank = g->ThisRank();
    MPI_File fh;
    int ierr = MPI_File_open(g->communicator,
                             const_cast<char *>(auxfile.c_str()),
                             MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (ierr != MPI_SUCCESS) {
      std::cout << GridLogError << "DTXQCDCheckpointer: aux sidecar "
                << auxfile << " missing or unreadable." << std::endl;
      abort();
    }
    char hdr[16];
    ierr = MPI_File_read_at_all(fh, 0, hdr, 16, MPI_BYTE, MPI_STATUS_IGNORE);
    GRID_ASSERT(ierr == MPI_SUCCESS);
    std::memcpy(&magic,   hdr + 0, sizeof(magic));
    std::memcpy(&version, hdr + 4, sizeof(version));
#else
    std::ifstream ifs(auxfile, std::ios::binary);
    if (!ifs) {
      std::cout << GridLogError << "DTXQCDCheckpointer: aux sidecar "
                << auxfile << " missing." << std::endl;
      abort();
    }
    ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char *>(&version), sizeof(version));
    uint64_t pad = 0;
    ifs.read(reinterpret_cast<char *>(&pad), sizeof(pad));
#endif

    if (magic != kAuxMagic) {
      std::cout << GridLogError << "DTXQCDCheckpointer: bad aux magic in "
                << auxfile << " (got 0x" << std::hex << magic << std::dec
                << ", expected DTX3 = 0x44545833).  This may be a v1 'DTXA'"
                   " config; v2 does not support reading them."
                << std::endl;
      abort();
    }
    if (version != kAuxVersion) {
      std::cout << GridLogError << "DTXQCDCheckpointer: aux file " << auxfile
                << " has version " << version
                << " but this build expects " << kAuxVersion << std::endl;
      abort();
    }

    std::vector<double> buf(local_nsites * kSiteDoubles);
#if defined(GRID_COMMS_MPI3)
    MPI_Offset offset = 16 + (MPI_Offset)my_rank * (MPI_Offset)local_bytes;
    ierr = MPI_File_read_at_all(fh, offset, buf.data(),
                                 local_bytes, MPI_BYTE, MPI_STATUS_IGNORE);
    GRID_ASSERT(ierr == MPI_SUCCESS);
    MPI_File_close(&fh);
#else
    ifs.read(reinterpret_cast<char *>(buf.data()), local_bytes);
#endif
    BinaryIO::be64toh_v((void *)buf.data(), buf.size() * sizeof(double));

    typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SigSobj;
    typedef typename LatticeDtxqcdPi::vector_object::scalar_object    PiSobj;
    typedef typename LatticeDtxqcdD::vector_object::scalar_object     DSobj;
    typedef typename LatticeDtxqcdN::vector_object::scalar_object     NSobj;
    typedef typename LatticeDtxqcdS::vector_object::scalar_object     SSobj;
    typedef typename LatticeDtxqcdP::vector_object::scalar_object     PSobj;

    std::vector<SigSobj> sig_s(local_nsites);
    std::vector<PiSobj>  pi_s(local_nsites);
    std::vector<DSobj>   d_s(local_nsites);
    std::vector<NSobj>   n_s(local_nsites);
    std::vector<SSobj>   s_s(local_nsites);
    std::vector<PSobj>   p_s(local_nsites);

    thread_for(x, local_nsites, {
      UnpackSite(&buf[x * kSiteDoubles],
                 sig_s[x], pi_s[x], d_s[x], n_s[x], s_s[x], p_s[x]);
    });

    vectorizeFromLexOrdArray(sig_s, U.sigma);
    vectorizeFromLexOrdArray(pi_s,  U.pi);
    vectorizeFromLexOrdArray(d_s,   U.d);
    vectorizeFromLexOrdArray(n_s,   U.n);
    vectorizeFromLexOrdArray(s_s,   U.s);
    vectorizeFromLexOrdArray(p_s,   U.p);

    // Re-apply Hermitian + traceless + real-projection to absorb any
    // minor numerical drift from the disk round trip.
    DTXQCDCompositeImpl::Project(U);

    std::cout << GridLogMessage << "DTXQCDCheckpointer: restored packed daux "
              << auxfile << std::endl;
  }
};

NAMESPACE_END(Grid);
