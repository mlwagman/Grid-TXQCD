#pragma once
// Checkpointer for the TXQCD composite Field (gauge + 5 auxiliary fields).
//
// Writes:
//   <config_prefix>.N         gauge, NERSC binary (interoperable with stock Grid)
//   <config_prefix>_aux.N     packed aux sidecar: only independent real DOF of
//                             each Hermitian/antisymmetric field, IEEE64BIG.
//   <rng_prefix>.N            RNG state (NERSC writer)
//
// Packed layout per site (80 doubles = 640 bytes):
//   sigma:  Nf^2 = 4  reals (diagonal, then upper-triangle re,im)
//   pi:     Nf^2 = 4  reals
//   s:      Nc^2 = 9  reals
//   p:      Nc^2 = 9  reals
//   t:  6 × Nc^2 = 54 reals (mu<nu blocks only, each packed as Hermitian)
//
// CheckpointRestore fails loudly if the aux sidecar is missing.
//
// Multi-rank correctness: each rank's `unvectorizeToLexOrdArray` produces its
// rank-LOCAL lex-ordered slice of the lattice (size lSites).  v3 layout writes
// these slices in RANK-MAJOR order via MPI-IO collective writes — every rank
// writes its own offset.  v2 wrote only rank-0's slice (loses 3/4 of aux at
// mpi=1.1.1.4, was silently broken for any multi-rank layout) and is rejected.
// Single-rank (mpi=1.1.1.1) files written before this fix happen to be valid
// v3 layout (rank 0 is the only rank, lSites = gSites), so re-reading them
// against v3 expectations just works once the version bump is in place.

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <cstring>
#if defined(GRID_COMMS_MPI3)
#include <mpi.h>
#endif

NAMESPACE_BEGIN(Grid);

class TXQCDCheckpointer : public BaseHmcCheckpointer<TXQCDCompositeImpl> {
 private:
  CheckpointerParameters Params;

  static constexpr uint32_t kAuxMagic   = 0x54585141;  // 'TXQA'
  // v3 = packed Hermitian, RANK-MAJOR lex-local layout (multi-rank correct).
  // v2 was rank-0-only (broken for multi-rank); we still accept v2 ONLY when
  // running on a single rank (where v2 and v3 layouts are bit-identical).
  static constexpr uint32_t kAuxVersion = 3;

  // Per-site packed sizes (in doubles)
  static constexpr int kSigmaPacked = TxqcdNf * TxqcdNf;   // 4
  static constexpr int kPiPacked    = TxqcdNf * TxqcdNf;   // 4
  static constexpr int kSPacked     = Nc * Nc;              // 9
  static constexpr int kPPacked     = Nc * Nc;              // 9
  static constexpr int kNtPairs     = Nd * (Nd - 1) / 2;   // 6
  static constexpr int kTPacked     = kNtPairs * Nc * Nc;   // 54
  static constexpr int kSiteDoubles =
      kSigmaPacked + kPiPacked + kSPacked + kPPacked + kTPacked;  // 80

  std::string aux_filename(int traj) const {
    std::ostringstream os;
    os << Params.config_prefix << "_aux." << traj;
    return os.str();
  }

  // Pack an N×N Hermitian complex matrix into N^2 doubles:
  //   [M_00.re, M_11.re, ..., M_01.re, M_01.im, M_02.re, M_02.im, ...]
  template <int N, class ctype>
  static void PackHermitian(const iScalar<iScalar<iMatrix<ctype, N>>> &M,
                            double *buf) {
    int k = 0;
    for (int i = 0; i < N; ++i)
      buf[k++] = M()()(i, i).real();
    for (int i = 0; i < N; ++i)
      for (int j = i + 1; j < N; ++j) {
        buf[k++] = M()()(i, j).real();
        buf[k++] = M()()(i, j).imag();
      }
  }

  template <int N, class ctype>
  static void UnpackHermitian(const double *buf,
                              iScalar<iScalar<iMatrix<ctype, N>>> &M) {
    int k = 0;
    for (int i = 0; i < N; ++i)
      M()()(i, i) = ctype(buf[k++], 0.0);
    for (int i = 0; i < N; ++i)
      for (int j = i + 1; j < N; ++j) {
        double re = buf[k++], im = buf[k++];
        M()()(i, j) = ctype(re, im);
        M()()(j, i) = ctype(re, -im);
      }
  }

  // Pack one site's worth of all five aux fields into kSiteDoubles doubles.
  template <class SigSobj, class PiSobj, class SSobj, class PSobj, class TSobj>
  static void PackSite(const SigSobj &sigma, const PiSobj &pi,
                       const SSobj &s, const PSobj &p, const TSobj &t,
                       double *buf) {
    int off = 0;
    PackHermitian<TxqcdNf>(sigma, buf + off);  off += kSigmaPacked;
    PackHermitian<TxqcdNf>(pi,    buf + off);  off += kPiPacked;
    PackHermitian<Nc>(s, buf + off);           off += kSPacked;
    PackHermitian<Nc>(p, buf + off);           off += kPPacked;
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu) {
        iScalar<iScalar<iMatrix<ComplexD, Nc>>> block;
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j)
            block()()(i, j) = t()(mu, nu)(i, j);
        PackHermitian<Nc>(block, buf + off);
        off += Nc * Nc;
      }
  }

  template <class SigSobj, class PiSobj, class SSobj, class PSobj, class TSobj>
  static void UnpackSite(const double *buf,
                         SigSobj &sigma, PiSobj &pi,
                         SSobj &s, PSobj &p, TSobj &t) {
    int off = 0;
    UnpackHermitian<TxqcdNf>(buf + off, sigma);  off += kSigmaPacked;
    UnpackHermitian<TxqcdNf>(buf + off, pi);     off += kPiPacked;
    UnpackHermitian<Nc>(buf + off, s);            off += kSPacked;
    UnpackHermitian<Nc>(buf + off, p);            off += kPPacked;
    // Zero full t, then fill upper triangle and antisymmetrize
    t = Zero();
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu) {
        iScalar<iScalar<iMatrix<ComplexD, Nc>>> block;
        UnpackHermitian<Nc>(buf + off, block);
        off += Nc * Nc;
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j) {
            t()(mu, nu)(i, j) = block()()(i, j);
            t()(nu, mu)(i, j) = -block()()(i, j);
          }
      }
  }

 public:
  typedef GaugeStatistics<PeriodicGimplR> GaugeStats;

  TXQCDCheckpointer(const CheckpointerParameters &Params_) {
    initialize(Params_);
  }

  void initialize(const CheckpointerParameters &Params_) override {
    Params = Params_;
    Params.format = "IEEE64BIG";
  }

  void TrajectoryComplete(int traj, TXQCDField &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG) override {
    if ((traj % Params.saveInterval) != 0) return;

    std::string config, rng, smr;
    this->build_filenames(traj, Params, config, smr, rng);
    std::string auxfile = aux_filename(traj);

    int precision32 = 1;
    int tworow = 0;
    NerscIO::writeRNGState(sRNG, pRNG, rng);
    NerscIO::writeConfiguration<GaugeStats>(U.U, config, tworow, precision32);

    // Unvectorize all aux fields to scalar site arrays
    typedef typename LatticeSigmaField::vector_object::scalar_object SigSobj;
    typedef typename LatticePiField::vector_object::scalar_object    PiSobj;
    typedef typename LatticeSFieldC::vector_object::scalar_object    SSobj;
    typedef typename LatticePFieldC::vector_object::scalar_object    PSobj;
    typedef typename LatticeTField::vector_object::scalar_object     TSobj;

    std::vector<SigSobj> sig_s;  unvectorizeToLexOrdArray(sig_s, U.sigma);
    std::vector<PiSobj>  pi_s;   unvectorizeToLexOrdArray(pi_s,  U.pi);
    std::vector<SSobj>   s_s;    unvectorizeToLexOrdArray(s_s,   U.s);
    std::vector<PSobj>   p_s;    unvectorizeToLexOrdArray(p_s,   U.p);
    std::vector<TSobj>   t_s;    unvectorizeToLexOrdArray(t_s,   U.t);

    uint64_t nsites = sig_s.size();
    std::vector<double> buf(nsites * kSiteDoubles);

    thread_for(x, nsites, {
      PackSite(sig_s[x], pi_s[x], s_s[x], p_s[x], t_s[x],
               &buf[x * kSiteDoubles]);
    });

    // Byte-swap to big-endian
    BinaryIO::htobe64_v((void *)buf.data(), buf.size() * sizeof(double));

    // Multi-rank-aware write: each rank writes its own lSites slice at its
    // rank-major offset using MPI-IO collective.  Rank 0 also writes the
    // 16-byte header.  File total size = 16 + gSites * kSiteDoubles * 8.
    GridBase *g = U.Grid();
    uint64_t local_bytes = nsites * kSiteDoubles * sizeof(double);
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
    // Truncate to exact expected size in case the file existed at a larger
    // length from a previous run (paranoia — saveInterval normally prevents
    // re-writing the same traj, but we don't want stale bytes if it does).
    ierr = MPI_File_set_size(fh, (MPI_Offset)(16 + global_bytes));
    GRID_ASSERT(ierr == MPI_SUCCESS);
    // Rank 0 writes the 16-byte header at offset 0.
    if (my_rank == 0) {
      uint32_t magic = kAuxMagic, version = kAuxVersion;
      uint64_t pad   = 0;
      char hdr[16];
      std::memcpy(hdr + 0,  &magic,   sizeof(magic));
      std::memcpy(hdr + 4,  &version, sizeof(version));
      std::memcpy(hdr + 8,  &pad,     sizeof(pad));
      ierr = MPI_File_write_at(fh, 0, hdr, 16, MPI_BYTE, MPI_STATUS_IGNORE);
      GRID_ASSERT(ierr == MPI_SUCCESS);
    }
    // Each rank writes its slice at offset 16 + my_rank * local_bytes.
    // local_bytes is the same for every rank (lSites is uniform in Grid),
    // so rank-major offsetting is trivial.
    MPI_Offset offset = 16 + (MPI_Offset)my_rank * (MPI_Offset)local_bytes;
    ierr = MPI_File_write_at_all(fh, offset, buf.data(),
                                  local_bytes, MPI_BYTE, MPI_STATUS_IGNORE);
    GRID_ASSERT(ierr == MPI_SUCCESS);
    MPI_File_close(&fh);
#else
    // No-MPI build: identical to old single-rank path.
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

    std::cout << GridLogMessage << "TXQCDCheckpointer: wrote packed aux "
              << auxfile << " (" << (16 + global_bytes)
              << " bytes, " << kSiteDoubles << " doubles/site, gSites="
              << global_nsites << " across " << g->ProcessorCount()
              << " ranks)" << std::endl;
  }

  void CheckpointRestore(int traj, TXQCDField &U, GridSerialRNG &sRNG,
                         GridParallelRNG &pRNG) override {
    std::string config, rng, smr;
    this->build_filenames(traj, Params, config, smr, rng);
    std::string auxfile = aux_filename(traj);
    ReadConfigFiles(U, sRNG, pRNG, config, rng, auxfile);
  }

  static void ReadConfig(TXQCDField &U, GridSerialRNG &sRNG,
                         GridParallelRNG &pRNG,
                         const std::string &cfg_prefix,
                         const std::string &rng_prefix,
                         int traj) {
    std::string ts = std::to_string(traj);
    ReadConfigFiles(U, sRNG, pRNG,
                    cfg_prefix + "." + ts,
                    rng_prefix + "." + ts,
                    cfg_prefix + "_aux." + ts);
  }

 private:
  static void ReadConfigFiles(TXQCDField &U, GridSerialRNG &sRNG,
                              GridParallelRNG &pRNG,
                              const std::string &config,
                              const std::string &rng,
                              const std::string &auxfile) {
    FieldMetaData header;
    NerscIO::readRNGState(sRNG, pRNG, header, rng);
    NerscIO::readConfiguration<GaugeStats>(U.U, header, config);

    GridBase *g = U.Grid();
    uint64_t local_nsites  = g->lSites();
    uint64_t local_bytes   = local_nsites * kSiteDoubles * sizeof(double);
    uint32_t magic = 0, version = 0;

#if defined(GRID_COMMS_MPI3)
    int my_rank = g->ThisRank();
    int nranks  = g->ProcessorCount();
    MPI_File fh;
    int ierr = MPI_File_open(g->communicator,
                             const_cast<char *>(auxfile.c_str()),
                             MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (ierr != MPI_SUCCESS) {
      std::cout << GridLogError
                << "TXQCDCheckpointer: aux sidecar " << auxfile
                << " missing or unreadable." << std::endl;
      abort();
    }
    // Header: all ranks read 16-byte prefix (cheap, avoids extra broadcast).
    char hdr[16];
    ierr = MPI_File_read_at_all(fh, 0, hdr, 16, MPI_BYTE, MPI_STATUS_IGNORE);
    GRID_ASSERT(ierr == MPI_SUCCESS);
    std::memcpy(&magic,   hdr + 0, sizeof(magic));
    std::memcpy(&version, hdr + 4, sizeof(version));
#else
    std::ifstream ifs(auxfile, std::ios::binary);
    if (!ifs) {
      std::cout << GridLogError
                << "TXQCDCheckpointer: aux sidecar " << auxfile
                << " missing." << std::endl;
      abort();
    }
    ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char *>(&version), sizeof(version));
    uint64_t pad = 0;
    ifs.read(reinterpret_cast<char *>(&pad), sizeof(pad));
#endif

    if (magic != kAuxMagic) {
      std::cout << GridLogError << "TXQCDCheckpointer: bad aux magic in "
                << auxfile << std::endl;
      abort();
    }
    // Accept v2 ONLY when running single-rank (v2 layout = rank-0-only, which
    // coincides with v3 rank-major layout when nranks == 1).  Reject otherwise.
    if (version != kAuxVersion) {
#if defined(GRID_COMMS_MPI3)
      bool single_rank = (g->ProcessorCount() == 1);
#else
      bool single_rank = true;
#endif
      if (!(version == 2 && single_rank)) {
        std::cout << GridLogError
                  << "TXQCDCheckpointer: aux file " << auxfile
                  << " has version " << version
                  << " but multi-rank read requires version " << kAuxVersion
                  << " — old v2 files written by mpi != 1.1.1.1 are corrupt"
                  << " (rank-0-only) and must be regenerated." << std::endl;
        abort();
      }
      std::cout << GridLogMessage
                << "TXQCDCheckpointer: accepting legacy v2 aux file (single-rank)"
                << std::endl;
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

    typedef typename LatticeSigmaField::vector_object::scalar_object SigSobj;
    typedef typename LatticePiField::vector_object::scalar_object    PiSobj;
    typedef typename LatticeSFieldC::vector_object::scalar_object    SSobj;
    typedef typename LatticePFieldC::vector_object::scalar_object    PSobj;
    typedef typename LatticeTField::vector_object::scalar_object     TSobj;

    std::vector<SigSobj> sig_s(local_nsites);
    std::vector<PiSobj>  pi_s(local_nsites);
    std::vector<SSobj>   s_s(local_nsites);
    std::vector<PSobj>   p_s(local_nsites);
    std::vector<TSobj>   t_s(local_nsites);

    thread_for(x, local_nsites, {
      UnpackSite(&buf[x * kSiteDoubles],
                 sig_s[x], pi_s[x], s_s[x], p_s[x], t_s[x]);
    });

    vectorizeFromLexOrdArray(sig_s, U.sigma);
    vectorizeFromLexOrdArray(pi_s,  U.pi);
    vectorizeFromLexOrdArray(s_s,   U.s);
    vectorizeFromLexOrdArray(p_s,   U.p);
    vectorizeFromLexOrdArray(t_s,   U.t);

    std::cout << GridLogMessage << "TXQCDCheckpointer: restored packed aux "
              << auxfile << std::endl;
  }
};

NAMESPACE_END(Grid);
