#pragma once
// Checkpointer for the DTXQCD composite Field (gauge + 5 aux fields).
//
// Writes:
//   <config_prefix>.N         gauge, NERSC binary (interoperable with stock Grid)
//   <config_prefix>_daux.N    packed aux sidecar (DTXQCD-specific layout)
//   <rng_prefix>.N            RNG state (NERSC writer)
//
// Sidecar magic 'DTXA' (0x44545841) co-exists with TXQCD's 'TXQA' so the two
// actions can live in the same checkpoint tree without collision.
//
// Packed layout per site (42 doubles = 336 bytes):
//   sigma:  3 reals (Pauli triplet real parts)
//   pi:     3 reals
//   t:      6 (mu<nu) x 3 = 18 reals
//   d:      Nc^2 = 9 reals (Hermitian: diag re, then upper-tri re,im)
//   n:      Nc^2 = 9 reals
//
// Multi-rank layout follows TXQCDCheckpointer's v3: each rank writes its
// rank-local lex-ordered slice at offset 16 + my_rank * local_bytes via
// MPI-IO collective.

#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>
#include <cstring>
#if defined(GRID_COMMS_MPI3)
#include <mpi.h>
#endif

NAMESPACE_BEGIN(Grid);

class DTXQCDCheckpointer : public BaseHmcCheckpointer<DTXQCDCompositeImpl> {
 private:
  CheckpointerParameters Params;

  static constexpr uint32_t kAuxMagic   = 0x44545841;  // 'DTXA'
  static constexpr uint32_t kAuxVersion = 1;

  static constexpr int kTripletPacked = DtxqcdNTriplet;          // 3
  static constexpr int kSigmaPacked   = kTripletPacked;           // 3
  static constexpr int kPiPacked      = kTripletPacked;           // 3
  static constexpr int kNtPairs       = Nd * (Nd - 1) / 2;       // 6
  static constexpr int kTPacked       = kNtPairs * kTripletPacked; // 18
  static constexpr int kDPacked       = Nc * Nc;                   // 9
  static constexpr int kNPacked       = Nc * Nc;                   // 9
  static constexpr int kSiteDoubles =
      kSigmaPacked + kPiPacked + kTPacked + kDPacked + kNPacked;   // 42

  std::string aux_filename(int traj) const {
    std::ostringstream os;
    os << Params.config_prefix << "_daux." << traj;
    return os.str();
  }

  // Pack a Pauli triplet (iScalar<iScalar<iVector<ctype,K>>>) as K doubles
  // — the real part of each component (imag is held to zero on disk).
  template <int K, class ctype>
  static void PackTriplet(const iScalar<iScalar<iVector<ctype, K>>> &V,
                          double *buf) {
    for (int a = 0; a < K; ++a) buf[a] = V()()(a).real();
  }
  template <int K, class ctype>
  static void UnpackTriplet(const double *buf,
                            iScalar<iScalar<iVector<ctype, K>>> &V) {
    for (int a = 0; a < K; ++a) V()()(a) = ctype(buf[a], 0.0);
  }

  // Pack/unpack a Hermitian matrix — same convention as TXQCDCheckpointer.
  //   [M_00.re, M_11.re, ..., M_01.re, M_01.im, M_02.re, M_02.im, ...]
  template <int N, class ctype>
  static void PackHermitian(const iScalar<iScalar<iMatrix<ctype, N>>> &M,
                            double *buf) {
    int k = 0;
    for (int i = 0; i < N; ++i) buf[k++] = M()()(i, i).real();
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
    for (int i = 0; i < N; ++i) M()()(i, i) = ctype(buf[k++], 0.0);
    for (int i = 0; i < N; ++i)
      for (int j = i + 1; j < N; ++j) {
        double re = buf[k++], im = buf[k++];
        M()()(i, j) = ctype(re, im);
        M()()(j, i) = ctype(re, -im);
      }
  }

  template <class SigSobj, class PiSobj, class TSobj, class DSobj, class NSobj>
  static void PackSite(const SigSobj &sigma, const PiSobj &pi,
                       const TSobj &t, const DSobj &d, const NSobj &n,
                       double *buf) {
    int off = 0;
    PackTriplet<kTripletPacked>(sigma, buf + off);  off += kSigmaPacked;
    PackTriplet<kTripletPacked>(pi,    buf + off);  off += kPiPacked;
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu) {
        iScalar<iScalar<iVector<ComplexD, kTripletPacked>>> triplet;
        for (int a = 0; a < kTripletPacked; ++a) triplet()()(a) = t()(mu, nu)(a);
        PackTriplet<kTripletPacked>(triplet, buf + off);
        off += kTripletPacked;
      }
    PackHermitian<Nc>(d, buf + off);  off += kDPacked;
    PackHermitian<Nc>(n, buf + off);  off += kNPacked;
  }

  template <class SigSobj, class PiSobj, class TSobj, class DSobj, class NSobj>
  static void UnpackSite(const double *buf,
                         SigSobj &sigma, PiSobj &pi, TSobj &t,
                         DSobj &d, NSobj &n) {
    int off = 0;
    UnpackTriplet<kTripletPacked>(buf + off, sigma);  off += kSigmaPacked;
    UnpackTriplet<kTripletPacked>(buf + off, pi);     off += kPiPacked;
    t = Zero();
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu) {
        iScalar<iScalar<iVector<ComplexD, kTripletPacked>>> triplet;
        UnpackTriplet<kTripletPacked>(buf + off, triplet);
        off += kTripletPacked;
        for (int a = 0; a < kTripletPacked; ++a) {
          t()(mu, nu)(a) =  triplet()()(a);
          t()(nu, mu)(a) = -triplet()()(a);
        }
      }
    UnpackHermitian<Nc>(buf + off, d);  off += kDPacked;
    UnpackHermitian<Nc>(buf + off, n);  off += kNPacked;
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
    typedef typename LatticeDtxqcdT::vector_object::scalar_object     TSobj;
    typedef typename LatticeDtxqcdD::vector_object::scalar_object     DSobj;
    typedef typename LatticeDtxqcdN::vector_object::scalar_object     NSobj;

    std::vector<SigSobj> sig_s;  unvectorizeToLexOrdArray(sig_s, U.sigma);
    std::vector<PiSobj>  pi_s;   unvectorizeToLexOrdArray(pi_s,  U.pi);
    std::vector<TSobj>   t_s;    unvectorizeToLexOrdArray(t_s,   U.t);
    std::vector<DSobj>   d_s;    unvectorizeToLexOrdArray(d_s,   U.d);
    std::vector<NSobj>   n_s;    unvectorizeToLexOrdArray(n_s,   U.n);

    uint64_t nsites = sig_s.size();
    std::vector<double> buf(nsites * kSiteDoubles);
    thread_for(x, nsites, {
      PackSite(sig_s[x], pi_s[x], t_s[x], d_s[x], n_s[x],
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
                << auxfile << std::endl;
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
    typedef typename LatticeDtxqcdT::vector_object::scalar_object     TSobj;
    typedef typename LatticeDtxqcdD::vector_object::scalar_object     DSobj;
    typedef typename LatticeDtxqcdN::vector_object::scalar_object     NSobj;

    std::vector<SigSobj> sig_s(local_nsites);
    std::vector<PiSobj>  pi_s(local_nsites);
    std::vector<TSobj>   t_s(local_nsites);
    std::vector<DSobj>   d_s(local_nsites);
    std::vector<NSobj>   n_s(local_nsites);

    thread_for(x, local_nsites, {
      UnpackSite(&buf[x * kSiteDoubles],
                 sig_s[x], pi_s[x], t_s[x], d_s[x], n_s[x]);
    });

    vectorizeFromLexOrdArray(sig_s, U.sigma);
    vectorizeFromLexOrdArray(pi_s,  U.pi);
    vectorizeFromLexOrdArray(t_s,   U.t);
    vectorizeFromLexOrdArray(d_s,   U.d);
    vectorizeFromLexOrdArray(n_s,   U.n);

    std::cout << GridLogMessage << "DTXQCDCheckpointer: restored packed daux "
              << auxfile << std::endl;
  }
};

NAMESPACE_END(Grid);
