#pragma once
// Checkpointer for the TXQCD composite Field (gauge + 5 auxiliary fields).
//
// Writes:
//   <config_prefix>.N         gauge, NERSC binary (interoperable with stock Grid)
//   <config_prefix>_aux.N     sidecar: sigma, pi, s, p, t concatenated, with
//                             a small header ("TXQCDAUX", uint32 version).
//   <rng_prefix>.N            RNG state (NERSC writer)
//
// CheckpointRestore fails loudly if the aux sidecar is missing -- per the
// design decision to never silently re-heat aux fields on restart.

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>

NAMESPACE_BEGIN(Grid);

class TXQCDCheckpointer : public BaseHmcCheckpointer<TXQCDCompositeImpl> {
 private:
  CheckpointerParameters Params;

  // Fixed on-disk format for aux fields: IEEE64BIG (matches gauge NERSC default).
  static constexpr const char *kAuxFormat = "IEEE64BIG";
  static constexpr uint32_t kAuxMagic = 0x54585141;  // 'TXQA' (TXQcd Aux)
  static constexpr uint32_t kAuxVersion = 1;

  std::string aux_filename(int traj) const {
    std::ostringstream os;
    os << Params.config_prefix << "_aux." << traj;
    return os.str();
  }

  // Write one Lattice to the open-file sidecar at 'offset'. Advances offset
  // by the number of bytes written. Uses BinaryIO::writeLatticeObject with
  // an identity (BinarySimple) munger at IEEE64BIG.
  template <class Lat>
  void write_aux_lattice(Lat &X, const std::string &file, uint64_t &offset) {
    typedef typename Lat::vector_object vobj;
    typedef typename vobj::scalar_object sobj;
    BinarySimpleUnmunger<sobj, sobj> munge;
    uint32_t nersc_csum = 0, scidac_a = 0, scidac_b = 0;
    BinaryIO::writeLatticeObject<vobj, sobj>(X, file, munge, offset,
                                             std::string(kAuxFormat),
                                             nersc_csum, scidac_a, scidac_b);
    offset += X.Grid()->gSites() * sizeof(sobj);
    std::cout << GridLogMessage
              << "TXQCDCheckpointer: aux write checksum " << std::hex
              << nersc_csum << std::dec << std::endl;
  }

  template <class Lat>
  void read_aux_lattice(Lat &X, const std::string &file, uint64_t &offset) {
    typedef typename Lat::vector_object vobj;
    typedef typename vobj::scalar_object sobj;
    BinarySimpleMunger<sobj, sobj> munge;
    uint32_t nersc_csum = 0, scidac_a = 0, scidac_b = 0;
    BinaryIO::readLatticeObject<vobj, sobj>(X, file, munge, offset,
                                            std::string(kAuxFormat),
                                            nersc_csum, scidac_a, scidac_b);
    offset += X.Grid()->gSites() * sizeof(sobj);
  }

  // Small header written by rank 0, read by all.
  void write_aux_header(const std::string &file) {
    std::ofstream ofs(file, std::ios::binary | std::ios::trunc);
    uint32_t magic = kAuxMagic, version = kAuxVersion;
    ofs.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char *>(&version), sizeof(version));
    // Pad to 16 bytes so downstream BinaryIO writes land on an aligned offset.
    uint64_t pad = 0;
    ofs.write(reinterpret_cast<const char *>(&pad), sizeof(pad));
  }

  void read_aux_header(const std::string &file) {
    std::ifstream ifs(file, std::ios::binary);
    if (!ifs) {
      std::cout << GridLogError
                << "TXQCDCheckpointer: aux sidecar " << file
                << " missing. Refusing to silently re-heat aux fields."
                << std::endl;
      abort();
    }
    uint32_t magic = 0, version = 0;
    ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (magic != kAuxMagic) {
      std::cout << GridLogError
                << "TXQCDCheckpointer: bad aux sidecar magic in " << file
                << std::endl;
      abort();
    }
    if (version != kAuxVersion) {
      std::cout << GridLogError
                << "TXQCDCheckpointer: aux sidecar version " << version
                << " != expected " << kAuxVersion << std::endl;
      abort();
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

    // Aux sidecar: header then 5 lattices.
    if (U.Grid()->IsBoss()) {
      write_aux_header(auxfile);
    }
    U.Grid()->Barrier();
    uint64_t offset = 16;
    write_aux_lattice(U.sigma, auxfile, offset);
    write_aux_lattice(U.pi,    auxfile, offset);
    write_aux_lattice(U.s,     auxfile, offset);
    write_aux_lattice(U.p,     auxfile, offset);
    write_aux_lattice(U.t,     auxfile, offset);
    std::cout << GridLogMessage << "TXQCDCheckpointer: wrote aux sidecar "
              << auxfile << " (" << offset << " bytes)" << std::endl;
  }

  void CheckpointRestore(int traj, TXQCDField &U, GridSerialRNG &sRNG,
                         GridParallelRNG &pRNG) override {
    std::string config, rng, smr;
    this->build_filenames(traj, Params, config, smr, rng);
    std::string auxfile = aux_filename(traj);

    this->check_filename(rng);
    this->check_filename(config);
    this->check_filename(auxfile);

    FieldMetaData header;
    NerscIO::readRNGState(sRNG, pRNG, header, rng);
    NerscIO::readConfiguration<GaugeStats>(U.U, header, config);

    read_aux_header(auxfile);
    uint64_t offset = 16;
    read_aux_lattice(U.sigma, auxfile, offset);
    read_aux_lattice(U.pi,    auxfile, offset);
    read_aux_lattice(U.s,     auxfile, offset);
    read_aux_lattice(U.p,     auxfile, offset);
    read_aux_lattice(U.t,     auxfile, offset);
    std::cout << GridLogMessage << "TXQCDCheckpointer: restored aux sidecar "
              << auxfile << std::endl;
  }
};

NAMESPACE_END(Grid);
