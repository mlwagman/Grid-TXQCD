// Read NERSC (our ckpoint_lat.*) or ILDG/SciDAC LIME (chroma) gauge configs
// and compute plaquette.  Auto-detects format from the file's magic bytes.
//
// Usage:
//   ./compute_plaq <cfg>...  --grid L.L.L.T --mpi mx.my.mz.mt
//
// NERSC files start with the text "BEGIN_HEADER"; LIME files start with the
// 4-byte magic 0x45 0x67 0x89 0xAB.

#include <Grid/Grid.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <cstdio>
#include <cstring>

using namespace Grid;

enum class CfgFormat { NERSC, LIME_ILDG, UNKNOWN };

static CfgFormat detect_format(const std::string &path) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) return CfgFormat::UNKNOWN;
  unsigned char buf[16] = {0};
  std::fread(buf, 1, sizeof(buf), f);
  std::fclose(f);
  if (std::memcmp(buf, "BEGIN_HEADER", 12) == 0) return CfgFormat::NERSC;
  if (buf[0] == 0x45 && buf[1] == 0x67 && buf[2] == 0x89 && buf[3] == 0xAB)
    return CfgFormat::LIME_ILDG;
  return CfgFormat::UNKNOWN;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);

  // Collect filenames from CLI (skip --grid / --mpi and their values).
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--", 0) == 0) { ++i; continue; }
    files.push_back(a);
  }
  if (files.empty()) {
    std::cerr << "Usage: compute_plaq <cfg>... --grid L.L.L.T --mpi mx.my.mz.mt\n";
    Grid_finalize();
    return 1;
  }

  LatticeGaugeField Umu(&Grid);
  typedef GaugeStatistics<PeriodicGimplR> GS;
  for (const auto &f : files) {
    CfgFormat fmt = detect_format(f);
    FieldMetaData header;

    if (fmt == CfgFormat::NERSC) {
      NerscIO::readConfiguration<GS>(Umu, header, f);
    } else if (fmt == CfgFormat::LIME_ILDG) {
      IldgReader reader;
      reader.open(f);
      reader.readConfiguration(Umu, header);
      reader.close();
    } else {
      std::cerr << "compute_plaq: unknown format for " << f << std::endl;
      continue;
    }

    RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);
    RealD link = WilsonLoops<PeriodicGimplR>::linkTrace(Umu);
    std::cout << GridLogMessage
              << f << "\tplaq = " << std::setprecision(10) << plaq
              << "\tlink_trace = " << link << std::endl;
  }

  Grid_finalize();
  return 0;
}
