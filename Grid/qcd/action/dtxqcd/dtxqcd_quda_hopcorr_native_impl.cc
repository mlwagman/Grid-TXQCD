// Native t-hop correction kernel for the eta-deformed M~ — strong
// definitions.  See the decl header for the contract and
// production/dtxqcd_deform_quda.h for the operator/dagger derivation.
// Pattern: gamma5/pre/post native TUs (trove isolation, FloatNOrder
// accessors, UKQCD<->DR conversion per site).

#include <Grid/Grid.h>
#include <quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_hopcorr_native_decl.h>

namespace Grid {
namespace DtxqcdQudaHopCorrNative {

using AccessorTy = quda::colorspinor::FloatNOrder<double, 4, 3, 2>;
using Cd = quda::complex<double>;

// UKQCD (QUDA non-relativistic) -> DR (Grid gamma conventions) and back —
// identical to the pre/post kernels.
static accelerator_inline void toRel_(const Cd in[12], Cd out[12]) {
  const double inv_sqrt2 = 0.70710678118654752440;
  for (int c = 0; c < 3; ++c) {
    out[0 * 3 + c] = inv_sqrt2 * (-in[1 * 3 + c] - in[3 * 3 + c]);
    out[1 * 3 + c] = inv_sqrt2 * (in[0 * 3 + c] + in[2 * 3 + c]);
    out[2 * 3 + c] = inv_sqrt2 * (-in[1 * 3 + c] + in[3 * 3 + c]);
    out[3 * 3 + c] = inv_sqrt2 * (in[0 * 3 + c] - in[2 * 3 + c]);
  }
}
static accelerator_inline void toNonRelHalf_(const Cd in[12], Cd out[12]) {
  const double inv_sqrt2 = 0.70710678118654752440;
  for (int c = 0; c < 3; ++c) {
    out[0 * 3 + c] = inv_sqrt2 * (in[1 * 3 + c] + in[3 * 3 + c]);
    out[1 * 3 + c] = inv_sqrt2 * (-in[0 * 3 + c] - in[2 * 3 + c]);
    out[2 * 3 + c] = inv_sqrt2 * (in[1 * 3 + c] - in[3 * 3 + c]);
    out[3 * 3 + c] = inv_sqrt2 * (-in[0 * 3 + c] + in[2 * 3 + c]);
  }
}

// (x_cb, parity) <-> coords, QUDA cb convention (x0 fastest, X0 even).
static accelerator_inline void coordsFromCb_(int x_cb, int parity, int X0,
                                             int X1, int X2, int X3, int x[4]) {
  int za = x_cb / (X0 / 2);
  int zb = za / X1;
  x[1] = za - zb * X1;
  x[3] = zb / X2;
  x[2] = zb - x[3] * X2;
  int x0odd = (x[1] + x[2] + x[3] + parity) & 1;
  x[0] = 2 * (x_cb % (X0 / 2)) + x0odd;
}
static accelerator_inline int cbFromCoords_(const int x[4], int X0, int X1,
                                            int X2, int X3) {
  return (x[0] + X0 * (x[1] + X1 * (x[2] + X2 * x[3]))) >> 1;
}

// DR-basis GammaT: (g_t psi)_{0,1,2,3} = psi_{2,3,0,1}.  proj sign s = +1
// applies (1 - g_t) on fwd / (1 + g_t) on bwd; s = -1 flips both (dagger).
static accelerator_inline void projFwd_(const Cd in[12], Cd out[12],
                                        double s) {
  for (int c = 0; c < 3; ++c) {
    out[0 * 3 + c] = in[0 * 3 + c] - s * in[2 * 3 + c];
    out[1 * 3 + c] = in[1 * 3 + c] - s * in[3 * 3 + c];
    out[2 * 3 + c] = in[2 * 3 + c] - s * in[0 * 3 + c];
    out[3 * 3 + c] = in[3 * 3 + c] - s * in[1 * 3 + c];
  }
}
static accelerator_inline void projBwd_(const Cd in[12], Cd out[12],
                                        double s) {
  for (int c = 0; c < 3; ++c) {
    out[0 * 3 + c] = in[0 * 3 + c] + s * in[2 * 3 + c];
    out[1 * 3 + c] = in[1 * 3 + c] + s * in[3 * 3 + c];
    out[2 * 3 + c] = in[2 * 3 + c] + s * in[0 * 3 + c];
    out[3 * 3 + c] = in[3 * 3 + c] + s * in[1 * 3 + c];
  }
}

void UploadHopGauge(const double *w3_host, const double *w7_host,
                    const int X[4], HopGauge &g) {
  FreeHopGauge(g);
  for (int d = 0; d < 4; ++d) g.X[d] = X[d];
  g.vol = (std::size_t)X[0] * X[1] * X[2] * X[3];
  const std::size_t bytes = g.vol * 9 * sizeof(Cd);
  g.w3_d = acceleratorAllocDevice(bytes);
  g.w7_d = acceleratorAllocDevice(bytes);
  acceleratorCopyToDevice((void *)w3_host, g.w3_d, bytes);
  acceleratorCopyToDevice((void *)w7_host, g.w7_d, bytes);
}
void FreeHopGauge(HopGauge &g) {
  if (g.w3_d) { acceleratorFreeDevice(g.w3_d); g.w3_d = nullptr; }
  if (g.w7_d) { acceleratorFreeDevice(g.w7_d); g.w7_d = nullptr; }
}
double *UploadCoeffs(const double *host, int T) {
  double *d = (double *)acceleratorAllocDevice(T * sizeof(double));
  acceleratorCopyToDevice((void *)host, (void *)d, T * sizeof(double));
  return d;
}
void FreeCoeffs(double *d) {
  if (d) acceleratorFreeDevice((void *)d);
}

void ApplyHopCorrNative(quda::ColorSpinorField &in_csf,
                        quda::ColorSpinorField &out_csf, const HopGauge &g,
                        const double *cf_d, const double *cb_d,
                        bool conj_gauge, bool flip_proj) {
  AccessorTy in_acc(in_csf);
  AccessorTy out_acc(out_csf);

  const int volumeCB = in_csf.VolumeCB();
  const int nParity =
      (in_csf.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? 2 : 1;
  const std::size_t N = (std::size_t)nParity * (std::size_t)volumeCB;
  const int X0 = g.X[0], X1 = g.X[1], X2 = g.X[2], X3 = g.X[3];
  const Cd *w3 = (const Cd *)g.w3_d;
  const Cd *w7 = (const Cd *)g.w7_d;
  const double ps = flip_proj ? -1.0 : +1.0;
  const double cg = conj_gauge ? -1.0 : +1.0;  // conj = flip imag of W

  accelerator_for(idx, N, 1, {
    const int parity = idx / volumeCB;
    const int x_cb = idx % volumeCB;
    int x[4];
    coordsFromCb_(x_cb, parity, X0, X1, X2, X3, x);
    const int t = x[3];

    // Neighbors in t (periodic; APBC lives in W3's baked phase).
    int xp[4] = {x[0], x[1], x[2], (t + 1) % X3};
    int xm[4] = {x[0], x[1], x[2], (t - 1 + X3) % X3};
    const int nb_par = 1 - parity;  // odd t-shift flips parity
    const int xp_cb = cbFromCoords_(xp, X0, X1, X2, X3);
    const int xm_cb = cbFromCoords_(xm, X0, X1, X2, X3);

    // Gauge site index (parity-blocked cb order, matching the upload).
    const std::size_t gsite = (std::size_t)parity * (g.vol / 2) + x_cb;
    const Cd *W3s = w3 + gsite * 9;
    const Cd *W7s = w7 + gsite * 9;

    Cd nb_UK[12], nb_DR[12], pr[12], tf[12], tb[12];

    // fwd: cf[t] * (-1/2)(1 -+ g_t) W3 psi(x+that)
    in_acc.load(nb_UK, xp_cb, nb_par);
    toRel_(nb_UK, nb_DR);
    projFwd_(nb_DR, pr, ps);
    for (int s = 0; s < 4; ++s)
      for (int c = 0; c < 3; ++c) {
        Cd acc(0.0, 0.0);
        for (int b = 0; b < 3; ++b) {
          Cd wcb = W3s[c * 3 + b];
          wcb = Cd(wcb.real(), cg * wcb.imag());
          acc += wcb * pr[s * 3 + b];
        }
        tf[s * 3 + c] = acc;
      }

    // bwd: cb[t] * (-1/2)(1 +- g_t) W7 psi(x-that)
    in_acc.load(nb_UK, xm_cb, nb_par);
    toRel_(nb_UK, nb_DR);
    projBwd_(nb_DR, pr, ps);
    for (int s = 0; s < 4; ++s)
      for (int c = 0; c < 3; ++c) {
        Cd acc(0.0, 0.0);
        for (int b = 0; b < 3; ++b) {
          Cd wcb = W7s[c * 3 + b];
          wcb = Cd(wcb.real(), cg * wcb.imag());
          acc += wcb * pr[s * 3 + b];
        }
        tb[s * 3 + c] = acc;
      }

    // out += -1/2 (cf tf + cb tb), converted back to UKQCD.
    Cd add_DR[12], add_UK[12], out_UK[12];
    const double cft = cf_d[t], cbt = cb_d[t];
    for (int i = 0; i < 12; ++i)
      add_DR[i] = Cd(-0.5, 0.0) * (cft * tf[i] + cbt * tb[i]);
    toNonRelHalf_(add_DR, add_UK);
    out_acc.load(out_UK, x_cb, parity);
    for (int i = 0; i < 12; ++i) out_UK[i] += add_UK[i];
    out_acc.save(out_UK, x_cb, parity);
  });
}

void ApplyHopCorrNativeEO(quda::ColorSpinorField &in_csf,
                          quda::ColorSpinorField &out_csf, const HopGauge &g,
                          const double *cf_d, const double *cb_d,
                          bool conj_gauge, bool flip_proj, int out_parity) {
  AccessorTy in_acc(in_csf);
  AccessorTy out_acc(out_csf);

  const int volumeCB = out_csf.VolumeCB();
  const int X0 = g.X[0], X1 = g.X[1], X2 = g.X[2], X3 = g.X[3];
  const Cd *w3 = (const Cd *)g.w3_d;
  const Cd *w7 = (const Cd *)g.w7_d;
  const double ps = flip_proj ? -1.0 : +1.0;
  const double cg = conj_gauge ? -1.0 : +1.0;
  const std::size_t halfvol = g.vol / 2;
  const int opar = out_parity;

  accelerator_for(x_cb, (std::size_t)volumeCB, 1, {
    int x[4];
    coordsFromCb_((int)x_cb, opar, X0, X1, X2, X3, x);
    const int t = x[3];

    int xp[4] = {x[0], x[1], x[2], (t + 1) % X3};
    int xm[4] = {x[0], x[1], x[2], (t - 1 + X3) % X3};
    const int xp_cb = cbFromCoords_(xp, X0, X1, X2, X3);
    const int xm_cb = cbFromCoords_(xm, X0, X1, X2, X3);

    const std::size_t gsite = (std::size_t)opar * halfvol + x_cb;
    const Cd *W3s = w3 + gsite * 9;
    const Cd *W7s = w7 + gsite * 9;

    Cd nb_UK[12], nb_DR[12], pr[12], tf[12], tb[12];

    in_acc.load(nb_UK, xp_cb, 0);  // single-parity CSF: parity index 0
    toRel_(nb_UK, nb_DR);
    projFwd_(nb_DR, pr, ps);
    for (int s = 0; s < 4; ++s)
      for (int c = 0; c < 3; ++c) {
        Cd acc(0.0, 0.0);
        for (int b = 0; b < 3; ++b) {
          Cd wcb = W3s[c * 3 + b];
          wcb = Cd(wcb.real(), cg * wcb.imag());
          acc += wcb * pr[s * 3 + b];
        }
        tf[s * 3 + c] = acc;
      }

    in_acc.load(nb_UK, xm_cb, 0);
    toRel_(nb_UK, nb_DR);
    projBwd_(nb_DR, pr, ps);
    for (int s = 0; s < 4; ++s)
      for (int c = 0; c < 3; ++c) {
        Cd acc(0.0, 0.0);
        for (int b = 0; b < 3; ++b) {
          Cd wcb = W7s[c * 3 + b];
          wcb = Cd(wcb.real(), cg * wcb.imag());
          acc += wcb * pr[s * 3 + b];
        }
        tb[s * 3 + c] = acc;
      }

    Cd add_DR[12], add_UK[12], out_UK[12];
    const double cft = cf_d[t], cbt = cb_d[t];
    for (int i = 0; i < 12; ++i)
      add_DR[i] = Cd(-0.5, 0.0) * (cft * tf[i] + cbt * tb[i]);
    toNonRelHalf_(add_DR, add_UK);
    out_acc.load(out_UK, x_cb, 0);
    for (int i = 0; i < 12; ++i) out_UK[i] += add_UK[i];
    out_acc.save(out_UK, x_cb, 0);
  });
}

}  // namespace DtxqcdQudaHopCorrNative
}  // namespace Grid
