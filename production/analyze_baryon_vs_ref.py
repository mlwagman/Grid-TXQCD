#!/usr/bin/env python3
"""Baryon SD/Fierz amplitude check: aux baryon x lambda^4  vs  QCD nucleon.

Fierz makes the aux-diquark baryon B = eps.eps d q identical IN EXPECTATION to
the 3-quark nucleon (lambda^2 d == qq), so
    <signPf . C_aux_baryon>/<signPf> . lambda^4  ~=  O(1) . C_nucleon_ref
with NO extra V3 (verified 2026-07-09: single-cfg ratio ref/aux ~2, not ~4096;
the code's 1/V3^2 wall-wall norm is already point-source-comparable).  The O(1)
is the baryon analog of the pion's calibrated x2 doubling factor.

Averages baryon_all16_avg (recomputed from the saved SB_2d with the CORRECTED
bwd APBC alignment) over an ensemble of configs, x lambda^4, and tabulates
alongside the reference nucleon.  lambda>=4 has signPf=+1 so no reweighting;
add it here when measuring lambda<=3.

Usage: analyze_baryon_vs_ref.py <baryon_h5_dir> [--lam 4] [--ntsrc auto]
"""
import glob
import sys
import numpy as np
import h5py

REF = ("/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt/"
       "qcd_chroma_64src_ref/conn_qcd_11100.h5")


def getc(h, n):
    v = np.array(h[n])
    return (v["re"] + 1j * v["im"]) if (v.dtype.names and "re" in v.dtype.names) \
        else v.astype(complex)


def main():
    args = sys.argv[1:]
    d = args[0]
    lam = 4.0
    if "--lam" in args:
        lam = float(args[args.index("--lam") + 1])
    lam4 = lam ** 4
    T = 48
    V3 = 16 ** 3

    files = sorted(glob.glob(d.rstrip("/") + "/baryon_dtxqcd_*.h5"),
                   key=lambda s: int(s.split("_")[-1].split(".")[0]))
    if not files:
        sys.exit(f"no baryon h5 in {d}")

    def fold(SB, backward, tsrc_list, norm):
        align = -1.0 if backward else 1.0
        C = np.zeros(T, complex)
        for dt in range(T):
            for ts in tsrc_list:
                if not backward:
                    tl = (dt + ts) % T
                    sgn = -1.0 if (dt + ts) >= T else 1.0
                else:
                    tl = ((ts - dt) % T + T) % T
                    sgn = -1.0 if (dt - ts) > 0 else 1.0
                C[dt] += sgn * SB[ts][tl]
            C[dt] *= align * norm
        return C

    per_cfg = []
    for f in files:
        with h5py.File(f) as h:
            # infer tsrc_list from which rows of SB are nonzero
            SB0 = getc(h, "baryon_SB_uu_p_up_2d").reshape(T, T)
            tsrc_list = [ts for ts in range(T) if np.any(np.abs(SB0[ts]) > 0)]
            norm = 1.0 / (V3 * V3 * len(tsrc_list))
            comps = []
            # uu only (2026-07-11): pre-fix h5 also carried ll datasets, but
            # those were the grading-odd cross propagator S_lu (ensemble-zero)
            # — and the pre-fix uu was gauge-variant (Elitzur-zero; the sink
            # conjugation bug).  Only h5 produced by the FIXED meas_baryon
            # (gauge-invariance-gated, uu-only schema) carry signal.
            #
            # fwd only (2026-07-11): the P+ BACKWARD tower is the
            # opposite-parity (N*) spectrum, NOT the nucleon — the connected
            # script's fbavg pairs fwd-P+ with bwd-P-, which this measurement
            # does not compute.  Averaging bwd in dilutes the nucleon with
            # parity-partner contamination, so the nucleon estimator is the
            # fwd 4-average {p,n}x{up,dn}.
            for blk in ["uu"]:
                for fl in ["p", "n"]:
                    for sp in ["up", "dn"]:
                        SB = getc(h, f"baryon_SB_{blk}_{fl}_{sp}_2d").reshape(T, T)
                        comps.append(fold(SB, False, tsrc_list, norm))
            per_cfg.append(np.mean(comps, axis=0))
    per_cfg = np.array(per_cfg)            # (ncfg, T)
    ncfg = len(per_cfg)
    aux = per_cfg.mean(axis=0) * lam4
    aux_err = per_cfg.std(axis=0, ddof=1) / np.sqrt(ncfg) * lam4 if ncfg > 1 \
        else np.zeros(T)

    with h5py.File(REF) as h:
        ref = getc(h, "nucleon")

    print(f"aux baryon ensemble: {ncfg} cfg   lambda={lam} (l4={lam4:.0f})   "
          f"dir={d}")
    print(f"{'t':>3} | {'ref nucleon':>12} | {'ref/2':>12} | "
          f"{'aux fwd-4 x l4':>20} | {'(ref/2)/aux':>11}")
    print("-" * 72)
    for t in range(0, 13):
        r = ref[t].real
        a = aux[t].real
        rr = (r / 2.0) / a if abs(a) > 1e-30 else float("nan")
        print(f"{t:3d} | {r:+12.4e} | {r/2:+12.4e} | "
              f"{a:+12.4e}+-{aux_err[t]:.1e} | {rr:+11.2f}")
    print(f"\nref meff:  " +
          " ".join(f"{np.log(abs(ref[t]/ref[t+1])):+.3f}" for t in range(1, 10)))
    print(f"aux |meff|:" +
          " ".join(f"{np.log(abs(aux[t]/aux[t+1])):+.3f}"
                   if abs(aux[t+1]) > 0 else " nan" for t in range(1, 10)))
    print("\nExpect: aux fwd-4 -> ref/2 on the ensemble, possibly x(-1) "
          "(overall source-convention sign; ref is spin-SUMMED Tr[P+ NNbar], "
          "ours spin-AVERAGED).  fwd only: the P+ backward tower is the "
          "parity-partner (N*) channel, excluded from the nucleon average.  "
          "NO V3 (verified on the fixed estimator).")


if __name__ == "__main__":
    main()
