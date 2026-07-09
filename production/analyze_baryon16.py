#!/usr/bin/env python3
"""Validate/inspect the 16-component aux baryon output (baryon_dtxqcd_*.h5).

16 = block{uu,ll} x flavor{p,n} x spin{up,dn} x dir{fwd,bwd}, plus
baryon_all16_avg.  Checks (single config = structural validation, not
statistics):
  - all 16 present, nonzero, comparable magnitude
  - uu vs ll consistency (||uu||==||ll|| expected from the cross-block test;
    per-config the ll is the C=parity-flip partner -> compare magnitudes and
    the effective mass, not the raw values)
  - fwd vs bwd both show a sensible (positive, decaying) baryon signal
  - effective mass m_eff(t) = log(C(t)/C(t+1)) around the plateau
Prints a compact table + the 16-average effective mass.

Usage: analyze_baryon16.py FILE.h5
"""
import sys
import numpy as np
import h5py

BLK = ["uu", "ll"]
FL = ["p", "n"]
SP = ["up", "dn"]
DR = ["fwd", "bwd"]


def meff(c):
    c = np.real(c)
    out = []
    T = len(c)
    for t in range(T - 1):
        a, b = c[t], c[t + 1]
        out.append(np.log(a / b) if (a > 0 and b > 0) else float("nan"))
    return np.array(out)


def main():
    f = sys.argv[1]
    with h5py.File(f, "r") as h:
        def get(name):
            v = np.array(h[name])
            if v.dtype.names and "re" in v.dtype.names:
                return v["re"] + 1j * v["im"]
            return v
        # Grid's Hdf5Writer nests scalars awkwardly; derive T from an array.
        _ = None
        T = len(get("baryon_all16_avg"))
        def scal(nm, default="?"):
            try:
                return float(np.array(h[nm]).ravel()[0])
            except Exception:
                return default
        print(f"file={f}  T={T}  lambda={scal('lambda')}  traj={scal('traj')}")
        print(f"{'component':22} {'|C(2)|':>10} {'|C(6)|':>10} "
              f"{'|C(10)|':>10} {'meff[4:9]':>10}")
        mags = {}
        for blk in BLK:
            for fl in FL:
                for sp in SP:
                    for dr in DR:
                        nm = f"baryon_{blk}_{fl}_{sp}_{dr}"
                        if nm not in h:
                            print(f"{nm:22} MISSING"); continue
                        c = get(nm)
                        mags[nm] = np.abs(c)
                        me = meff(c)
                        mp = np.nanmean(me[4:9]) if len(me) > 9 else float("nan")
                        print(f"{nm:22} {abs(c[2]):10.3e} {abs(c[6]):10.3e} "
                              f"{abs(c[10]):10.3e} {mp:10.4f}")
        avg = get("baryon_all16_avg")
        me = meff(avg)
        print(f"\n{'baryon_all16_avg':22} meff:", end="")
        for t in range(2, min(14, len(me))):
            print(f" {me[t]:.3f}", end="")
        print()

        # uu vs ll magnitude consistency (per cross-block ||uu||==||ll||)
        print("\n-- uu vs ll relative magnitude (expect ~1 from cross-block) --")
        for fl in FL:
            for sp in SP:
                for dr in DR:
                    u = mags.get(f"baryon_uu_{fl}_{sp}_{dr}")
                    l = mags.get(f"baryon_ll_{fl}_{sp}_{dr}")
                    if u is None or l is None:
                        continue
                    rel = np.nanmean((l[2:12] + 1e-30) / (u[2:12] + 1e-30))
                    print(f"  {fl}_{sp}_{dr}: <|ll|/|uu|>[2:12] = {rel:.3f}")

        # fwd vs bwd (should both be sensible; their avg = the fold)
        print("\n-- fwd vs bwd meff plateau (both should be ~equal, positive) --")
        for blk in BLK:
            for fl in FL:
                for sp in SP:
                    cf = get(f"baryon_{blk}_{fl}_{sp}_fwd")
                    cb = get(f"baryon_{blk}_{fl}_{sp}_bwd")
                    mf = np.nanmean(meff(cf)[4:9])
                    mb = np.nanmean(meff(cb)[4:9])
                    print(f"  {blk}_{fl}_{sp}: meff_fwd={mf:.4f} "
                          f"meff_bwd={mb:.4f}")


if __name__ == "__main__":
    main()
