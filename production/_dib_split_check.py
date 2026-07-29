#!/usr/bin/env python3
"""Half-split falsification of the N=60/100 dt=1 signal (site class)."""
import glob
import numpy as np
import h5py

P = "/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt/dtxqcd_lam5_dibaryon_aux_sm2"


def getc(v):
    v = np.array(v)
    return (v["re"] + 1j * v["im"]) if (v.dtype.names and "re" in v.dtype.names) \
        else v.astype(complex)


files = sorted(glob.glob(f"{P}/dibaryon_aux_dtxqcd_*.h5"),
               key=lambda s: int(s.split("_")[-1].split(".")[0]))
for sm in (60, 100):
    S = np.array([getc(h5py.File(f)[f"dib_I1_sm{sm}_S_0"]) for f in files])
    X = np.mean(np.roll(S, -1, axis=1) * np.conj(S), axis=1).real
    n = len(X)
    h1, h2 = X[: n // 2], X[n // 2:]

    def sig(a):
        return a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))

    print(f"sm{sm} site dt=1: full {sig(X):+.2f}s | "
          f"1st half {sig(h1):+.2f}s | 2nd half {sig(h2):+.2f}s | "
          f"even cfgs {sig(X[::2]):+.2f}s | odd cfgs {sig(X[1::2]):+.2f}s")
