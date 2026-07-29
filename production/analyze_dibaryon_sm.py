#!/usr/bin/env python3
"""Smeared-operator dibaryon scan: significance per (smear level, class, dt)
plus per-level class Hotelling p and split-half matched filter.
Usage: analyze_dibaryon_sm.py [h5dir] [--fam I1|I0]"""
import glob
import re
import sys
import numpy as np
import h5py

d5 = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else \
    "/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt/dtxqcd_lam5_dibaryon_aux_sm"
FAM = sys.argv[sys.argv.index("--fam") + 1] if "--fam" in sys.argv else "I1"
LAM6 = 5.0 ** 6
rng = np.random.default_rng(31)

try:
    from scipy.stats import f as fdist
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def getc(v):
    v = np.array(v)
    return (v["re"] + 1j * v["im"]) if (v.dtype.names and "re" in v.dtype.names) \
        else v.astype(complex)


def classify(r1, r3):
    n1, n3 = np.abs(r1).sum(), np.abs(r3).sum()
    L = max(n1, n3)
    if n1 == 0 and n3 == 0:
        return ("site", 0)
    if n1 == 0 or n3 == 0:
        return ("disp1", L)
    if np.all(r1 == -r3):
        return ("collinear", L)
    if np.all(r1 == r3):
        return ("sameside", L)
    return ("planar", L)


files = sorted(glob.glob(f"{d5}/dibaryon_aux_dtxqcd_*.h5"),
               key=lambda s: int(s.split("_")[-1].split(".")[0]))
with h5py.File(files[0]) as h:
    sms = sorted({int(m.group(1)) for k in h.keys()
                  for m in [re.match(rf"dib_{FAM}_sm(\d+)_S_\d+", k)] if m})
    ng = sum(1 for k in h.keys() if re.match(rf"dib_{FAM}_sm{sms[0]}_S_", k))
    g = np.array(h[f"geo_{FAM}"]).reshape(2, ng, 3)
    meta = [(g[0, i], g[1, i]) for i in range(ng)]
S = {sm: [] for sm in sms}
for f in files:
    with h5py.File(f) as h:
        for sm in sms:
            S[sm].append([getc(h[f"dib_{FAM}_sm{sm}_S_{i}"]) for i in range(ng)])
S = {sm: np.array(v) for sm, v in S.items()}
N = len(files)
classes = {}
for i, (r1, r3) in enumerate(meta):
    classes.setdefault(classify(np.array(r1), np.array(r3)), []).append(i)
cidx = sorted(classes, key=lambda k: (k[1], k[0]))
NC = len(cidx)
print(f"{FAM}: {N} configs, {ng} geometries, {NC} classes, smears {sms}")

for sm in sms:
    print(f"\n===== smear N={sm} =====")
    print(f"{'class':>10} {'L':>2} | " +
          " | ".join(f"{'dt=%d' % dt:>18}" for dt in (1, 2, 3, 4)))
    best = (0.0, None)
    for (cls, L) in cidx:
        idx = classes[(cls, L)]
        row = f"{cls:>10} {L:>2} |"
        for dt in (1, 2, 3, 4):
            X = np.mean(np.roll(S[sm], -dt, axis=2) * np.conj(S[sm]),
                        axis=2).real[:, idx].mean(axis=1) * LAM6
            m = X.mean()
            jk = np.array([np.delete(X, i).mean() for i in range(N)])
            e = np.sqrt((N - 1) / N * np.sum((jk - jk.mean()) ** 2))
            sig = abs(m) / e if e > 0 else 0
            if dt == 1 and sig > best[0]:
                best = (sig, (cls, L))
            row += f" {m:+9.2e}({sig:4.1f}s) |"
        print(row)
    # class-level Hotelling per dt
    hps = []
    for dt in (1, 2, 3, 4):
        X = np.mean(np.roll(S[sm], -dt, axis=2) * np.conj(S[sm]), axis=2).real
        Y = np.stack([X[:, classes[c]].mean(axis=1) for c in cidx], axis=1)
        yb = Y.mean(axis=0)
        Sg = np.cov(Y.T)
        T2 = N * yb @ np.linalg.solve(Sg, yb)
        F = (N - NC) / (NC * (N - 1)) * T2
        hps.append(1 - fdist.cdf(F, NC, N - NC) if HAVE_SCIPY else float("nan"))
    print(f"  Hotelling p (dt=1..4): " +
          "  ".join(f"{p:.3f}" for p in hps) +
          f"   best dt=1 class: {best[1]} at {best[0]:.1f} sigma")
