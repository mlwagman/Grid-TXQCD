#!/usr/bin/env python3
"""FIRST PRODUCTION graded-estimator pass: pure-aux dibaryon with the
layer-1 reweighting anchors (t0=0 tent, farm rwf_farmA10).

Estimator (trough at the d*^3 end, fwd/bwd averaged, t0=0 only):
  X_c(dt)  = Re[(S_c(+dt)+S_c(-dt))/2 conj(S_c(0))]
  R(dt)    = Re< e^{-3 eta p(dt)} X_c w_c > / Re<w_c>,   w_c = e^{dlogdet_c + i darg_c}
R estimates C(dt) directly; per-sample spread is graded by e^{-3 eta p(dt)}.
Reports per eta: R(dt) +- jackknife err, the unreweighted same-subset
values, the 175-config all-TSRC reference, and error-contraction ratios."""
import glob
import re
import numpy as np
import h5py

OUTD = "/lustre2/nplqcd/Grid-DTXQCD/production/rwf_farmA10"
DIB = ("/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt/"
       "dtxqcd_lam5_dibaryon_aux_sm2")
pat = re.compile(r"STOCHPF traj=(\d+) eta=(\S+) dlogdet= (\S+) \+- (\S+) "
                 r"darg= (\S+) \+- (\S+)")
T, ETAS = 48, {1: 0.2, 2: 0.4, 3: 0.6}
p = np.array([min(t, T - t) for t in range(T)], float)

# ---- anchors: seed-combine per (traj, seg), then cumulative sum ----------
raw = {}
for f in glob.glob(f"{OUTD}/rwf_tr*_seg*_j*.log"):
    m = re.search(r"tr(\d+)_seg(\d)_j(\d+)", f)
    r = pat.findall(open(f, errors="ignore").read())
    if m and r:
        _, _, dld, dlde, da, dae = r[-1]
        raw.setdefault((int(m.group(1)), int(m.group(2))), []).append(
            (float(dld), float(dlde), float(da), float(dae)))


def ivar(vals, k):
    w = np.array([1.0 / v[k + 1] ** 2 for v in vals])
    x = np.array([v[k] for v in vals])
    return (w * x).sum() / w.sum()


W = {}   # W[eta_index][traj] = complex log-weight (cumulative)
for (t, s), vals in sorted(raw.items()):
    pass
for ei in (1, 2, 3):
    W[ei] = {}
    trajs = sorted({t for (t, s) in raw})
    for t in trajs:
        if not all((t, s) in raw for s in range(1, ei + 1)):
            continue
        dld = sum(ivar(raw[(t, s)], 0) for s in range(1, ei + 1))
        da = sum(ivar(raw[(t, s)], 2) for s in range(1, ei + 1))
        W[ei][t] = complex(dld, da)

# ---- dibaryon slice sums --------------------------------------------------


def getc(v):
    v = np.array(v)
    return (v["re"] + 1j * v["im"]) if (v.dtype.names and "re" in v.dtype.names) \
        else v.astype(complex)


files = sorted(glob.glob(f"{DIB}/dibaryon_aux_dtxqcd_*.h5"),
               key=lambda s: int(s.split("_")[-1].split(".")[0]))
alltr = [int(f.split("_")[-1].split(".")[0]) for f in files]
S = {t: getc(h5py.File(f)["dib_I1_sm60_S_0"]) for t, f in zip(alltr, files)}

# reference: all configs, all TSRC
Sall = np.array([S[t] for t in alltr])
print(f"reference: {len(alltr)} configs all-TSRC")
ref = {}
for dt in (1, 2, 3):
    X = np.mean(0.5 * (np.roll(Sall, -dt, axis=1) + np.roll(Sall, dt, axis=1)) *
                np.conj(Sall), axis=1).real
    ref[dt] = (X.mean(), X.std(ddof=1) / np.sqrt(len(X)))
    print(f"  dt={dt}: C = {ref[dt][0]:+9.2e} ± {ref[dt][1]:8.2e}")


def jackR(num, den):
    n = len(num)
    full = num.sum().real / den.sum().real
    sub = np.array([(num.sum() - num[i]).real / (den.sum() - den[i]).real
                    for i in range(n)])
    return full, np.sqrt((n - 1) / n * ((sub - full) ** 2).sum())


for ei, eta in ETAS.items():
    trs = sorted(set(W[ei]) & set(S))
    if len(trs) < 3:
        print(f"eta={eta}: only {len(trs)} configs -- skip")
        continue
    lw = np.array([W[ei][t] for t in trs])
    lw -= complex(lw.real.mean(), 0.0)          # common-scale normalization
    w = np.exp(lw)
    print(f"\neta={eta} (n={len(trs)}, trajs {trs}):  |w| spread "
          f"{np.std(np.abs(w), ddof=1)/np.mean(np.abs(w)):.2f}, "
          f"phase spread {np.std(lw.imag, ddof=1):.2f} rad")
    for dt in (1, 2, 3):
        g = np.exp(-3 * eta * p[dt])
        # ALL same-slope pairs (mwagman 2026-09-04): one fixed contour grades
        # every pair (a,b) with |p(a)-p(b)| = dt at the full rate; orient the
        # CONJUGATED end at the lower-p point.  ~2*(T/2-dt) pairs per config
        # vs the 2 directions of the old t0=0-source-only estimator.
        pairs = [(a, b) for a in range(T) for b in range(T)
                 if abs(p[a] - p[b]) == dt and p[a] > p[b]]
        Xp = np.array([np.mean([(S[t][a] * np.conj(S[t][b])).real
                                for a, b in pairs]) for t in trs])
        # unreweighted, same subset, t0=0 only
        mu, eu = Xp.mean(), Xp.std(ddof=1) / np.sqrt(len(Xp))
        # graded + reweighted: R estimates C(dt) DIRECTLY (Cauchy identity)
        R, eR = jackR(g * Xp * w, w)
        print(f"  dt={dt}: unrew {mu:+9.2e} ± {eu:8.2e}   "
              f"graded+w R = {R:+9.2e} ± {eR:8.2e}   err ratio {eR/eu:6.3f}"
              f" (grading alone {g:6.3f})")
