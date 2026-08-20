#!/usr/bin/env python3
"""Assemble the rwf_farmA10 production reweighting factors.

Per config: average seed-jobs within each segment (inverse-variance),
sum segments -> cumulative (dlogdet, darg) at eta = 0.2 / 0.4 / 0.6.
Reports per-config w's, the config-to-config spreads (measurement-
corrected intrinsic), and the resulting ESS per eta -- the numbers the
graded baryon/dibaryon estimators consume.  Compare intrinsic spreads
against the 4-config smoke values (0.38 / 0.27 / ~0+phase)."""
import glob
import re
import numpy as np

OUTD = "/lustre2/nplqcd/Grid-DTXQCD/production/rwf_farmA10"
pat = re.compile(r"STOCHPF traj=(\d+) eta=(\S+) dlogdet= (\S+) \+- (\S+) "
                 r"darg= (\S+) \+- (\S+)")

# data[traj][seg] = list of (dld, dlde, da, dae)
data = {}
for f in glob.glob(f"{OUTD}/rwf_tr*_seg*_j*.log"):
    m = re.search(r"tr(\d+)_seg(\d)_j(\d+)", f)
    txt = open(f, errors="ignore").read()
    r = pat.findall(txt)
    if not (m and r):
        continue
    t, s = int(m.group(1)), int(m.group(2))
    _, _, dld, dlde, da, dae = r[-1]
    data.setdefault(t, {}).setdefault(s, []).append(
        (float(dld), float(dlde), float(da), float(dae)))


def ivar(vals):  # inverse-variance combine [(v, e), ...]
    w = np.array([1.0 / e**2 for v, e in vals])
    v = np.array([v for v, e in vals])
    return (w * v).sum() / w.sum(), 1.0 / np.sqrt(w.sum())


etas = {1: 0.2, 2: 0.4, 3: 0.6}
print(f"configs with data: {sorted(data)}")
cum = {}
for t in sorted(data):
    row = {}
    dld_c, var_d, da_c, var_a = 0.0, 0.0, 0.0, 0.0
    for s in (1, 2, 3):
        if s not in data[t]:
            break
        d, de = ivar([(x[0], x[1]) for x in data[t][s]])
        a, ae = ivar([(x[2], x[3]) for x in data[t][s]])
        dld_c += d; var_d += de**2; da_c += a; var_a += ae**2
        row[s] = (dld_c, np.sqrt(var_d), da_c, np.sqrt(var_a))
    cum[t] = row

for s in (1, 2, 3):
    have = [t for t in sorted(cum) if s in cum[t]]
    if len(have) < 2:
        print(f"eta={etas[s]}: {len(have)} configs complete -- waiting")
        continue
    dld = np.array([cum[t][s][0] for t in have])
    ed = np.array([cum[t][s][1] for t in have])
    da = np.array([cum[t][s][2] for t in have])
    ea = np.array([cum[t][s][3] for t in have])
    sR = dld.std(ddof=1); mR = np.sqrt((ed**2).mean())
    sT = da.std(ddof=1); mT = np.sqrt((ea**2).mean())
    iR = np.sqrt(max(sR**2 - mR**2, 0)); iT = np.sqrt(max(sT**2 - mT**2, 0))
    # ESS/n for the normalized weights (lognormal + phase approx)
    ess = np.exp(-(iR**2 + iT**2))
    print(f"eta={etas[s]} (n={len(have)}): "
          f"sigma_R obs {sR:.2f} meas {mR:.2f} intr {iR:.2f} | "
          f"sigma_th obs {sT:.2f} meas {mT:.2f} intr {iT:.2f} | "
          f"intrinsic ESS/cfg ~ {ess:.3f}")
    for t in have:
        print(f"   traj {t}: dlogdet {cum[t][s][0]:+7.2f} ± {cum[t][s][1]:.2f}"
              f"   darg {cum[t][s][2]:+7.2f} ± {cum[t][s][3]:.2f}")
