#!/usr/bin/env python3
"""Per-offset reweighting-factor reconstruction from the TSRC-ratio circuits.

Input: rwf_tsrc_tr<T>_e<K>_j*.log containing TSRCNODE lines
  [.. ] TSRCNODE m=<i> s=<s> w=<wq> Tr[Sinv dS/ds] = <re> +- <ree>  (im <im> +- <ime>)
= d(log det)/ds sampled on 2-pt Gauss-Legendre nodes per unit s-interval over
the full period.  dlogPf/ds = re/2; darg contribution from im (already the
Pf-phase derivative convention of the estimator -- relative shapes are what
matter here).

Reconstruction: cumulative C(s) = sum_{nodes<=s} wq * f;  closure C(T) must
be 0 (periodicity); bridge-correct  \\hat C(s) = C(s) - (s/T) C(T).
Per-offset error: independent node noise + Brownian-bridge propagation
  var(s) = [sum_{<=s} (wq*err)^2] corrected for the subtracted drift.

Output per circuit: closure z-score, per-offset (dlogPf shift, phase shift)
at s = 0,4,...,44 with errors.  With BOTH eta arms: the [0.2,0.4]-segment
per-offset shifts = arm(0.4) - arm(0.2).

Usage: analyze_rwf_tsrc.py <traj> [<traj> ...]
"""
import glob
import re
import sys
import numpy as np

OUTD = "/lustre2/nplqcd/Grid-DTXQCD/production/rwf_farmA10"
NODE = re.compile(r"TSRCNODE m=(\d+) s=([0-9.eE+-]+) w=([0-9.eE+-]+) "
                  r"Tr\[Sinv dS/ds\] = ([0-9.eE+-]+) \+- ([0-9.eE+-]+)\s+"
                  r"\(im ([0-9.eE+-]+) \+- ([0-9.eE+-]+)\)")
T = 48
OFFS = list(range(0, T, 4))


def circuit(path):
    rows = []
    for line in open(path, errors="replace"):
        m = NODE.search(line)
        if m:
            rows.append([float(m.group(k)) for k in (2, 3, 4, 5, 6, 7)])
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    s = np.array([r[0] for r in rows])
    wq = np.array([r[1] for r in rows])
    re_, ree = np.array([r[2] for r in rows]) / 2.0, np.array([r[3] for r in rows]) / 2.0
    im_, ime = np.array([r[4] for r in rows]) / 2.0, np.array([r[5] for r in rows]) / 2.0
    return s, wq, re_, ree, im_, ime


def bridge(s, wq, f, fe):
    C = np.cumsum(wq * f)
    V = np.cumsum((wq * fe) ** 2)
    Ctot, Vtot = C[-1], V[-1]
    out = {}
    for off in OFFS:
        k = np.searchsorted(s, off) - 1
        if k < 0:
            out[off] = (0.0, 0.0)
            continue
        c = C[k] - (off / T) * Ctot
        v = V[k] * (1 - off / T) ** 2 + (Vtot - V[k]) * (off / T) ** 2
        out[off] = (c, np.sqrt(v))
    return out, Ctot, np.sqrt(Vtot)


for traj in sys.argv[1:]:
    arms = {}
    for k, eta in ((1, 0.2), (2, 0.4)):
        fs = sorted(glob.glob(f"{OUTD}/rwf_tsrc_tr{traj}_e{k}_j*.log"))
        if not fs:
            continue
        c = circuit(fs[0])
        if c is None:
            continue
        s, wq, re_, ree, im_, ime = c
        R, CtR, CeR = bridge(s, wq, re_, ree)
        I, CtI, CeI = bridge(s, wq, im_, ime)
        arms[eta] = (R, I)
        print(f"== traj {traj} eta={eta}: {len(s)} nodes; closure "
              f"re {CtR:+.2f}±{CeR:.2f} (z={abs(CtR)/max(CeR,1e-12):.1f})  "
              f"im {CtI:+.2f}±{CeI:.2f} (z={abs(CtI)/max(CeI,1e-12):.1f})")
        print("   s : dlogPf(s)-dlogPf(0)          phase(s)-phase(0)")
        for off in OFFS[::2]:
            print(f"  {off:2d} : {R[off][0]:+7.2f} ± {R[off][1]:5.2f}"
                  f"          {I[off][0]:+7.2f} ± {I[off][1]:5.2f}")
    if 0.2 in arms and 0.4 in arms:
        print(f"== traj {traj} SEGMENT [0.2,0.4] per-offset shifts "
              f"(arm 0.4 - arm 0.2):")
        for off in OFFS[::2]:
            d = arms[0.4][0][off][0] - arms[0.2][0][off][0]
            e = np.hypot(arms[0.4][0][off][1], arms[0.2][0][off][1])
            print(f"  {off:2d} : {d:+7.2f} ± {e:5.2f}")
