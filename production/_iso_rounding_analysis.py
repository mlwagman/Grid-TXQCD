#!/usr/bin/env python3
"""Rounding-ladder verdict: per-profile config spread of the O(eta) phase
coefficient (im column), measurement-corrected, on both lattices; plus the
within-config sharp-vs-rounded structure (shared-solve correlations)."""
import re
import numpy as np

LOG = ("/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/"
       "iso_rounding.1308616.out")
pat = re.compile(
    r"LATT=(\S+) TRAJ=(\d+) .*ISO-LINEAR (\S+)\s+c = (\S+) \+- (\S+)"
    r"\s+im (\S+) \+- (\S+)")
data = {}
for line in open(LOG, errors="ignore"):
    m = pat.search(line)
    if m:
        latt, traj, prof = m.group(1), int(m.group(2)), m.group(3)
        data.setdefault((latt, prof), []).append(
            (traj, float(m.group(6)), float(m.group(7))))

for latt in ("4.4.4.4", "8.8.8.16"):
    print(f"== {latt} ==")
    base = None
    for prof in ("tent_sharp", "round_1mpi", "round_2mpi", "round_3mpi"):
        rows = sorted(data.get((latt, prof), []))
        if not rows:
            continue
        im = np.array([r[1] for r in rows])
        er = np.array([r[2] for r in rows])
        n = len(im)
        obs = im.std(ddof=1)
        meas = np.sqrt((er ** 2).mean())
        intr2 = obs ** 2 - meas ** 2
        intr = np.sqrt(max(intr2, 0.0))
        line = (f"  {prof:11s} n={n:2d}  obs_std={obs:8.2f}  meas={meas:8.2f}"
                f"  intrinsic={intr:8.2f}")
        if prof == "tent_sharp":
            base = im
        else:
            r = np.corrcoef(base, im)[0, 1]
            d = im - base
            line += f"  corr(sharp)={r:+.3f}  <|round|-|sharp|>={np.mean(np.abs(im)-np.abs(base)):+8.2f}"
        print(line)
