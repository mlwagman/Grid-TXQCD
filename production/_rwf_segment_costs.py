#!/usr/bin/env python3
"""Per-segment reweighting-factor cost table from the tent smoke logs:
runtime (last timestamp), N_NOISE, achieved dlogdet error per config, and
the implied cost to reach sigma_meas targets (error ~ 1/sqrt(N)).
Anchors + tsrc halves where present."""
import glob
import re
import numpy as np

LOGD = "/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs"
SEGS = ["seg00_02", "seg02_04", "seg04_06", "seg06_07", "seg07_08"]

def harvest(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        txt = open(f, errors="ignore").read()
        nn = re.search(r"N_NOISE=(\d+)", txt)
        res = re.findall(r"Message : ([0-9.]+) s : STOCHPF traj=(\d+) eta=(\S+) "
                         r"dlogdet= (\S+) \+- (\S+) darg= (\S+) \+- (\S+)", txt)
        if not (nn and res):
            continue
        t, traj, eta, dld, dlde, da, dae = res[-1]
        out.append(dict(f=f.split("/")[-1], nn=int(nn.group(1)),
                        wall=float(t), traj=int(traj),
                        dld=float(dld), dlde=float(dlde),
                        da=float(da), dae=float(dae)))
    return out

for seg in SEGS:
    # prefer highest-stat tent variant available per segment
    for tag in ("_tent_hs2", "_tent_hs1", "_tent"):
        rows = harvest(f"{LOGD}/rwf_smoke_{seg}{tag}_tr*_anchor.log")
        if rows:
            break
    if not rows:
        print(f"{seg}: no tent anchor logs")
        continue
    nn = rows[0]["nn"]
    wall = np.mean([r["wall"] for r in rows]) / 3600.0
    edld = np.mean([r["dlde"] for r in rows])
    edarg = np.mean([r["dae"] for r in rows])
    dlds = [r["dld"] for r in rows]
    spread = np.std(dlds, ddof=1) if len(dlds) > 1 else float("nan")
    spread_i = np.sqrt(max(spread**2 - edld**2, 0)) if len(dlds) > 1 else float("nan")
    # cost per solve and cost to reach sigma targets
    per_solve = wall * 3600.0 / (2 * nn)          # N_GL=2 nodes
    print(f"{seg}{tag}: n={len(rows)} cfgs  N_NOISE={nn}  wall={wall:.2f} h"
          f"  ({per_solve:.0f} s/solve)")
    print(f"   err(dlogdet)={edld:.3f}  err(darg)={edarg:.3f}"
          f"  cfg-spread={spread:.2f} (intrinsic~{spread_i:.2f})")
    for tgt in (0.5, 0.3):
        scale = (edld / tgt) ** 2
        print(f"   -> sigma_meas={tgt}: {wall*scale:6.2f} GPU-h/config"
              f"  (N_NOISE~{int(nn*scale)})")
