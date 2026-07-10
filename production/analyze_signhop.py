#!/usr/bin/env python3
"""<signPf>(lambda) from the 4^4 sign-record chains ([SIGNHOP] lines).

Parses per-traj lines
  [SIGNHOP] traj=N signPf=+-1 n_neg=... logabsdet=... min_abs_eval=...
from slurm logs, discards the first NTHERM trajectories, and reports per log:
  <signPf> with binned (blocked) errors, flip rate, min|eval| stats.

Usage: analyze_signhop.py [--ntherm 50] LOG [LOG ...]
"""
import re
import sys
import numpy as np

LINE = re.compile(r"\[SIGNHOP\] traj=(\d+) signPf=([+-]1) n_neg=\d+ "
                  r"logabsdet=[0-9.eE+-]+ min_abs_eval=([0-9.eE+-]+)")
LAM = re.compile(r"(?:lam(?:bda)?|_l)[=_]?([0-9p]+(?:\.[0-9]+)?)")


def analyze(path, ntherm):
    trajs, signs, mev = [], [], []
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if m:
            trajs.append(int(m.group(1)))
            signs.append(int(m.group(2)))
            mev.append(float(m.group(3)))
    if not trajs:
        return None
    t = np.array(trajs); s = np.array(signs, dtype=float); e = np.array(mev)
    keep = t > ntherm
    s, e = s[keep], e[keep]
    n = len(s)
    if n == 0:
        return dict(n=0, ntot=len(t), tmax=t.max())
    mean = s.mean()
    # blocked errors (block=20 trajs) for autocorrelation
    nb = max(1, n // 20)
    blocks = np.array_split(s, nb)
    bm = np.array([b.mean() for b in blocks])
    err = bm.std(ddof=1) / np.sqrt(len(bm)) if len(bm) > 1 else float("nan")
    flips = int(np.sum(s[1:] != s[:-1]))
    return dict(n=n, ntot=len(t), tmax=t.max(), mean=mean, err=err,
                fplus=float(np.mean(s > 0)), flips=flips,
                fliprate=flips / max(1, n - 1),
                mev_med=float(np.median(e)), mev_min=float(e.min()))


def main():
    args = sys.argv[1:]
    ntherm = 50
    if args and args[0] == "--ntherm":
        ntherm = int(args[1]); args = args[2:]
    for path in args:
        lamm = LAM.search(path)
        lam = lamm.group(1).replace("p", ".") if lamm else "?"
        r = analyze(path, ntherm)
        if r is None:
            print(f"lam={lam:<5} {path}: no SIGNHOP lines")
            continue
        if r["n"] == 0:
            print(f"lam={lam:<5} traj<= {r['tmax']}: all inside ntherm={ntherm}")
            continue
        print(f"lam={lam:<5} trajs={r['ntot']:<5} used={r['n']:<5} "
              f"<signPf>={r['mean']:+.3f}({r['err']:.3f})  "
              f"f(+)={r['fplus']:.3f}  fliprate={r['fliprate']:.3f}  "
              f"min|eval|: med={r['mev_med']:.3e} min={r['mev_min']:.3e}")


if __name__ == "__main__":
    main()
