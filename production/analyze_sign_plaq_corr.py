#!/usr/bin/env python3
"""Sign-plaq correlation from 4^4 signhop logs.

For each log, pair the per-traj [SIGNHOP] signPf with the same traj's
"Traj N plaq =" line, then report (post-ntherm):
  <plaq>_+ , <plaq>_-  (sector-conditioned means +- blocked errors)
  <plaq>_unsigned      (what the |Pf| chain gives)
  <plaq>_signed = <s*p>/<s>  (the reweighted physical value; huge error if
                              <s> ~ 0 -- that IS the sign problem)
If <plaq>_+ == <plaq>_- the sign decouples from the gauge observable and
reweighting is a no-op for plaq even when <sign> < 1.

Usage: analyze_sign_plaq_corr.py [--ntherm 50] LOG [LOG ...]
"""
import re
import sys
import numpy as np

SGN = re.compile(r"\[SIGNHOP\] traj=(\d+) signPf=([+-]1) ")
PLQ = re.compile(r"Traj (\d+) plaq = ([0-9.eE+-]+)")


def blocked_err(x, bs=20):
    nb = max(1, len(x) // bs)
    bm = np.array([b.mean() for b in np.array_split(x, nb)])
    return bm.std(ddof=1) / np.sqrt(len(bm)) if len(bm) > 1 else float("nan")


def main():
    args = sys.argv[1:]
    ntherm = 50
    if args and args[0] == "--ntherm":
        ntherm = int(args[1]); args = args[2:]
    for path in args:
        signs, plaqs = {}, {}
        for line in open(path, errors="replace"):
            m = SGN.search(line)
            if m:
                signs[int(m.group(1))] = int(m.group(2))
                continue
            m = PLQ.search(line)
            if m and "PlaqPlusRectangle" not in line:
                plaqs[int(m.group(1))] = float(m.group(2))
        common = sorted(t for t in signs if t in plaqs and t > ntherm)
        if len(common) < 20:
            print(f"{path}: only {len(common)} paired trajs")
            continue
        s = np.array([signs[t] for t in common], dtype=float)
        p = np.array([plaqs[t] for t in common])
        pp, pm = p[s > 0], p[s < 0]
        smean = s.mean()
        signed = np.sum(s * p) / np.sum(s) if abs(np.sum(s)) > 0 else float("nan")
        print(f"{path.split('/')[-1]}")
        print(f"  n={len(common)}  n+={len(pp)}  n-={len(pm)}  <sign>={smean:+.3f}")
        print(f"  <plaq>_+        = {pp.mean():.6f} ({blocked_err(pp):.6f})")
        if len(pm):
            print(f"  <plaq>_-        = {pm.mean():.6f} ({blocked_err(pm):.6f})")
            print(f"  delta(+ vs -)   = {pp.mean() - pm.mean():+.6f}")
        print(f"  <plaq>_unsigned = {p.mean():.6f} ({blocked_err(p):.6f})")
        print(f"  <plaq>_signed   = {signed:.6f}   (<s*p>/<s>, err blows up as <s>->0)")


if __name__ == "__main__":
    main()
