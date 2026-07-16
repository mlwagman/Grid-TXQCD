#!/usr/bin/env python3
"""Cosine-profile mode-count census -> threshold-law fit.

Parses gridmg_setups/logs/coscen_e<ETA>.log (probe_deform_spectrum, NEV=48
ord=1001 uniform instrument) for the full eval lists of the deformed
Mpc~^dag Mpc~, counts near-zero modes below a set of cuts, and fits the
Fermi-threshold form

    N(eta) = c * sum_t max(0, mu(t)^2 - M*^2)^{3/2},
    mu(t)  = eta * (pi/2) * |sin(2 pi t / T)|        (cos profile, T=48)

— the local-density picture behind the MG window-volume law and the
solver wall at cos eta >= 0.7.  Counts that saturate NEV=48 are censored
(lower bounds) and excluded from the fit.  numpy-only (grid search over
M*, linear solve for c).
"""
import glob
import math
import os
import re
import sys

import numpy as np

LOGD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "gridmg_setups", "logs")
T = 48
NEV = 48
CUTS = [1e-4, 2e-4, 5e-4, 1e-3]


def parse():
    data = {}
    for f in sorted(glob.glob(os.path.join(LOGD, "coscen_e*.log"))):
        m = re.search(r"coscen_e([0-9.]+)\.log", f)
        if not m:
            continue
        eta = float(m.group(1))
        evs = [float(x) for x in
               re.findall(r"ev\[\d+\] = ([0-9.eE+-]+)", open(f).read())]
        if evs:
            data[eta] = np.sort(np.array(evs))
    return data


def window_sum(eta, mstar):
    t = np.arange(T)
    mu = eta * (math.pi / 2.0) * np.abs(np.sin(2 * math.pi * t / T))
    x = mu * mu - mstar * mstar
    return float(np.sum(np.where(x > 0, x, 0.0) ** 1.5))


def fit(etas, counts):
    """Grid-search M*, linear-solve c; return (c, mstar, resid)."""
    best = None
    for mstar in np.linspace(0.0, max(etas) * math.pi / 2.0, 400):
        w = np.array([window_sum(e, mstar) for e in etas])
        if np.all(w == 0):
            continue
        c = float(np.dot(w, counts) / max(np.dot(w, w), 1e-300))
        r = float(np.sum((c * w - counts) ** 2))
        if best is None or r < best[2]:
            best = (c, mstar, r)
    return best


def main():
    data = parse()
    if not data:
        print("no census logs found in", LOGD)
        sys.exit(1)
    etas = sorted(data)
    print(f"{'eta':>6} {'ev[0]':>12} {'ev[47]':>12} " +
          " ".join(f"N<{c:g}" .rjust(8) for c in CUTS))
    for e in etas:
        evs = data[e]
        row = [f"{e:6.2f}", f"{evs[0]:12.4e}", f"{evs[-1]:12.4e}"]
        for c in CUTS:
            n = int(np.sum(evs < c))
            row.append((f">={n}" if n >= NEV else f"{n}").rjust(8))
        print(" ".join(row))
    print()
    for cut in CUTS:
        pts = [(e, int(np.sum(data[e] < cut))) for e in etas]
        keep = [(e, n) for e, n in pts if 0 < n < NEV]
        if len(keep) < 3:
            print(f"cut {cut:g}: {len(keep)} uncensored points — skip fit")
            continue
        xe = np.array([e for e, _ in keep])
        yn = np.array([float(n) for _, n in keep], dtype=float)
        c, mstar, r = fit(xe, yn)
        print(f"cut {cut:g}: N = c*sum_t(mu^2-M*^2)^(3/2)  ->  "
              f"c={c:.4g}  M*={mstar:.4f}  (eta_onset={2*mstar/math.pi:.3f})"
              f"  rms={math.sqrt(r/len(keep)):.2f} over {len(keep)} pts")
        for e, n in pts:
            pred = c * window_sum(e, mstar)
            tag = " (censored)" if n >= NEV else ""
            print(f"    eta={e:4.2f}  N={n:3d}{tag}  pred={pred:7.1f}")


if __name__ == "__main__":
    main()
