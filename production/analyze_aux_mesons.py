#!/usr/bin/env python3
"""Aux meson correlator C(1)/C(2) signal scan across channels, whole vs recent.

Passive check for the lambda=3 (and any lambda) aux-correlator streams.  Per
channel reports the vacuum-subtracted C(t=1) and C(t=2) with blocked errors +
the significance |C(1)|/err, for the WHOLE ensemble (therm cut) and the RECENT
~N configs (to catch thermalization transients).

Channels:
  - isovector pion (pseudoscalar): pi_plus (=pi_minus by H.c.), pi_zero, and
    the 3-charge-state ISO-AVERAGE (pi_plus+pi_minus+pi_zero)/3.
  - isovector a0 (scalar, has sigma VEV -> vac-sub): a0_plus, a0_zero, iso-avg.
  - isoscalar SINGLETS (S/N-favorable, 1 real DOF/site so ~2x less noisy than
    the isovector matrix channels): p (pseudoscalar, eta'-like, no VEV) and
    s (scalar, sigma/f0, VEV -> vac-sub).  These show hints first
    (2026-07-09: at lambda=3/200traj p=-0.27(9) 2.9sig, s=+0.42(16) 2.6sig,
    both stable vs recent-50, while the isovector pion stays ~0.4sig).

Vacuum subtraction = subtract the per-config t=T/2 plateau (= the disconnected
<W><W*> constant; parity-odd pi/p have <W>=0 so the sub is ~0, harmless).

Usage: analyze_aux_mesons.py <diag_dir> [--ntherm 20] [--recent 50]
"""
import h5py, glob, sys, numpy as np

args = sys.argv[1:]
d = args[0]
ntherm = int(args[args.index("--ntherm") + 1]) if "--ntherm" in args else 20
recent = int(args[args.index("--recent") + 1]) if "--recent" in args else 50
T = 48
files = sorted(glob.glob(d.rstrip("/") + "/hmc_diagnostics.*.h5"),
               key=lambda s: int(s.split(".")[-2]))


def load(chan):
    trajs, rows = [], []
    for f in files:
        thi = int(f.split(".")[-2])
        with h5py.File(f) as h:
            key = "aux_C_" + chan
            if key not in h:
                return None, None
            v = h[key]
            n = v.shape[0]
            for i in range(n):
                trajs.append(thi - n + 1 + i)
                rows.append(np.array(v[i]["re"]))
    return np.array(trajs), np.array(rows)


def blocked(x, bs=15):
    x = np.asarray(x)
    nb = max(1, len(x) // bs)
    bm = np.array([b.mean() for b in np.array_split(x, nb)])
    return x.mean(), (bm.std(ddof=1) / np.sqrt(len(bm)) if len(bm) > 1 else np.nan)


# per-config correlator builder (with optional iso-average over charge states)
def build(chans, vacsub):
    tj = None
    acc = None
    for c in chans:
        t, C = load(c)
        if C is None:
            return None, None
        tj = t
        acc = C if acc is None else acc + C
    acc = acc / len(chans)
    if vacsub:
        acc = acc - acc[:, T // 2][:, None]   # subtract disconnected plateau
    return tj, acc


CHANNELS = [
    ("pi_plus",      ["pi_plus"],                         False),
    ("pi_zero",      ["pi_zero"],                         False),
    ("pi_iso3(vsub)", ["pi_plus", "pi_minus", "pi_zero"], True),
    ("p (singlet)",  ["p"],                               False),
    ("a0_plus(vsub)", ["a0_plus"],                        True),
    ("a0_iso3(vsub)", ["a0_plus", "a0_minus", "a0_zero"], True),
    ("s (singlet,vsub)", ["s"],                           True),
]


def main():
    maxtraj = 0
    for _, chans, _ in CHANNELS:
        tj, _ = load(chans[0])
        if tj is not None:
            maxtraj = max(maxtraj, tj.max())
    print(f"dir={d}   maxtraj={maxtraj}")
    hdr = (f"{'channel':17} {'N':>4} | {'C(1)':>9} {'err':>7} {'sig':>4} | "
           f"{'C(2)':>9} {'err':>7}")
    for label, sel in [(f"WHOLE (therm>{ntherm})", lambda t: t > ntherm),
                       (f"RECENT ~{recent}", lambda t: t > maxtraj - recent)]:
        print(f"\n== {label} ==\n{hdr}")
        for name, chans, vacsub in CHANNELS:
            tj, C = build(chans, vacsub)
            if C is None:
                continue
            keep = sel(tj)
            C = C[keep]
            if len(C) < 10:
                continue
            m1, e1 = blocked(C[:, 1]); m2, e2 = blocked(C[:, 2])
            sig = abs(m1) / e1 if e1 > 0 else 0
            print(f"{name:17} {len(C):>4} | {m1:+9.4f} {e1:7.4f} {sig:4.1f} | "
                  f"{m2:+9.4f} {e2:7.4f}")


if __name__ == "__main__":
    main()
