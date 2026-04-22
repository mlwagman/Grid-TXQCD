#!/usr/bin/env python3
"""Live monitoring of our HMC streams against the chroma reference ensemble.

Reads hmc_diagnostics.<traj>.h5 files from cfgs/qcd_s<ID>/ (one per stream),
extracts plaquette and vev_trminv time series, and compares against the
chroma cl3_16_48_b6p1_m0p2450 reference:

  Reference plaquette (thermalized, traj >= 500): 0.5135 ± 0.0005
  Reference plaquette (traj 100, still thermalizing): 0.5153

Usage:
  ./monitor_stream.py                         # print current state of all 4 streams
  ./monitor_stream.py --watch                 # re-print every 60s
  ./monitor_stream.py --stream 0              # only stream 0
  ./monitor_stream.py --cfg-root cfgs/qcd     # single-stream (no STREAM_ID)
"""

import argparse
import glob
import os
import sys
import time

try:
    import h5py
    import numpy as np
except ImportError as e:
    print(f"Need h5py and numpy: {e}", file=sys.stderr)
    sys.exit(1)


# Chroma reference values for cl3_16_48_b6p1_m0p2450 (computed by chroma_plaq.py
# on configs at trajectories 100, 200, 500, 1000, 2000, 5000, 10000).
CHROMA_PLAQ_THERMAL = 0.5135       # asymptotic (traj >= 500)
CHROMA_PLAQ_THERMAL_STD = 0.0005   # spread across trajectories
CHROMA_PLAQ_TRAJ100 = 0.5153       # mid-thermalization reference


def load_stream(cfg_dir):
    """Return (trajs, plaqs, vev_trminv) arrays concatenated across all diag files."""
    files = sorted(glob.glob(os.path.join(cfg_dir, "hmc_diagnostics.*.h5")),
                   key=lambda f: int(f.split(".")[-2]))
    if not files:
        return None, None, None
    t_all, p_all, v_all = [], [], []
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                t_all.extend(h["traj"][:])
                p_all.extend(h["plaq"][:])
                v_all.extend(h["vev_trminv"][:])
        except (OSError, KeyError):
            continue
    return np.array(t_all), np.array(p_all), np.array(v_all)


def summarize(name, trajs, plaqs, vevs):
    if trajs is None or len(trajs) == 0:
        print(f"  {name}: no data yet")
        return
    n = len(trajs)
    latest_t = trajs[-1]
    latest_p = plaqs[-1]
    latest_v = vevs[-1]
    # Running mean/std on the last 20 trajectories
    tail = plaqs[-20:] if n >= 5 else plaqs
    tail_v = vevs[-20:] if n >= 5 else vevs
    p_mean, p_std = float(np.mean(tail)), float(np.std(tail))
    v_mean, v_std = float(np.mean(tail_v)), float(np.std(tail_v))
    # chroma-comparison tag
    diff = p_mean - CHROMA_PLAQ_THERMAL
    diff_sig = diff / CHROMA_PLAQ_THERMAL_STD if CHROMA_PLAQ_THERMAL_STD > 0 else 0.0
    tag = " converged" if abs(diff) < 3 * CHROMA_PLAQ_THERMAL_STD else " (thermalizing)"
    print(f"  {name}: n={n:3d}  latest traj={latest_t:4d}  "
          f"plaq={latest_p:.5f}  vev_trminv={latest_v:.4f}")
    print(f"    tail(20): plaq={p_mean:.5f}±{p_std:.5f}  vev={v_mean:.4f}±{v_std:.4f}"
          f"  Δplaq vs chroma = {diff:+.5f} ({diff_sig:+.1f}σ){tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-root", default="cfgs",
                    help="parent of the per-stream cfg dirs (default: cfgs)")
    ap.add_argument("--stream", type=int, default=-1,
                    help="single stream id (default: all 0..3)")
    ap.add_argument("--watch", action="store_true", help="re-print every 60 s")
    ap.add_argument("--interval", type=float, default=60.0)
    args = ap.parse_args()

    streams = [args.stream] if args.stream >= 0 else [0, 1, 2, 3]

    while True:
        print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  chroma ref plaq = "
              f"{CHROMA_PLAQ_THERMAL:.4f} ± {CHROMA_PLAQ_THERMAL_STD:.4f} (thermal) ===")
        for s in streams:
            cfg_dir = os.path.join(args.cfg_root, f"qcd_s{s}")
            if not os.path.isdir(cfg_dir):
                # fall back to single-stream layout
                cfg_dir = os.path.join(args.cfg_root, "qcd")
            trajs, plaqs, vevs = load_stream(cfg_dir)
            summarize(f"stream {s} [{cfg_dir}]", trajs, plaqs, vevs)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
