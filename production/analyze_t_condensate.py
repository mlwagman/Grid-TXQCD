#!/usr/bin/env python3
"""
Aggregate per-traj <Tr σ, π, s, p, t> measurements from meas_aux_txqcd
output h5 files and produce a side-by-side comparison across the three
overnight runs (λ=0.1 cold, λ=0.1 hot, λ=3.0 cold).

Quick condensate diagnostic: if <t_raw> stays at zero (within stderr) across
all three ensembles → no condensate at laptop scale.  If <t_raw> for the
λ=0.1 runs is significantly nonzero and structured → condensate confirmed.
"""

import glob
import h5py
import numpy as np
import os
import sys

ENSEMBLES = [
    ("A_lam0p1_cold", "meas_2pt/txqcd_lam0.1000_cold"),
    ("B_lam0p1_hot",  "meas_2pt/txqcd_lam0.1000_hot"),
    ("C_lam3p0_cold", "meas_2pt/txqcd_lam3.0000_cold"),
]


def load_aux_ensemble(directory):
    files = sorted(glob.glob(os.path.join(directory, "aux_txqcd_*.h5")))
    if not files:
        return None
    traces = {"sigma": [], "pi": [], "s": [], "p": [], "t": []}
    for f in files:
        with h5py.File(f, "r") as h:
            for k in traces:
                key = f"aux_{k}_raw"
                if key in h:
                    arr = h[key][:]
                    if arr.dtype.names and "re" in arr.dtype.names:
                        arr = arr["re"] + 1j * arr["im"]
                    traces[k].append(arr)
    out = {}
    for k, v in traces.items():
        if v:
            arr = np.stack(v)
            out[k] = {
                "n_cfg": len(arr),
                "shape": arr[0].shape,
                "mean": arr.mean(axis=0),
                "stderr": arr.std(axis=0, ddof=1) / np.sqrt(len(arr))
                if len(arr) > 1 else np.zeros_like(arr[0]),
                "raw": arr,
            }
    return out


def fmt_mean_err(mean, err, label):
    if np.iscomplexobj(mean):
        mag = np.abs(mean)
        sig = (mag / np.maximum(err, 1e-30)).max() if err.size else 0
        print(f"      {label}:  |mean|_max = {mag.max():.4e}  err_max = {err.max():.4e}  "
              f"max(|mean|/err) = {sig:.2f}")
    else:
        sig = (np.abs(mean) / np.maximum(err, 1e-30)).max() if err.size else 0
        print(f"      {label}:  |mean|_max = {np.abs(mean).max():.4e}  err_max = {err.max():.4e}  "
              f"max(|mean|/err) = {sig:.2f}")


print("=" * 60)
print("TXQCD tensor-condensate aggregator")
print("=" * 60)

for tag, directory in ENSEMBLES:
    print(f"\n[{tag}]  {directory}")
    if not os.path.isdir(directory):
        print("  (no data — gen run did not produce this ensemble)")
        continue
    aux = load_aux_ensemble(directory)
    if aux is None:
        print("  (empty)")
        continue
    for channel, info in aux.items():
        print(f"  {channel}  n_cfg={info['n_cfg']}  shape={info['shape']}")
        fmt_mean_err(info["mean"], info["stderr"], "ensemble-avg")

print()
print("=" * 60)
print("Interpretation hint:")
print("  If max(|mean|/err) >> 1 for the t channel in the λ=0.1 ensembles")
print("  (A and B) but ≈ 1 in the λ=3.0 control (C), the condensate is real")
print("  and laptop-scale.  If A and B disagree, the condensate has multiple")
print("  minima (ergodicity issue, cluster's worth of MD steps may be needed).")
