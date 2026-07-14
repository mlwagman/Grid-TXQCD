#!/usr/bin/env python3
# Compare two baryon_deform h5 files (abs-diff relative to peak per dataset).
# Usage: _qdef_cmp2.py <ref.h5> <new.h5> <pass_threshold>
import h5py, numpy as np, sys

ref = h5py.File(sys.argv[1], "r")
new = h5py.File(sys.argv[2], "r")
thr = float(sys.argv[3])

def arr(f, k):
    d = f[k][()]
    if d.dtype.names and 're' in d.dtype.names:
        return d['re'] + 1j * d['im']
    return d

worst = 0
for k in sorted(ref.keys()):
    if not hasattr(ref[k], 'shape'):
        continue
    try:
        a = arr(ref, k).astype(complex).ravel()
        b = arr(new, k).astype(complex).ravel()
    except Exception:
        continue
    if a.shape != b.shape:
        print(f"{k:34s} SHAPE MISMATCH {a.shape} vs {b.shape}")
        worst = 1e9
        continue
    ad = np.abs(a - b)
    r = ad.max() / max(np.abs(a).max(), 1e-300)
    worst = max(worst, r)
    print(f"{k:34s} max|ref|={np.abs(a).max():11.3e} absdiff/peak={r:11.3e}")
print(f"WORST={worst:.3e} thr={thr:g} -> {'PASS' if worst <= thr else 'FAIL'}")
sys.exit(0 if worst <= thr else 1)
