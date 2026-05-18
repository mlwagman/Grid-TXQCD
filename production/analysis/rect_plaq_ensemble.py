#!/usr/bin/env python3
"""Ensemble-averaged plaq, rect, and rect-vs-plaq slope for OUR vs CHROMA.

Builds on rect_plaq_ratio.py (validated parser).  Averages over N cfgs each
side and, crucially, fits rect vs plaq WITHIN each ensemble: if both ensembles
share the same gauge action they lie on the SAME rect(plaq) line, so the
chroma points should fall on our fitted line (and vice-versa).  A different
gauge action displaces the whole line.
"""
import sys, glob, numpy as np
sys.path.insert(0, 'analysis')
from rect_plaq_ratio import read_nersc, read_ildg, avg_plaq, avg_rect

def collect_nersc(paths):
    out = []
    for p in paths:
        U, _, rec = read_nersc(p)
        pl = avg_plaq(U); rc = avg_rect(U)
        assert abs(pl - rec) < 1e-5, (p, pl, rec)
        out.append((pl, rc))
    return np.array(out)

def collect_ildg(paths):
    out = []
    for p in paths:
        U, _ = read_ildg(p)
        pl = avg_plaq(U); rc = avg_rect(U)
        out.append((pl, rc))
    return np.array(out)

if __name__ == '__main__':
    nersc = sorted(glob.glob('cfgs/qcd_s702_nf2p1_mdscan_mds10_fork_v2/ckpoint_lat.*'),
                   key=lambda f: int(f.split('.')[-1]))[-12:]   # last 12 (post-plateau)
    chroma = sorted(glob.glob('/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/*.lime'))[-12:]

    O = collect_nersc(nersc)
    C = collect_ildg(chroma)

    def stat(A, lab):
        p, r = A[:, 0], A[:, 1]
        print(f"  {lab}: N={len(A)}  <plaq>={p.mean():.6f}±{p.std(ddof=1)/len(p)**.5:.6f}"
              f"  <rect>={r.mean():.6f}±{r.std(ddof=1)/len(r)**.5:.6f}"
              f"  <rect/plaq>={(r/p).mean():.6f}")
        return p, r
    print("=== ensemble means ===")
    op, orr = stat(O, 'ours  ')
    cp, cr = stat(C, 'chroma')

    # Fit rect = a + b*plaq within OUR ensemble; see where chroma sits vs that line.
    b, a = np.polyfit(op, orr, 1)
    pred_c = a + b * cp.mean()
    resid = cr.mean() - pred_c
    sigma_r = cr.std(ddof=1) / len(cr) ** .5
    print("\n=== same-gauge-action test (rect vs plaq locus) ===")
    print(f"  our fit:  rect = {a:.5f} + {b:.4f}*plaq   (slope b={b:.3f})")
    print(f"  chroma <plaq>={cp.mean():.6f} -> our-line predicts rect={pred_c:.6f}")
    print(f"  chroma actual <rect>={cr.mean():.6f}   residual={resid:+.6f}"
          f"  ({resid/sigma_r:+.1f} sigma_rect)")
    print("  |residual| within ~1-2 sigma  => SAME gauge action; 0.0008 plaq")
    print("    offset is fermion-sector (different effective coupling).")
    print("  residual many sigma off       => gauge action convention differs.")
