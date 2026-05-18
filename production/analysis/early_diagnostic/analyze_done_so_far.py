#!/usr/bin/env python3
"""Analysis of all 64-src measurements done so far (new FB+iso schema).

Computes effective masses (pos-only, FB-averaged, isospin+FB-averaged) and
SEM ratios.  Highlights:
- Cross-λ pion + nucleon spectrum
- Per-λ noise reduction from FB and iso averaging
- λ=6 basin pair (cfg.40 LO vs cfg.50 HI) — same stream, different basin
- TXQCD vs QCD noise comparison
"""
import h5py
import numpy as np
import os
from glob import glob

T = 48
T_LATT = T


def re(a):
    a = np.array(a)
    return a["re"] if a.dtype.names else a.real


def load_meas(h5path, kind):
    with h5py.File(h5path, "r") as f:
        keys = list(f.keys())
        out = {}
        out["pion_per_src"] = re(f["pion_per_src"])
        if kind == "txqcd" and "proton_pos_per_src" in keys:
            out["p_pos_s"] = re(f["proton_pos_per_src"])
            out["p_neg_s"] = re(f["proton_neg_per_src"])
            out["n_pos_s"] = re(f["neutron_pos_per_src"])
            out["n_neg_s"] = re(f["neutron_neg_per_src"])
            out["has_iso"] = True
        else:
            # legacy or QCD: single nucleon, possibly with neg parity
            out["p_pos_s"] = re(f["nucleon_per_src"])
            if "nucleon_neg_per_src" in keys:
                out["p_neg_s"] = re(f["nucleon_neg_per_src"])
                out["has_fb"] = True
            else:
                out["has_fb"] = False
            out["has_iso"] = False
    return out


def shift_per_src(corr, src_ts):
    out = np.zeros_like(corr)
    for s in range(corr.shape[0]):
        for t in range(T):
            rel = (t - src_ts[s]) % T
            out[s, rel] = corr[s, t]
    return out


def stats(corr_shifted):
    """Mean and SEM along the source axis (axis 0)."""
    mean = corr_shifted.mean(axis=0)
    sem = corr_shifted.std(axis=0, ddof=1) / np.sqrt(corr_shifted.shape[0])
    return mean, sem


def effective_mass(c):
    em = np.full(T, np.nan)
    for t in range(T - 1):
        if c[t] > 0 and c[t + 1] > 0:
            em[t] = np.log(c[t] / c[t + 1])
    return em


def plateau(em, t0, t1):
    vals = em[t0:t1]
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0: return (np.nan, np.nan)
    return (np.mean(finite), np.std(finite, ddof=1) / np.sqrt(len(finite)))


def analyze_cfg(label, h5path, kind, sx=2, st=8):
    nsrc_expected = sx ** 3 * st
    if not os.path.exists(h5path):
        return None
    sz = os.path.getsize(h5path)
    if sz < 50000:  # too small = old schema, skip
        return None
    m = load_meas(h5path, kind)
    nsrc = m["pion_per_src"].shape[0]
    if nsrc != nsrc_expected:
        return None
    # Source positions: t_s = (s % st) * (T // st)
    src_ts = np.array([(s % st) * (T // st) for s in range(nsrc)])

    # Pion: shift+avg
    pi_s = shift_per_src(m["pion_per_src"], src_ts)
    pi_mean, pi_sem = stats(pi_s)

    # Nucleon pos-only (proton): shift+avg
    p_pos_s = shift_per_src(m["p_pos_s"], src_ts)
    p_pos_mean, p_pos_sem = stats(p_pos_s)

    # FB-averaged per source, then average
    if "p_neg_s" in m:
        p_neg_s = shift_per_src(m["p_neg_s"], src_ts)
        p_fb_per_src = np.zeros_like(p_pos_s)
        for s in range(nsrc):
            for t in range(T):
                p_fb_per_src[s, t] = 0.5 * (p_pos_s[s, t] - p_neg_s[s, (T - t) % T])
        p_fb_mean, p_fb_sem = stats(p_fb_per_src)
    else:
        p_fb_mean = p_fb_sem = None

    # Isospin+FB avg (TXQCD only)
    if m.get("has_iso"):
        n_pos_s = shift_per_src(m["n_pos_s"], src_ts)
        n_neg_s = shift_per_src(m["n_neg_s"], src_ts)
        iso_fb_per_src = np.zeros_like(p_pos_s)
        for s in range(nsrc):
            for t in range(T):
                iso_fb_per_src[s, t] = 0.25 * (
                    p_pos_s[s, t] - p_neg_s[s, (T - t) % T] +
                    n_pos_s[s, t] - n_neg_s[s, (T - t) % T]
                )
        iso_fb_mean, iso_fb_sem = stats(iso_fb_per_src)
    else:
        iso_fb_mean = iso_fb_sem = None

    return {
        "label": label,
        "pi_mean": pi_mean.real, "pi_sem": pi_sem.real,
        "p_pos_mean": p_pos_mean.real, "p_pos_sem": p_pos_sem.real,
        "p_fb_mean": p_fb_mean.real if p_fb_mean is not None else None,
        "p_fb_sem":  p_fb_sem.real  if p_fb_sem  is not None else None,
        "iso_fb_mean": iso_fb_mean.real if iso_fb_mean is not None else None,
        "iso_fb_sem":  iso_fb_sem.real  if iso_fb_sem  is not None else None,
    }


SPECS = [
    # md10 (latest + next)
    ("λ=6 cfg.40 md10 (BASIN_LO)", "../../meas_2pt/txqcd_lam6.0000_fromchroma_md10/conn_txqcd_40.h5", "txqcd"),
    ("λ=6 cfg.50 md10 (BASIN_HI)", "../../meas_2pt/txqcd_lam6.0000_fromchroma_md10/conn_txqcd_50.h5", "txqcd"),
    ("λ=7 cfg.30 md10",            "../../meas_2pt/txqcd_lam7.0000_fromchroma_md10/conn_txqcd_30.h5", "txqcd"),
    ("λ=7 cfg.40 md10",            "../../meas_2pt/txqcd_lam7.0000_fromchroma_md10/conn_txqcd_40.h5", "txqcd"),
    ("λ=12 cfg.50 md10",           "../../meas_2pt/txqcd_lam12.0000_fromchroma_md10/conn_txqcd_50.h5", "txqcd"),
    ("λ=12 cfg.60 md10",           "../../meas_2pt/txqcd_lam12.0000_fromchroma_md10/conn_txqcd_60.h5", "txqcd"),
    ("λ=16 cfg.40 md10",           "../../meas_2pt/txqcd_lam16.0000_fromchroma_md10/conn_txqcd_40.h5", "txqcd"),
    ("λ=16 cfg.50 md10",           "../../meas_2pt/txqcd_lam16.0000_fromchroma_md10/conn_txqcd_50.h5", "txqcd"),
    # md20 cfg.20
    ("λ=5 cfg.20 md20",            "../../meas_2pt/txqcd_lam5.0000_fromchroma_md20/conn_txqcd_20.h5", "txqcd"),
    ("λ=6 cfg.20 md20",            "../../meas_2pt/txqcd_lam6.0000_fromchroma_md20/conn_txqcd_20.h5", "txqcd"),
    ("λ=6.5 cfg.20 md20",          "../../meas_2pt/txqcd_lam6.5000_fromchroma_md20/conn_txqcd_20.h5", "txqcd"),
    ("λ=7 cfg.20 md20",            "../../meas_2pt/txqcd_lam7.0000_fromchroma_md20/conn_txqcd_20.h5", "txqcd"),
    ("λ=7.5 cfg.20 md20",          "../../meas_2pt/txqcd_lam7.5000_fromchroma_md20/conn_txqcd_20.h5", "txqcd"),
    ("λ=8 cfg.20 md20",            "../../meas_2pt/txqcd_lam8.0000_fromchroma_md20/conn_txqcd_20.h5", "txqcd"),
    # QCD ref
    ("QCD chroma 11100",           "../../meas_2pt/qcd_chroma_64src/conn_qcd_11100.h5", "qcd"),
]

results = []
for label, path, kind in SPECS:
    r = analyze_cfg(label, path, kind)
    if r is None:
        print(f"SKIP (missing/old): {label}")
        continue
    results.append(r)

PLATEAU_T0, PLATEAU_T1 = 8, 16

print()
print("=" * 120)
print(f"{'config':<28} | {'m_π':>8} | {'m_N(pos)':>10} | {'m_N(fb)':>10} | {'m_N(iso+fb)':>12} | "
      f"{'σ/|C_π|@10':>10} | {'σ/|C_pos|@10':>13} | {'σ/|C_iso|@10':>13}")
print("-" * 120)
T_PROBE = 10
for r in results:
    em_pi = effective_mass(r["pi_mean"])
    em_pos = effective_mass(r["p_pos_mean"])
    em_fb  = effective_mass(r["p_fb_mean"])  if r["p_fb_mean"]  is not None else None
    em_iso = effective_mass(r["iso_fb_mean"]) if r["iso_fb_mean"] is not None else None
    mpi, _ = plateau(em_pi, PLATEAU_T0, PLATEAU_T1)
    mN_p, _ = plateau(em_pos, PLATEAU_T0, PLATEAU_T1)
    mN_fb, _ = plateau(em_fb, PLATEAU_T0, PLATEAU_T1) if em_fb is not None else (np.nan, 0)
    mN_iso, _ = plateau(em_iso, PLATEAU_T0, PLATEAU_T1) if em_iso is not None else (np.nan, 0)
    rel_pi  = r["pi_sem"][T_PROBE]   / abs(r["pi_mean"][T_PROBE])    if r["pi_mean"][T_PROBE]    != 0 else np.nan
    rel_pos = r["p_pos_sem"][T_PROBE] / abs(r["p_pos_mean"][T_PROBE]) if r["p_pos_mean"][T_PROBE] != 0 else np.nan
    rel_iso = (r["iso_fb_sem"][T_PROBE] / abs(r["iso_fb_mean"][T_PROBE])
               if r["iso_fb_mean"] is not None and r["iso_fb_mean"][T_PROBE] != 0 else np.nan)
    print(f"{r['label']:<28} | {mpi:>8.4f} | {mN_p:>10.4f} | "
          f"{mN_fb if not np.isnan(mN_fb) else 'NA':>10} | "
          f"{mN_iso if not np.isnan(mN_iso) else 'NA':>12} | "
          f"{rel_pi:>10.4f} | {rel_pos:>13.4f} | {rel_iso if not np.isnan(rel_iso) else 'NA':>13}")

# Compute noise reduction ratios where we have iso (TXQCD) + a QCD ref
print()
print("=" * 80)
print("Noise reduction (σ/|C|) at t=10:  TXQCD/QCD ratios")
print("-" * 80)
qcd_ref = next((r for r in results if "QCD" in r["label"]), None)
if qcd_ref is not None:
    qcd_rel_pos = qcd_ref["p_pos_sem"][T_PROBE] / abs(qcd_ref["p_pos_mean"][T_PROBE])
    qcd_rel_pi  = qcd_ref["pi_sem"][T_PROBE]    / abs(qcd_ref["pi_mean"][T_PROBE])
    print(f"  QCD ref: σ_π/|C_π|={qcd_rel_pi:.4f},  σ_N(pos)/|C|={qcd_rel_pos:.4f}")
    print(f"{'config':<28} | {'σ_π / σ_π_QCD':>15} | {'σ_N(pos) / σ_N_QCD':>20} | {'σ_N(iso+fb) / σ_N_QCD':>22}")
    print("-" * 90)
    for r in results:
        if r is qcd_ref: continue
        ratio_pi = (r["pi_sem"][T_PROBE] / abs(r["pi_mean"][T_PROBE])) / qcd_rel_pi
        ratio_pos = (r["p_pos_sem"][T_PROBE] / abs(r["p_pos_mean"][T_PROBE])) / qcd_rel_pos
        if r["iso_fb_mean"] is not None:
            ratio_iso = (r["iso_fb_sem"][T_PROBE] / abs(r["iso_fb_mean"][T_PROBE])) / qcd_rel_pos
        else:
            ratio_iso = np.nan
        print(f"{r['label']:<28} | {ratio_pi:>15.3f} | {ratio_pos:>20.3f} | "
              f"{ratio_iso if not np.isnan(ratio_iso) else 'NA':>22}")

# Effective mass plateau study for λ=6 basin pair
print()
print("=" * 80)
print("λ=6 BASIN PAIR: effective mass at each t (pos-only) — same stream, different basin")
print("-" * 80)
lo = next((r for r in results if "BASIN_LO" in r["label"]), None)
hi = next((r for r in results if "BASIN_HI" in r["label"]), None)
if lo and hi:
    em_lo = effective_mass(lo["p_pos_mean"])
    em_hi = effective_mass(hi["p_pos_mean"])
    em_pi_lo = effective_mass(lo["pi_mean"])
    em_pi_hi = effective_mass(hi["pi_mean"])
    print(f"  {'t':>3} | {'m_N (LO)':>10} | {'m_N (HI)':>10} | {'m_π (LO)':>10} | {'m_π (HI)':>10}")
    for t in [4, 6, 8, 10, 12, 14, 16]:
        print(f"  {t:>3} | {em_lo[t]:>10.4f} | {em_hi[t]:>10.4f} | {em_pi_lo[t]:>10.4f} | {em_pi_hi[t]:>10.4f}")
