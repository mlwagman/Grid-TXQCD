#!/usr/bin/env python3
"""Quick Z-scan analysis on QCD chroma cfg 11610 — connected pion m_eff.

Compares λ=7 with kinetic filter Z=0, 10, 1e6 against the regular qcd_chroma_ref
baseline for the same cfg (and lam5/lam7-HALF historical points for context).
"""
import os
import h5py
import numpy as np

BASE = "/lustre2/nplqcd/Grid-TXQCD/production/meas_2pt"
QCD_DIR = os.path.join(BASE, "qcd_chroma_ref_shift")
TX_TEST_DIRS = [
    ("lam7 Z=0   full-sigma",  "txqcd_lam7.0000_on_qcd_chroma_z0_test"),
    ("lam7 Z=10  filter",      "txqcd_lam7.0000_on_qcd_chroma_z10_test"),
    ("lam7 Z=1e6 frozen",      "txqcd_lam7.0000_on_qcd_chroma_zinf_test"),
]
CFG = 11610
T = 48
PLATEAU = (9, 16)


def load_conn(path, key):
    with h5py.File(path, "r") as f:
        return np.array(f[key], dtype=float)


def fold(c):
    cf = c.copy()
    for t in range(1, T):
        cf[t] = 0.5 * (c[t] + c[(T - t) % T])
    return cf


def cosh_meff_plateau(c):
    em = []
    for t in range(PLATEAU[0], PLATEAU[1]):
        num = c[t + 1] + c[t - 1]
        den = 2.0 * c[t]
        ratio = num / den
        if ratio > 1.0:
            em.append(np.arccosh(ratio))
    return np.mean(em) if em else float("nan")


qcd_path = os.path.join(QCD_DIR, f"conn_qcd_{CFG}.h5")
if not os.path.exists(qcd_path):
    print(f"QCD baseline missing: {qcd_path}"); exit(1)
qcd_pi = fold(load_conn(qcd_path, "pion_conn"))
m_qcd = cosh_meff_plateau(qcd_pi)

print(f"cfg {CFG} — t∈{PLATEAU} cosh m_eff plateau\n")
print(f"  QCD baseline  m_pi = {m_qcd:.4f}\n")
print(f"  {'config':<24} {'m_pi(TX)':>10}  {'Δ vs QCD':>10}")
print(f"  {'-'*24} {'-'*10}  {'-'*10}")
for label, sub in TX_TEST_DIRS:
    path = os.path.join(BASE, sub, f"conn_txqcd_{CFG}.h5")
    if not os.path.exists(path):
        print(f"  {label:<24} {'NA':>10}  {'NA':>10}    (file missing)")
        continue
    tx_pi = fold(load_conn(path, "pion_conn"))
    m_tx = cosh_meff_plateau(tx_pi)
    print(f"  {label:<24} {m_tx:>10.4f}  {m_tx - m_qcd:>+10.4f}")
