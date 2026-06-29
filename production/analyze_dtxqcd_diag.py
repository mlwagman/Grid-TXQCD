#!/usr/bin/env python3
"""Autocorrelation + aux-correlator analysis for DTXQCD HMC diagnostics.

The signed-Pfaffian diagnostics observer (production/dtxqcd_diag.h) writes one
`hmc_diagnostics.<traj>.h5` per flush into a stream's cfg dir.  Each flush is a
DISJOINT chunk (the observer clears its buffers after writing), and small flushes
store the per-traj scalars (`traj`, `plaq`, ...) as HDF5 *attributes* while large
ones store them as datasets -- so a stream's full time series must be rebuilt by
globbing + concatenating all its h5 files.  This tool does that, then computes:

  (1) Integrated autocorrelation time tau_int (Sokal automatic windowing, with
      the Madras-Sokal error) for plaq + the aux scalar VEVs + the spectral gap,
      reported in trajectories and (if --trajL given) in MD-time units.  This is
      the quantity that sets measurement flow, and its lambda-scaling is the open
      question (`--scan` tabulates tau_int(plaq) across a set of streams).

  (2) Jackknifed aux correlators C(t): time-folded, effective mass, and (for the
      VEV-dominated scalar channels) plateau-subtracted connected correlators.

Usage:
  analyze_dtxqcd_diag.py CFGDIR [--therm N] [--trajL X] [--corr]
  analyze_dtxqcd_diag.py --scan 'cfgs/dtxqcd_lam*_nonEO_*' [--therm N] [--frac F]

--therm N    discard the first N trajectories (thermalization).  --frac F instead
             discards the first fraction F (default 0.3) -- use until you know
             where each lambda equilibrates (QCD/TXQCD took ~500 traj).
Notes: tau_int needs O(50+) post-therm trajs to be meaningful; the tool flags
short series.  Aux correlators are bosonic -> periodic in T, folded accordingly.
"""
import sys, os, glob, argparse
import numpy as np
import h5py

NS_DEFAULT = 16 * 16 * 16 * 48  # total sites; aux_wall_* are summed over spatial vol/timeslice
T_DEFAULT = 48


def _to_complex(a):
    a = np.asarray(a)
    if a.dtype.names and 're' in a.dtype.names:
        return a['re'] + 1j * a['im']
    return a


def _get(f, name):
    """Dataset-or-attribute fetch (complex-aware); None if absent."""
    if name in f.keys():
        return _to_complex(f[name][:])
    if name in f.attrs:
        return _to_complex(np.atleast_1d(f.attrs[name]))
    return None


def load_stream(cfgdir, ns=NS_DEFAULT):
    """Glob + concatenate all hmc_diagnostics.*.h5 in cfgdir into one time series.

    Returns dict of per-traj arrays (sorted by traj, dups resolved to last) plus
    'corr': {name: (ntraj, T) real} for each aux_C_* channel.
    """
    files = glob.glob(os.path.join(cfgdir, "hmc_diagnostics.*.h5"))

    def _tj(p):
        try:
            return int(os.path.basename(p).split('.')[-2])
        except ValueError:
            return -1
    files = sorted(files, key=_tj)
    if not files:
        raise SystemExit(f"no hmc_diagnostics.*.h5 in {cfgdir}")

    scal_keys = ["plaq", "vev_trminv", "mdagm_lmin", "mdagm_lmax",
                 "norm_sigma", "norm_pi", "norm_s", "norm_p"]
    wall_map = {"s": "aux_wall_s", "trsig": "aux_wall_trsig",
                "p": "aux_wall_p", "trpi": "aux_wall_trpi"}
    acc = {"traj": []}
    for k in scal_keys:
        acc[k] = []
    for k in wall_map:
        acc["vev_" + k] = []
    acc["g5M_absmin"] = []
    corr = {}

    for fn in files:
        with h5py.File(fn, "r") as f:
            traj = _get(f, "traj")
            if traj is None:
                continue
            traj = np.asarray(traj.real, dtype=int)
            n = len(traj)
            acc["traj"].append(traj)
            for k in scal_keys:
                v = _get(f, k)
                acc[k].append(np.asarray(v.real, float) if v is not None
                              else np.full(n, np.nan))
            for short, wall in wall_map.items():
                w = _get(f, wall)
                acc["vev_" + short].append(
                    w.real.reshape(n, -1).sum(axis=1) / ns if w is not None
                    else np.full(n, np.nan))
            g = _get(f, "g5M_evals")
            acc["g5M_absmin"].append(np.abs(g.real).min(axis=1) if g is not None
                                     else np.full(n, np.nan))
            for k in f.keys():
                if k.startswith("aux_C_"):
                    corr.setdefault(k, []).append(_to_complex(f[k][:]).real)

    out = {k: np.concatenate(v) for k, v in acc.items()}
    corr = {k: np.concatenate(v, axis=0) for k, v in corr.items()}
    # sort by traj, drop duplicate traj numbers (keep last occurrence)
    order = np.argsort(out["traj"], kind="stable")
    for k in out:
        out[k] = out[k][order]
    for k in corr:
        corr[k] = corr[k][order]
    _, keep = np.unique(out["traj"][::-1], return_index=True)
    keep = np.sort(len(out["traj"]) - 1 - keep)
    for k in out:
        out[k] = out[k][keep]
    for k in corr:
        corr[k] = corr[k][keep]
    out["corr"] = corr
    return out


# ---------------------------------------------------------------------------
# integrated autocorrelation time (Sokal automatic windowing)
# ---------------------------------------------------------------------------
def tau_int(x, S=5.0):
    """Madras-Sokal integrated autocorrelation time of 1D series x.

    Returns (tau, dtau, W, N): tau_int, its error, the window, and the series
    length.  Sokal self-consistent window: smallest W with W >= S*tau_int(W).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    N = len(x)
    if N < 4:
        return np.nan, np.nan, 0, N
    xm = x.mean()
    dx = x - xm
    g0 = np.dot(dx, dx) / N
    if g0 <= 0:
        return 0.5, 0.0, 0, N            # constant series -> no autocorr
    rho = np.empty(N)
    rho[0] = 1.0
    for t in range(1, N):
        rho[t] = np.dot(dx[:N - t], dx[t:]) / ((N - t) * g0)
    tau = 0.5
    W = N - 1
    for w in range(1, N):
        tau += rho[w]
        if w >= S * tau:
            W = w
            break
    tau = max(tau, 0.5)
    dtau = tau * np.sqrt((4.0 * W + 2.0) / N)   # Madras-Sokal
    return tau, dtau, W, N


# ---------------------------------------------------------------------------
# jackknife correlator analysis
# ---------------------------------------------------------------------------
def _jack_means(samples):
    """Leave-one-out jackknife resamples of the column-mean. samples: (N, T)."""
    N = samples.shape[0]
    tot = samples.sum(axis=0)
    return (tot - samples) / (N - 1)            # (N, T)


def jack_corr(C, fold=True, T=T_DEFAULT):
    """Jackknife mean+err of C(t) and the log-ratio effective mass.

    C: (ntraj, T).  Returns dict with c, dc, meff, dmeff, (folded if fold).
    """
    N = C.shape[0]
    jk = _jack_means(C)                          # (N, T)
    cm = C.mean(axis=0)
    if fold:
        idx = (T - np.arange(T)) % T
        cm = 0.5 * (cm + cm[idx])
        jk = 0.5 * (jk + jk[:, idx])
    dc = np.sqrt((N - 1) / N * ((jk - jk.mean(0)) ** 2).sum(0))

    def meff_of(c):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(c[:-1] / c[1:])
    mjk = np.array([meff_of(jk[i]) for i in range(N)])
    mm = meff_of(cm)
    dm = np.sqrt((N - 1) / N * ((mjk - mjk.mean(0)) ** 2).sum(0))
    return dict(c=cm, dc=dc, meff=mm, dmeff=dm, N=N)


def connected_scalar(C, T=T_DEFAULT):
    """VEV-dominated scalar channel: subtract the large-t plateau (disconnected
    constant) estimated as the mean of C(t) over the central half."""
    lo, hi = T // 4, 3 * T // 4
    plateau = C[:, lo:hi].mean(axis=1, keepdims=True)
    return C - plateau


# ---------------------------------------------------------------------------
def _therm_cut(d, therm, frac):
    n = len(d["traj"])
    cut = therm if therm is not None else int(np.floor(frac * n))
    return cut, n


def analyze_one(cfgdir, therm=None, frac=0.3, trajL=None, do_corr=False):
    d = load_stream(cfgdir)
    cut, n = _therm_cut(d, therm, frac)
    name = os.path.basename(os.path.normpath(cfgdir))
    tr = d["traj"]
    print(f"=== {name}")
    print(f"  trajs {tr.min()}..{tr.max()}  (n={n}); discarding first {cut} as therm "
          f"-> {n - cut} for analysis")
    if n - cut < 50:
        print("  [!] < 50 post-therm trajs: tau_int is NOT yet reliable (need O(100s)).")

    obs = [("plaq", "plaq"), ("Tr_sigma", "vev_trsig"), ("s", "vev_s"),
           ("Tr_pi", "vev_trpi"), ("mdagm_lmin", "mdagm_lmin"),
           ("g5M_absmin", "g5M_absmin")]
    print(f"  {'observable':12s} {'mean':>12s} {'tau_int(traj)':>16s} {'W':>4s}"
          + ("  tau_int(MDtime)" if trajL else ""))
    res = {}
    for label, key in obs:
        x = d.get(key)
        if x is None or not np.any(np.isfinite(x)):
            continue
        xc = x[cut:]
        tau, dtau, W, Nn = tau_int(xc)
        res[label] = (tau, dtau)
        mdt = f"  {tau * trajL:8.3f} +- {dtau * trajL:.3f}" if trajL else ""
        mean = np.nanmean(xc)
        print(f"  {label:12s} {mean:12.5g} {tau:8.2f} +- {dtau:5.2f} {W:4d}{mdt}")
    if do_corr:
        analyze_corr(d, cut)
    return res


def analyze_corr(d, cut):
    corr = d["corr"]
    if not corr:
        print("  (no aux_C_* correlators in this stream)")
        return
    print("  --- aux correlators (jackknife, folded) ---")
    chans = [("pi_zero", "aux_C_pi_zero", False),
             ("Tr_pi", "aux_C_trpi", False),
             ("p", "aux_C_p", False),
             ("a0_zero", "aux_C_a0_zero", False),
             ("s(conn)", "aux_C_s", True),
             ("Tr_sig(conn)", "aux_C_trsig", True)]
    for label, key, scalar in chans:
        if key not in corr:
            continue
        C = corr[key][cut:]
        if C.shape[0] < 4:
            continue
        if scalar:
            C = connected_scalar(C)
        r = jack_corr(C)
        c, dc, meff, dmeff = r["c"], r["dc"], r["meff"], r["dmeff"]
        snr0 = abs(c[0]) / (dc[0] + 1e-30)
        # first t-range where the signal stays > 2 sigma
        good = [t for t in range(1, min(12, len(c))) if abs(c[t]) > 2 * dc[t]]
        gmax = max(good) if good else 0
        print(f"  {label:14s} C0={c[0]:+.3e}+-{dc[0]:.1e} SNR0={snr0:6.1f} "
              f" signal>2sig to t={gmax}")
        line = "    meff: " + " ".join(
            f"{meff[t]:+.2f}({dmeff[t]:.2f})" for t in range(1, min(7, len(meff))))
        print(line)


def do_scan(pattern, therm=None, frac=0.3):
    dirs = sorted(glob.glob(pattern))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise SystemExit(f"no dirs match {pattern}")

    def lam_of(p):
        b = os.path.basename(p)
        try:
            return float(b.split("lam")[1][:6])
        except (IndexError, ValueError):
            return np.nan
    rows = []
    for cd in dirs:
        try:
            d = load_stream(cd)
        except SystemExit:
            continue
        cut, n = _therm_cut(d, therm, frac)
        x = d["plaq"][cut:]
        tau, dtau, W, Nn = tau_int(x)
        rows.append((lam_of(cd), os.path.basename(cd), n - cut,
                     np.nanmean(x), tau, dtau))
    rows.sort(key=lambda r: (np.isnan(r[0]), r[0]))
    print(f"{'lambda':>8s} {'Npost':>6s} {'<plaq>':>9s} {'tau_int(plaq,traj)':>20s}   stream")
    for lam, nm, npost, pm, tau, dtau in rows:
        flag = "" if npost >= 50 else "  [short]"
        print(f"{lam:8.3f} {npost:6d} {pm:9.5f} {tau:9.2f} +- {dtau:5.2f}{flag}   {nm}")
    print("\n(tau_int in trajectories; multiply by each stream's trajL for MD-time, "
          "by trajL*MDS*const for cost.  Need Npost >> 50 for reliable values.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cfgdir", nargs="?", help="stream cfg dir (contains hmc_diagnostics.*.h5)")
    ap.add_argument("--scan", metavar="GLOB", help="tabulate tau_int(plaq) across matching dirs")
    ap.add_argument("--therm", type=int, default=None, help="discard first N trajs")
    ap.add_argument("--frac", type=float, default=0.3, help="discard first fraction (default 0.3)")
    ap.add_argument("--trajL", type=float, default=None, help="trajectory length -> MD-time units")
    ap.add_argument("--corr", action="store_true", help="also analyze aux correlators")
    a = ap.parse_args()
    if a.scan:
        do_scan(a.scan, therm=a.therm, frac=a.frac)
    elif a.cfgdir:
        analyze_one(a.cfgdir, therm=a.therm, frac=a.frac, trajL=a.trajL, do_corr=a.corr)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
