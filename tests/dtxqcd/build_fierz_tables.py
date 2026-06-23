#!/usr/bin/env python3
"""Build LaTeX tables, splitting prod into non-EO and EO subphases."""
import re, math

def parse_plaq(path, tag):
    pat = re.compile(rf'\[{re.escape(tag)} plaq traj (\d+)\] ([\d\.e\+\-]+)')
    return [(int(m.group(1)), float(m.group(2)))
            for line in open(path) for m in [pat.search(line)] if m]

def parse_fierz_avg(path, phase_filter=None):
    """Parse FierzAvg lines. phase_filter is a (start_marker, end_marker)
    tuple — only return lines between matching markers."""
    pat = re.compile(
        r'\[FierzAvg traj (\d+)\]\s+Σ_DTX=([\d\.e\+\-]+)\s+Σ_W=([\d\.e\+\-]+)\s+ratio=([\d\.e\+\-]+)\s+⟨Tr s⟩=([\d\.e\+\-]+)\s+⟨Tr σ⟩=([\d\.e\+\-]+)')
    samples = []
    in_phase = (phase_filter is None)
    start_m, end_m = (None, None) if phase_filter is None else phase_filter
    for line in open(path):
        if start_m and re.search(start_m, line):
            in_phase = True; continue
        if end_m and re.search(end_m, line):
            in_phase = False; continue
        if not in_phase: continue
        m = pat.search(line)
        if m:
            samples.append(tuple(float(m.group(i)) for i in range(2,7)))
    return samples  # list of (sigdtx, sigw, ratio, tr_s, tr_sigma)

def block_se(xs, binsize=20):
    nb = len(xs) // binsize
    if nb < 2: return (sum(xs)/len(xs) if xs else 0, 0, nb)
    blocks = [sum(xs[i*binsize:(i+1)*binsize])/binsize for i in range(nb)]
    m = sum(blocks)/nb
    v = sum((b-m)**2 for b in blocks)/(nb-1)
    return m, math.sqrt(v/nb), nb

B = "/tmp/dtxqcd_hmc_scout/"

# Each entry: (display name, log, n_therm, qcd_plaq_tag, dtxqcd_plaq_tag, fierz_phase_filter)
runs = [
  ("csw=0 RHMC (orig)", B+"fierz_full_qcd_600.log",       100, None,"DTXQCD", None),
  ("csw=0 HMC short",   B+"fierz_full_qcd_csw0_hmc.log",  200, "QCD","DTXQCD", None),
  ("csw=0 HMC long",    B+"fierz_csw0_hmc_long.log",      500, "QCD","DTXQCD", None),
  ("csw=1 HMC",         B+"fierz_full_qcd_csw1_v3.log",   200, "QCD","DTXQCD", None),
  ("csw=1 RHMC",        B+"fierz_full_qcd_csw1_rhmc.log", 200, "QCD","DTXQCD", None),
  ("prod HMC non-EO",   B+"fierz_prod_full_v2.log",       200, "QCD","DTXQCD-noneo",
   (r"Phase 2 DTXQCD non-EO", r"Phase 3 DTXQCD EO")),
  ("prod HMC EO",       B+"fierz_prod_full_v2.log",       200, "QCD","DTXQCD-eo",
   (r"Phase 3 DTXQCD EO", None)),
  ("prod RHMC non-EO",  B+"fierz_prod_full_rhmc.log",     200, "QCD","DTXQCD-noneo",
   (r"Phase 2 DTXQCD non-EO", r"Phase 3 DTXQCD EO")),
  ("prod RHMC EO",      B+"fierz_prod_full_rhmc.log",     200, "QCD","DTXQCD-eo",
   (r"Phase 3 DTXQCD EO", None)),
]

def fmt(x, se, decimals=5):
    err_scale = 10**decimals
    return f"${x:.{decimals}f}({int(round(se*err_scale)):2d})$"

print(r"% Auto-generated Fierz comparison tables, 4^4 lattice, lambda=10")
print(r"% Errors are bin=20 block-averaged for plaq, bin=4 for FierzAvg samples")
print()

# ---------- Table 1: plaq ----------
print(r"\begin{table}")
print(r"\caption{Plaquette comparison.  Bin=20 block averaging absorbs the per-traj autocorrelation in the gauge action.  $\Delta_\mathrm{plaq}=\langle P\rangle_\mathrm{DTXQCD}-\langle P\rangle_\mathrm{QCD}$; consistency with zero is the exact DTXQCD-QCD Fierz requirement.}")
print(r"\begin{tabular}{lrcccc}")
print(r"\hline")
print(r"variant & $N_\mathrm{prod}$ & $\langle P\rangle_\mathrm{QCD}$ & $\langle P\rangle_\mathrm{DTXQCD}$ & $\Delta_\mathrm{plaq}$ & $\sigma$ \\")
print(r"\hline")
for (name, path, nt, qt, dt, _) in runs:
    try:
        if qt is None:
            # original 1-phase: same log has only DTXQCD samples
            continue
        qprod = [v for (t,v) in parse_plaq(path, qt) if t >= nt]
        dprod = [v for (t,v) in parse_plaq(path, dt) if t >= nt]
        if not qprod or not dprod: continue
        mq, sq, _ = block_se(qprod, 20)
        md, sd, _ = block_se(dprod, 20)
        d = md - mq; se = math.sqrt(sq**2 + sd**2)
        sig = abs(d)/se if se>0 else 0
        n = min(len(qprod), len(dprod))
        print(rf"{name} & {n} & {fmt(mq, sq)} & {fmt(md, sd)} & {fmt(d, se)} & ${sig:.2f}$ \\")
    except FileNotFoundError:
        pass
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
print()

# ---------- Table 2: quark VEV Sigma_DTX vs Sigma_W ----------
print(r"\begin{table}")
print(r"\caption{Quark condensate $\Sigma = \langle\bar q q\rangle$ comparison (FierzAvg observer with $N_\mathrm{noise}=16$ per measurement, stride 5 trajs).  $\Sigma_\mathrm{DTX}$ is the in-line DTXQCD stochastic estimate; $\Sigma_W$ is plain Wilson on the same gauge configuration.  By Fierz they must agree to stochastic precision.  Bin=4 block averaging over the strided samples.}")
print(r"\begin{tabular}{lrcccc}")
print(r"\hline")
print(r"variant & $N_\mathrm{meas}$ & $\Sigma_\mathrm{DTX}$ & $\Sigma_W$ & ratio $-1$ & $\sigma$ \\")
print(r"\hline")
for (name, path, nt, qt, dt, pf) in runs:
    try:
        s = parse_fierz_avg(path, pf)
        if not s: continue
        sigdtx = [x[0] for x in s]; sigw = [x[1] for x in s]; ratios = [x[2] for x in s]
        ms,  ses, _ = block_se(sigdtx, 4)
        mw,  sew, _ = block_se(sigw,   4)
        mr,  ser, _ = block_se(ratios, 4)
        dev = mr - 1.0
        sig_lvl = abs(dev)/ser if ser>0 else 0
        print(rf"{name} & {len(s)} & {fmt(ms,ses)} & {fmt(mw,sew)} & {fmt(dev,ser)} & ${sig_lvl:.2f}$ \\")
    except FileNotFoundError:
        pass
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
print()

# ---------- Table 3: Scalar aux VEVs vs saddle ----------
print(r"\begin{table}")
print(r"\caption{Scalar auxiliary VEVs $\langle\mathrm{Tr}\,s\rangle$ and $\langle\mathrm{Tr}\,\sigma\rangle$ compared to the saddle prediction $N_f\Sigma_W/\lambda^2$ ($N_f=2$, $\lambda=10$, so saddle $\approx 0.02\,\Sigma_W$).  Deviations from saddle are stochastic-noise-dominated at large $\lambda$.}")
print(r"\begin{tabular}{lrcccc}")
print(r"\hline")
print(r"variant & $N_\mathrm{meas}$ & $\langle\mathrm{Tr}\,s\rangle$ & $\langle\mathrm{Tr}\,\sigma\rangle$ & $N_f\Sigma_W/\lambda^2$ & ${\rm dev}_s$, ${\rm dev}_\sigma$ \\")
print(r"\hline")
LAMBDA = 10.0; NF = 2.0
for (name, path, nt, qt, dt, pf) in runs:
    try:
        s = parse_fierz_avg(path, pf)
        if not s: continue
        sigw = [x[1] for x in s]; trs = [x[3] for x in s]; trsi = [x[4] for x in s]
        mw,  _,    _ = block_se(sigw, 4)
        msd      = NF * mw / (LAMBDA*LAMBDA)
        ms,  ses,  _ = block_se(trs,  4)
        msi, sesi, _ = block_se(trsi, 4)
        ds = ms - msd; dsi = msi - msd
        print(rf"{name} & {len(s)} & {fmt(ms,ses,4)} & {fmt(msi,sesi,4)} & ${msd:.5f}$ & $s:{ds:+.4f}$,\,$\sigma:{dsi:+.4f}$ \\")
    except FileNotFoundError:
        pass
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
