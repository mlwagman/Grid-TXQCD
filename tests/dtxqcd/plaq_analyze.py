#!/usr/bin/env python3
"""Per-traj plaq thermalization + binned + autocorrelation analysis."""
import re, sys, math

def read_plaq(path, tag):
    pat = re.compile(rf'\[{re.escape(tag)} plaq traj (\d+)\] ([\d\.e\+\-]+)')
    samples = []
    for line in open(path):
        m = pat.search(line)
        if m: samples.append((int(m.group(1)), float(m.group(2))))
    return samples

def block_mean_se(xs, binsize):
    # Block-average over binsize then take mean / SE of the block-means.
    nb = len(xs) // binsize
    if nb < 2: return 0,0,nb
    blocks = [sum(xs[i*binsize:(i+1)*binsize])/binsize for i in range(nb)]
    m = sum(blocks)/nb
    v = sum((b-m)**2 for b in blocks)/(nb-1)
    se = math.sqrt(v/nb)
    return m, se, nb

def autocorr_tau(xs, max_lag=None):
    n = len(xs); m = sum(xs)/n
    var = sum((x-m)**2 for x in xs)/n
    if var == 0 or n < 10: return 0.5
    if max_lag is None: max_lag = min(n//4, 200)
    tau = 0.5
    for t in range(1, max_lag):
        c = sum((xs[i]-m)*(xs[i+t]-m) for i in range(n-t))/((n-t)*var)
        if c < 0: break
        tau += c
    return tau

def analyze(path, tag, n_therm, name=""):
    s = read_plaq(path, tag)
    if not s:
        print(f"  {name}: no '{tag}' samples"); return None
    prod = [v for (t,v) in s if t >= n_therm]
    if len(prod) < 20:
        print(f"  {name}: only N_prod={len(prod)} — too short")
        return None
    nq = len(prod)//4
    quarters = [sum(prod[i*nq:(i+1)*nq])/nq for i in range(4)] if nq else []
    naive_m = sum(prod)/len(prod)
    naive_se = math.sqrt(sum((x-naive_m)**2 for x in prod)/(len(prod)*(len(prod)-1)))
    tau = autocorr_tau(prod)
    bin10_m, bin10_se, nb10 = block_mean_se(prod, 10)
    bin20_m, bin20_se, nb20 = block_mean_se(prod, 20)
    print(f"  {name} [{tag}]: N_prod={len(prod)}  τ_int≈{tau:.1f}")
    print(f"    quarters: " + " ".join(f"{q:.5f}" for q in quarters))
    print(f"    naive  : {naive_m:.6f} ± {naive_se:.6f}")
    print(f"    bin=10 : {bin10_m:.6f} ± {bin10_se:.6f}  (Nb={nb10})")
    print(f"    bin=20 : {bin20_m:.6f} ± {bin20_se:.6f}  (Nb={nb20})")
    return (bin20_m, bin20_se)  # use bin=20 as the headline

print("Effective error budget — block-binned (bin=20 = headline)\n")

specs = [
  # (log, qcd_tag, dtxqcd_tag, n_therm, name)
  ("fierz_full_qcd_600.log",         "QCD","DTXQCD",100,"csw=0 RHMC (orig 1-phase)"),
  ("fierz_full_qcd_csw0_hmc.log",    "QCD","DTXQCD",200,"csw=0 HMC"),
  ("fierz_csw0_hmc_long.log","QCD","DTXQCD",500,"csw=0 HMC long"),
  ("fierz_full_qcd_csw1_v3.log",     "QCD","DTXQCD",200,"csw=1 HMC"),
  ("fierz_full_qcd_csw1_rhmc.log",   "QCD","DTXQCD",200,"csw=1 RHMC"),
  ("fierz_prod_full_v2.log",         "QCD","DTXQCD-noneo",200,"prod HMC (noneo)"),
  ("fierz_prod_full_v2.log",         "QCD","DTXQCD-eo",   200,"prod HMC (eo)"),
  ("fierz_prod_full_rhmc.log",       "QCD","DTXQCD-noneo",200,"prod RHMC (noneo)"),
  ("fierz_prod_full_rhmc.log",       "QCD","DTXQCD-eo",   200,"prod RHMC (eo)"),
  ("fierz_prod_full_hmc_long.log",   "QCD","DTXQCD-noneo",500,"prod HMC long (noneo)"),
]
base = "/tmp/dtxqcd_hmc_scout/"
for (logname, qcd_tag, dtx_tag, nt, name) in specs:
    path = base + logname
    print(f"--- {name} ---")
    rq = analyze(path, qcd_tag, nt, "QCD")
    rd = analyze(path, dtx_tag, nt, "DTXQCD")
    if rq and rd:
        d = rd[0]-rq[0]; se = math.sqrt(rq[1]**2+rd[1]**2)
        sig = abs(d)/se if se>0 else 0
        print(f"    Δplaq (bin=20) = {d:+.6f} ± {se:.6f}   ({sig:.2f} σ)")
    print()
