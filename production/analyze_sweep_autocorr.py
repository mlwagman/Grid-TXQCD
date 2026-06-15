#!/usr/bin/env python3
"""Quick autocorr on aux VEVs across the λ sweep.

Reads hmc_diagnostics.*.h5 in each ensemble dir.  Computes integrated
autocorr τ_int via Madras-Sokal windowing (Wmax = first crossing).
Sample sizes are modest (10-30 trajs) so these are rough τ estimates.
"""
import glob, os, h5py, numpy as np

V3 = 64
LAMBDAS = [5.0, 2.0, 1.0, 0.5, 0.25, 0.1]

def load(d):
    files = sorted(glob.glob(f'configs_2pt_dtxqcd_v2_lam{d}_sweep/hmc_diagnostics.*.h5'),
                   key=lambda p: int(os.path.basename(p).split('.')[1]))
    out = {'traj':[], 'plaq':[], 'vev_trminv':[],
           'wall_s':[], 'wall_p':[], 'wall_trsig':[], 'wall_trpi':[]}
    for f in files:
        with h5py.File(f, 'r') as h:
            if 'aux_wall_s' not in h: continue
            out['traj'].extend(h['traj'][...].tolist())
            out['plaq'].extend(h['plaq'][...].tolist())
            out['vev_trminv'].extend(h['vev_trminv'][...].tolist())
            for k in ['aux_wall_s','aux_wall_p','aux_wall_trsig','aux_wall_trpi']:
                arr = h[k][...]
                vals = np.mean(arr['re'], axis=1).tolist()
                outkey = 'wall_' + k.split('_')[-1]
                out[outkey].extend(vals)
    return out

def tau_int(x, Wmax=None):
    """Madras-Sokal integrated autocorr (rho window until first negative)."""
    n = len(x)
    if n < 4: return 0.5, n
    xc = x - np.mean(x)
    var = float(np.var(xc, ddof=0))
    if var == 0: return 0.5, 0
    if Wmax is None: Wmax = n // 4
    rho = []
    for k in range(Wmax + 1):
        c = float(np.mean(xc[:n-k] * xc[k:])) / var
        if k > 0 and c < 0:
            break
        rho.append(c)
    tau = 0.5 + sum(rho[1:])
    return tau, len(rho) - 1

print(f"{'λ':>6} {'N':>5} {'plaq mean':>10} {'<s>':>10} {'<p>':>9} {'<Trπ>':>9} "
      f"{'<vev_trm>':>11} {'τ(<p>)':>8} {'τ(<s>)':>8}")
print('-' * 95)
for L in LAMBDAS:
    d = load(L)
    N = len(d['traj'])
    if N == 0:
        print(f'{L:>6.2f} {N:>5}    (no flushed h5 yet)')
        continue
    s    = np.array(d['wall_s'])    / V3
    p    = np.array(d['wall_p'])    / V3
    trpi = np.array(d['wall_trpi']) / V3
    plaq = np.array(d['plaq'])
    trm  = np.array(d['vev_trminv'])
    t_p, _ = tau_int(p)
    t_s, _ = tau_int(s)
    print(f'{L:>6.2f} {N:>5} {np.mean(plaq):>10.5f} {np.mean(s):>+10.3f} '
          f'{np.mean(p):>+9.4f} {np.mean(trpi):>+9.4f} {np.mean(trm):>11.4f} '
          f'{t_p:>8.2f} {t_s:>8.2f}')
