#!/usr/bin/env python3
"""Survey aux VEVs across λ for any interesting condensates / channel structure.
Focus on what's NEW relative to the old picture:
  - per-channel σ_{ab}, π_{ab}, d_{ab}, n_{ab} wall means
  - per-quark Σ_DTXQCD via M48
  - check for diquark condensate ⟨n⟩, ⟨d⟩ at low λ
  - parity-odd VEVs ⟨p⟩, ⟨d⟩ (γ5-paired) by channel
"""
import glob, os, h5py, numpy as np

V3 = 64
NF = 2  # flavor

def load(d, max_traj=None):
    files = sorted(glob.glob(f'configs_2pt_dtxqcd_v2_lam{d}_sweep/hmc_diagnostics.*.h5'),
                   key=lambda p: int(os.path.basename(p).split('.')[1]))
    out = {'traj':[], 'plaq':[], 'vev_trminv':[],
           'wall_s':[], 'wall_p':[], 'wall_trsig':[], 'wall_trpi':[],
           'sig_ab':[], 'pi_ab':[], 'd_ab':[], 'n_ab':[]}
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
            # Matrix-level walls: each row is Nf²·T flat.  Reshape to (Ntraj, Nf², T)
            for prefix, key in [('sig_ab', 'aux_wall_sig_ab'),
                                ('pi_ab',  'aux_wall_pi_ab'),
                                ('d_ab',   'aux_wall_d_ab'),
                                ('n_ab',   'aux_wall_n_ab')]:
                if key in h:
                    arr = h[key][...]  # complex compound
                    ntraj = arr.shape[0]
                    re = arr['re'].reshape(ntraj, NF*NF, -1)  # (Ntraj, 4, T)
                    out[prefix].append(re)
    if out['sig_ab']:
        for k in ['sig_ab','pi_ab','d_ab','n_ab']:
            out[k] = np.concatenate(out[k], axis=0) if out[k] else np.zeros((0,4,8))
    return out

print(f"{'λ':>6} {'N':>4} {'plaq':>8} {'Σ_M48':>7} {'⟨s⟩':>8} {'⟨p⟩':>8} "
      f"{'⟨Trσ⟩':>8} {'⟨Trπ⟩':>8} ")
print('-'*70)
for L in [5.0, 2.0, 1.0, 0.5, 0.25, 0.1]:
    d = load(L)
    N = len(d['traj'])
    if N == 0:
        print(f'{L:>6.2f} {N:>4}   (no flush)')
        continue
    s    = np.array(d['wall_s'])    / V3
    p    = np.array(d['wall_p'])    / V3
    trsig = np.array(d['wall_trsig']) / V3
    trpi  = np.array(d['wall_trpi']) / V3
    plaq = np.array(d['plaq'])
    trm  = np.array(d['vev_trminv'])
    print(f'{L:>6.2f} {N:>4} {np.mean(plaq):>8.5f} {np.mean(trm):>7.3f} '
          f'{np.mean(s):>+8.3f} {np.mean(p):>+8.4f} {np.mean(trsig):>+8.4f} {np.mean(trpi):>+8.4f}')

# Per-channel σ_ab matrix: look for non-singlet bias.  Wall mean over space-time
# of each (a,b) flavor entry.
print()
print("=== σ_ab wall means (re part, vol-avg per traj, ensemble-averaged) ===")
print(f"{'λ':>6} {'N':>4}    σ_00     σ_01     σ_10     σ_11    sym (σ_01+σ_10)/2")
print('-'*75)
for L in [5.0, 2.0, 1.0, 0.5, 0.25, 0.1]:
    d = load(L)
    if not isinstance(d.get('sig_ab'), np.ndarray) or d['sig_ab'].size == 0:
        print(f'{L:>6.2f}   no matrix data'); continue
    arr = d['sig_ab']  # (Ntraj, 4=Nf², T)
    # Vol-avg = sum over T, normalize by V3·T = V
    means = np.mean(arr, axis=(0, 2)) / V3
    print(f'{L:>6.2f} {arr.shape[0]:>4}  {means[0]:+8.4f} {means[1]:+8.4f} '
          f'{means[2]:+8.4f} {means[3]:+8.4f}   {(means[1]+means[2])/2:+8.4f}')

print()
print("=== ⟨n_ab⟩ (parity-EVEN diquark): nonzero ⇒ baryon-no breaking ===")
print(f"{'λ':>6} {'N':>4}    n_00     n_01     n_10     n_11    sym")
print('-'*75)
for L in [5.0, 2.0, 1.0, 0.5, 0.25, 0.1]:
    d = load(L)
    if not isinstance(d.get('n_ab'), np.ndarray) or d['n_ab'].size == 0:
        print(f'{L:>6.2f}   no matrix data'); continue
    arr = d['n_ab']
    means = np.mean(arr, axis=(0, 2)) / V3
    print(f'{L:>6.2f} {arr.shape[0]:>4}  {means[0]:+8.4f} {means[1]:+8.4f} '
          f'{means[2]:+8.4f} {means[3]:+8.4f}   {(means[1]+means[2])/2:+8.4f}')

print()
print("=== ⟨d_ab⟩ (parity-ODD diquark): pseudoscalar diquark condensate ===")
print(f"{'λ':>6} {'N':>4}    d_00     d_01     d_10     d_11    sym")
print('-'*75)
for L in [5.0, 2.0, 1.0, 0.5, 0.25, 0.1]:
    d = load(L)
    if not isinstance(d.get('d_ab'), np.ndarray) or d['d_ab'].size == 0:
        print(f'{L:>6.2f}   no matrix data'); continue
    arr = d['d_ab']
    means = np.mean(arr, axis=(0, 2)) / V3
    print(f'{L:>6.2f} {arr.shape[0]:>4}  {means[0]:+8.4f} {means[1]:+8.4f} '
          f'{means[2]:+8.4f} {means[3]:+8.4f}   {(means[1]+means[2])/2:+8.4f}')
