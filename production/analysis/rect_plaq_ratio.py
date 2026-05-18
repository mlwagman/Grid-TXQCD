#!/usr/bin/env python3
"""Standalone (no Grid, no GPU) plaquette + 1x2 rectangle reader.

Purpose: test whether our HMC ensemble and chroma's reference ensemble share
the same gauge action by comparing the equilibrium rectangle/plaquette ratio
(an action fingerprint, independent of absolute normalization / fermion sea).

Validation discipline: the NERSC header records PLAQUETTE exactly.  We require
our computed avgPlaquette to reproduce it (to ~1e-6) before trusting the
rectangle.  Reconstruction/index convention is auto-resolved by trying the
small set of (matrix transpose) x (link order) variants and keeping the one
that matches the recorded plaq.

avgPlaquette / avgRectangle here use the cold->1 normalization:
  P = <(1/Nc) Re Tr [ U_mu(x) U_nu(x+mu) U_mu(x+nu)^dag U_nu(x)^dag ]>
  R = <(1/Nc) Re Tr [ 2x1 Wilson loop ]>            (averaged over mu!=nu)
"""
import sys, struct, numpy as np

Nc = 3

def _su3_from_doubles(buf, nlinks):
    """buf: big-endian float64 array, 18 per link (9 complex, row-major).
    Returns complex128 array (nlinks, 3, 3)."""
    a = np.frombuffer(buf, dtype='>f8').astype(np.float64)
    a = a.reshape(nlinks, 9, 2)
    m = a[:, :, 0] + 1j * a[:, :, 1]
    return m.reshape(nlinks, 3, 3)

def read_nersc(path):
    with open(path, 'rb') as f:
        raw = f.read()
    end = raw.index(b'END_HEADER\n') + len(b'END_HEADER\n')
    hdr = raw[:end].decode('ascii', 'replace')
    H = {}
    for line in hdr.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            H[k.strip()] = v.strip()
    Lx = int(H['DIMENSION_1']); Ly = int(H['DIMENSION_2'])
    Lz = int(H['DIMENSION_3']); Lt = int(H['DIMENSION_4'])
    assert H['FLOATING_POINT'] == 'IEEE64BIG', H['FLOATING_POINT']
    assert H['DATATYPE'] == '4D_SU3_GAUGE_3x3', H['DATATYPE']
    vol = Lx * Ly * Lz * Lt
    body = raw[end:]
    links = _su3_from_doubles(body[:vol * 4 * 18 * 8], vol * 4)
    # Grid lexicographic: idx = x + Lx*(y + Ly*(z + Lz*t)), link order mu=0..3
    U = links.reshape(Lt, Lz, Ly, Lx, 4, 3, 3)
    return U, (Lx, Ly, Lz, Lt), float(H['PLAQUETTE'])

def read_ildg(path):
    with open(path, 'rb') as f:
        while True:
            h = f.read(144)
            if len(h) < 144: raise RuntimeError("no ildg-binary-data")
            magic, = struct.unpack('>I', h[0:4])
            assert magic == 0x456789ab
            nb, = struct.unpack('>q', h[8:16])
            rt = h[16:144].split(b'\x00')[0].decode('ascii', 'replace')
            pad = (8 - (nb % 8)) % 8
            if rt == 'ildg-binary-data':
                data = f.read(nb)
                break
            f.seek(nb + pad, 1)
    # ILDG standard: idx = x + Lx*(y+Ly*(z+Lz*t)), mu=0..3=x,y,z,t, big-endian
    # dims from earlier inspection: 16 16 16 48
    Lx = Ly = Lz = 16; Lt = 48
    vol = Lx * Ly * Lz * Lt
    links = _su3_from_doubles(data[:vol * 4 * 18 * 8], vol * 4)
    U = links.reshape(Lt, Lz, Ly, Lx, 4, 3, 3)
    return U, (Lx, Ly, Lz, Lt)

def mul(a, b):       return np.matmul(a, b)
def dag(a):          return np.conjugate(np.swapaxes(a, -1, -2))
def shift(A, mu, n=1):
    # axes: (t,z,y,x,...)  mu 0,1,2,3 = x,y,z,t -> axis 3,2,1,0
    ax = {0: 3, 1: 2, 2: 1, 3: 0}[mu]
    return np.roll(A, -n, axis=ax)

def avg_plaq(U):
    Lt, Lz, Ly, Lx = U.shape[:4]
    nsite = Lt * Lz * Ly * Lx
    tot = 0.0; cnt = 0
    Um = [U[..., mu, :, :] for mu in range(4)]
    for mu in range(4):
        for nu in range(mu):
            a = Um[mu]
            b = shift(Um[nu], mu)
            c = dag(shift(Um[mu], nu))
            d = dag(Um[nu])
            P = mul(mul(a, b), mul(c, d))
            tr = np.trace(P, axis1=-2, axis2=-1).real
            tot += tr.sum(); cnt += nsite
    return tot / cnt / Nc

def avg_rect(U):
    """2x1 Wilson loop, averaged over all ordered mu!=nu (Nd(Nd-1) orient.)."""
    Lt, Lz, Ly, Lx = U.shape[:4]
    nsite = Lt * Lz * Ly * Lx
    tot = 0.0; cnt = 0
    Um = [U[..., mu, :, :] for mu in range(4)]
    for mu in range(4):
        for nu in range(4):
            if mu == nu: continue
            # path: mu, mu, nu, -mu, -mu, -nu  (2 long in mu, 1 in nu)
            s = Um[mu]
            s = mul(s, shift(Um[mu], mu))                       # U_mu(x)U_mu(x+mu)
            s = mul(s, shift(shift(Um[nu], mu), mu))            # U_nu(x+2mu)
            s = mul(s, dag(shift(shift(Um[mu], nu), mu)))       # U_mu(x+nu+mu)^d
            s = mul(s, dag(shift(Um[mu], nu)))                  # U_mu(x+nu)^d
            s = mul(s, dag(Um[nu]))                             # U_nu(x)^d
            tr = np.trace(s, axis1=-2, axis2=-1).real
            tot += tr.sum(); cnt += nsite
    return tot / cnt / Nc

def analyze(label, U, target_plaq=None):
    variants = {
        'asis':            lambda x: x,
        'transpose':       lambda x: np.swapaxes(x, -1, -2),
        'conj':            lambda x: np.conjugate(x),
        'dagger':          lambda x: np.conjugate(np.swapaxes(x, -1, -2)),
    }
    best = None
    for name, fn in variants.items():
        Uv = fn(U)
        p = avg_plaq(Uv)
        ok = (target_plaq is None) or abs(p - target_plaq) < 1e-5
        tag = ''
        if target_plaq is not None:
            tag = '  <-- MATCHES recorded' if ok else f'  (recorded {target_plaq:.9f})'
        print(f"  [{label}] variant={name:9s}  plaq={p:.9f}{tag}")
        if ok and best is None:
            best = (name, Uv, p)
    if best is None:
        if target_plaq is not None:
            print(f"  [{label}] !! no variant matched recorded plaq — parse/convention wrong")
            return None
        best = ('asis', U, avg_plaq(U))
    name, Uv, p = best
    r = avg_rect(Uv)
    print(f"  [{label}] CHOSEN variant={name}  plaq={p:.9f}  rect={r:.9f}  rect/plaq={r/p:.6f}")
    return p, r, r / p

if __name__ == '__main__':
    nersc = sys.argv[1] if len(sys.argv) > 1 else \
        'cfgs/qcd_s702_nf2p1_mdscan_mds10_fork_v2/ckpoint_lat.840'
    ildg = sys.argv[2] if len(sys.argv) > 2 else \
        '/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime'

    print(f"=== OUR ensemble (NERSC): {nersc} ===")
    Un, dims, prec = read_nersc(nersc)
    rn = analyze('ours', Un, target_plaq=prec)

    print(f"\n=== CHROMA reference (ILDG): {ildg} ===")
    Uc, dimc = read_ildg(ildg)
    rc = analyze('chroma', Uc, target_plaq=None)  # validate via plaq ~0.513-0.514

    if rn and rc:
        print("\n=== VERDICT ===")
        print(f"  ours  : plaq={rn[0]:.6f}  rect={rn[1]:.6f}  rect/plaq={rn[2]:.6f}")
        print(f"  chroma: plaq={rc[0]:.6f}  rect={rc[1]:.6f}  rect/plaq={rc[2]:.6f}")
        dr = abs(rn[2] - rc[2]) / rc[2] * 100
        print(f"  rect/plaq ratio difference: {dr:.3f}%")
        print("  same gauge action  -> ratios agree (<~0.1%); 0.0008 plaq offset is fermion-sector")
        print("  diff gauge action  -> ratios disagree; gauge convention IS the bug")
