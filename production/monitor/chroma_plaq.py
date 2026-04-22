#!/usr/bin/env python3
"""Compute plaquette from chroma ILDG/SciDAC LIME gauge configs.

Minimal LIME reader — SciDAC records have an 144-byte header:
  magic(4) + version(2) + reserved(2) + data_length(8) + type(128)
  followed by data_length bytes, padded to 8-byte boundary.

The gauge field is stored in the "ildg-binary-data" record as a flat array
of QDP_D3_ColorMatrix, ordered lexicographically (x fastest, t slowest),
with 4 links per site (μ=0..3).  Double precision (16 bytes per complex),
so 9 × 16 = 144 bytes per link, big-endian on disk.
"""

import argparse
import numpy as np
import struct
import sys

LIME_MAGIC = b"\x45\x67\x89\xab"


def read_lime_records(path):
    """Yield (record_type, payload_bytes) for each LIME record in a file."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(144)
            if len(hdr) < 144:
                return
            if hdr[:4] != LIME_MAGIC:
                raise ValueError(f"Not a LIME file: bad magic at offset {f.tell()-144}")
            # big-endian 8-byte data length at offset 8
            data_len = struct.unpack(">Q", hdr[8:16])[0]
            rec_type = hdr[16:].split(b"\x00", 1)[0].decode("ascii", "replace")
            data = f.read(data_len)
            # pad to 8-byte boundary
            pad = (8 - data_len % 8) % 8
            f.read(pad)
            yield rec_type, data


def extract_gauge(path, L, T):
    """Extract the ildg-binary-data record and reshape to (T,Z,Y,X,4,3,3) complex."""
    for rt, data in read_lime_records(path):
        if rt == "ildg-binary-data":
            # big-endian double precision
            n_doubles = len(data) // 8
            arr = np.frombuffer(data, dtype=">f8", count=n_doubles).astype(np.float64)
            # 4 links × 9 complex × 2 reals = 72 doubles per site
            sites = L**3 * T
            expected = sites * 72
            if arr.size != expected:
                raise ValueError(
                    f"Expected {expected} doubles for {L}^3x{T} × 4 links × DP complex, "
                    f"got {arr.size}"
                )
            # Real/imag interleaved → complex
            cmplx = arr.view(np.complex128 if False else None)
            cmplx = arr[::2] + 1j * arr[1::2]  # pair consecutive as Re/Im
            # Layout: site-lex order × μ × color_row × color_col
            U = cmplx.reshape(T, L, L, L, 4, 3, 3)
            return U
    raise ValueError("No ildg-binary-data record found")


def plaquette(U):
    """Average plaquette = (1/6Vol) Σ_{μ<ν} Re Tr[U_μ(x) U_ν(x+μ) U_μ†(x+ν) U_ν†(x)]."""
    T, Lz, Ly, Lx, _, _, _ = U.shape
    total = 0.0
    ndirs = 0
    # axes: U[t, z, y, x, mu, i, j]  with mu order (x,y,z,t) = (0,1,2,3) per chroma
    # shifts: axis mapping mu → (x=3, y=2, z=1, t=0) for numpy.roll
    axis_of = {0: 3, 1: 2, 2: 1, 3: 0}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            Umu = U[..., mu, :, :]
            Unu = U[..., nu, :, :]
            # U_nu(x+mu): shift nu-field by -1 along the mu axis
            Unu_shift_mu = np.roll(Unu, -1, axis=axis_of[mu])
            # U_mu(x+nu): shift mu-field by -1 along the nu axis
            Umu_shift_nu = np.roll(Umu, -1, axis=axis_of[nu])
            # Plaquette = Umu · Unu(x+mu) · Umu†(x+nu) · Unu†
            P = np.einsum("...ij,...jk,...lk,...ml->...im",
                          Umu, Unu_shift_mu,
                          np.conj(Umu_shift_nu), np.conj(Unu))
            # take trace over the i==m and real part
            tr = np.einsum("...ii->...", P).real / 3.0
            total += tr.mean()
            ndirs += 1
    return total / ndirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="chroma LIME gauge config(s)")
    ap.add_argument("--L", type=int, default=16, help="spatial lattice size")
    ap.add_argument("--T", type=int, default=48, help="temporal lattice size")
    args = ap.parse_args()
    for f in args.files:
        U = extract_gauge(f, args.L, args.T)
        p = plaquette(U)
        print(f"{f}\tplaq = {p:.8f}")


if __name__ == "__main__":
    main()
