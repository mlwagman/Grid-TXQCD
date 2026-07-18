#!/usr/bin/env python3
"""F(t,t') map analysis -- the t-dilution scheme designer.

Parses FMAP lines (DTXQCD_STOCHPF_TSLICE=2: per probe pair (z,z'), one
solve per masked slice t' of z', slice-t partials f(t,t') of <z, A z'|t'>).
For circular gaussians with E|z_i|^2 = 2:
    F(t,t') = mean over pairs of |f(t,t')|^2 / 4
            = sum_{i in t, j in t'} |A_ij|^2,
the exact building block of every t-dilution variance:
    per-sweep Var(scheme) = sum_b sum_{t,t' in b} F(t,t'),
    equal-budget sigma^2  = D * per-sweep Var;   undiluted = sum_all F.
Cross-check: sqrt(sum_all F) must match the measured undiluted
per-sample sigma (~540 at ep=0.4).

Usage: python3 analyze_fmap.py <log> [<log2> ...]
"""
import math
import re
import sys

T = 48
re_line = re.compile(r"FMAP h=(\d+) tp=(\d+) (re|im)((?:\s+-?[0-9.eE+-]+)+)")
pairs = {}
for path in sys.argv[1:]:
    for line in open(path, errors="ignore"):
        m = re_line.search(line)
        if not m:
            continue
        key = (path, int(m.group(1)), int(m.group(2)))
        vals = [float(x) for x in m.group(4).split()]
        pairs.setdefault(key, {})[m.group(3)] = vals

F = [[0.0] * T for _ in range(T)]
cnt = [[0] * T for _ in range(T)]
npairs = set()
for (path, h, tp), d in pairs.items():
    if "re" not in d or "im" not in d or len(d["re"]) != T:
        continue
    npairs.add((path, h))
    for t in range(T):
        F[t][tp] += (d["re"][t] ** 2 + d["im"][t] ** 2) / 4.0
        cnt[t][tp] += 1
np_ = len(npairs)
print(f"probe pairs: {np_}")
for t in range(T):
    for tp in range(T):
        if cnt[t][tp]:
            F[t][tp] /= cnt[t][tp]

tot = sum(sum(r) for r in F)
print(f"sum F = {tot:.4e}  -> undiluted per-sample sigma = "
      f"{math.sqrt(max(tot, 0)):.1f}  (measured ~500-540: cross-check)")

f0 = sum(F[t][t] for t in range(T)) / T
print("\nF(Dt) profile (avg over t, normalized to Dt=0):")
for d in range(0, 25):
    fd = sum(F[t][(t + d) % T] for t in range(T)) / T
    print(f"  Dt={d:2d}  {fd / max(f0, 1e-300):8.4f}")

def scheme_sigma(blocks):
    D = len(blocks)
    v = sum(F[t][u] for b in blocks for t in b for u in b)
    return math.sqrt(max(D * v, 0.0))

base = math.sqrt(max(tot, 0))
print("\nequal-budget sigma predictions (ratio vs undiluted):")
print(f"  {'scheme':<16} {'sigma_eff':>10} {'ratio':>7}")
print(f"  {'undiluted':<16} {base:10.1f} {1.00:7.2f}")
for k in [2, 3, 4, 6, 8, 12, 16, 24]:
    if T % k:
        continue
    blocks = [[t for t in range(T) if t % k == r] for r in range(k)]
    s = scheme_sigma(blocks)
    print(f"  {'comb k=' + str(k):<16} {s:10.1f} {s / base:7.2f}")
for B in [24, 12, 8, 6, 4, 3, 2]:
    blocks = [list(range(b * B, (b + 1) * B)) for b in range(T // B)]
    s = scheme_sigma(blocks)
    print(f"  {'block B=' + str(B):<16} {s:10.1f} {s / base:7.2f}")
s = scheme_sigma([[t] for t in range(T)])
print(f"  {'full t (D=T)':<16} {s:10.1f} {s / base:7.2f}")
