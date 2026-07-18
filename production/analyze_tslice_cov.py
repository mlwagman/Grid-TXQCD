#!/usr/bin/env python3
"""t-slice covariance analysis of the stochastic reweighting estimator.

Parses TSLICE lines (per-sample per-timeslice partials s_t of the
remainder estimate; DTXQCD_STOCHPF_TSLICE=1) and forms the sample
covariance  C(t,t') = Cov(s_t, conj s_t'),  which for circular gaussian
noise estimates the t-resolved Frobenius mass of the remainder operator:
    C(t,t') ~ F(t,t') = sum_{i in t, j in t'} |A_ij|^2   (A = S~^-1 P dS~)
Every t-dilution scheme's variance is a partial sum of F:
    per-sweep Var(scheme) = sum_b sum_{t,t' in b} F(t,t'),
    equal-budget sigma^2  = D * per-sweep Var.
So one undiluted run predicts blocks, combs, and the optimum.

Usage: python3 analyze_tslice_cov.py <log> [T=48]
Prints: F(Dt) decay profile, per-scheme equal-budget sigma predictions
(vs the undiluted baseline), heavy-tail diagnostics.
"""
import math
import re
import sys

log = sys.argv[1]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 48

re_line = re.compile(r"TSLICE n=(\d+) (re|im)((?:\s+-?[0-9.eE+-]+)+)")
samples = {}
for line in open(log, errors="ignore"):
    m = re_line.search(line)
    if not m:
        continue
    n, part = int(m.group(1)), m.group(2)
    vals = [float(x) for x in m.group(3).split()]
    samples.setdefault(n, {})[part] = vals

S = []
for n in sorted(samples):
    d = samples[n]
    if "re" in d and "im" in d and len(d["re"]) == T:
        S.append([complex(a, b) for a, b in zip(d["re"], d["im"])])
N = len(S)
print(f"samples: {N}  T: {T}")

mean = [sum(s[t] for s in S) / N for t in range(T)]
C = [[0.0] * T for _ in range(T)]
for s in S:
    for t in range(T):
        dt_ = s[t] - mean[t]
        for u in range(T):
            C[t][u] += (dt_ * (s[u] - mean[u]).conjugate()).real
for t in range(T):
    for u in range(T):
        C[t][u] /= (N - 1)

tot = sum(sum(row) for row in C)
print(f"total per-sample Var (= sigma^2 undiluted): {tot:.4e}"
      f"  sigma = {math.sqrt(max(tot, 0)):.1f}")

# heavy-tail diagnostic on the per-sample totals
tots = [sum(s).real for s in S]
mu = sum(tots) / N
m2 = sum((x - mu) ** 2 for x in tots) / N
m4 = sum((x - mu) ** 4 for x in tots) / N
print(f"per-sample total: mean {mu:.2f} sigma {math.sqrt(m2):.1f} "
      f"kurtosis {m4 / max(m2 * m2, 1e-300):.1f} (gaussian=3; large => "
      f"error bars at small N are optimistic)")

# F(Dt) profile (periodic average)
print("\nF(Dt) profile (avg over t, normalized to Dt=0):")
f0 = sum(C[t][t] for t in range(T)) / T
for d in range(0, min(13, T // 2 + 1)):
    fd = sum(C[t][(t + d) % T] for t in range(T)) / T
    print(f"  Dt={d:2d}  {fd / max(f0, 1e-300):+8.4f}")

def scheme_sigma(blocks):
    D = len(blocks)
    v = 0.0
    for b in blocks:
        bs = set(b)
        v += sum(C[t][u] for t in b for u in bs)
    return math.sqrt(max(D * v, 0.0))

base = math.sqrt(max(tot, 0))
print("\nequal-budget sigma predictions (ratio vs undiluted):")
print(f"  {'scheme':<16} {'sigma_eff':>10} {'ratio':>7}")
print(f"  {'undiluted':<16} {base:10.1f} {1.0:7.2f}")
for k in [2, 3, 4, 6, 8, 12]:
    if T % k:
        continue
    blocks = [[t for t in range(T) if t % k == r] for r in range(k)]
    s = scheme_sigma(blocks)
    print(f"  {'comb k=' + str(k):<16} {s:10.1f} {s / base:7.2f}")
for B in [24, 12, 8, 6, 4, 2]:
    if T % B:
        continue
    blocks = [list(range(b * B, (b + 1) * B)) for b in range(T // B)]
    s = scheme_sigma(blocks)
    print(f"  {'block B=' + str(B):<16} {s:10.1f} {s / base:7.2f}")
blocks = [[t] for t in range(T)]
print(f"  {'full t (D=T)':<16} {scheme_sigma(blocks):10.1f} "
      f"{scheme_sigma(blocks) / base:7.2f}")
