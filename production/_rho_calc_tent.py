import math
import re

for ep in ["0.16", "0.47"]:
    A, B = [], []
    path = ("/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/"
            "rwf_rho_tent_ep%s.log" % ep)
    try:
        fh = open(path, errors="ignore")
    except OSError:
        print(ep, "missing")
        continue
    for line in fh:
        m = re.search(r"RHOPAIR h=\d+ A (\S+) (\S+) B (\S+) (\S+)", line)
        if m:
            A.append(complex(float(m.group(1)), float(m.group(2))))
            B.append(complex(float(m.group(3)), float(m.group(4))))
    n = len(A)
    if n < 3:
        print(ep, "insufficient", n)
        continue
    mA = sum(A) / n
    mB = sum(B) / n
    vA = sum(abs(a - mA) ** 2 for a in A) / n
    vB = sum(abs(b - mB) ** 2 for b in B) / n
    cov = sum(((a - mA) * (b - mB).conjugate()).real
              for a, b in zip(A, B)) / n
    rho = cov / math.sqrt(vA * vB)
    vD = sum(abs((b - a) - (mB - mA)) ** 2 for a, b in zip(A, B)) / n
    print(f"ep={ep}: n={n} rho={rho:.4f} sigmaA={math.sqrt(vA):.0f} "
          f"sigmaB={math.sqrt(vB):.0f} sigma_diff={math.sqrt(vD):.0f} "
          f"varratio diff/full={vD / vA:.4f}")
