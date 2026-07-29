import h5py
import numpy as np


def g(f, k):
    a = np.array(h5py.File(f)[k])
    return a["re"] + 1j * a["im"] if a.dtype.names else a


P = "/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt"
old = g(f"{P}/dtxqcd_lam5_dibaryon_aux/dibaryon_aux_dtxqcd_30.h5", "dib_I1_S_5")
new = g(f"{P}/dtxqcd_lam5_dibaryon_aux_sm/dibaryon_aux_dtxqcd_30.h5",
        "dib_I1_sm0_S_5")
print("max|old-new| =", np.abs(old - new).max(),
      "  max|old| =", np.abs(old).max())
