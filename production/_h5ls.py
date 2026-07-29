import h5py
import sys
with h5py.File(sys.argv[1]) as f:
    f.visit(print)
