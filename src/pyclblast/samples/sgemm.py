#!/usr/bin/env python

# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0.
# This file follows the PEP8 Python style guide and uses a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import numpy as np
import pyopencl as cl
from pyopencl.array import Array
import pyclblast
from datetime import datetime

# Settings for this sample
dtype = 'float32'

print("# Setting up OpenCL")
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

print("# Setting up Numpy arrays")
m, n, k = 128, 256, 512
a = np.random.rand(m, k).astype(dtype=dtype)
b = np.random.rand(k, n).astype(dtype=dtype)
c = np.random.rand(m, n).astype(dtype=dtype)

print("# Setting up OpenCL arrays")
cla = Array(queue, a.shape, a.dtype)
clb = Array(queue, b.shape, b.dtype)
clc = Array(queue, c.shape, c.dtype)
cla.set(a)
clb.set(b)
clc.set(c)

print("# Example level-3 operation: GEMM")
pyclblast.gemm(queue, m, n, k, cla, clb, clc, a_ld=k, b_ld=n, c_ld=n)
print("# PyCLBlast matrix result is correct?:", np.allclose(clc.get(), np.dot(a, b)))

print("# GFLOPS when tuned with different values of MWG:")
params = { "KWG": 32, "KWI": 2, "MDIMA": 8, "MDIMC": 8, "MWG": 64, "NDIMB": 8, "NDIMC": 8, "NWG": 64, "SA": 0, "SB": 0, "STRM": 0, "STRN": 0, "VWM": 4, "VWN": 1 }
mwg = 1
while mwg <= 256:
    params["MWG"] = mwg
    pyclblast.override_parameters(ctx.devices[0], 'Xgemm', 32, params)
    for i in range(100):
        if i == 10:
            t0 = datetime.now()
        pyclblast.gemm(queue, m, n, k, cla, clb, clc, a_ld=k, b_ld=n, c_ld=n)
    print("#\tMWG = %-3d : %4d" % (mwg,  int(2 * m * n * k / ((datetime.now() - t0).total_seconds() / 100) / 1024 ** 3)))
    mwg *= 4
