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

# Settings for this sample
dtype = 'float32'

print("# Setting up OpenCL")
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

print("# Setting up Numpy arrays")
m, n, k = 2, 3, 4
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
queue.finish()
print("# Matrix C result: %s" % clc.get())
print("# Expected result: %s" % (np.dot(a, b)))
