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
m, n = 4, 3
alpha = 1.0
beta = 0.0

print("# Setting up OpenCL")
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

print("# Setting up Numpy arrays")
a = np.random.rand(m, n).astype(dtype=dtype)
x = np.random.rand(n).astype(dtype=dtype)
y = np.random.rand(m).astype(dtype=dtype)

print("# Setting up OpenCL arrays")
cla = Array(queue, a.shape, a.dtype)
clx = Array(queue, x.shape, x.dtype)
cly = Array(queue, y.shape, y.dtype)
cla.set(a)
clx.set(x)
cly.set(y)

print("# Example level-2 operation: GEMV")
pyclblast.gemv(queue, m, n, cla, clx, cly, a_ld=n, alpha=alpha, beta=beta)
queue.finish()
print("# Result for vector y: %s" % cly.get())
print("# Expected result:     %s" % (alpha * np.dot(a, x) + beta * y))
