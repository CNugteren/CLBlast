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

# Settings for this sample:
batch_count = 2
dtype = 'float32'
alphas = [1.5, 1.0]
n = 4

print("# Setting up OpenCL")
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

print("# Setting up Numpy arrays")
x = np.random.rand(n * batch_count).astype(dtype=dtype)
y = np.random.rand(n * batch_count).astype(dtype=dtype)

print("# Batch offsets: next after each other")
x_offsets = [0, n]
y_offsets = [0, n]

print("# Setting up OpenCL arrays")
clx = Array(queue, x.shape, x.dtype)
cly = Array(queue, y.shape, y.dtype)
clx.set(x)
cly.set(y)

print("# Example level-1 batched operation: AXPY-batched")
assert len(alphas) == len(x_offsets) == len(y_offsets) == batch_count
pyclblast.axpyBatched(queue, n, clx, cly, alphas, x_offsets, y_offsets)
queue.finish()

print("# Full result for vector y: %s" % str(cly.get()))
for i in range(batch_count):
	result = alphas[i] * x[x_offsets[i]:x_offsets[i] + n] + y[y_offsets[i]:y_offsets[i] + n]
	print("# Expected result batch #%d: %s" % (i, str(result)))
