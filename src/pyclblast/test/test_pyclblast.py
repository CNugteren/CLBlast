
####################################################################################################
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# This file test PyCLBlast: the Python interface to CLBlast. It is not exhaustive. For full testing
# it is recommended to run the regular CLBlast tests, this is just a small smoke test.
#
####################################################################################################

import unittest

import numpy as np
import pyopencl as cl
from pyopencl.array import Array

import pyclblast


class TestPyCLBlast(unittest.TestCase):

    @staticmethod
    def setup(sizes, dtype):
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        host_arrays, device_arrays = [], []
        for size in sizes:
            numpy_array = np.random.rand(*size).astype(dtype=dtype)
            opencl_array = Array(queue, numpy_array.shape, numpy_array.dtype)
            opencl_array.set(numpy_array)
            host_arrays.append(numpy_array)
            device_arrays.append(opencl_array)
        queue.finish()
        return queue, host_arrays, device_arrays

    def test_axpy(self):
        for dtype in ["float32", "complex64"]:
            for alpha in [1.0, 3.1]:
                for n in [1, 7, 32]:
                    queue, h, d = self.setup([(n,), (n,)], dtype=dtype)
                    pyclblast.axpy(queue, n, d[0], d[1], alpha=alpha)
                    queue.finish()
                    result = d[1].get()
                    reference = alpha * h[0] + h[1]
                    for i in range(n):
                        self.assertAlmostEqual(reference[i], result[i], places=3)

    def test_gemv(self):
        for dtype in ["float32", "complex64"]:
            for beta in [1.0]:
                for alpha in [1.0, 3.1]:
                    for m in [1, 7, 32]:
                        for n in [1, 7, 32]:
                            queue, h, d = self.setup([(m, n), (n,), (m,)], dtype=dtype)
                            pyclblast.gemv(queue, m, n, d[0], d[1], d[2],
                                           a_ld=n, alpha=alpha, beta=beta)
                            queue.finish()
                            result = d[2].get()
                            reference = alpha * np.dot(h[0], h[1]) + beta * h[2]
                            for i in range(m):
                                self.assertAlmostEqual(reference[i], result[i], places=3)

    def test_gemm(self):
        for dtype in ["float32", "complex64"]:
            for beta in [1.0]:
                for alpha in [1.0, 3.1]:
                    for m in [1, 7, 32]:
                        for n in [1, 7, 32]:
                            for k in [1, 7, 32]:
                                queue, h, d = self.setup([(m, k), (k, n), (m, n)], dtype=dtype)
                                pyclblast.gemm(queue, m, n, k, d[0], d[1], d[2],
                                               a_ld=k, b_ld=n, c_ld=n, alpha=alpha, beta=beta)
                                queue.finish()
                                result = d[2].get()
                                reference = alpha * np.dot(h[0], h[1]) + beta * h[2]
                                for i in range(m):
                                    for j in range(n):
                                        self.assertAlmostEqual(reference[i, j], result[i, j],
                                                               places=3)
