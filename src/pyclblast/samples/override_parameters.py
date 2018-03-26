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

if __name__ == "__main__":

    # Set up pyopencl:
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Set up a basic sgemm example:
    m, n, k = 2, 3, 4
    a = np.random.rand(m, k).astype(dtype=np.float32)
    b = np.random.rand(k, n).astype(dtype=np.float32)
    c = np.empty((m, n), np.float32)
    cla = Array(queue, a.shape, a.dtype)
    clb = Array(queue, b.shape, b.dtype)
    clc = Array(queue, c.shape, c.dtype)
    cla.set(a)
    clb.set(b)
    clc.set(c)

    # Perform sgemm on these matrices, overriding the CLBlast parameters. In this example, we'll
    # just change the 'MWG' parameter a couple of times:
    params = { "KWG": 32, "KWI": 2, "MDIMA": 8, "MDIMC": 8, "MWG": 64, "NDIMB": 8, "NDIMC": 8,
            "NWG": 64, "SA": 0, "SB": 0, "STRM": 0, "STRN": 0, "VWM": 4, "VWN": 1 }
    for mwg in (32, 64, 256):
        print("Running sgemm tuned with MWG = %d" % mwg)
        params["MWG"] = mwg
        pyclblast.override_parameters(ctx.devices[0], 'Xgemm', 32, params)
        pyclblast.gemm(queue, m, n, k, cla, clb, clc, a_ld=k, b_ld=n, c_ld=n)
        assert np.allclose(clc.get(), a.dot(b)), "uh-oh, xgemm isn't behaving correctly"
