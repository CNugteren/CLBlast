#!/usr/bin/env python

# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import utils


AXPY = {
    "label_names": ["CLBlast", "clBLAS"],
    "num_rows": 2, "num_cols": 3,
    "benchmarks": [
        {
            "name": "axpy",
            "title": "multiples of 256K",
            "x_label": "vector sizes (n)", "x_keys": ["n"],
            "y_label": "GB/s (higher is better)", "y_keys": ["GBs_1", "GBs_2"],
            "arguments": [{"n": utils.k(256), "incx": 1, "incy": 1, "step": utils.k(256), "num_steps": 16}],
        },
        {
            "name": "axpy",
            "title": "multiples of 256K+1",
            "x_label": "vector sizes (n)", "x_keys": ["n"],
            "y_label": "GB/s (higher is better)", "y_keys": ["GBs_1", "GBs_2"],
            "arguments": [{"n": utils.k(256) + 1, "incx": 1, "incy": 1, "step": utils.k(256) + 1, "num_steps": 16}],
        },
        {
            "name": "axpy",
            "title": "around n=1M",
            "x_label": "vector sizes (n)", "x_keys": ["n"],
            "y_label": "GB/s (higher is better)", "y_keys": ["GBs_1", "GBs_2"],
            "arguments": [{"n": utils.m(1), "incx": 1, "incy": 1, "step": 1, "num_steps": 16}],
        },
        {
            "name": "axpy",
            "title": "around n=16M",
            "x_label": "vector sizes (n)", "x_keys": ["n"],
            "y_label": "GB/s (higher is better)", "y_keys": ["GBs_1", "GBs_2"],
            "arguments": [{"n": utils.m(16), "incx": 1, "incy": 1, "step": 1, "num_steps": 16}],
        },
        {
            "name": "axpy",
            "title": "strides (n=8M)",
            "x_label": "increments/strides for x,y", "x_keys": ["incx", "incy"],
            "y_label": "GB/s (higher is better)", "y_keys": ["GBs_1", "GBs_2"],
            "arguments": [{"n": utils.m(8), "incx": inc_x, "incy": inc_y, "step": 0, "num_steps": 1}
                          for inc_x in [1, 2, 4] for inc_y in [1, 2, 4]],
        },
        {
            "name": "axpy",
            "title": "powers of 2",
            "x_label": "vector sizes (n)", "x_keys": ["n"],
            "y_label": "GB/s (higher is better)", "y_keys": ["GBs_1", "GBs_2"],
            "arguments": [{"n": n, "incx": 1, "incy": 1, "step": 0, "num_steps": 1}
                          for n in utils.powers_of_2(utils.k(32), utils.m(64))],
        }
    ]
}

GEMV = {
    "label_names": ["CLBlast", "clBLAS"],
    "num_rows": 2, "num_cols": 3,
    "benchmarks": [
        {
            "name": "gemv",
            "title": "multiples of 256",
            "x_label": "matrix/vector sizes (n=m)", "x_keys": ["n"],
            "y_label": "GB/s (higher is better)", "y_keys": ["GBs_1", "GBs_2"],
            "arguments": [{"n": 256, "m": 256, "incx": 1, "incy": 1, "layout": 102, "step": 256, "num_steps": 20}],
        },
        {
            "name": "gemv",
            "title": "multiples of 257",
            "x_label": "matrix/vector sizes (n=m)", "x_keys": ["n"],
            "y_label": "GB/s (higher is better)", "y_keys": ["GBs_1", "GBs_2"],
            "arguments": [{"n": 257, "m": 257, "incx": 1, "incy": 1, "layout": 102, "step": 257, "num_steps": 20}],
        },
        {
            "name": "gemv",
            "title": "around n=m=4K",
            "x_label": "matrix/vector sizes (n=m)", "x_keys": ["n"],
            "y_label": "GB/s (higher is better)", "y_keys": ["GBs_1", "GBs_2"],
            "arguments": [{"n": 4096, "m": 4096, "incx": 1, "incy": 1, "layout": 102, "step": 1, "num_steps": 16}],
        },
        {
            "name": "gemv",
            "title": "multiples of 256 rotated",
            "x_label": "matrix/vector sizes (n=m)", "x_keys": ["n"],
            "y_label": "GB/s (higher is better)", "y_keys": ["GBs_1", "GBs_2"],
            "arguments": [{"n": 256, "m": 256, "incx": 1, "incy": 1, "layout": 101, "step": 256, "num_steps": 20}],
        },
        {
            "name": "gemv",
            "title": "multiples of 257 rotated",
            "x_label": "matrix/vector sizes (n=m)", "x_keys": ["n"],
            "y_label": "GB/s (higher is better)", "y_keys": ["GBs_1", "GBs_2"],
            "arguments": [{"n": 257, "m": 257, "incx": 1, "incy": 1, "layout": 101, "step": 257, "num_steps": 20}],
        },
        {
            "name": "gemv",
            "title": "strides (n=m=4K)",
            "x_label": "increments/strides for x,y", "x_keys": ["incx", "incy"],
            "y_label": "GB/s (higher is better)", "y_keys": ["GBs_1", "GBs_2"],
            "arguments": [{"n": 4096, "m": 4096, "incx": inc_x, "incy": inc_y, "layout": 102, "step": 0, "num_steps": 1}
                          for inc_x in [1, 2, 4] for inc_y in [1, 2, 4]],
        }
    ]
}

GEMM = {
    "label_names": ["CLBlast", "clBLAS"],
    "num_rows": 2, "num_cols": 3,
    "benchmarks": [
        {
            "name": "gemm",
            "title": "multiples of 128",
            "x_label": "matrix sizes (m=n=k)", "x_keys": ["m"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": 128, "n": 128, "k": 128, "layout": 102,
                           "transA": 111, "transB": 111, "step": 128, "num_steps": 20}],
        },
        {
            "name": "gemm",
            "title": "multiples of 129",
            "x_label": "matrix sizes (m=n=k)", "x_keys": ["m"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": 129, "n": 129, "k": 129, "layout": 102,
                           "transA": 111, "transB": 111, "step": 129, "num_steps": 20}],
        },
        {
            "name": "gemm",
            "title": "around m=n=k=512",
            "x_label": "matrix sizes (m=n=k)", "x_keys": ["m"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": 512, "n": 512, "k": 512, "layout": 102,
                           "transA": 111, "transB": 111, "step": 1, "num_steps": 16}],
        },
        {
            "name": "gemm",
            "title": "around m=n=k=2048",
            "x_label": "matrix sizes (m=n=k)", "x_keys": ["m"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": 2048, "n": 2048, "k": 2048, "layout": 102,
                           "transA": 111, "transB": 111, "step": 1, "num_steps": 16}],
        },
        {
            "name": "gemm",
            "title": "layouts/transposing (m=n=k=1K)",
            "x_label": "layout, transA, transB", "x_keys": ["layout", "transA", "transB"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": 1024, "n": 1024, "k": 1024, "layout": layout,
                           "transA": transA, "transB": transB, "step": 0, "num_steps": 1}
                          for layout in [101, 102] for transA in [111, 112] for transB in [111, 112]],
        },
        {
            "name": "gemm",
            "title": "powers of 2",
            "x_label": "matrix sizes (m=n=k)", "x_keys": ["m"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": n, "n": n, "k": n, "layout": 102,
                           "transA": 111, "transB": 111, "step": 0, "num_steps": 1}
                          for n in utils.powers_of_2(8, utils.k(4))],
        }
    ]
}

GEMM_SMALL = {
    "label_names": ["CLBlast", "clBLAS"],
    "num_rows": 2, "num_cols": 1,
    "benchmarks": [
        {
            "name": "gemm",
            "title": "small matrices in steps of 16",
            "x_label": "matrix sizes (m=n=k)", "x_keys": ["m"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": 128, "n": 128, "k": 128, "layout": 102,
                           "transA": 111, "transB": 111, "step": 16, "num_steps": 57}],
        },
        {
            "name": "gemm",
            "title": "small matrices in steps of 1",
            "x_label": "matrix sizes (m=n=k)", "x_keys": ["m"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": 128, "n": 128, "k": 128, "layout": 102,
                           "transA": 111, "transB": 111, "step": 1, "num_steps": 385}],
        },

    ]
}

SYMM = {
    "label_names": ["CLBlast", "clBLAS"],
    "num_rows": 2, "num_cols": 3,
    "benchmarks": [
        {
            "name": "symm",
            "title": "multiples of 128",
            "x_label": "matrix sizes (m=n)", "x_keys": ["m"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": 128, "n": 128, "layout": 102,
                           "side": 141, "triangle": 121, "step": 128, "num_steps": 20}],
        },
        {
            "name": "symm",
            "title": "multiples of 129",
            "x_label": "matrix sizes (m=n)", "x_keys": ["m"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": 129, "n": 129, "layout": 102,
                           "side": 141, "triangle": 121, "step": 129, "num_steps": 20}],
        },
        {
            "name": "symm",
            "title": "around m=n=512",
            "x_label": "matrix sizes (m=n)", "x_keys": ["m"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": 512, "n": 512, "layout": 102,
                           "side": 141, "triangle": 121, "step": 1, "num_steps": 16}],
        },
        {
            "name": "symm",
            "title": "around m=n=2048",
            "x_label": "matrix sizes (m=n)", "x_keys": ["m"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": 2048, "n": 2048, "layout": 102,
                           "side": 141, "triangle": 121, "step": 1, "num_steps": 16}],
        },
        {
            "name": "symm",
            "title": "layouts/sides/triangles (m=n=1K)",
            "x_label": "layout, side, triangle", "x_keys": ["layout", "side", "triangle"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": 1024, "n": 1024, "layout": layout,
                           "side": side, "triangle": triangle, "step": 0, "num_steps": 1}
                          for layout in [101, 102] for side in [141, 142] for triangle in [121, 122]],
        },
        {
            "name": "symm",
            "title": "powers of 2",
            "x_label": "matrix sizes (m=n)", "x_keys": ["m"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"m": n, "n": n, "layout": 102,
                           "side": 141, "triangle": 121, "step": 0, "num_steps": 1}
                          for n in utils.powers_of_2(8, utils.k(4))],
        }
    ]
}

SYRK = {
    "label_names": ["CLBlast", "clBLAS"],
    "num_rows": 2, "num_cols": 3,
    "benchmarks": [
        {
            "name": "syrk",
            "title": "multiples of 128",
            "x_label": "matrix sizes (n=k)", "x_keys": ["n"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"n": 128, "k": 128, "layout": 102,
                           "side": 141, "triangle": 121, "step": 128, "num_steps": 20}],
        },
        {
            "name": "syrk",
            "title": "multiples of 129",
            "x_label": "matrix sizes (n=k)", "x_keys": ["n"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"n": 129, "k": 129, "layout": 102,
                           "side": 141, "triangle": 121, "step": 129, "num_steps": 20}],
        },
        {
            "name": "syrk",
            "title": "around n=k=512",
            "x_label": "matrix sizes (n=k)", "x_keys": ["n"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"n": 512, "k": 512, "layout": 102,
                           "side": 141, "triangle": 121, "step": 1, "num_steps": 16}],
        },
        {
            "name": "syrk",
            "title": "around n=k=2048",
            "x_label": "matrix sizes (n=k)", "x_keys": ["n"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"n": 2048, "k": 2048, "layout": 102,
                           "side": 141, "triangle": 121, "step": 1, "num_steps": 16}],
        },
        {
            "name": "syrk",
            "title": "layouts/sides/triangles (n=k=1K)",
            "x_label": "layout, triangle, transA", "x_keys": ["layout", "triangle", "transA"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"n": 1024, "k": 1024, "layout": layout,
                           "triangle": triangle, "transA": transA, "step": 0, "num_steps": 1}
                          for layout in [101, 102] for triangle in [121, 122] for transA in [111, 112]],
        },
        {
            "name": "syrk",
            "title": "powers of 2",
            "x_label": "matrix sizes (n=k)", "x_keys": ["n"],
            "y_label": "GFLOPS (higher is better)", "y_keys": ["GFLOPS_1", "GFLOPS_2"],
            "arguments": [{"n": n, "k": n, "layout": 102,
                           "side": 141, "triangle": 121, "step": 0, "num_steps": 1}
                          for n in utils.powers_of_2(8, utils.k(4))],
        }
    ]
}

SUMMARY = {
    "label_names": ["CLBlast", "clBLAS"],
    "num_rows": 4, "num_cols": 2,
    "benchmarks": [
        AXPY["benchmarks"][0],
        AXPY["benchmarks"][1],
        GEMV["benchmarks"][0],
        GEMV["benchmarks"][1],
        GEMM["benchmarks"][0],
        GEMM["benchmarks"][1],
        SYMM["benchmarks"][0],
        SYMM["benchmarks"][1],
    ]
}
