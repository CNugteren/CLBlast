#!/usr/bin/env python

# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# This script automatically generates the bodies of the following files, creating the full CLBlast API interface and
# implementation (C, C++, and reference BLAS wrappers):
#    clblast.h
#    clblast.cpp
#    clblast_c.h
#    clblast_c.cpp
#    clblast_cuda.h
#    clblast_cuda.cpp
#    clblast_netlib_c.h
#    clblast_netlib_c.cpp
#    wrapper_clblas.h
#    wrapper_cblas.h
#    pyclblast.pyx
# It also generates the main functions for the correctness and performance tests as found in
#    test/correctness/routines/levelX/xYYYY.cpp
#    test/performance/routines/levelX/xYYYY.cpp
# It also produces the API documentation found in doc/clblast.md


import sys
import os.path
import argparse

import generator.cpp as cpp
import generator.doc as doc
import generator.pyclblast as pyclblast
from generator.routine import Routine
from generator.datatype import H, S, D, C, Z, Sc, Dz, iH, iS, iD, iC, iZ, Css, Zdd, Ccs, Zzd, T, Tc, TU

FILES = [
    "/include/clblast.h",
    "/src/clblast.cpp",
    "/include/clblast_c.h",
    "/src/clblast_c.cpp",
    "/test/wrapper_clblas.hpp",
    "/test/wrapper_cblas.hpp",
    "/test/wrapper_cublas.hpp",
    "/include/clblast_netlib_c.h",
    "/src/clblast_netlib_c.cpp",
    "/include/clblast_cuda.h",
    "/src/clblast_cuda.cpp",
    "/src/pyclblast/src/pyclblast.pyx"
]
HEADER_LINES = [124, 21, 128, 24, 29, 45, 29, 66, 40, 96, 21, 327]
FOOTER_LINES = [98, 57, 112, 275, 6, 6, 6, 9, 2, 41, 56, 37]
HEADER_LINES_DOC = 0
FOOTER_LINES_DOC = 232

# Different possibilities for requirements
ald_m = "The value of `a_ld` must be at least `m`."
ald_n = "The value of `a_ld` must be at least `n`."
ald_k_one = "The value of `a_ld` must be at least `k + 1`."
ald_kl_ku_one = "The value of `a_ld` must be at least `kl + ku + 1`."
ald_transa_m_k = "When `transpose_a == Transpose::kNo`, then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `k`."
ald_trans_n_k = "When `transpose == Transpose::kNo`, then `a_ld` must be at least `n`, otherwise `a_ld` must be at least `k`."
ald_side_m_n = "When `side = Side::kLeft` then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `n`."
bld_m = "The value of `b_ld` must be at least `m`."
bld_n = "The value of `b_ld` must be at least `n`."
bld_transb_k_n = "When `transpose_b == Transpose::kNo`, then `b_ld` must be at least `k`, otherwise `b_ld` must be at least `n`."
bld_trans_n_k = "When `transpose == Transpose::kNo`, then `b_ld` must be at least `n`, otherwise `b_ld` must be at least `k`."
cld_m = "The value of `c_ld` must be at least `m`."
cld_n = "The value of `c_ld` must be at least `n`."


# Helper functions to compute vector and matrix sizes
def size_helper(condition, size_one, size_two, multiplier):
    length = "(" + condition + ")" + " ? " + size_one + " * " + multiplier + " : " + size_two + " * " + multiplier
    return length


def layout_transpose_condition(prefix):
    return "(layout == CLBlastLayoutColMajor && " + prefix + "_transpose != CLBlastTransposeNo) || " +\
           "(layout == CLBlastLayoutRowMajor && " + prefix + "_transpose == CLBlastTransposeNo)"


# Different possibilities for the vector and matrix sizes
xn = "n * x_inc"
xm = "m * x_inc"
yn = "n * y_inc"
ym = "m * y_inc"
zn = "n * z_inc"
an = "n * a_ld"
apn = "((n*(n+1)) / 2)"
cn = "n * c_ld"
xmn = size_helper("a_transpose != CLBlastTransposeNo", "m", "n", "x_inc")
ynm = size_helper("a_transpose != CLBlastTransposeNo", "n", "m", "y_inc")
amn = size_helper("layout == CLBlastLayoutRowMajor", "m", "n", "a_ld")
amns = size_helper("side == CLBlastSideLeft", "m", "n", "a_ld")
amk = size_helper(layout_transpose_condition("a"), "m", "k", "a_ld")
ank = size_helper(layout_transpose_condition("a"), "n", "k", "a_ld")
ankab = size_helper(layout_transpose_condition("ab"), "n", "k", "a_ld")
bkn = size_helper(layout_transpose_condition("b"), "k", "n", "b_ld")
bnkab = size_helper(layout_transpose_condition("ab"), "n", "k", "b_ld")
bmn = size_helper("layout == CLBlastLayoutRowMajor", "m", "n", "b_ld")
bnma = size_helper(layout_transpose_condition("a"), "n", "m", "b_ld")
cmn = size_helper("layout == CLBlastLayoutRowMajor", "m", "n", "c_ld")
ammn = size_helper("layout == CLBlastLayoutRowMajor", "m", "((side == CLBlastSideLeft) ? m : n)", "a_ld")
bmnn = size_helper("layout == CLBlastLayoutRowMajor", "((side == CLBlastSideLeft) ? m : n)", "n", "b_ld")
im = "height * width * channels"
col = "height * width * channels"
imb = "height * width * channels * batch_count"
kernel = "kernel_h * kernel_w * num_kernels"
result = "height_out * width_out * num_kernels * batch_count"


# ==================================================================================================

# Populates a list of routines
im2col_constants = ["channels", "height", "width", "kernel_h", "kernel_w", "pad_h", "pad_w", "stride_h", "stride_w", "dilation_h", "dilation_w"]
convgemm_constants = im2col_constants + ["num_kernels", "batch_count"]
ROUTINES = [
[  # Level 1: vector-vector
  Routine(False, True,  0, False, "1", "rotg",  T, [S,D],            [],                  [],                                                     [],         ["sa","sb","sc","ss"],        ["1","1","1","1"], [],       "",    "Generate givens plane rotation", "", []),
  Routine(False, True,  0, False, "1", "rotmg", T, [S,D],            [],                  [],                                                     ["sy1"],    ["sd1","sd2","sx1","sparam"], ["1","1","1","1","1"], [],   "",    "Generate modified givens plane rotation", "", []),
  Routine(False, True,  0, False, "1", "rot",   T, [S,D],            ["n"],               [],                                                     [],         ["x","y"],                    [xn,yn],       ["cos","sin"],"",    "Apply givens plane rotation", "", []),
  Routine(False, True,  0, False, "1", "rotm",  T, [S,D],            ["n"],               [],                                                     [],         ["x","y","sparam"],           [xn,yn,"1"],   [],           "",    "Apply modified givens plane rotation", "", []),
  Routine(True,  True,  0, False, "1", "swap",  T, [S,D,C,Z,H],      ["n"],               [],                                                     [],         ["x","y"],                    [xn,yn],       [],           "",    "Swap two vectors", "Interchanges _n_ elements of vectors _x_ and _y_.", []),
  Routine(True,  True,  0, False, "1", "scal",  T, [S,D,C,Z,H],      ["n"],               [],                                                     [],         ["x"],                        [xn],          ["alpha"],    "",    "Vector scaling", "Multiplies _n_ elements of vector _x_ by a scalar constant _alpha_.", []),
  Routine(True,  True,  0, False, "1", "copy",  T, [S,D,C,Z,H],      ["n"],               [],                                                     ["x"],      ["y"],                        [xn,yn],       [],           "",    "Vector copy", "Copies the contents of vector _x_ into vector _y_.", []),
  Routine(True,  True,  0, False, "1", "axpy",  T, [S,D,C,Z,H],      ["n"],               [],                                                     ["x"],      ["y"],                        [xn,yn],       ["alpha"],    "",    "Vector-times-constant plus vector", "Performs the operation _y = alpha * x + y_, in which _x_ and _y_ are vectors and _alpha_ is a scalar constant.", []),
  Routine(True,  True,  0, False, "1", "dot",   T, [S,D,H],          ["n"],               [],                                                     ["x","y"],  ["dot"],                      [xn,yn,"1"],   [],           "n",   "Dot product of two vectors", "Multiplies _n_ elements of the vectors _x_ and _y_ element-wise and accumulates the results. The sum is stored in the _dot_ buffer.", []),
  Routine(True,  True,  0, False, "1", "dotu",  T, [C,Z],            ["n"],               [],                                                     ["x","y"],  ["dot"],                      [xn,yn,"1"],   [],           "n",   "Dot product of two complex vectors", "See the regular xDOT routine.", []),
  Routine(True,  True,  0, False, "1", "dotc",  T, [C,Z],            ["n"],               [],                                                     ["x","y"],  ["dot"],                      [xn,yn,"1"],   [],           "n",   "Dot product of two complex vectors, one conjugated", "See the regular xDOT routine.", []),
  Routine(True,  True,  0, False, "1", "nrm2",  T, [S,D,Sc,Dz,H],    ["n"],               [],                                                     ["x"],      ["nrm2"],                     [xn,"1"],      [],           "2*n", "Euclidian norm of a vector", "Accumulates the square of _n_ elements in the _x_ vector and takes the square root. The resulting L2 norm is stored in the _nrm2_ buffer.", []),
  Routine(True,  True,  0, False, "1", "asum",  T, [S,D,Sc,Dz,H],    ["n"],               [],                                                     ["x"],      ["asum"],                     [xn,"1"],      [],           "n",   "Absolute sum of values in a vector", "Accumulates the absolute value of _n_ elements in the _x_ vector. The results are stored in the _asum_ buffer.", []),
  Routine(True,  False, 0, False, "1", "sum",   T, [S,D,Sc,Dz,H],    ["n"],               [],                                                     ["x"],      ["sum"],                      [xn,"1"],      [],           "n",   "Sum of values in a vector (non-BLAS function)", "Accumulates the values of _n_ elements in the _x_ vector. The results are stored in the _sum_ buffer. This routine is the non-absolute version of the xASUM BLAS routine.", []),
  Routine(True,  True,  0, False, "1", "amax",  T, [iS,iD,iC,iZ,iH], ["n"],               [],                                                     ["x"],      ["imax"],                     [xn,"1"],      [],           "2*n", "Index of absolute maximum value in a vector", "Finds the index of the maximum of the absolute values in the _x_ vector. The resulting integer index is stored in the _imax_ buffer.", []),
  Routine(True,  False, 0, False, "1", "amin",  T, [iS,iD,iC,iZ,iH], ["n"],               [],                                                     ["x"],      ["imin"],                     [xn,"1"],      [],           "2*n", "Index of absolute minimum value in a vector (non-BLAS function)", "Finds the index of the minimum of the absolute values in the _x_ vector. The resulting integer index is stored in the _imin_ buffer.", []),
  Routine(True,  False, 0, False, "1", "max",   T, [iS,iD,iC,iZ,iH], ["n"],               [],                                                     ["x"],      ["imax"],                     [xn,"1"],      [],           "2*n", "Index of maximum value in a vector (non-BLAS function)", "Finds the index of the maximum of the values in the _x_ vector. The resulting integer index is stored in the _imax_ buffer. This routine is the non-absolute version of the IxAMAX BLAS routine.", []),
  Routine(True,  False, 0, False, "1", "min",   T, [iS,iD,iC,iZ,iH], ["n"],               [],                                                     ["x"],      ["imin"],                     [xn,"1"],      [],           "2*n", "Index of minimum value in a vector (non-BLAS function)", "Finds the index of the minimum of the values in the _x_ vector. The resulting integer index is stored in the _imin_ buffer. This routine is the non-absolute minimum version of the IxAMAX BLAS routine.", []),
],
[  # Level 2: matrix-vector
  Routine(True,  True,  0, False, "2a", "gemv",  T,  [S,D,C,Z,H],    ["m","n"],           ["layout","a_transpose"],                               ["a","x"],  ["y"],                        [amn,xmn,ynm], ["alpha","beta"], "",    "General matrix-vector multiplication", "Performs the operation _y = alpha * A * x + beta * y_, in which _x_ is an input vector, _y_ is an input and output vector, _A_ is an input matrix, and _alpha_ and _beta_ are scalars. The matrix _A_ can optionally be transposed before performing the operation.", [ald_m]),
  Routine(True,  True,  0, False, "2a", "gbmv",  T,  [S,D,C,Z,H],    ["m","n","kl","ku"], ["layout","a_transpose"],                               ["a","x"],  ["y"],                        [amn,xmn,ynm], ["alpha","beta"], "",    "General banded matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is banded instead.", [ald_kl_ku_one]),
  Routine(True,  True,  0, False, "2a", "hemv",  T,  [C,Z],          ["n"],               ["layout","triangle"],                                  ["a","x"],  ["y"],                        [an,xn,yn],    ["alpha","beta"], "",    "Hermitian matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is an Hermitian matrix instead.", [ald_n]),
  Routine(True,  True,  0, False, "2a", "hbmv",  T,  [C,Z],          ["n","k"],           ["layout","triangle"],                                  ["a","x"],  ["y"],                        [an,xn,yn],    ["alpha","beta"], "",    "Hermitian banded matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is an Hermitian banded matrix instead.", [ald_k_one]),
  Routine(True,  True,  0, False, "2a", "hpmv",  T,  [C,Z],          ["n"],               ["layout","triangle"],                                  ["ap","x"], ["y"],                        [apn,xn,yn],   ["alpha","beta"], "",    "Hermitian packed matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is an Hermitian packed matrix instead and represented as _AP_.", []),
  Routine(True,  True,  0, False, "2a", "symv",  T,  [S,D,H],        ["n"],               ["layout","triangle"],                                  ["a","x"],  ["y"],                        [an,xn,yn],    ["alpha","beta"], "",    "Symmetric matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is symmetric instead.", [ald_n]),
  Routine(True,  True,  0, False, "2a", "sbmv",  T,  [S,D,H],        ["n","k"],           ["layout","triangle"],                                  ["a","x"],  ["y"],                        [an,xn,yn],    ["alpha","beta"], "",    "Symmetric banded matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is symmetric and banded instead.", [ald_k_one]),
  Routine(True,  True,  0, False, "2a", "spmv",  T,  [S,D,H],        ["n"],               ["layout","triangle"],                                  ["ap","x"], ["y"],                        [apn,xn,yn],   ["alpha","beta"], "",    "Symmetric packed matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is a symmetric packed matrix instead and represented as _AP_.", []),
  Routine(True,  True,  0, False, "2a", "trmv",  T,  [S,D,C,Z,H],    ["n"],               ["layout","triangle","a_transpose","diagonal"],         ["a"],      ["x"],                        [an,xn],       [],               "n",   "Triangular matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is triangular instead.", [ald_n]),
  Routine(True,  True,  0, False, "2a", "tbmv",  T,  [S,D,C,Z,H],    ["n","k"],           ["layout","triangle","a_transpose","diagonal"],         ["a"],      ["x"],                        [an,xn],       [],               "n",   "Triangular banded matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is triangular and banded instead.", [ald_k_one]),
  Routine(True,  True,  0, False, "2a", "tpmv",  T,  [S,D,C,Z,H],    ["n"],               ["layout","triangle","a_transpose","diagonal"],         ["ap"],     ["x"],                        [apn,xn],      [],               "n",   "Triangular packed matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is a triangular packed matrix instead and repreented as _AP_.", []),
  Routine(True,  True,  0, False, "2a", "trsv",  T,  [S,D,C,Z],      ["n"],               ["layout","triangle","a_transpose","diagonal"],         ["a"],      ["x"],                        [an,xn],       [],               "",    "Solves a triangular system of equations", "", []),
  Routine(False, True,  0, False, "2a", "tbsv",  T,  [S,D,C,Z],      ["n","k"],           ["layout","triangle","a_transpose","diagonal"],         ["a"],      ["x"],                        [an,xn],       [],               "",    "Solves a banded triangular system of equations", "", [ald_k_one]),
  Routine(False, True,  0, False, "2a", "tpsv",  T,  [S,D,C,Z],      ["n"],               ["layout","triangle","a_transpose","diagonal"],         ["ap"],     ["x"],                        [apn,xn],      [],               "",    "Solves a packed triangular system of equations", "", []),
  # Level 2: matrix update
  Routine(True,  True,  0, False, "2b", "ger",   T,  [S,D,H],        ["m","n"],           ["layout"],                                             ["x","y"],  ["a"],                        [xm,yn,amn],   ["alpha"],        "",    "General rank-1 matrix update", "Performs the operation _A = alpha * x * y^T + A_, in which _x_ is an input vector, _y^T_ is the transpose of the input vector _y_, _A_ is the matrix to be updated, and _alpha_ is a scalar value.", [ald_m]),
  Routine(True,  True,  0, False, "2b", "geru",  T,  [C,Z],          ["m","n"],           ["layout"],                                             ["x","y"],  ["a"],                        [xm,yn,amn],   ["alpha"],        "",    "General rank-1 complex matrix update", "Same operation as xGER, but with complex data-types.", [ald_m]),
  Routine(True,  True,  0, False, "2b", "gerc",  T,  [C,Z],          ["m","n"],           ["layout"],                                             ["x","y"],  ["a"],                        [xm,yn,amn],   ["alpha"],        "",    "General rank-1 complex conjugated matrix update", "Same operation as xGERU, but the update is done based on the complex conjugate of the input vectors.", [ald_m]),
  Routine(True,  True,  0, False, "2b", "her",   Tc, [Css,Zdd],      ["n"],               ["layout","triangle"],                                  ["x"],      ["a"],                        [xn,an],       ["alpha"],        "",    "Hermitian rank-1 matrix update", "Performs the operation _A = alpha * x * x^T + A_, in which x is an input vector, x^T is the transpose of this vector, _A_ is the triangular Hermetian matrix to be updated, and alpha is a scalar value.", [ald_n]),
  Routine(True,  True,  0, False, "2b", "hpr",   Tc, [Css,Zdd],      ["n"],               ["layout","triangle"],                                  ["x"],      ["ap"],                       [xn,apn],      ["alpha"],        "",    "Hermitian packed rank-1 matrix update", "Same operation as xHER, but matrix _A_ is an Hermitian packed matrix instead and represented as _AP_.", []),
  Routine(True,  True,  0, False, "2b", "her2",  T,  [C,Z],          ["n"],               ["layout","triangle"],                                  ["x","y"],  ["a"],                        [xn,yn,an],    ["alpha"],        "",    "Hermitian rank-2 matrix update", "Performs the operation _A = alpha * x * y^T + conj(alpha) * y * x^T + A_, in which _x_ is an input vector and _x^T_ its transpose, _y_ is an input vector and _y^T_ its transpose, _A_ is the triangular Hermetian matrix to be updated, _alpha_ is a scalar value and _conj(alpha)_ its complex conjugate.", [ald_n]),
  Routine(True,  True,  0, False, "2b", "hpr2",  T,  [C,Z],          ["n"],               ["layout","triangle"],                                  ["x","y"],  ["ap"],                       [xn,yn,apn],   ["alpha"],        "",    "Hermitian packed rank-2 matrix update", "Same operation as xHER2, but matrix _A_ is an Hermitian packed matrix instead and represented as _AP_.", []),
  Routine(True,  True,  0, False, "2b", "syr",   T,  [S,D,H],        ["n"],               ["layout","triangle"],                                  ["x"],      ["a"],                        [xn,an],       ["alpha"],        "",    "Symmetric rank-1 matrix update", "Same operation as xHER, but matrix A is a symmetric matrix instead.", [ald_n]),
  Routine(True,  True,  0, False, "2b", "spr",   T,  [S,D,H],        ["n"],               ["layout","triangle"],                                  ["x"],      ["ap"],                       [xn,apn],      ["alpha"],        "",    "Symmetric packed rank-1 matrix update", "Same operation as xSPR, but matrix _A_ is a symmetric packed matrix instead and represented as _AP_.", []),
  Routine(True,  True,  0, False, "2b", "syr2",  T,  [S,D,H],        ["n"],               ["layout","triangle"],                                  ["x","y"],  ["a"],                        [xn,yn,an],    ["alpha"],        "",    "Symmetric rank-2 matrix update", "Same operation as xHER2, but matrix _A_ is a symmetric matrix instead.", [ald_n]),
  Routine(True,  True,  0, False, "2b", "spr2",  T,  [S,D,H],        ["n"],               ["layout","triangle"],                                  ["x","y"],  ["ap"],                       [xn,yn,apn],   ["alpha"],        "",    "Symmetric packed rank-2 matrix update", "Same operation as xSPR2, but matrix _A_ is a symmetric packed matrix instead and represented as _AP_.", []),
],
[  # Level 3: matrix-matrix
  Routine(True,  True,  0, True,  "3", "gemm",  T,  [S,D,C,Z,H],     ["m","n","k"],        ["layout","a_transpose","b_transpose"],                ["a","b"],  ["c"],                        [amk,bkn,cmn],   ["alpha","beta"], "",    "General matrix-matrix multiplication", "Performs the matrix product _C = alpha * A * B + beta * C_, in which _A_ (_m_ by _k_) and _B_ (_k_ by _n_) are two general rectangular input matrices, _C_ (_m_ by _n_) is the matrix to be updated, and _alpha_ and _beta_ are scalar values. The matrices _A_ and/or _B_ can optionally be transposed before performing the operation.", [ald_transa_m_k, bld_transb_k_n, cld_m]),
  Routine(True,  True,  0, False, "3", "symm",  T,  [S,D,C,Z,H],     ["m","n"],            ["layout","side","triangle"],                          ["a","b"],  ["c"],                        [ammn,bmnn,cmn], ["alpha","beta"], "",    "Symmetric matrix-matrix multiplication", "Same operation as xGEMM, but _A_ is symmetric instead. In case of `side == kLeft`, _A_ is a symmetric _m_ by _m_ matrix and _C = alpha * A * B + beta * C_ is performed. Otherwise, in case of `side == kRight`, _A_ is a symmtric _n_ by _n_ matrix and _C = alpha * B * A + beta * C_ is performed.", [ald_side_m_n, bld_m, cld_m]),
  Routine(True,  True,  0, False, "3", "hemm",  T,  [C,Z],           ["m","n"],            ["layout","side","triangle"],                          ["a","b"],  ["c"],                        [ammn,bmnn,cmn], ["alpha","beta"], "",    "Hermitian matrix-matrix multiplication", "Same operation as xSYMM, but _A_ is an Hermitian matrix instead.", [ald_side_m_n, bld_m, cld_m]),
  Routine(True,  True,  0, False, "3", "syrk",  T,  [S,D,C,Z,H],     ["n","k"],            ["layout","triangle","a_transpose"],                   ["a"],      ["c"],                        [ank,cn],        ["alpha","beta"], "",    "Rank-K update of a symmetric matrix", "Performs the matrix product _C = alpha * A * A^T + beta * C_ or _C = alpha * A^T * A + beta * C_, in which _A_ is a general matrix and _A^T_ is its transpose, _C_ (_n_ by _n_) is the symmetric matrix to be updated, and _alpha_ and _beta_ are scalar values.", [ald_trans_n_k, cld_m]),
  Routine(True,  True,  0, False, "3", "herk",  Tc, [Css,Zdd],       ["n","k"],            ["layout","triangle","a_transpose"],                   ["a"],      ["c"],                        [ank,cn],        ["alpha","beta"], "",    "Rank-K update of a hermitian matrix", "Same operation as xSYRK, but _C_ is an Hermitian matrix instead.", [ald_trans_n_k, cld_m]),
  Routine(True,  True,  0, False, "3", "syr2k", T,  [S,D,C,Z,H],     ["n","k"],            ["layout","triangle","ab_transpose"],                  ["a","b"],  ["c"],                        [ankab,bnkab,cn],["alpha","beta"], "",    "Rank-2K update of a symmetric matrix", "Performs the matrix product _C = alpha * A * B^T + alpha * B * A^T + beta * C_ or _C = alpha * A^T * B + alpha * B^T * A + beta * C_, in which _A_ and _B_ are general matrices and _A^T_ and _B^T_ are their transposed versions, _C_ (_n_ by _n_) is the symmetric matrix to be updated, and _alpha_ and _beta_ are scalar values.", [ald_trans_n_k, bld_trans_n_k, cld_n]),
  Routine(True,  True,  0, False, "3", "her2k", TU, [Ccs,Zzd],       ["n","k"],            ["layout","triangle","ab_transpose"],                  ["a","b"],  ["c"],                        [ankab,bnkab,cn],["alpha","beta"], "",    "Rank-2K update of a hermitian matrix", "Same operation as xSYR2K, but _C_ is an Hermitian matrix instead.", [ald_trans_n_k, bld_trans_n_k, cld_n]),
  Routine(True,  True,  0, False, "3", "trmm",  T,  [S,D,C,Z,H],     ["m","n"],            ["layout","side","triangle","a_transpose","diagonal"], ["a"],      ["b"],                        [amns,bmn],      ["alpha"],        "",    "Triangular matrix-matrix multiplication", "Performs the matrix product _B = alpha * A * B_ or _B = alpha * B * A_, in which _A_ is a unit or non-unit triangular matrix, _B_ (_m_ by _n_) is the general matrix to be updated, and _alpha_ is a scalar value.", [ald_side_m_n, bld_m]),
  Routine(True,  True,  0, False, "3", "trsm",  T,  [S,D,C,Z],       ["m","n"],            ["layout","side","triangle","a_transpose","diagonal"], ["a"],      ["b"],                        [amns,bmn],      ["alpha"],        "",    "Solves a triangular system of equations", "Solves the equation _A * X = alpha * B_ for the unknown _m_ by _n_ matrix X, in which _A_ is an _n_ by _n_ unit or non-unit triangular matrix and B is an _m_ by _n_ matrix. The matrix _B_ is overwritten by the solution _X_.", []),
],
[  # Level X: extra routines (not part of BLAS)
  # Special routines:
  Routine(True,  True,  0, False, "x", "had",      T, [S,D,C,Z,H],   ["n"],                [],                                                    ["x","y"],  ["z"],                        [xn,yn,zn],      ["alpha","beta"], "",    "Element-wise vector product (Hadamard)", "Performs the Hadamard element-wise product _z = alpha * x * y + beta * z_, in which _x_, _y_, and _z_ are vectors and _alpha_ and _beta_ are scalar constants.", []),
  Routine(True,  True,  0, False, "x", "omatcopy", T, [S,D,C,Z,H],   ["m","n"],            ["layout","a_transpose"],                              ["a"],      ["b"],                        [amn,bnma],      ["alpha"],        "",    "Scaling and out-place transpose/copy (non-BLAS function)", "Performs scaling and out-of-place transposition/copying of matrices according to _B = alpha*op(A)_, in which _A_ is an input matrix (_m_ rows by _n_ columns), _B_ an output matrix, and _alpha_ a scalar value. The operation _op_ can be a normal matrix copy, a transposition or a conjugate transposition.", [ald_m, bld_n]),
  Routine(True,  True,  0, False, "x", "im2col",   T, [S,D,C,Z,H],   im2col_constants,     ["kernel_mode"],                                       ["im"],     ["col"],                      [im,col],        [""],             "",    "Im2col function (non-BLAS function)", "Performs the im2col algorithm, in which _im_ is the input matrix and _col_ is the output matrix. Overwrites any existing values in the _col_ buffer", []),
  Routine(True,  True,  0, False, "x", "col2im",   T, [S,D,C,Z,H],   im2col_constants,     ["kernel_mode"],                                       ["col"],    ["im"],                       [col,im],        [""],             "",    "Col2im function (non-BLAS function)", "Performs the col2im algorithm, in which _col_ is the input matrix and _im_ is the output matrix. Accumulates results on top of the existing values in the _im_ buffer.", []),
  Routine(True,  True,  0, False, "x", "convgemm", T, [S,D,H],       convgemm_constants,   ["kernel_mode"],                                       ["im","kernel"], ["result"],              [imb,kernel,result],[""],          "",    "Batched convolution as GEMM (non-BLAS function)", "Integrates im2col and GEMM for batched 3D convolution, in which _im_ is the 4D input tensor (NCHW - batch-channelin-height-width), _kernel_ the 4D kernel weights tensor (KCHW - channelout-channelin-height-width), and _result_ the 4D output tensor (NCHW - batch-channelout-height-width).", []),
  # Batched routines:
  Routine(True,  True,  1, False, "x", "axpy",     T, [S,D,C,Z,H],   ["n"],                [],                                                    ["x"],      ["y"],                        [xn,yn],         ["alpha"],        "",    "Batched version of AXPY", "As AXPY, but multiple operations are batched together for better performance.", []),
  Routine(True,  True,  1, False, "x", "gemm",     T, [S,D,C,Z,H],   ["m","n","k"],        ["layout","a_transpose","b_transpose"],                ["a","b"],  ["c"],                        [amk,bkn,cmn],   ["alpha","beta"], "",    "Batched version of GEMM", "As GEMM, but multiple operations are batched together for better performance.", [ald_transa_m_k, bld_transb_k_n, cld_m]),
  Routine(True,  True,  2, False, "x", "gemm",     T, [S,D,C,Z,H],   ["m","n","k"],        ["layout","a_transpose","b_transpose"],                ["a","b"],  ["c"],                        [amk,bkn,cmn],   ["alpha","beta"], "",    "StridedBatched version of GEMM", "As GEMM, but multiple strided operations are batched together for better performance.", [ald_transa_m_k, bld_transb_k_n, cld_m]),
]]


def main(argv):

    # Parses the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("clblast_root", help="Root of the CLBlast sources")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase verbosity of the script")
    cl_args = parser.parse_args(argv)
    library_root = cl_args.clblast_root

    # Checks whether the command-line arguments are valid; exists otherwise
    for f in FILES:
        if not os.path.isfile(library_root + f):
            print("[ERROR] The path '" + library_root + "' does not point to the root of the CLBlast library")
            sys.exit()

    # Iterates over all regular files to output
    for i in range(0, len(FILES)):

        # Stores the header and the footer of the original file
        with open(library_root + FILES[i]) as f:
            original = f.readlines()
        file_header = original[:HEADER_LINES[i]]
        file_footer = original[-FOOTER_LINES[i]:]

        # Re-writes the body of the file
        with open(library_root + FILES[i], "w") as f:
            body = ""
            levels = [1, 2, 3] if (i == 4 or i == 5 or i == 6) else [1, 2, 3, 4]
            for level in levels:
                if i not in [11]:
                    body += cpp.LEVEL_SEPARATORS[level - 1] + "\n"
                for routine in ROUTINES[level - 1]:
                    if i == 0:
                        body += cpp.clblast_h(routine)
                    if i == 1:
                        body += cpp.clblast_cc(routine)
                    if i == 2:
                        body += cpp.clblast_c_h(routine)
                    if i == 3:
                        body += cpp.clblast_c_cc(routine)
                    if i == 4:
                        body += cpp.wrapper_clblas(routine)
                    if i == 5:
                        body += cpp.wrapper_cblas(routine)
                    if i == 6:
                        body += cpp.wrapper_cublas(routine)
                    if i == 7:
                        if routine.batched == 0 and routine.name not in ["convgemm"]:
                            body += cpp.clblast_netlib_c_h(routine)
                    if i == 8:
                        if routine.batched == 0 and routine.name not in ["convgemm"]:
                            body += cpp.clblast_netlib_c_cc(routine)
                    if i == 9:
                        body += cpp.clblast_h(routine, cuda=True)
                    if i == 10:
                        body += cpp.clblast_cc(routine, cuda=True)
                    if i == 11:
                        body += pyclblast.generate_pyx(routine)
            f.write("".join(file_header))
            f.write(body)
            f.write("".join(file_footer))

    # Outputs all the test implementations
    for level in [1, 2, 3, 4]:
        for routine in ROUTINES[level - 1]:
            if routine.has_tests:
                level_string = cpp.LEVEL_NAMES[level - 1]
                routine_suffix = "level" + level_string + "/x" + routine.lowercase_name() + ".cpp"

                # Correctness tests
                filename = library_root + "/test/correctness/routines/" + routine_suffix
                with open(filename, "w") as f:
                    f.write(cpp.HEADER + "\n")
                    f.write(cpp.correctness_test(routine, level_string))
                    f.write(cpp.FOOTER)

                # Performance tests
                filename = library_root + "/test/performance/routines/" + routine_suffix
                with open(filename, "w") as f:
                    f.write(cpp.HEADER + "\n")
                    f.write(cpp.performance_test(routine, level_string))
                    f.write(cpp.FOOTER)

    # API documentation
    filename = cl_args.clblast_root + "/doc/api.md"

    # Stores the header and the footer of the original documentation file
    with open(filename) as f:
        original = f.readlines()
    file_header = original[:HEADER_LINES_DOC]
    file_footer = original[-FOOTER_LINES_DOC:]

    # Outputs the API documentation
    with open(filename, "w") as f:

        # Outputs the header
        f.write("".join(file_header))
        doc_header = doc.header()
        f.write(doc_header)

        # Generates the documentation for each routine
        for level in [1, 2, 3, 4]:
            for routine in ROUTINES[level - 1]:
                if routine.implemented:
                    doc_routine = doc.generate(routine)
                    f.write(doc_routine)

        # Outputs the footer
        f.write("".join(file_footer))

if __name__ == '__main__':
    main(sys.argv[1:])
