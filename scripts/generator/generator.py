#!/usr/bin/env python

# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# This script automatically generates the bodies of the following files, creating the full CLBlast
# API interface and implementation (C, C++, and reference BLAS wrappers):
#    clblast.h
#    clblast.cc
#    clblast_c.h
#    clblast_c.cc
#    wrapper_clblas.h
#    wrapper_cblas.h
# It also generates the main functions for the correctness and performance tests as found in
#    test/correctness/routines/levelX/xYYYY.cc
#    test/performance/routines/levelX/xYYYY.cc
# It also produces the API documentation found in doc/clblast.md
#
# ==================================================================================================

# System modules
import sys
import os.path

# Local files
from routine import Routine
from datatype import DataType, HLF, FLT, DBL, FLT2, DBL2, HCL, F2CL, D2CL

# ==================================================================================================

# Regular data-types
H = DataType("H", "H", HLF,  [HLF,  HLF,  HCL,  HCL],  HLF ) # half (16)
S = DataType("S", "S", FLT,  [FLT,  FLT,  FLT,  FLT],  FLT ) # single (32)
D = DataType("D", "D", DBL,  [DBL,  DBL,  DBL,  DBL],  DBL ) # double (64)
C = DataType("C", "C", FLT2, [FLT2, FLT2, F2CL, F2CL], FLT2) # single-complex (3232)
Z = DataType("Z", "Z", DBL2, [DBL2, DBL2, D2CL, D2CL], DBL2) # double-complex (6464)

# Special cases
Sc = DataType("C", "Sc", FLT2,         [FLT2, FLT2, FLT2, FLT2], FLT2) # As C, but with real output
Dz = DataType("Z", "Dz", DBL2,         [DBL2, DBL2, DBL2, DBL2], DBL2) # As Z, but with real output
iH = DataType("H", "iH", HLF,          [HLF,  HLF,  HLF,  HLF],  HLF ) # As H, but with integer output
iS = DataType("S", "iS", FLT,          [FLT,  FLT,  FLT,  FLT],  FLT ) # As S, but with integer output
iD = DataType("D", "iD", DBL,          [DBL,  DBL,  DBL,  DBL],  DBL ) # As D, but with integer output
iC = DataType("C", "iC", FLT2,         [FLT2, FLT2, F2CL, F2CL], FLT2) # As C, but with integer output
iZ = DataType("Z", "iZ", DBL2,         [DBL2, DBL2, D2CL, D2CL], DBL2) # As Z, but with integer output
Css = DataType("C", "C", FLT,          [FLT,  FLT,  FLT,  FLT], FLT2) # As C, but with constants from S
Zdd = DataType("Z", "Z", DBL,          [DBL,  DBL,  DBL,  DBL], DBL2) # As Z, but with constants from D
Ccs = DataType("C", "C", FLT2+","+FLT, [FLT2, FLT,  F2CL, FLT], FLT2) # As C, but with one constant from S
Zzd = DataType("Z", "Z", DBL2+","+DBL, [DBL2, DBL,  D2CL, DBL], DBL2) # As Z, but with one constant from D

# C++ template data-types
T = DataType("T", "typename T", "T", ["T", "T", "T", "T"], "T") # regular routine
Tc = DataType("Tc", "typename T", "std::complex<T>,T", ["T", "T", "T", "T"], "std::complex<T>") # for herk
TU = DataType("TU", "typename T, typename U", "T,U", ["T", "U", "T", "U"], "T") # for her2k

# ==================================================================================================

# Different possibilities for requirements
ald_m = "The value of `a_ld` must be at least `m`."
ald_n = "The value of `a_ld` must be at least `n`."
ald_k_one = "The value of `a_ld` must be at least `k + 1`."
ald_kl_ku_one = "The value of `a_ld` must be at least `kl + ku + 1`."
ald_transa_m_k = "When `transpose_a == Transpose::kNo`, then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `k`."
ald_trans_n_k = "When `transpose == Transpose::kNo`, then `a_ld` must be at least `n`, otherwise `a_ld` must be at least `k`."
ald_side_m_n = "When `side = Side::kLeft` then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `n`."
bld_m = "The value of `b_ld` must be at least `m`."
bld_transb_k_n = "When `transpose_b == Transpose::kNo`, then `b_ld` must be at least `k`, otherwise `b_ld` must be at least `n`."
bld_trans_n_k = "When `transpose == Transpose::kNo`, then `b_ld` must be at least `n`, otherwise `b_ld` must be at least `k`."
cld_m = "The value of `c_ld` must be at least `m`."
cld_n = "The value of `c_ld` must be at least `n`."

# ==================================================================================================

# Populates a list of routines
routines = [
[ # Level 1: vector-vector
  Routine(False, True,  "1", "rotg",  T, [S,D],            [], [], [], ["sa","sb","sc","ss"], [], "", "Generate givens plane rotation", "", []),
  Routine(False, True,  "1", "rotmg", T, [S,D],            [], [], ["sy1"], ["sd1","sd2","sx1","sparam"], [], "", "Generate modified givens plane rotation", "", []),
  Routine(False, True,  "1", "rot",   T, [S,D],            ["n"], [], [], ["x","y"], ["cos","sin"], "", "Apply givens plane rotation", "", []),
  Routine(False, True,  "1", "rotm",  T, [S,D],            ["n"], [], [], ["x","y","sparam"], [], "", "Apply modified givens plane rotation", "", []),
  Routine(True,  True,  "1", "swap",  T, [S,D,C,Z,H],      ["n"], [], [], ["x","y"], [], "", "Swap two vectors", "Interchanges _n_ elements of vectors _x_ and _y_.", []),
  Routine(True,  True,  "1", "scal",  T, [S,D,C,Z,H],      ["n"], [], [], ["x"], ["alpha"], "", "Vector scaling", "Multiplies _n_ elements of vector _x_ by a scalar constant _alpha_.", []),
  Routine(True,  True,  "1", "copy",  T, [S,D,C,Z,H],      ["n"], [], ["x"], ["y"], [], "", "Vector copy", "Copies the contents of vector _x_ into vector _y_.", []),
  Routine(True,  True,  "1", "axpy",  T, [S,D,C,Z,H],      ["n"], [], ["x"], ["y"], ["alpha"], "", "Vector-times-constant plus vector", "Performs the operation _y = alpha * x + y_, in which _x_ and _y_ are vectors and _alpha_ is a scalar constant.", []),
  Routine(True,  True,  "1", "dot",   T, [S,D,H],          ["n"], [], ["x","y"], ["dot"], [], "n", "Dot product of two vectors", "Multiplies _n_ elements of the vectors _x_ and _y_ element-wise and accumulates the results. The sum is stored in the _dot_ buffer.", []),
  Routine(True,  True,  "1", "dotu",  T, [C,Z],            ["n"], [], ["x","y"], ["dot"], [], "n", "Dot product of two complex vectors", "See the regular xDOT routine.", []),
  Routine(True,  True,  "1", "dotc",  T, [C,Z],            ["n"], [], ["x","y"], ["dot"], [], "n", "Dot product of two complex vectors, one conjugated", "See the regular xDOT routine.", []),
  Routine(True,  True,  "1", "nrm2",  T, [S,D,Sc,Dz,H],    ["n"], [], ["x"], ["nrm2"], [], "2*n", "Euclidian norm of a vector", "Accumulates the square of _n_ elements in the _x_ vector and takes the square root. The resulting L2 norm is stored in the _nrm2_ buffer.", []),
  Routine(True,  True,  "1", "asum",  T, [S,D,Sc,Dz,H],    ["n"], [], ["x"], ["asum"], [], "n", "Absolute sum of values in a vector", "Accumulates the absolute value of _n_ elements in the _x_ vector. The results are stored in the _asum_ buffer.", []),
  Routine(True,  False, "1", "sum",   T, [S,D,Sc,Dz,H],    ["n"], [], ["x"], ["sum"], [], "n", "Sum of values in a vector (non-BLAS function)", "Accumulates the values of _n_ elements in the _x_ vector. The results are stored in the _sum_ buffer. This routine is the non-absolute version of the xASUM BLAS routine.", []),
  Routine(True,  True,  "1", "amax",  T, [iS,iD,iC,iZ,iH], ["n"], [], ["x"], ["imax"], [], "2*n", "Index of absolute maximum value in a vector", "Finds the index of the maximum of the absolute values in the _x_ vector. The resulting integer index is stored in the _imax_ buffer.", []),
  Routine(True,  False, "1", "max",   T, [iS,iD,iC,iZ,iH], ["n"], [], ["x"], ["imax"], [], "2*n", "Index of maximum value in a vector (non-BLAS function)", "Finds the index of the maximum of the values in the _x_ vector. The resulting integer index is stored in the _imax_ buffer. This routine is the non-absolute version of the IxAMAX BLAS routine.", []),
  Routine(True,  False, "1", "min",   T, [iS,iD,iC,iZ,iH], ["n"], [], ["x"], ["imin"], [], "2*n", "Index of minimum value in a vector (non-BLAS function)", "Finds the index of the minimum of the values in the _x_ vector. The resulting integer index is stored in the _imin_ buffer. This routine is the non-absolute minimum version of the IxAMAX BLAS routine.", []),
],
[ # Level 2: matrix-vector
  Routine(True,  True,  "2a", "gemv",  T,  [S,D,C,Z,H], ["m","n"], ["layout","a_transpose"], ["a","x"], ["y"], ["alpha","beta"], "", "General matrix-vector multiplication", "Performs the operation _y = alpha * A * x + beta * y_, in which _x_ is an input vector, _y_ is an input and output vector, _A_ is an input matrix, and _alpha_ and _beta_ are scalars. The matrix _A_ can optionally be transposed before performing the operation.", [ald_m]),
  Routine(True,  True,  "2a", "gbmv",  T,  [S,D,C,Z,H], ["m","n","kl","ku"], ["layout","a_transpose"], ["a","x"], ["y"], ["alpha","beta"], "", "General banded matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is banded instead.", [ald_kl_ku_one]),
  Routine(True,  True,  "2a", "hemv",  T,  [C,Z],       ["n"], ["layout","triangle"], ["a","x"], ["y"], ["alpha","beta"], "", "Hermitian matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is an Hermitian matrix instead.", [ald_n]),
  Routine(True,  True,  "2a", "hbmv",  T,  [C,Z],       ["n","k"], ["layout","triangle"], ["a","x"], ["y"], ["alpha","beta"], "", "Hermitian banded matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is an Hermitian banded matrix instead.", [ald_k_one]),
  Routine(True,  True,  "2a", "hpmv",  T,  [C,Z],       ["n"], ["layout","triangle"], ["ap","x"], ["y"], ["alpha","beta"], "", "Hermitian packed matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is an Hermitian packed matrix instead and represented as _AP_.", []),
  Routine(True,  True,  "2a", "symv",  T,  [S,D,H],     ["n"], ["layout","triangle"], ["a","x"], ["y"], ["alpha","beta"], "", "Symmetric matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is symmetric instead.", [ald_n]),
  Routine(True,  True,  "2a", "sbmv",  T,  [S,D,H],     ["n","k"], ["layout","triangle"], ["a","x"], ["y"], ["alpha","beta"], "", "Symmetric banded matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is symmetric and banded instead.", [ald_k_one]),
  Routine(True,  True,  "2a", "spmv",  T,  [S,D,H],     ["n"], ["layout","triangle"], ["ap","x"], ["y"], ["alpha","beta"], "", "Symmetric packed matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is a symmetric packed matrix instead and represented as _AP_.", []),
  Routine(True,  True,  "2a", "trmv",  T,  [S,D,C,Z,H], ["n"], ["layout","triangle","a_transpose","diagonal"], ["a"], ["x"], [], "n", "Triangular matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is triangular instead.", [ald_n]),
  Routine(True,  True,  "2a", "tbmv",  T,  [S,D,C,Z,H], ["n","k"], ["layout","triangle","a_transpose","diagonal"], ["a"], ["x"], [], "n", "Triangular banded matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is triangular and banded instead.", [ald_k_one]),
  Routine(True,  True,  "2a", "tpmv",  T,  [S,D,C,Z,H], ["n"], ["layout","triangle","a_transpose","diagonal"], ["ap"], ["x"], [], "n", "Triangular packed matrix-vector multiplication", "Same operation as xGEMV, but matrix _A_ is a triangular packed matrix instead and repreented as _AP_.", []),
  Routine(False, True,  "2a", "trsv",  T,  [S,D,C,Z],   ["n"], ["layout","triangle","a_transpose","diagonal"], ["a"], ["x"], [], "", "Solves a triangular system of equations", "", []),
  Routine(False, True,  "2a", "tbsv",  T,  [S,D,C,Z],   ["n","k"], ["layout","triangle","a_transpose","diagonal"], ["a"], ["x"], [], "", "Solves a banded triangular system of equations", "", [ald_k_one]),
  Routine(False, True,  "2a", "tpsv",  T,  [S,D,C,Z],   ["n"], ["layout","triangle","a_transpose","diagonal"], ["ap"], ["x"], [], "", "Solves a packed triangular system of equations", "", []),
  # Level 2: matrix update
  Routine(True,  True,  "2b", "ger",   T,  [S,D,H],     ["m","n"], ["layout"], ["x","y"], ["a"], ["alpha"], "", "General rank-1 matrix update", "Performs the operation _A = alpha * x * y^T + A_, in which _x_ is an input vector, _y^T_ is the transpose of the input vector _y_, _A_ is the matrix to be updated, and _alpha_ is a scalar value.", [ald_m]),
  Routine(True,  True,  "2b", "geru",  T,  [C,Z],       ["m","n"], ["layout"], ["x","y"], ["a"], ["alpha"], "", "General rank-1 complex matrix update", "Same operation as xGER, but with complex data-types.", [ald_m]),
  Routine(True,  True,  "2b", "gerc",  T,  [C,Z],       ["m","n"], ["layout"], ["x","y"], ["a"], ["alpha"], "", "General rank-1 complex conjugated matrix update", "Same operation as xGERU, but the update is done based on the complex conjugate of the input vectors.", [ald_m]),
  Routine(True,  True,  "2b", "her",   Tc, [Css,Zdd],   ["n"], ["layout","triangle"], ["x"], ["a"], ["alpha"], "", "Hermitian rank-1 matrix update", "Performs the operation _A = alpha * x * x^T + A_, in which x is an input vector, x^T is the transpose of this vector, _A_ is the triangular Hermetian matrix to be updated, and alpha is a scalar value.", [ald_n]),
  Routine(True,  True,  "2b", "hpr",   Tc, [Css,Zdd],   ["n"], ["layout","triangle"], ["x"], ["ap"], ["alpha"], "", "Hermitian packed rank-1 matrix update", "Same operation as xHER, but matrix _A_ is an Hermitian packed matrix instead and represented as _AP_.", []),
  Routine(True,  True,  "2b", "her2",  T,  [C,Z],       ["n"], ["layout","triangle"], ["x","y"], ["a"], ["alpha"], "", "Hermitian rank-2 matrix update", "Performs the operation _A = alpha * x * y^T + conj(alpha) * y * x^T + A_, in which _x_ is an input vector and _x^T_ its transpose, _y_ is an input vector and _y^T_ its transpose, _A_ is the triangular Hermetian matrix to be updated, _alpha_ is a scalar value and _conj(alpha)_ its complex conjugate.", [ald_n]),
  Routine(True,  True,  "2b", "hpr2",  T,  [C,Z],       ["n"], ["layout","triangle"], ["x","y"], ["ap"], ["alpha"], "", "Hermitian packed rank-2 matrix update", "Same operation as xHER2, but matrix _A_ is an Hermitian packed matrix instead and represented as _AP_.", []),
  Routine(True,  True,  "2b", "syr",   T,  [S,D,H],     ["n"], ["layout","triangle"], ["x"], ["a"], ["alpha"], "", "Symmetric rank-1 matrix update", "Same operation as xHER, but matrix A is a symmetric matrix instead.", [ald_n]),
  Routine(True,  True,  "2b", "spr",   T,  [S,D,H],     ["n"], ["layout","triangle"], ["x"], ["ap"], ["alpha"], "", "Symmetric packed rank-1 matrix update", "Same operation as xSPR, but matrix _A_ is a symmetric packed matrix instead and represented as _AP_.", []),
  Routine(True,  True,  "2b", "syr2",  T,  [S,D,H],     ["n"], ["layout","triangle"], ["x","y"], ["a"], ["alpha"], "", "Symmetric rank-2 matrix update", "Same operation as xHER2, but matrix _A_ is a symmetric matrix instead.", [ald_n]),
  Routine(True,  True,  "2b", "spr2",  T,  [S,D,H],     ["n"], ["layout","triangle"], ["x","y"], ["ap"], ["alpha"], "", "Symmetric packed rank-2 matrix update", "Same operation as xSPR2, but matrix _A_ is a symmetric packed matrix instead and represented as _AP_.", []),
],
[ # Level 3: matrix-matrix
  Routine(True,  True,  "3", "gemm",  T,  [S,D,C,Z,H], ["m","n","k"], ["layout","a_transpose","b_transpose"], ["a","b"], ["c"], ["alpha","beta"], "", "General matrix-matrix multiplication", "Performs the matrix product _C = alpha * A * B + beta * C_, in which _A_ (_m_ by _k_) and _B_ (_k_ by _n_) are two general rectangular input matrices, _C_ (_m_ by _n_) is the matrix to be updated, and _alpha_ and _beta_ are scalar values. The matrices _A_ and/or _B_ can optionally be transposed before performing the operation.", [ald_transa_m_k, bld_transb_k_n, cld_m]),
  Routine(True,  True,  "3", "symm",  T,  [S,D,C,Z,H], ["m","n"], ["layout","side","triangle"], ["a","b"], ["c"], ["alpha","beta"], "", "Symmetric matrix-matrix multiplication", "Same operation as xGEMM, but _A_ is symmetric instead. In case of `side == kLeft`, _A_ is a symmetric _m_ by _m_ matrix and _C = alpha * A * B + beta * C_ is performed. Otherwise, in case of `side == kRight`, _A_ is a symmtric _n_ by _n_ matrix and _C = alpha * B * A + beta * C_ is performed.", [ald_side_m_n, bld_m, cld_m]),
  Routine(True,  True,  "3", "hemm",  T,  [C,Z],       ["m","n"], ["layout","side","triangle"], ["a","b"], ["c"], ["alpha","beta"], "", "Hermitian matrix-matrix multiplication", "Same operation as xSYMM, but _A_ is an Hermitian matrix instead.", [ald_side_m_n, bld_m, cld_m]),
  Routine(True,  True,  "3", "syrk",  T,  [S,D,C,Z,H], ["n","k"], ["layout","triangle","a_transpose"], ["a"], ["c"], ["alpha","beta"], "", "Rank-K update of a symmetric matrix", "Performs the matrix product _C = alpha * A * A^T + beta * C_ or _C = alpha * A^T * A + beta * C_, in which _A_ is a general matrix and _A^T_ is its transpose, _C_ (_n_ by _n_) is the symmetric matrix to be updated, and _alpha_ and _beta_ are scalar values.", [ald_trans_n_k, cld_m]),
  Routine(True,  True,  "3", "herk",  Tc, [Css,Zdd],   ["n","k"], ["layout","triangle","a_transpose"], ["a"], ["c"], ["alpha","beta"], "", "Rank-K update of a hermitian matrix", "Same operation as xSYRK, but _C_ is an Hermitian matrix instead.", [ald_trans_n_k, cld_m]),
  Routine(True,  True,  "3", "syr2k", T,  [S,D,C,Z,H], ["n","k"], ["layout","triangle","ab_transpose"], ["a","b"], ["c"], ["alpha","beta"], "", "Rank-2K update of a symmetric matrix", "Performs the matrix product _C = alpha * A * B^T + alpha * B * A^T + beta * C_ or _C = alpha * A^T * B + alpha * B^T * A + beta * C_, in which _A_ and _B_ are general matrices and _A^T_ and _B^T_ are their transposed versions, _C_ (_n_ by _n_) is the symmetric matrix to be updated, and _alpha_ and _beta_ are scalar values.", [ald_trans_n_k, bld_trans_n_k, cld_n]),
  Routine(True,  True,  "3", "her2k", TU, [Ccs,Zzd],   ["n","k"], ["layout","triangle","ab_transpose"], ["a","b"], ["c"], ["alpha","beta"], "", "Rank-2K update of a hermitian matrix", "Same operation as xSYR2K, but _C_ is an Hermitian matrix instead.", [ald_trans_n_k, bld_trans_n_k, cld_n]),
  Routine(True,  True,  "3", "trmm",  T,  [S,D,C,Z,H], ["m","n"], ["layout","side","triangle","a_transpose","diagonal"], ["a"], ["b"], ["alpha"], "", "Triangular matrix-matrix multiplication", "Performs the matrix product _B = alpha * A * B_ or _B = alpha * B * A_, in which _A_ is a unit or non-unit triangular matrix, _B_ (_m_ by _n_) is the general matrix to be updated, and _alpha_ is a scalar value.", [ald_side_m_n, bld_m]),
  Routine(False, True,  "3", "trsm",  T,  [S,D,C,Z,H], ["m","n"], ["layout","side","triangle","a_transpose","diagonal"], ["a"], ["b"], ["alpha"], "", "Solves a triangular system of equations", "", []),
]]

# ==================================================================================================
# Translates an option name to a CLBlast data-type
def PrecisionToFullName(x):
	return {
		'H': "Half",
		'S': "Single",
		'D': "Double",
		'C': "ComplexSingle",
		'Z': "ComplexDouble",
	}[x]

# ==================================================================================================
# Separators for the BLAS levels
separators = ["""
// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================""",
"""
// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================""",
"""
// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// ================================================================================================="""]

# Main header/footer for source files
header = """
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================
"""
footer = """
// =================================================================================================
"""

# ==================================================================================================

# The C++ API header (.h)
def clblast_h(routines):
	result = ""
	for routine in routines:
		result += "\n// "+routine.description+": "+routine.ShortNames()+"\n"
		result += routine.RoutineHeaderCPP(12, " = nullptr")+";\n"
	return result

# The C++ API implementation (.cc)
def clblast_cc(routines):
	result = ""
	for routine in routines:
		indent1 = " "*(20 + routine.Length())
		result += "\n// "+routine.description+": "+routine.ShortNames()+"\n"
		if routine.implemented:
			result += routine.RoutineHeaderCPP(12, "")+" {\n"
			result += "  auto queue_cpp = Queue(*queue);\n"
			result += "  auto routine = X"+routine.name+"<"+routine.template.template+">(queue_cpp, event);\n"
			result += "  auto status = routine.SetUp();\n"
			result += "  if (status != StatusCode::kSuccess) { return status; }\n"
			result += "  return routine.Do"+routine.name.capitalize()+"("
			result += (",\n"+indent1).join([a for a in routine.ArgumentsCladuc(routine.template, indent1)])
			result += ");\n"
		else:
			result += routine.RoutineHeaderTypeCPP(12)+" {\n"
			result += "  return StatusCode::kNotImplemented;\n"
		result += "}\n"
		for flavour in routine.flavours:
			indent2 = " "*(34 + routine.Length() + len(flavour.template))
			result += "template StatusCode PUBLIC_API "+routine.name.capitalize()+"<"+flavour.template+">("
			result += (",\n"+indent2).join([a for a in routine.ArgumentsType(flavour)])
			result += ",\n"+indent2+"cl_command_queue*, cl_event*);\n"
	return result

# ==================================================================================================

# The C API header (.h)
def clblast_c_h(routines):
	result = ""
	for routine in routines:
		result += "\n// "+routine.description+": "+routine.ShortNames()+"\n"
		for flavour in routine.flavours:
			result += routine.RoutineHeaderC(flavour, 31, " PUBLIC_API")+";\n"
	return result

# The C API implementation (.cc)
def clblast_c_cc(routines):
	result = ""
	for routine in routines:
		result += "\n// "+routine.name.upper()+"\n"
		for flavour in routine.flavours:
			template = "<"+flavour.template+">" if routine.NoScalars() else ""
			indent = " "*(26 + routine.Length() + len(template))
			result += routine.RoutineHeaderC(flavour, 20, "")+" {\n"
			result += "  auto status = clblast::"+routine.name.capitalize()+template+"("
			result += (",\n"+indent).join([a for a in routine.ArgumentsCast(flavour, indent)])
			result += ",\n"+indent+"queue, event);"
			result += "\n  return static_cast<StatusCode>(status);\n}\n"
	return result

# ==================================================================================================

# The wrapper to the reference clBLAS routines (for performance/correctness testing)
def wrapper_clblas(routines):
	result = ""
	for routine in routines:
		if routine.has_tests:
			result += "\n// Forwards the clBLAS calls for %s\n" % (routine.ShortNamesTested())
			if routine.NoScalars():
				result += routine.RoutineHeaderWrapperCL(routine.template, True, 21)+";\n"
			for flavour in routine.flavours:
				result += routine.RoutineHeaderWrapperCL(flavour, False, 21)+" {\n"

				# There is a version available in clBLAS
				if flavour.precision_name in ["S","D","C","Z"]:
					indent = " "*(17 + routine.Length())
					arguments = routine.ArgumentsWrapperCL(flavour)
					if routine.scratch:
						result += "  auto queue = Queue(queues[0]);\n"
						result += "  auto context = queue.GetContext();\n"
						result += "  auto scratch_buffer = Buffer<"+flavour.template+">(context, "+routine.scratch+");\n"
						arguments += ["scratch_buffer()"]
					result += "  return clblas"+flavour.name+routine.name+"("
					result += (",\n"+indent).join([a for a in arguments])
					result += ",\n"+indent+"num_queues, queues, num_wait_events, wait_events, events);"

				# There is no clBLAS available, forward the call to one of the available functions
				else: # Half-precision
					indent = " "*(24 + routine.Length())

					# Convert to float (note: also integer buffers are stored as half/float)
					for buf in routine.inputs + routine.outputs:
						result += "  auto "+buf+"_buffer_bis = HalfToFloatBuffer("+buf+"_buffer, queues[0]);\n"

					# Call the float routine
					result += "  auto status = clblasX"+routine.name+"("
					result += (",\n"+indent).join([a for a in routine.ArgumentsHalf()])
					result += ",\n"+indent+"num_queues, queues, num_wait_events, wait_events, events);"
					result += "\n"

					# Convert back to half
					for buf in routine.outputs:
						result += "  FloatToHalfBuffer("+buf+"_buffer, "+buf+"_buffer_bis, queues[0]);\n"
					result += "  return status;"

				# Complete
				result += "\n}\n"
	return result

# The wrapper to the reference CBLAS routines (for performance/correctness testing)
def wrapper_cblas(routines):
	result = ""
	for routine in routines:
		if routine.has_tests:
			result += "\n// Forwards the Netlib BLAS calls for %s\n" % (routine.ShortNamesTested())
			for flavour in routine.flavours:
				result += routine.RoutineHeaderWrapperC(flavour, False, 12)+" {\n"

				# There is a version available in CBLAS
				if flavour.precision_name in ["S","D","C","Z"]:
					indent = " "*(10 + routine.Length())
					arguments = routine.ArgumentsWrapperC(flavour)

					# Complex scalars
					for scalar in routine.scalars:
						if flavour.IsComplex(scalar):
							result += "  const auto "+scalar+"_array = std::vector<"+flavour.buffertype[:-1]+">{"+scalar+".real(), "+scalar+".imag()};\n"

					# Special case for scalar outputs
					assignment = ""
					postfix = ""
					endofline = ""
					extra_argument = ""
					for output_buffer in routine.outputs:
						if output_buffer in routine.ScalarBuffersFirst():
							if flavour in [C,Z]:
								postfix += "_sub"
								indent += "    "
								extra_argument += ",\n"+indent+"reinterpret_cast<return_pointer_"+flavour.buffertype[:-1]+">(&"+output_buffer+"_buffer["+output_buffer+"_offset])"
							elif output_buffer in routine.IndexBuffers():
								assignment = "((int*)&"+output_buffer+"_buffer[0])["+output_buffer+"_offset] = "
								indent += " "*len(assignment)
							else:
								assignment = output_buffer+"_buffer["+output_buffer+"_offset]"
								if (flavour.name in ["Sc","Dz"]):
									assignment = assignment+".real("
									endofline += ")"
								else:
									assignment = assignment+" = "
								indent += " "*len(assignment)

					result += "  "+assignment+"cblas_"+flavour.name.lower()+routine.name+postfix+"("
					result += (",\n"+indent).join([a for a in arguments])
					result += extra_argument+endofline+");\n"

				# There is no CBLAS available, forward the call to one of the available functions
				else: # Half-precision
					indent = " "*(9 + routine.Length())

					# Convert to float (note: also integer buffers are stored as half/float)
					for buf in routine.inputs + routine.outputs:
						result += "  auto "+buf+"_buffer_bis = HalfToFloatBuffer("+buf+"_buffer);\n"

					# Call the float routine
					result += "  cblasX"+routine.name+"("
					result += (",\n"+indent).join([a for a in routine.ArgumentsHalf()])
					result += ");\n"

					# Convert back to half
					for buf in routine.outputs:
						result += "  FloatToHalfBuffer("+buf+"_buffer, "+buf+"_buffer_bis);\n"

				# Complete
				result += "}\n"
	return result

# ==================================================================================================

# Checks for the number of command-line arguments
if len(sys.argv) != 2:
	print "[ERROR] Usage: generator.py <root_of_clblast>"
	sys.exit()

# Parses the command-line arguments
path_clblast = sys.argv[1]
files = [
  path_clblast+"/include/clblast.h",
  path_clblast+"/src/clblast.cc",
  path_clblast+"/include/clblast_c.h",
  path_clblast+"/src/clblast_c.cc",
  path_clblast+"/test/wrapper_clblas.h",
  path_clblast+"/test/wrapper_cblas.h",
]
header_lines = [84, 71, 93, 22, 29, 41]
footer_lines = [17, 71, 19, 14, 6, 6]

# Checks whether the command-line arguments are valid; exists otherwise
for f in files:
	if not os.path.isfile(f):
		print "[ERROR] The path '"+path_clblast+"' does not point to the root of the CLBlast library"
		sys.exit()

# ==================================================================================================

# Iterates over all files to output
for i in xrange(0,len(files)):

	# Stores the header and the footer of the original file
	with open(files[i]) as f:
		original = f.readlines()
	file_header = original[:header_lines[i]]
	file_footer = original[-footer_lines[i]:]

	# Re-writes the body of the file
	with open(files[i], "w") as f:
		body = ""
		for level in [1,2,3]:
			body += separators[level-1]+"\n"
			if i == 0:
				body += clblast_h(routines[level-1])
			if i == 1:
				body += clblast_cc(routines[level-1])
			if i == 2:
				body += clblast_c_h(routines[level-1])
			if i == 3:
				body += clblast_c_cc(routines[level-1])
			if i == 4:
				body += wrapper_clblas(routines[level-1])
			if i == 5:
				body += wrapper_cblas(routines[level-1])
		f.write("".join(file_header))
		f.write(body)
		f.write("".join(file_footer))

# ==================================================================================================

# Outputs all the correctness-test implementations
for level in [1,2,3]:
	for routine in routines[level-1]:
		if routine.has_tests:
			filename = path_clblast+"/test/correctness/routines/level"+str(level)+"/x"+routine.name+".cc"
			with open(filename, "w") as f:
				body = ""
				body += "#include \"correctness/testblas.h\"\n"
				body += "#include \"routines/level"+str(level)+"/x"+routine.name+".h\"\n\n"
				body += "// Shortcuts to the clblast namespace\n"
				body += "using float2 = clblast::float2;\n"
				body += "using double2 = clblast::double2;\n\n"
				body += "// Main function (not within the clblast namespace)\n"
				body += "int main(int argc, char *argv[]) {\n"
				body += "  auto errors = size_t{0};\n"
				not_first = "false"
				for flavour in routine.flavours:
					body += "  errors += clblast::RunTests<clblast::TestX"+routine.name+flavour.TestTemplate()
					body += ">(argc, argv, "+not_first+", \""+flavour.name+routine.name.upper()+"\");\n"
					not_first = "true"
				body += "  if (errors > 0) { return 1; } else { return 0; }\n"
				body += "}\n"
				f.write(header+"\n")
				f.write(body)
				f.write(footer)

# Outputs all the performance-test implementations
for level in [1,2,3]:
	for routine in routines[level-1]:
		if routine.has_tests:
			filename = path_clblast+"/test/performance/routines/level"+str(level)+"/x"+routine.name+".cc"
			with open(filename, "w") as f:
				body = ""
				body += "#include \"performance/client.h\"\n"
				body += "#include \"routines/level"+str(level)+"/x"+routine.name+".h\"\n\n"
				body += "// Shortcuts to the clblast namespace\n"
				body += "using float2 = clblast::float2;\n"
				body += "using double2 = clblast::double2;\n\n"
				body += "// Main function (not within the clblast namespace)\n"
				body += "int main(int argc, char *argv[]) {\n"
				default = PrecisionToFullName(routine.flavours[0].precision_name)
				body += "  switch(clblast::GetPrecision(argc, argv, clblast::Precision::k"+default+")) {\n"
				for precision in ["H","S","D","C","Z"]:
					body += "    case clblast::Precision::k"+PrecisionToFullName(precision)+":"
					found = False
					for flavour in routine.flavours:
						if flavour.precision_name == precision:
							body += "\n      clblast::RunClient<clblast::TestX"+routine.name+flavour.TestTemplate()
							body += ">(argc, argv); break;\n"
							found = True
					if not found:
						body += " throw std::runtime_error(\"Unsupported precision mode\");\n"
				body += "  }\n"
				body += "  return 0;\n"
				body += "}\n"
				f.write(header+"\n")
				f.write(body)
				f.write(footer)

# ==================================================================================================

# Outputs the API documentation
filename = path_clblast+"/doc/clblast.md"
with open(filename, "w") as f:

	# Outputs the header
	f.write("CLBlast: API reference\n")
	f.write("================\n")
	f.write("\n\n")

	# Loops over the routines
	for level in [1,2,3]:
		for routine in routines[level-1]:
			if routine.implemented:

				# Routine header
				f.write("x"+routine.name.upper()+": "+routine.description+"\n")
				f.write("-------------\n")
				f.write("\n")
				f.write(routine.details+"\n")
				f.write("\n")

				# Routine API
				f.write("C++ API:\n")
				f.write("```\n")
				f.write(routine.RoutineHeaderCPP(12, "")+"\n")
				f.write("```\n")
				f.write("\n")
				f.write("C API:\n")
				f.write("```\n")
				for flavour in routine.flavours:
					f.write(routine.RoutineHeaderC(flavour, 20, "")+"\n")
				f.write("```\n")
				f.write("\n")

				# Routine arguments
				f.write("Arguments to "+routine.name.upper()+":\n")
				f.write("\n")
				for argument in routine.ArgumentsDoc():
					f.write("* "+argument+"\n")
				f.write("* `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.\n")
				f.write("* `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.\n")
				f.write("\n")

				# Routine requirements
				if len(routine.RequirementsDoc()) > 0:
					f.write("Requirements for "+routine.name.upper()+":\n")
					f.write("\n")
					for requirement in routine.RequirementsDoc():
						f.write("* "+requirement+"\n")
					f.write("\n")

				# Routine footer
				f.write("\n\n")


# ==================================================================================================
