
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>


def precision_to_full_name(x):
    """Translates an option name to a CLBlast data-type"""
    return {
        'H': "Half",
        'S': "Single",
        'D': "Double",
        'C': "ComplexSingle",
        'Z': "ComplexDouble",
    }[x]


def option_to_clblast(x):
    """Translates an option name to a CLBlast data-type"""
    return {
        'layout': "Layout",
        'a_transpose': "Transpose",
        'b_transpose': "Transpose",
        'ab_transpose': "Transpose",
        'side': "Side",
        'triangle': "Triangle",
        'diagonal': "Diagonal",
        'kernel_mode': "KernelMode",
    }[x]


def option_to_clblas(x):
    """As above, but for clBLAS data-types"""
    return {
        'layout': "clblasOrder",
        'a_transpose': "clblasTranspose",
        'b_transpose': "clblasTranspose",
        'ab_transpose': "clblasTranspose",
        'side': "clblasSide",
        'triangle': "clblasUplo",
        'diagonal': "clblasDiag",
    }[x]


def option_to_cblas(x):
    """As above, but for CBLAS data-types"""
    return {
        'layout': "CBLAS_ORDER",
        'a_transpose': "CBLAS_TRANSPOSE",
        'b_transpose': "CBLAS_TRANSPOSE",
        'ab_transpose': "CBLAS_TRANSPOSE",
        'side': "CBLAS_SIDE",
        'triangle': "CBLAS_UPLO",
        'diagonal': "CBLAS_DIAG",
    }[x]


def option_to_cublas(x):
    """As above, but for clBLAS data-types"""
    return {
        'layout': "Layout",
        'a_transpose': "cublasOperation_t",
        'b_transpose': "cublasOperation_t",
        'ab_transpose': "cublasOperation_t",
        'side': "cublasSideMode_t",
        'triangle': "cublasFillMode_t",
        'diagonal': "cublasDiagType_t",
    }[x]


def option_to_documentation(x):
    """Translates an option name to a documentation string"""
    return {
        'layout': "Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.",
        'a_transpose': "Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.",
        'b_transpose': "Transposing the input matrix B, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.",
        'ab_transpose': "Transposing the packed input matrix AP, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.",
        'side': "The position of the triangular matrix in the operation, either on the `Side::kLeft` (141) or `Side::kRight` (142).",
        'triangle': "The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).",
        'diagonal': "The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for non-unit values on the diagonal or `Diagonal::kUnit` (132) for unit values on the diagonal.",
        'kernel_mode': "The kernel mode, either `KernelMode::kCrossCorrelation` for the normal mode, or `KernelMode::kConvolution` for the convolution mode that flips a kernel along `h` and `w` axes.",
    }[x]
