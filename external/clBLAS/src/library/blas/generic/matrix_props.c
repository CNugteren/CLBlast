/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


/*
 * Implementation of functions for determining matrix properties
 */

#include "matrix_props.h"

static bool
gemmIsTrans(KernelExtraFlags flags, MatrixRole mrole)
{
    bool trans = false;
    bool order = false;

    switch (mrole) {
    case MATRIX_A:
        trans = ((flags & KEXTRA_TRANS_A) != 0);
        order = ((flags & KEXTRA_COLUMN_MAJOR) != 0);
        break;
    case MATRIX_B:
        trans = ((flags & KEXTRA_TRANS_B) != 0);
        order = !(flags & KEXTRA_COLUMN_MAJOR);
        break;
    case MATRIX_C:
        trans = false;
        order = ((flags & KEXTRA_COLUMN_MAJOR) != 0);
        break;
    default:
        break;
    }

    // each initial flag "flip" resulting need transposing flag
    return (trans ^ order);
}

static bool
trxmIsTrans(KernelExtraFlags flags, MatrixRole mrole)
{
    bool trans = false;
    bool order = false;
    bool side = ((flags & KEXTRA_SIDE_RIGHT) != 0);
    bool ret;

    switch (mrole) {
    case MATRIX_A:
        trans = ((flags & KEXTRA_TRANS_A) != 0);
        order = ((flags & KEXTRA_COLUMN_MAJOR) != 0);
        break;
    case MATRIX_B:
    case MATRIX_C:
        order = !(flags & KEXTRA_COLUMN_MAJOR); // row major
        break;
    default:
        break;
    }

    // each initial flag "flip" resulting need transposing flag
    ret = trans ^ order ^ side;

    if (mrole == MATRIX_C) {
        /*
         * the output matrix always has inverted transposing flags against
         * matrix B
         */
        ret = !ret;
    }

    return ret;
}

static bool
syrkIsTrans(KernelExtraFlags flags, MatrixRole mrole)
{
    bool ret = false;

    switch (mrole) {
    case MATRIX_A:
    case MATRIX_B:
    {
        bool trans = ((flags & KEXTRA_TRANS_A) != 0);
        bool order = ((flags & KEXTRA_COLUMN_MAJOR) != 0);

        ret = (trans && !order) || (!trans && order);
        break;
    }
    case MATRIX_C:
        ret = ((flags & KEXTRA_COLUMN_MAJOR) != 0);
        break;
    default:
        break;
    }

    return ret;
}

static bool
l2IsTrans(KernelExtraFlags flags, MatrixRole mrole)
{
    bool ret;

    if (mrole == MATRIX_A) {
        bool trans = ((flags & KEXTRA_TRANS_A) != 0);
        bool order = ((flags & KEXTRA_COLUMN_MAJOR) != 0);

        ret = (trans && !order) || (!trans && order);
    }
    else {
        ret = false;
    }

    return ret;
}

bool
isMatrixConj(KernelExtraFlags flags, MatrixRole mrole)
{
    bool ret = false;

    switch (mrole) {
    case MATRIX_A:
        ret = ((flags & KEXTRA_CONJUGATE_A) != 0);
        break;
    case MATRIX_B:
        ret = ((flags & KEXTRA_CONJUGATE_B) != 0);
        break;
    default:
        ret = false;
        break;
    }

    return ret;
}

bool
isMatrixAccessColMaj(
    BlasFunctionID funcID,
    KernelExtraFlags flags,
    MatrixRole mrole)
{
    bool ret = false;

    switch (funcID) {
	case CLBLAS_SYMM:
    case CLBLAS_GEMM:
	case CLBLAS_GEMM2:
        ret = gemmIsTrans(flags, mrole);
        break;
    case CLBLAS_TRMM:
    case CLBLAS_TRSM:
        ret = trxmIsTrans(flags, mrole);
        break;
    case CLBLAS_SYRK:
    case CLBLAS_SYR2K:
        ret = syrkIsTrans(flags, mrole);
        break;
	case CLBLAS_TRMV:
	case CLBLAS_TRSV:
	case CLBLAS_TRSV_GEMV:
		ret = true;
		break;
    case CLBLAS_GEMV:
    case CLBLAS_SYMV:
        ret = l2IsTrans(flags, mrole);
    default:
        break;
    }

    return ret;
}
