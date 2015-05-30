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


#include <matrix_dims.h>

static __inline bool
isRightSide(BlasFunctionID funcID, clblasSide side)
{
    return ((funcID == CLBLAS_TRMM || funcID == CLBLAS_TRSM) &&
            side == clblasRight);
}

void VISIBILITY_HIDDEN
swapDimXY(SubproblemDim *dim)
{
    size_t tmp;

    tmp = dim->itemX;
    dim->itemX = dim->itemY;
    dim->itemY = tmp;
    tmp = dim->x;
    dim->x = dim->y;
    dim->y = tmp;
}

size_t VISIBILITY_HIDDEN
matrBlockPitch(
    const SubproblemDim *dim,
    MatrixRole mrole,
    DataType dtype,
    clblasSide side)
{
    size_t tsize = dtypeSize(dtype);
    size_t nfloats = tsize / sizeof(cl_float);
    size_t rowLen = 0;

    switch (mrole) {
    case MATRIX_A:
    case MATRIX_B:
        rowLen = dim->bwidth;
        break;
    case MATRIX_C:
        rowLen = (side == clblasLeft) ? dim->x : dim->y;
        break;
    default:
        break;
    }

    rowLen = fl4RowWidth(rowLen, tsize) * FLOAT4_VECLEN / nfloats;

    return rowLen;
}

cl_ulong VISIBILITY_HIDDEN
matrBlockSize(
    SubproblemDim *dim,
    MatrixRole mrole,
    DataType dtype,
    clblasSide side)
{
    size_t height, pitch;

    pitch = matrBlockPitch(dim, mrole, dtype, side);
    height = matrBlockHeight(dim, mrole, side);

    return (cl_ulong)height * pitch;
}

size_t VISIBILITY_HIDDEN
matrBlockHeight(
    SubproblemDim *dim,
    MatrixRole mrole,
    clblasSide side)
{
    size_t ret = 0;

    switch (mrole) {
    case MATRIX_A:
        ret = dim->y;
        break;
    case MATRIX_B:
        ret = dim->x;
        break;
    case MATRIX_C:
        ret = (side == clblasLeft) ? dim->y : dim->x;
        break;
    default:
        break;
    }

    return ret;
}

void VISIBILITY_HIDDEN
kargsToProbDims(
    SubproblemDim *probDim,
    BlasFunctionID funcID,
    const CLBlasKargs *kargs,
    bool offset)
{

    if (funcID == CLBLAS_SYMV) {
        if (offset) {
            probDim->y = kargs->offsetN;
            probDim->x = 0;
            probDim->bwidth = 0;
        }
        else {
            probDim->y = kargs->N;
            probDim->x = kargs->N;
            probDim->bwidth = kargs->K;
        }
    }
    else {
        if (offset) {
            probDim->y = kargs->offsetM;
            probDim->x = kargs->offsetN;
        }
        else {
            probDim->y = kargs->M;
            probDim->x = kargs->N;
        }

        if (isRightSide(funcID, kargs->side)) {
            swapDimXY(probDim);
        }
        if (funcID == CLBLAS_GEMV) {
            if (kargs->transA != clblasNoTrans) {
                swapDimXY(probDim);
            }
            probDim->bwidth = (offset) ? 0 : probDim->x;
        }
        else {
            probDim->bwidth = (offset) ? 0 : kargs->K;
        }
    }
}

void VISIBILITY_HIDDEN
probDimsToKargs(
    CLBlasKargs *kargs,
    BlasFunctionID funcID,
    SubproblemDim *probDim,
    bool offset)
{
    size_t *m, *n;
    SubproblemDim tmpDim;

    if (offset) {
        m = &kargs->offsetM;
        n = &kargs->offsetN;
    }
    else {
        m = &kargs->M;
        n = &kargs->N;
        kargs->K = probDim->bwidth;
    }

    tmpDim = *probDim;

    if (isRightSide(funcID, kargs->side)) {
        swapDimXY(&tmpDim);
    }
    if (funcID == CLBLAS_GEMV) {
        if (kargs->transA != clblasNoTrans) {
            swapDimXY(&tmpDim);
        }
    }
    *m = tmpDim.y;
    *n = tmpDim.x;
}

