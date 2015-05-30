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


// Problem iterator to scatter solving, for passing over matrix A

#include <assert.h>
#include <sys/types.h>
#include <clblas_stddef.h>

#include "matrix_dims.h"
#include "problem_iter.h"

void VISIBILITY_HIDDEN
initProblemIterator(
    ProblemIterator *iter,
    BlasFunctionID funcID,
    MatrixRole mrole,
    CLBlasKargs *kargs,
    size_t maxPanels,
    size_t maxBlocks,
    SubproblemDim *topDim)
{
    SubproblemDim tmp;

    iter->mrole = mrole;
    iter->funcID = funcID;
    kargsToProbDims(&tmp, funcID, kargs, false);
    iter->size = matrBlockHeight(&tmp, mrole, kargs->side);
    iter->globPitch = matrBlockPitch(&tmp, mrole, kargs->dtype, kargs->side);
    iter->maxPanels = maxPanels;
    iter->maxBlocks = maxBlocks;
    iter->uplo = kargs->uplo;
    iter->side = kargs->side;
    iter->dtype = kargs->dtype;
    iter->bpitch = matrBlockPitch(topDim, mrole, kargs->dtype, kargs->side);
    iter->bheight = matrBlockHeight(topDim, mrole, kargs->side);
    iteratorReset(iter);
}

void VISIBILITY_HIDDEN
iteratorReset(ProblemIterator *iter)
{
    if (isIterBackward(iter)) {
        iter->pos = iter->size;
        iter->prevPos = iter->size;
    }
    else {
        iter->pos = 0;
        iter->prevPos = 0;
    }
}

bool VISIBILITY_HIDDEN
isIterBackward(ProblemIterator *iter)
{
    bool ret = false;

    if (iter->funcID != CLBLAS_GEMM) {
        ret = (iter->side == clblasLeft && iter->uplo == clblasLower) ||
              (iter->side == clblasRight && iter->uplo == clblasUpper);
        if (iter->funcID == CLBLAS_TRSM) {
            ret = !ret;
        }
    }

    return ret;
}

int VISIBILITY_HIDDEN
iterateProblem(ProblemIterator *iter)
{
    bool backward;
    size_t dy = 0;

    backward = isIterBackward(iter);

    if (((iter->funcID != CLBLAS_TRSM) && (!iter->maxPanels)) ||
            ((iter->funcID == CLBLAS_TRSM) && (!iter->maxBlocks))) {
        iter->pos = (backward) ? 0 : iter->size;
        return 1;
    }

    iter->prevPos = iter->pos;

    if ((iter->funcID != CLBLAS_TRSM)) {
        dy = iter->maxPanels * iter->bheight;
        assert(dy != 0);
    }
    if (backward) {
        dy = szmin(iter->pos, dy);
        iter->pos -= dy;
    }
    else {
        dy = szmin(dy, iter->size - iter->pos);
        iter->pos += dy;
    }

    return (int)(backward && iter->pos == 0) ||
                (!backward && iter->pos == iter->size);
}

size_t VISIBILITY_HIDDEN
iterLastOffset(ProblemIterator *iter)
{
    return (iter->pos > iter->prevPos) ? (iter->pos - iter->prevPos) :
           (iter->prevPos - iter->pos);
}
