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


#ifndef PROBLEM_ITERATOR_H_
#define PROBLEM_ITERATOR_H_

#include <kerngen.h>

#include "clblas-internal.h"
#include "blas_funcs.h"

// Problem iterator to scatter solving, for passing over matrix A

typedef struct ProblemIterator {
    MatrixRole mrole;
    size_t pos;
    size_t prevPos;
    size_t size;
    size_t globPitch;
    BlasFunctionID funcID;
    clblasUplo uplo;
    clblasSide side;
    DataType dtype;
    size_t maxPanels;
    size_t maxBlocks;
    size_t bpitch;
    size_t bheight;
} ProblemIterator;

/*
 * @maxBlocks: maximal number of blocks to iterate with;
 *             There is as little as 1 iteration if it is
 *             set to 0.
 */
void VISIBILITY_HIDDEN
initProblemIterator(
    ProblemIterator *iter,
    BlasFunctionID funcID,
    MatrixRole mrole,
    CLBlasKargs *kargs,
    size_t maxPanels,
    size_t maxBlocks,
    SubproblemDim *topDim);

void VISIBILITY_HIDDEN
iteratorReset(ProblemIterator *iter);

bool VISIBILITY_HIDDEN
isIterBackward(ProblemIterator *iter);

/*
 * Iterate in some dimension based on maximal blocks info;
 * Iteration for the 'SDIM_BWIDTH' component is prohibited.
 * Returns 1 when achieve the end position
 */
int VISIBILITY_HIDDEN
iterateProblem(ProblemIterator *iter);

size_t VISIBILITY_HIDDEN
iterLastOffset(ProblemIterator *iter);

#endif /* PROBLEM_ITERATOR_H_ */
