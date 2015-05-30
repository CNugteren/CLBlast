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


#ifndef MATRIX_DIMS_H_
#define MATRIX_DIMS_H_

#include <defbool.h>
#include <clblas-internal.h>
#include <matrix_props.h>
#include <kerngen.h>

#ifdef __cplusplus
extern "C" {
#endif

void
swapDimXY(SubproblemDim *dim);

size_t
matrBlockPitch(
    const SubproblemDim *dim,
    MatrixRole mrole,
    DataType dtype,
    clblasSide side);

cl_ulong
matrBlockSize(
    SubproblemDim *dim,
    MatrixRole mrole,
    DataType dtype,
    clblasSide side);

size_t
matrBlockHeight(
    SubproblemDim *dim,
    MatrixRole mrole,
    clblasSide side);

/*
 * Transform respective kernel arguments to problem dimension.
 * if 'offset' is set to true, then it transform starting offsets
 * to process matrices from, otherwise it transforms matrix sizes.
 * It ignores 'bwidth' field in offset mode.
 */
void
kargsToProbDims(
    SubproblemDim *probDim,
    BlasFunctionID funcID,
    const CLBlasKargs *kargs,
    bool offset);

/*
 * Transform problem dimensions to respective kernel arguments.
 * In the offset mode it ignore 'offsetK' and always sets it to 0
 */
void
probDimsToKargs(
    CLBlasKargs *kargs,
    BlasFunctionID funcID,
    SubproblemDim *blasDim,
    bool offset);

#ifdef __cplusplus
}
#endif

#endif /* MATRIX_DIMS_H_ */
