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


#ifndef MATRIX_PROPS_H_
#define MATRIX_PROPS_H_

#include <defbool.h>

#include "clblas-internal.h"
#include "blas_funcs.h"
#include "matrix_props.h"

typedef enum MatrixRole {
    MATRIX_A,
    MATRIX_B,
    MATRIX_C,
    MATRIX_ROLES_NUMBER
} MatrixRole;

/*
 * Functions to deal with kernel extra flags
 */

// Is a matrix should be conjugated
bool
isMatrixConj(KernelExtraFlags flags, MatrixRole mrole);

/*
 * Is a matrix accessed in the column-major order
 */
bool
isMatrixAccessColMaj(
    BlasFunctionID funcID,
    KernelExtraFlags flags,
    MatrixRole mrole);

/*
 * Triangularity type at the physical layout with account
 * of solution element indices the largest part makes
 * a contribution to. That means a right-side, non transposed,
 * upper diagonal matrix is considered as the lower triangular
 * since the largest part make a contribution to solution elements
 * with a highest index.
 */
static __inline bool
isMatrixUpper(KernelExtraFlags kflags);

static __inline bool
isMatrixUpper(KernelExtraFlags kflags)
{
    return (((kflags & KEXTRA_UPPER_TRIANG) != 0) ^
            ((kflags & KEXTRA_TRANS_A) != 0) ^
            ((kflags & KEXTRA_SIDE_RIGHT) != 0));
}

#endif /* MATRIX_PROPS_H_ */
