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


#include <blas-math.h>
#include <common.h>

// Type-dependant constants
template <class T>
static cl_double DELTA_0();
template<>
__template_static cl_double DELTA_0<cl_float>()       { return pow(2.0, -20); }
template<>
__template_static cl_double DELTA_0<cl_double>()      { return pow(2.0, -50); }
template<>
__template_static cl_double DELTA_0<FloatComplex>()   { return pow(2.0, -20); }
template<>
__template_static cl_double DELTA_0<DoubleComplex>()  { return pow(2.0, -50); }

size_t
trsmBlockSize(size_t elemSize)
{
    /* TODO: Right now TRSM generators use block size of 16 elements for the
     *       double complex type, and of 32 elements for another types.
     *       If this changes, we have to fetch block size from TRSM generator
     *       somehow.
     */
    return (elemSize == sizeof(DoubleComplex)) ? 16 : 32;
}

template <typename T>
void
trsmDelta(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    T *A,
    size_t lda,
    T *B,
    size_t ldb,
    T alpha,
    cl_double *delta)
{
    cl_double *deltaCLBLAS, s;
    int i, k, j, jStart, jEnd, idx;
    int zinc;
    size_t z = 0;
    size_t bsize;
    bool isUpper;
    T v;

    isUpper = ((uplo == clblasUpper) && (transA == clblasNoTrans)) ||
              ((uplo == clblasLower) && (transA != clblasNoTrans));

    deltaCLBLAS = new cl_double[M * N];
    bsize = trsmBlockSize(sizeof(T));

    if (side == clblasLeft) {
        // Calculate delta of TRSM evaluated with the Gauss' method

        for (k = 0; k < (int)N; k++) {
            if (isUpper) {
                for (i = (int)M - 1; i >= 0; i--) {
                    v = getElement<T>(order, clblasNoTrans, i, k, B, ldb);
                    if (diag == clblasNonUnit) {
                        v = v / getElement<T>(order, transA, i, i, A, lda);
                    }
                    s = module(v) * DELTA_0<T>() * module(alpha);
                    if (i == (int)(M - 1)) {
                        delta[i * N + k] = s;
                    }
                    else {
                        delta[i * N + k] = s + delta[(i + 1) * N + k];
                    }
                    assert(delta[i* N + k] >= 0);
                }
            }
            else {
                for (i = 0; i < (int)M; i++) {
                    v = getElement<T>(order, clblasNoTrans, i, k, B, ldb);
                    if (diag == clblasNonUnit) {
                        v = v / getElement<T>(order, transA, i, i, A, lda);
                    }
                    s = module(v) * DELTA_0<T>() * module(alpha);
                    if (i == 0) {
                        delta[i * N + k] = s;
                    }
                    else {
                        delta[i * N + k] = s + delta[(i - 1) * N + k];
                    }
                    assert(delta[i* N + k] >= 0);
                }
            }
        }

        // Calculate clblas TRSM delta

        for (k = 0; k < (int)N; k++) {
            for (i = 0; i < (int)M; i++) {
                s = 0.0;

                /*
                 *  For the upper triangular matrix the solving process proceeds
                 *  from the bottom to the top, and the bottommost block's
                 *  delta influents most of all. For the lower triangular matrix
                 *  the situation is opposite.
                 */
                if (isUpper) {
                    jStart = i / (int)bsize;
                    // index of the block just after the last matrix block
                    jEnd = ((int)M + (int)bsize - 1) / (int)bsize;
                    z = 1;
                    zinc = 1;
                }
                else {
                    jStart = 0;
                    jEnd = i / (int)bsize + 1;
                    z = jEnd - jStart;
                    zinc = -1;
                }

                for (j = jStart; j < jEnd; j++) {
                    idx = j * (int)bsize + i % (int)bsize;
                    if (idx >= (int)M) {
                        continue;
                    }
                    s += z * delta[idx * N + k];
                    z += zinc;
                }

                deltaCLBLAS[i * N + k] = s * bsize;
                assert(deltaCLBLAS[i* N + k] >= 0);
            }
        }
    }
    else {
        // Calculate delta of TRSM evaluated with the Gauss' method

        for (i = 0; i < (int)M; i++) {
            if (isUpper) {
                for (k = 0; k < (int)N; k++) {
                    v = getElement<T>(order, clblasNoTrans, i, k, B, ldb);
                    if (diag == clblasNonUnit) {
                        v = v / getElement<T>(order, transA, k, k, A, lda);
                    }
                    s = module(v) * DELTA_0<T>() * module(alpha);
                    if (k == 0) {
                        delta[i * N + k] = s;
                    }
                    else {
                        delta[i * N + k] = s + delta[i * N + (k - 1)];
                    }
                    assert(delta[i* N + k] >= 0);
                }
            }
            else {
                for (k = (int)N - 1; k >= 0; k--) {
                    v = getElement<T>(order, clblasNoTrans, i, k, B, ldb);
                    if (diag == clblasNonUnit) {
                        v = v / getElement<T>(order, transA, k, k, A, lda);
                    }
                    s = module(v) * DELTA_0<T>() * module(alpha);
                    if (k == (int)(N - 1)) {
                        delta[i * N + k] = s;
                    }
                    else {
                        delta[i * N + k] = s + delta[i * N + (k + 1)];
                    }
                    assert(delta[i* N + k] >= 0);
                }
            }
        }

        // Calculate clblas TRSM delta

        for (i = 0; i < (int)M; i++) {
            for (k = 0; k < (int)N; k++) {
                s = 0.0;

                /*
                 * Approach is the same as for the left side matrix, but delta
                 * is calculated over the rows rather than the columns.
                 * Now, since the matrices are swapped, the largest and
                 * tightest blocks are swapped as well. Therefore, pass
                 * direction for the upper and lower triangular matrix is also
                 * swapped.
                 */
                if (isUpper) {
                    jStart = 0;
                    jEnd = k / (int)bsize + 1;
                    z = jEnd - jStart;
                    zinc = -1;
                }
                else {
                    jStart = k / (int)bsize;
                    jEnd = (k + (int)bsize - 1) / (int)bsize;
                    z = 1;
                    zinc = 1;
                }

                for (j = jStart; j < jEnd; j++) {
                    idx = j * (int)bsize + k % (int)bsize;
                    if (idx >= (int)N) {
                        continue;
                    }
                    s += z * delta[i * N + idx];
                    z += zinc;
                }

                deltaCLBLAS[i * N + k] = s * bsize;
                assert(deltaCLBLAS[i* N + k] >= 0);
            }
        }
    }

    for (k = 0; k < (int)N; k++) {
        for (i = 0; i < (int)M; i++) {
            delta[i * N + k] += deltaCLBLAS[i * N + k];
        }
    }

    delete[] deltaCLBLAS;
}
