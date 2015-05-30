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

#ifndef TRSV_DELTA_H_
#define TRSV_DELTA_H_

#include "delta.h"

static size_t
trsvBlockSize(size_t elemSize)
{
    /* TODO: Right now TRSV generators use block size of 16 elements for the
     *       double complex type, and of 32 elements for another types.
     *       If this changes, we have to fetch block size from TRSV generator
     *       somehow.
     */
    return (elemSize == sizeof(DoubleComplex)) ? 16 : 32;
}

template <typename T>
void
trsvDelta(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    T *A,
    size_t lda,
    T *X,
	int incx,
    cl_double *delta)
{
    cl_double *deltaCLBLAS, s;
    int i, j, jStart, jEnd, idx;
    int zinc;
    size_t z = 0;
    size_t bsize, lenX;
    bool isUpper = false;
	size_t previncxi=0;
    T v;

   	isUpper = ((uplo == clblasUpper) && (transA == clblasNoTrans)) ||
             ((uplo == clblasLower) && (transA != clblasNoTrans));
	// incx = abs(incx);
	lenX = 1 + (N-1)*abs(incx);
    deltaCLBLAS = new cl_double[lenX];
    bsize = trsvBlockSize(sizeof(T));

        // Calculate delta of TRSV evaluated with the Gauss' method

            if (isUpper) {
                for (i = (int)N - 1; i >= 0; i--) {
					size_t incxi;

					incxi = (incx > 0) ? (i*incx) : (N-1-i)*abs(incx);
                    v = getElement<T>(clblasColumnMajor, clblasNoTrans, incxi, 0, X, lenX);
                    if (diag == clblasNonUnit) {
                        T tempA;
                        if(lda > 0)
                        {
                            tempA = getElement<T>(order, transA, i, i, A, lda);
                    }
                        else
                        {
                            tempA = getElementPacked(order, clblasNoTrans, uplo, i, i, A, N);
                        }
                        v = v / tempA;
                    }
                    s = module(v) * DELTA_0<T>();
                    if (i == (int)(N - 1)) {
                        delta[ incxi ] = s;
                    }
                    else {
                        delta[ incxi ] = s + delta[ previncxi ];
                    }
                    assert(delta[ incxi ] >= 0);
					previncxi = incxi;
                }
            }
            else {
                for (i = 0; i < (int)N; i++) {
					size_t incxi;

					incxi = (incx > 0) ? (i*incx) : (N-1-i)*abs(incx);
                    v = getElement<T>(clblasColumnMajor, clblasNoTrans, incxi, 0, X, lenX);
                    if (diag == clblasNonUnit) {
                        T tempA;
                        if(lda > 0)
                        {
                            tempA = getElement<T>(order, transA, i, i, A, lda);
                    }
                        else
                        {
                            tempA = getElementPacked(order, clblasNoTrans, uplo, i, i, A, N);
                        }
                        v = v / tempA;
                    }
                    s = module(v) * DELTA_0<T>();
                    if (i == 0) {
                        delta[ incxi ] = s;
                    }
                    else {
                        delta[ incxi ] = s + delta[ previncxi ];
                    }
                    assert(delta[ incxi ] >= 0);
					previncxi = incxi;
                }
            }

        // Calculate clblas TRSV delta

            for (i = 0; i < (int)N; i++) {
				size_t incxi;
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
                    jEnd = ((int)N + (int)bsize - 1) / (int)bsize;
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
					size_t incxi;

                    idx = j * (int)bsize + i % (int)bsize;
                    if (idx >= (int)N) {
                        continue;
                    }
					incxi = (incx > 0) ? (idx*incx) : (N-1-idx)*abs(incx);
                    s += z * delta[ incxi ];
                    z += zinc;
                }

				incxi = (incx > 0) ? (i*incx) : (N-1-i)*abs(incx);
                deltaCLBLAS[ incxi ] = s * bsize;
                assert(deltaCLBLAS[ incxi ] >= 0);
            }

			for (i = 0; i < (int)N; i++) {
				size_t incxi;

				incxi = (incx > 0) ? (i*incx) : (N-1-i)*abs(incx);
				delta[ incxi ] += deltaCLBLAS[ incxi ];
			}

    delete[] deltaCLBLAS;
}

template <typename T>
void
tbsvDelta(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    size_t K,
    T *A,
    size_t lda,
    T *X,
    int incx,
    cl_double *delta)
{
    cl_double *deltaCLBLAS, s;
    int i, j, jStart, jEnd, idx;
    int zinc;
    size_t z = 0;
    size_t bsize, lenX;
    bool isUpper = false;
    size_t previncxi=0;
    T v;

    isUpper = ((uplo == clblasUpper) && (transA == clblasNoTrans)) ||
             ((uplo == clblasLower) && (transA != clblasNoTrans));
    lenX = 1 + (N-1)*abs(incx);
    deltaCLBLAS = new cl_double[lenX];
    bsize = trsvBlockSize(sizeof(T));

        // Calculate delta of TRSV evaluated with the Gauss' method

            if (isUpper) {
                for (i = (int)N - 1; i >= 0; i--) {
                    size_t incxi;

                    incxi = (incx > 0) ? (i*incx) : (N-1-i)*abs(incx);
                    v = getElement<T>(clblasColumnMajor, clblasNoTrans, incxi, 0, X, lenX);
                    if (diag == clblasNonUnit) {
                        v = v / getElementBanded<T>(order, uplo, i, i, K, A, lda);
                    }
                    s = module(v) * DELTA_0<T>();
                    if (i == (int)(N - 1)) {
                        delta[ incxi ] = s;
                    }
                    else {
                        delta[ incxi ] = s + delta[ previncxi ];
                    }
                    assert(delta[ incxi ] >= 0);
                    previncxi = incxi;
                }
            }
            else {
                for (i = 0; i < (int)N; i++) {
                    size_t incxi;

                    incxi = (incx > 0) ? (i*incx) : (N-1-i)*abs(incx);
                    v = getElement<T>(clblasColumnMajor, clblasNoTrans, incxi, 0, X, lenX);
                    if (diag == clblasNonUnit) {
                        v = v / getElementBanded<T>(order, uplo, i, i, K, A, lda);
                    }
                    s = module(v) * DELTA_0<T>();
                    if (i == 0) {
                        delta[ incxi ] = s;
                    }
                    else {
                        delta[ incxi ] = s + delta[ previncxi ];
                    }
                    assert(delta[ incxi ] >= 0);
                    previncxi = incxi;
                }
            }

        // Calculate clblas TRSV delta

            for (i = 0; i < (int)N; i++) {
                size_t incxi;
                s = 0.0;
                if (isUpper) {
                    jStart = i / (int)bsize;
                    // index of the block just after the last matrix block
                    jEnd = ((int)N + (int)bsize - 1) / (int)bsize;
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
                    size_t incxi;

                    idx = j * (int)bsize + i % (int)bsize;
                    if (idx >= (int)N) {
                        continue;
                    }
                    incxi = (incx > 0) ? (idx*incx) : (N-1-idx)*abs(incx);
                    s += z * delta[ incxi ];
                    z += zinc;
                }

                incxi = (incx > 0) ? (i*incx) : (N-1-i)*abs(incx);
                deltaCLBLAS[ incxi ] = s * bsize;
                assert(deltaCLBLAS[ incxi ] >= 0);
            }

            for (i = 0; i < (int)N; i++) {
                size_t incxi;

                incxi = (incx > 0) ? (i*incx) : (N-1-i)*abs(incx);
                delta[ incxi ] += deltaCLBLAS[ incxi ];
            }

    delete[] deltaCLBLAS;
}
#endif

