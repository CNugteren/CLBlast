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


#ifndef BLAS_RANDOM_H_
#define BLAS_RANDOM_H_

#include <clBLAS.h>
#include <math.h>       // sqrt()

#include <blas-math.h>
#include <test-limits.h>
#include <matrix.h>
#include <testDG.h>

template <typename T>
static void
randomGemmxMatrices(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    clblasTranspose transC,
    size_t M,
    size_t N,
    size_t K,
    bool useAlpha,
    T *alpha,
    T *A,
    size_t lda,
    T *B,
    size_t ldb,
    bool useBeta,
    T *beta,
    T *C,
    size_t ldc)
{
    size_t m, n, k;
    cl_double bound;

    if (!useAlpha) {
        *alpha = random<T>(100);
        if (module(*alpha) == 0.0) {
            *alpha = ONE<T>();
        }
    }

    bound = UPPER_BOUND<T>();
    bound = sqrt(((K - 1) * bound) / (module(*alpha) * K * K));

    for (m = 0; m < M; m++) {
        for (k = 0; k < K; k++) {
            setElement<T>(order, transA, m, k, A, lda, random<T>(bound));
        }
    }

    if (B != NULL) {
        for (k = 0; k < K; k++) {
            for (n = 0; n < N; n++) {
                setElement<T>(order, transB, k, n, B, ldb, random<T>(bound));
            }
        }
    }

    if ((!useBeta) && (beta != NULL)) {
        *beta = random<T>(100);
    }

    if (C != NULL) {
        // if C is not NULL, then beta must not be NULL.
        bound = UPPER_BOUND<T>();
        if (module(*beta) != 0.0) {
            bound = sqrt(bound / (module(*beta) * K));
        }

        for (m = 0; m < M; m++) {
            for (n = 0; n < N; n++) {
                setElement<T>(order, transC, m, n, C, ldc, random<T>(bound));
            }
        }
    }
}

template <typename T>
static void
randomGemmMatrices(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    bool useAlpha,
    T *alpha,
    T *A,
    size_t lda,
    T *B,
    size_t ldb,
    bool useBeta,
    T *beta,
    T *C,
    size_t ldc)
{
    randomGemmxMatrices<T>(order, transA, transB, clblasNoTrans, M, N, K,
                        useAlpha, alpha, A, lda, B, ldb, useBeta, beta, C, ldc);
}

template <typename T>
static void
randomTrmmMatrices(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasDiag diag,
    size_t M,
    size_t N,
    bool useAlpha,
    T *alpha,
    T *A,
    size_t lda,
    T *B,
    size_t ldb)
{
    size_t i, j;
    size_t limA = 0;        /* Matrix A boundary: M or N */

    switch (side) {
    case clblasLeft:
        randomGemmMatrices<T>(order, clblasNoTrans, clblasNoTrans, M, N, M,
            useAlpha, alpha, A, lda, B, ldb, false, NULL, NULL, 0);
        limA = M;
        break;
    case clblasRight:
        randomGemmMatrices<T>(order, clblasNoTrans, clblasNoTrans, M, N, N,
            useAlpha, alpha, B, ldb, A, lda, false, NULL, NULL, 0);
        limA = N;
        break;
    }

    // set to NAN elements which must not be accessed
    for (i = 0; i < limA; i++) {
        switch (uplo) {
        case clblasUpper:
            for (j = 0; j < i; j++) {
                setElement<T>(order, clblasNoTrans, i, j, A, lda, FNAN<T>());
            }
            break;
        case clblasLower:
            for (j = i + 1; j < limA; j++) {
                setElement<T>(order, clblasNoTrans, i, j, A, lda, FNAN<T>());
            }
            break;
        }
    }

    if (diag == clblasUnit) {
        for (i = 0; i < limA; i++) {
            setElement<T>(order, clblasNoTrans, i, i, A, lda, FNAN<T>());
        }
    }
}

template <typename T>
static void
randomTrsmMatrices(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasDiag diag,
    size_t M,
    size_t N,
    bool useAlpha,
    T *alpha,
    T *A,
    size_t lda,
    T *B,
    size_t ldb)
{
    size_t limA, i, j;
    T min, max, x, y;
    cl_double modMin, modMax, sum;

    min = ZERO<T>();
    max = ZERO<T>();

    if (side == clblasLeft) {
        limA = M;
    }
    else {
        limA = N;
    }

    /*
     * Generate max(|a_{ii}|). Determine min(|a_{ii}|).
     * Generate a_{ii} which are constrainted by min/max.
     */
    switch (diag) {
    case clblasUnit:
        for (i = 0; i < limA; i++) {
            // must not be accessed
            setElement<T>(order, clblasNoTrans, i, i, A, lda, ONE<T>());
        }
        break;
    case clblasNonUnit:
        /* Do not allow zeros on A's main diagonal */
        do {
            max = random<T>(TRSM_LIMIT_A<T>());
        } while (module(max) < 1);
        modMax = module(max);
        min = max / 100;
        modMin = module(min);
        setElement<T>(order, clblasNoTrans, 0, 0, A, lda, max);
        for (i = 1; i < limA; i++) {
            x = random<T>(modMin, modMax);
            if (module(x) == 0) {
                x = max;
            }
            setElement<T>(order, clblasNoTrans, i, i, A, lda, x);
        }
        break;
    }

    /* Generate a_{ij} for all j <> i. */
    for (i = 0; i < limA; i++) {
        if (diag == clblasUnit) {
            sum = module(ONE<T>());
        }
        else {
            sum = module(getElement<T>(order, clblasNoTrans, i, i, A, lda));
        }

        for (j = 0; j < limA; j++) {
            if (j == i) {
                continue;
            }

            if (((uplo == clblasUpper) && (j > i)) ||
                ((uplo == clblasLower) && (j < i))) {
                // useful element
                if (sum >= 1.0) {
                    x = random<T>(sum / sqrt((double)limA - j));
                    sum -= module(x);
                }
                else {
                    x = ZERO<T>();
                }
            }
            else {
                // must not be accessed
                x = FNAN<T>();
            }

            setElement<T>(order, clblasNoTrans, i, j, A, lda, x);
        }
    }

    /* Generate matrix B. */
    switch (side) {
    case clblasLeft:
        for (j = 0; j < N; j++) {
            sum = TRSM_LIMIT_B<T>();
            for (i = 0; i < M; i++) {
                x = getElement<T>(order, clblasNoTrans, i, i, A, lda);
                y = ZERO<T>();
                if (sum >= 0.0) {
                    y = random<T>(sum * module(x) / sqrt((double)M - i));
                    sum -= module(y) / module(x);
                }
                setElement<T>(order, clblasNoTrans, i, j, B, ldb, y);
                if ((i == 0) && (j == 0)) {
                    min = y;
                }
                else if (module(y) < module(min)) {
                    min = y;
                }
            }
        }
        break;
    case clblasRight:
        for (i = 0; i < M; i++) {
            sum = TRSM_LIMIT_B<T>();
            for (j = 0; j < N; j++) {
                x = getElement<T>(order, clblasNoTrans, j, j, A, lda);
                y = ZERO<T>();
                if (sum >= 0.0) {
                    y = random<T>(sum * module(x) / sqrt((double)N - j));
                    sum -= module(y) / module(x);
                }
                setElement<T>(order, clblasNoTrans, i, j, B, ldb, y);
                if ((i == 0) && (j == 0)) {
                    min = y;
                }
                else if (module(y) < module(min)) {
                    min = y;
                }
            }
        }
        break;
    }
    if (diag == clblasUnit) {
        for (i = 0; i < limA; i++) {
            // must not be accessed
            setElement<T>(order, clblasNoTrans, i, i, A, lda, FNAN<T>());
        }
    }

    /* Calculate alpha and adjust B accordingly */
    if (!useAlpha) {
        *alpha = ONE<T>();
    }
    if (module(min) > module(*alpha)) {
        /* FIXME: What exactly next three lines do? */
        *alpha = random<T>(module(min) - 2);
        *alpha = *alpha + ONE<T>();
        *alpha = *alpha + ONE<T>();

        if (module(*alpha) < 1.0) {
            *alpha = ONE<T>();
        }
    }
    if (module(*alpha) != 1.0) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                x = getElement<T>(order, clblasNoTrans, i, j, B, ldb);
                x = x / *alpha;
                setElement<T>(order, clblasNoTrans, i, j, B, ldb, x);
            }
        }
    }
}

template <typename T>
static void
randomTrsvMatrices(
    clblasOrder order,
	clblasUplo uplo,
    clblasDiag diag,
    size_t N,
    T *A,
    size_t lda,
    T *X,
    int incx)
{
	size_t i, j;
    T min, max, x, y;
    cl_double modMin, modMax, sum, maxDiag;

    min = ZERO<T>();
    max = ZERO<T>();
	incx = abs(incx);
    maxDiag = 1.0;

    cl_double bound;
    bound = (UPPER_BOUND<T>()/(N));

    switch (diag) {
    case clblasUnit:
        for (i = 0; i < N; i++) {
            // must not be accessed
            if(lda > 0)
            {
            setElement<T>(order, clblasNoTrans, i, i, A, lda, ONE/*FNAN*/<T>());
        }
            else //Packed case
            {
                setElementPacked<T>(order, clblasNoTrans, uplo, i, i, A, N, ONE/*FNAN*/<T>());
            }
        }
        break;
    case clblasNonUnit:
        /* Do not allow zeros on A's main diagonal and get a big number which is atleast greater than N/4*/
        maxDiag = ((N/4) > bound) ? (bound/4) : (N/4);
        maxDiag = (1 > (maxDiag)) ? 1 : maxDiag;
        do {
            max = randomTrsv<T>(bound);
        } while ((module(max) < (maxDiag)));
        modMax = module(max);
        min = max / 100;
        modMin = module(min);
        if(lda > 0)
        {
        setElement<T>(order, clblasNoTrans, 0, 0, A, lda, max);
        }
        else //Packed Case
        {
            setElementPacked<T>(order, clblasNoTrans, uplo, 0, 0, A, N, max);
        }
        //printf("Diagonals %d ", max);
        for (i = 1; i < N; i++) {
            x = randomTrsv<T>(modMin, modMax);
            if (module(x) < 1) {
                x = max;
            }
            //printf("%d ", x);
            /*if(module(x) < 1)
            {
                printf("WARNING: Diagonal less than one\n");
            }*/
            if(lda > 0)
            {
            setElement<T>(order, clblasNoTrans, i, i, A, lda, x);
        }
            else
            {
                setElementPacked<T>(order, clblasNoTrans, uplo, i, i, A, N, x);
            }
        }
       // printf("\n");
        break;
    }

    /* Generate a_{ij} for all j <> i. */
    for (i = 0; i < N; i++) {

        if (diag == clblasUnit) {
            sum = module(ONE<T>());
        }
        else {
            T temp;
            if(lda > 0)
            {
                temp = getElement<T>(order, clblasNoTrans, i, i, A, lda);
        }
            else
            {
                temp = getElementPacked<T>(order, clblasNoTrans, uplo, i, i, A, N);
            }
            sum = module(temp);
        }

        for (j = 0; j < N; j++) {
            if (j == i) {
                continue;
            }

            if (((uplo == clblasUpper) && (j > i)) ||
                ((uplo == clblasLower) && (j < i)))
            {
                x = randomTrsv<T>(sum/N);
                }
                else {
                // must not be accessed
                x = FNAN<T>();
            }
            if(lda > 0)
            {
            setElement<T>(order, clblasNoTrans, i, j, A, lda, x);
        }
            else //Packed Case.
            {
                setElementPacked<T>(order, clblasNoTrans, uplo, i, j, A, N, x);
    }
        }
    }

    /* Generate matrix X. */
    sum = TRSM_LIMIT_B<T>();
    for (i = 0; i < N; i++) {
        if(lda > 0)
        {
        x = getElement<T>(order, clblasNoTrans, i, i, A, lda);
        }
        else //Packed Case.
        {
            x = getElementPacked<T>(order, clblasNoTrans, uplo, i, i, A, N);
        }
        sum = module(x);
        y = randomTrsv<T>(sum/N);
        setElement<T>(clblasColumnMajor, clblasNoTrans, (i * abs(incx)), 0, X, (1 + (N-1)*abs(incx)), y);
        if (i == 0) {
            min = y;
        }
        else if (module(y) < module(min)) {
            min = y;
        }
    }
}

template <typename T>
static void
randomSyrMatrices(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    bool useAlpha,
    T *alpha,
    T *A,
    size_t lda,
    T *X,
	int incx
    )
{
    size_t i, j;
	size_t lengthX;
    cl_double bound;

    if (!useAlpha) {
        *alpha = random<T>(100);
        if (module(*alpha) == 0.0) {
            *alpha = ONE<T>();
        }
    }
	#ifdef DEBUG_SYR
	printf("ALPHA in randomSyrMatrices %f\n", *alpha);
	#endif

	// bound is calculated by solving the equation (alpha*x^2 + x - UPPER_BOUND) < 0

	bound = UPPER_BOUND<T>();

	if(module(*alpha) > (sqrt(bound) / (2.0)))
		*alpha = random<T>((sqrt(bound) / (2.0)));

	#ifdef DEBUG_SYR
	printf("ALPHA in randomSyrMatrices after check %f bound for alpha is %f\n", *alpha, (sqrt(bound) / (2.0)));
	#endif

	bound = bound / module(*alpha);

    bound = sqrt( ((((1.0) / module(*alpha)) / (4.0)) / module(*alpha)) + bound) - ((1.0) / ((2.0) * (*alpha)));

	#ifdef DEBUG_SYR
	printf("BOUND : %f alpha %f \n", bound, *alpha);
	#endif

     if( lda )
    {
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            setElement<T>(order, clblasNoTrans, i, j, A, lda, random<T>(bound));
        }
    }
    } else {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                setElementPacked<T>(order, clblasNoTrans, uplo, i, j, A, N, random<T>(bound));
            }
        }
    }


	lengthX = 1 + ((N - 1) * abs(incx));
    if (X != NULL) {
        for (i = 0; i < lengthX; i++) {
			X[i] = random<T>(bound);
        }
    }
}

template <typename T>
static void
randomSyr2Matrices(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    bool useAlpha,
    T *alpha,
    T *A,
    size_t lda,
    T *X,
	int incx,
	T *Y,
	int incy
    )
{
    size_t i, j;
	size_t lengthX;
    size_t lengthY;
	cl_double bound;

    if (!useAlpha) {
        *alpha = random<T>(100);
        if (module(*alpha) == 0.0) {
            *alpha = ONE<T>();
        }
    }
	#ifdef DEBUG_SYR2
	printf("ALPHA in randomSyr2Matrices %f\n", *alpha);
	#endif

	// bound is calculated by solving the equation (2*alpha*x^2 + x - UPPER_BOUND) < 0

	bound = UPPER_BOUND<T>();

	if(module(*alpha) > (sqrt(bound) / (4.0)))
		*alpha = random<T>((sqrt(bound) / (4.0)));

	#ifdef DEBUG_SYR2
	printf("ALPHA in randomSyrMatrices after check %f bound for alpha is %f\n", *alpha, (sqrt(bound) / (2.0)));
	#endif

	bound = bound / ( 2 * module(*alpha));

    bound = sqrt( ((((1.0) / module(*alpha)) / (16.0)) / module(*alpha)) + bound) - ((1.0) / ((4.0) * (*alpha)));

	#ifdef DEBUG_SYR2
	printf("BOUND : %f alpha %f \n", bound, *alpha);
	#endif

    if( lda )
    {
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            setElement<T>(order, clblasNoTrans, i, j, A, lda, random<T>(bound));
        }
    }
    } else {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                setElementPacked<T>(order, clblasNoTrans, uplo, i, j, A, N, random<T>(bound));
            }
        }
    }

	lengthX = 1 + ((N - 1) * abs(incx));
    if (X != NULL) {
        for (i = 0; i < lengthX; i++) {
			X[i] = random<T>(bound);
        }
    }
	lengthY = 1 + (N - 1) * abs(incy);
	if (Y != NULL) {
		for (i = 0; i < lengthY; i++) {
			Y[i] = random<T>(bound);
		}
	}
}

template <typename T>
static void
randomHemvMatrices(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    bool useAlpha,
    T *alpha,
    T *A,
    size_t lda,
    T *X,
    int incx,
	bool useBeta,
	T *beta,
    T *Y,
    int incy
    )
{
    size_t i, j;
    size_t lengthX;
    size_t lengthY;
    cl_double bound;
	cl_double fAlpha, fBeta;

    if (!useAlpha) {
        *alpha = random<T>(100);
        if (module(CREAL(*alpha)) == 0.0) {
            CREAL(*alpha) = 1.0;
        }
    }

	if (!useBeta) {
        *beta = random<T>(100);
        if (module(CREAL(*beta)) == 0.0) {
            CREAL(*beta) = 1.0;
        }
    }

    #ifdef DEBUG_HEMV
    printf("ALPHA in randomSyr2Matrices %f.%f\n", CREAL(*alpha), CIMAG(*alpha));
    printf("BETA in randomSyr2Matrices %f.%f\n", CREAL(*beta), CIMAG(*beta));
    #endif

    // bound is calculated by solving the equation (2*alpha*x^2 + x - UPPER_BOUND) < 0

    bound = UPPER_BOUND<T>();

    if((module(CREAL(*alpha)) > bound) || (module(CIMAG(*alpha)) > bound))
        *alpha = random<T>((sqrt(bound) / ((2.0) * N)));
	if (module(CREAL(*alpha)) == 0.0) {
            CREAL(*alpha) = 1.0;
    }

	if((module(CREAL(*beta)) > bound) || (module(CIMAG(*beta)) > bound))
        *beta = random<T>((sqrt(bound)));
	if (module(CREAL(*beta)) == 0.0) {
            CREAL(*beta) = 1.0;
    }

    #ifdef DEBUG_HEMV
    printf("ALPHA in randomSyrMatrices after check %f.%f bound for alpha is %f\n", CREAL(*alpha), CIMAG(*alpha), (sqrt(bound) / ((2.0) * N)));
    #endif

	fAlpha = (module(CREAL(*alpha)) > module(CIMAG(*alpha))) ? module(CREAL(*alpha)) : module(CIMAG(*alpha));
	fBeta  = (module(CREAL(*beta)) > module(CIMAG(*beta))) ? module(CREAL(*beta)) : module(CIMAG(*beta));

    bound = bound / (fAlpha * N);

    bound = sqrt( ((((((fBeta * fBeta)) / fAlpha) / (4.0)) / fAlpha) / (N * N)) + bound) - ((fBeta) / ((2.0) * (fAlpha) * N));

    #ifdef DEBUG_HEMV
    printf("BOUND : %f \n", bound);
    #endif

    if( lda )
    {
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            setElement<T>(order, clblasNoTrans, i, j, A, lda, random<T>(bound));
        }
    }
    } else {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                setElementPacked<T>(order, clblasNoTrans, uplo, i, j, A, N, random<T>(bound));
            }
        }
    }

    lengthX = 1 + ((N - 1) * abs(incx));
    if (X != NULL) {
        for (i = 0; i < lengthX; i++) {
            X[i] = random<T>(bound);
        }
    }
    lengthY = 1 + (N - 1) * abs(incy);
    if (Y != NULL) {
        for (i = 0; i < lengthY; i++) {
            Y[i] = random<T>(bound);
        }
    }
}

template <typename T>
static void randomVectors(
    size_t N,
    T *X,
    int incx,
    T *Y = NULL,
    int incy = 0,
    bool considerN=false
    )
{
    cl_double quotient = (considerN)? N: 1.0;
    cl_double bound = sqrt( UPPER_BOUND<T>()/quotient ) / 2;     // sqrt for the alpha factor and 2 for addition

    int length = 1 + ((N - 1) * abs(incx));
    for(int i=0; i<length; i++) {
        X[i] = random<T>(bound);
    }

    if(Y != NULL)
    {
        length = 1 + ((N - 1) * abs(incy));
        for(int i=0; i<length; i++) {
            Y[i] = random<T>(bound);
        }
    }
}

// testDG
template <typename T>
static void
setElementWithRandomData(T *p, int vectorLength, cl_double bound)
{
	for(int k=0; k<vectorLength; k++)
		p[k] = random<T>(bound);
}

template <typename T>
static void
setElementWithUnity(T *p, int vectorLength)
{
	p[0] = (T)1.0;
	if ( vectorLength == 2)
	{
		p[1] = 0.0f;
	}
}


template <typename T>
static void
setElementWithZero(T *p, int vectorLength)
{
	for(int k=0; k<vectorLength; k++)
		p[k] = (T)0.0;
}



template <typename T>
static void
setDiagonalUnityOrNonUnity(int unity, T* data, size_t rows, size_t cols, size_t lda, int vectorLength, int creationFlags, cl_double bound)
{

	if (creationFlags & PACKED_MATRIX)
	{

		// Rows = Cols for PACKED Matrix
		for(size_t i=0;i< rows;i++)
		{
			if (creationFlags & UPPER_HALF_ONLY)
    			{
				(unity==1)? setElementWithUnity( ((creationFlags & ROW_MAJOR_ORDER))?RMUPacked(i,i):RMLPacked(i,i), vectorLength):
                      			(unity == 0)? setElementWithZero( ((creationFlags & ROW_MAJOR_ORDER))?RMUPacked(i,i):RMLPacked(i,i), vectorLength):
                      				setElementWithRandomData( ((creationFlags & ROW_MAJOR_ORDER))?RMUPacked(i,i):RMLPacked(i,i), vectorLength, bound);
    			}
    			else
    			{
          			(unity==1)? setElementWithUnity( (creationFlags & ROW_MAJOR_ORDER)?RMLPacked(i,i):RMUPacked(i,i), vectorLength):
                      			(unity==0)? setElementWithZero( (creationFlags & ROW_MAJOR_ORDER)?RMLPacked(i,i):RMUPacked(i,i), vectorLength):
                      				setElementWithRandomData( (creationFlags & ROW_MAJOR_ORDER)?RMLPacked(i,i):RMUPacked(i,i) , vectorLength, bound);
    			}
		}
	}
	else
	{
		// Row Major - rows x lda
		// Col major - lda x cols
		size_t firstdimension;
		T *p;

		if (creationFlags & ROW_MAJOR_ORDER)
		{
			firstdimension = rows;
		} else {
			firstdimension = cols;
		}

		for(size_t i=0; i<firstdimension; i++)
		{
			p = (T *)data + (i*lda)*vectorLength;
			p += i*vectorLength;

			if (unity == 0)
			{
				setElementWithZero(p, vectorLength);
			}
			else if (unity == 1)
			{
				setElementWithUnity(p, vectorLength);
			}
			else
			{
				setElementWithRandomData(p, vectorLength, bound);
			}
		}
	}
}

template <typename T>
static void
setTriangularMatrixWithRandomData(char uplo, T* data, size_t rows, size_t cols, size_t lda, int vectorLength, int creationFlags, cl_double bound)
{

	// Packed Matrix
	if (creationFlags & PACKED_MATRIX)
	{
		if (uplo == 'L')
		{
			for( size_t i=0; i < rows; i++)
			{
				for( size_t j=0; j < i; j++) // Don't touch diagonals
				{
					//setRandom( (flags & ROW_MAJOR) ? RMLPacked(i,j) : CMLPacked(i,j));
					setElementWithRandomData( (creationFlags & ROW_MAJOR_ORDER) ? RMLPacked(i,j) : RMUPacked(j,i), vectorLength, bound);
				}
			}
		}
		else
		{
			for( size_t i=0; i < rows; i++)
			{
				for( size_t j=(i+1); j < cols; j++) // Don't touch diagonals
				{
					//printf("(i,j) -- (%d,%d) : Index : %d\n", i, j, ((i*((2*rows) + 1 - i))/2 + (j -i)));
					setElementWithRandomData( (creationFlags & ROW_MAJOR_ORDER) ? RMUPacked(i,j) : RMLPacked(j,i), vectorLength, bound);
				}
			}
		}
	}
	else
	{
		// Row Major - rows x lda
		// Col major - lda x cols
		size_t firstdimension, seconddimension;
		T *p;

		if ((uplo != 'U') && (uplo != 'L'))
		{
			throw -1;
		}

		if (creationFlags & ROW_MAJOR_ORDER)
		{
			firstdimension = rows;
			seconddimension = cols;
		} else {
			firstdimension = cols;
			seconddimension = rows;
			if (uplo == 'U')
			{
				uplo = 'L';
			} else {
				uplo = 'U';
			}
		}

		for(size_t i=0; i<firstdimension; i++)
		{
			size_t start, end;

			p = (T *)data + (i* lda)*vectorLength;

			// Fill the row
			if ((uplo == 'U') || (uplo == 'u'))
			{
				start = i+1;
				end = seconddimension;
			} else {
				start = 0;
				end = i;
			}
			for(size_t j=start; j<end; j++) // Don't populate the diagonal
			{
				setElementWithRandomData(p + j*vectorLength, vectorLength, bound);
			}
		}
	}
}



template <typename T>
static void
doTriangleOperation(TRIANGLE_OPERATIONS op, T* data, size_t rows, size_t cols, size_t lda, int vectorLength, int creationFlags )
{
        size_t firstdimension, seconddimension;
        T *p1, *p2;
		size_t start, end;

        if (creationFlags & ROW_MAJOR_ORDER)
        {
            firstdimension = rows;
            seconddimension = cols;
        } else {
            firstdimension = cols;
            seconddimension = rows;
        }

		for(size_t i=0; i<firstdimension; i++)
		{
			//
			// Get the correct Lower Triangle offsets for ROW
			// and COL major matrices
			//
			if (creationFlags & ROW_MAJOR_ORDER)
			{
				start =0; end = i;
			} else {
				start =i+1; end = seconddimension;
			}

			for(size_t j=start; j<end; j++)
			{
				p1 = (T *)data + i*lda*vectorLength + j*vectorLength; // LT Address
				p2 = (T *)data + j*lda*vectorLength + i*vectorLength; // UT Address
				switch(op)
				{
				case LTOU:
					for(int k=0; k<vectorLength; k++)
					{
						p2[k] = p1[k];
					}
					break;
				case UTOL:
					for(int k=0; k<vectorLength; k++)
					{
						p1[k] = p2[k];
					}
					break;
				case SWAP:
					for(int k=0; k<vectorLength; k++)
					{
						T temp;

						temp = p2[k];
						p1[k] = p2[k];
						p2[k] = temp;
					}
					break;
				default:
					throw -1;
				} // end switch
			}
		}
	}


// Handles float's and double's only
// Default is NO_FLAGS, Column-Major Order

template <typename T>
static void
doPopulate(T* data, size_t rows, size_t cols, size_t lda, int vectorLength, cl_double bound, int creationFlags = 0)
{
    bool triangularMatrix = ((creationFlags & LOWER_HALF_ONLY) ||
							(creationFlags & UPPER_HALF_ONLY));


	// Non-Square Matrix
	if( rows != cols)
	{
		// Row-Major
		if (creationFlags & ROW_MAJOR_ORDER)
		{
			for( size_t i=0; i < rows; i++)
			{
				for(size_t j=0; j < cols; j++)
				{

					T *p = (T *)data + i* lda*vectorLength + j*vectorLength;
					setElementWithRandomData(p, vectorLength , bound);

					if ( i == j)
					{
						if (creationFlags & UNIT_DIAGONAL)
						{
							setElementWithUnity(p, vectorLength);
						} else if (creationFlags & ZERO_DIAGONAL)
						{
							setElementWithZero(p, vectorLength);
						}
					}
				}
			}
		}
		else // Col-Major
		{
			for( size_t i=0; i < rows; i++)
			{
				for(size_t j=0; j < cols; j++)
				{
					T *p = (T *)data + j* lda*vectorLength + i*vectorLength;
					setElementWithRandomData(p, vectorLength, bound);
					if ( i == j)
					{
						if (creationFlags & UNIT_DIAGONAL)
						{
							setElementWithUnity(p, vectorLength);
						} else if (creationFlags & ZERO_DIAGONAL)
						{
							setElementWithZero(p, vectorLength);
						}
					}

				}
			}
		}
	}

	else if ( creationFlags & PACKED_MATRIX ) // SQUARE and PACKED
	{
		if (triangularMatrix)
		{
			if (creationFlags & UPPER_HALF_ONLY)
				setTriangularMatrixWithRandomData('U', data, rows, cols, lda, vectorLength, creationFlags, bound);
			if (creationFlags & LOWER_HALF_ONLY)
				{
					setTriangularMatrixWithRandomData('L', data, rows, cols, lda, vectorLength, creationFlags, bound);
				}
		}
		else
		{
			// FIXME: throw -1;
		}

		if (creationFlags & UNIT_DIAGONAL)
		{
			setDiagonalUnity();
		} else if (creationFlags & ZERO_DIAGONAL)
		{
			setDiagonalZero();
		} else
		{
			setDiagonalRandom();
		}


	} else // SQUARE
	{
		if (triangularMatrix)
		{
			if (creationFlags & UPPER_HALF_ONLY)
				setTriangularMatrixWithRandomData('U', data, rows, cols, lda, vectorLength, creationFlags, bound);
			if (creationFlags & LOWER_HALF_ONLY)
				setTriangularMatrixWithRandomData('L', data, rows, cols, lda, vectorLength, creationFlags, bound);
		} else {
			setTriangularMatrixWithRandomData('L', data, rows, cols, lda, vectorLength, creationFlags, bound);
			if (creationFlags & SYMMETRIC_MATRIX)
			{
				doTriangleOperation(LTOU, data, rows, cols, lda, vectorLength, creationFlags);
			} else {
				setTriangularMatrixWithRandomData('U', data, rows, cols, lda, vectorLength, creationFlags, bound);
			}
		}
		if (creationFlags & UNIT_DIAGONAL)
		{
			setDiagonalUnity();
		} else if (creationFlags & ZERO_DIAGONAL)
		{
			setDiagonalZero();
		} else
		{
			setDiagonalRandom();
		}

	}
}

template <typename T>
static void
populate(T* data, size_t rows, size_t cols, size_t lda, BlasRoutineID BlasFn, int creationFlags = 0)
{
    cl_double bound;
    bound = UPPER_BOUND<T>();
	cl_double biggest = (cl_double)std::max( rows, cols);

	switch( BlasFn )
	{
		case CLBLAS_TRMV:
							bound = sqrt( ((biggest - 1)* bound) / (biggest * biggest));
							break;

		case CLBLAS_SYMM:
		case CLBLAS_HER:
		case CLBLAS_HER2:
		case CLBLAS_HEMM:
		case CLBLAS_HERK:
		case CLBLAS_GER:	// Taking cube root because of Alpha factor- (alpha*X*Y)
							bound = pow( (((biggest - 1)* bound) / (biggest * biggest)), ((double)1/3) );
							break;

		default :			::std::cerr << "Invalid function ID sent to populate!" << ::std::endl;
	}
	doPopulate( data, rows, cols, lda, 1, bound, creationFlags);
}

template<>
__template_static void
populate<FloatComplex>(FloatComplex* data, size_t rows, size_t cols, size_t lda, BlasRoutineID BlasFn, int creationFlags)
{
    cl_double bound;
    bound = UPPER_BOUND<FloatComplex>();
	cl_double biggest = (cl_double)std::max( rows, cols);

	switch( BlasFn )
	{
		case CLBLAS_TRMV:
							bound = sqrt( ((biggest - 1)* bound) / (biggest * biggest)) / 2;
							break;

		case CLBLAS_SYMM:
		case CLBLAS_HER:
		case CLBLAS_HER2:
		case CLBLAS_HEMM:
		case CLBLAS_HERK:
		case CLBLAS_GER:	// Taking cube root because of Alpha factor- (alpha*X*Y)
							bound = pow( (((biggest - 1)* bound) / (biggest * biggest)), ((double)1/3) );
							break;

		default :			::std::cerr << "Invalid function ID sent to populate!" << ::std::endl;
	}
    doPopulate( (float*)data, rows, cols, lda, 2, bound, creationFlags);
}

template<>
__template_static  void
populate<DoubleComplex>(DoubleComplex* data, size_t rows, size_t cols, size_t lda,  BlasRoutineID BlasFn, int creationFlags )
{
    cl_double bound;
    bound = UPPER_BOUND<DoubleComplex>();
	cl_double biggest = (cl_double)std::max( rows, cols);

	switch( BlasFn )
	{
		case CLBLAS_TRMV:
							bound = sqrt( ((biggest - 1)* bound) / (biggest * biggest)) / 2;
							break;

		case CLBLAS_SYMM:
		case CLBLAS_HER:
		case CLBLAS_HER2:
		case CLBLAS_GER:
		case CLBLAS_HEMM:
		case CLBLAS_HERK:
		case CLBLAS_SYR:	// Taking cube root because of Alpha factor- (alpha*X*Y)
							bound = pow( (((biggest - 1)* bound) / (biggest * biggest)), ((double)1/3) );
							break;

		default :			::std::cerr << "Invalid function ID sent to populate!" << ::std::endl;
	}
    doPopulate( (double*)data, rows, cols, lda, 2, bound, creationFlags);
}

template <typename T>
static double maxVal( T elem )
{
    return (double)elem;
}

template <>
__template_static double maxVal<FloatComplex>( FloatComplex elem )
{
    return (cl_double)std::max( CREAL(elem), CIMAG(elem) );
}

template <>
__template_static double maxVal<DoubleComplex>( DoubleComplex elem )
{
    return (cl_double)std::max( CREAL(elem), CIMAG(elem) );
}


#endif  // BLAS_RANDOM_H_
