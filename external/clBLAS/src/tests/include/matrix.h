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


#ifndef MATRIX_H_
#define MATRIX_H_

#include <clBLAS.h>
#include <blas-math.h>
#include <stdio.h>
#include <iomanip>

// Data Generation
#include <testDG.h>

template <typename T>
static T
getElement(
    clblasOrder order,
    clblasTranspose trans,
    size_t row,
    size_t column,
    const T *A,
    size_t lda)
{
if ( lda > 0) // General case
{
    switch (order) {
    case clblasRowMajor:
        if (trans == clblasNoTrans) {
            A += lda * row;
            return A[column];
        }
        else {
            A += lda * column;
            return A[row];
        }
        break;
    case clblasColumnMajor:
        if (trans == clblasNoTrans) {
            A += lda * column;
            return A[row];
        }
        else {
            A += lda * row;
            return A[column];
        }
        break;
    }

    /* Unreachable point */
    return FNAN<T>();
}
else
{
	// Needed for Macro : testDG.h
	int vectorLength = 1;
	const T* data = A;

	if ( order == clblasRowMajor)
	{
		return *RMLPacked(row, column);
	}
	else
	{
  		// return CMLPacked(row, column);
		return FNAN<T>();
	}

}
}

template <typename T>
static void
setElement(
    clblasOrder order,
    clblasTranspose trans,
    size_t row,
    size_t column,
    T *A,
    size_t lda,
    T value)
{
    switch (order) {
    case clblasRowMajor:
        if (trans == clblasNoTrans) {
            A += lda * row;
            A[column] = value;
        }
        else {
            A += lda * column;
            A[row] = value;
        }
        break;
    case clblasColumnMajor:
        if (trans == clblasNoTrans) {
            A += lda * column;
            A[row] = value;
        }
        else {
            A += lda * row;
            A[column] = value;
        }
        break;
    }
}

template <typename T>
static void
setElementPacked(
    clblasOrder order,
    clblasTranspose trans,
    clblasUplo uplo,
    size_t row,
    size_t column,
    T *A,
    size_t rows,
    T value)
{
     // Needed for Macro : testDG.h
    int vectorLength = 1;
    const T* data = A;
    clblasUplo fUplo = (trans == clblasNoTrans) ? uplo : ((uplo == clblasUpper) ? clblasLower : clblasUpper);

    if(fUplo == clblasLower) //Should not access elements out of bounds.
    {
        if (column > row)
           return;
    }
    else
    {
        if (column < row)
            return;
    }
    switch (order) {
    case clblasRowMajor:
        if (fUplo == clblasLower)
        {
            *RMLPacked(row, column) = value;
        }
        else {
            *RMUPacked(row, column) = value;
        }
        break;
    case clblasColumnMajor:
        if (fUplo == clblasLower)
        {
            *CMLPacked(row, column) = value;
        }
        else {
            *CMUPacked(row, column) = value;
        }
        break;
    }
}

template <typename T>
static T
getElementPacked(
    clblasOrder order,
    clblasTranspose trans,
    clblasUplo uplo,
    size_t row,
    size_t column,
    T *A,
    size_t rows)
{
     // Needed for Macro : testDG.h
    int vectorLength = 1;
    const T* data = A;
    clblasUplo fUplo = (trans == clblasNoTrans) ? uplo : ((uplo == clblasUpper) ? clblasLower : clblasUpper);

    if(fUplo == clblasLower) //Should not access elements out of bounds.
    {
        if (column > row)
           return FNAN<T>();
    }
    else
    {
        if (column < row)
            return FNAN<T>();
    }
    switch (order) {
        case clblasRowMajor:
            if (fUplo == clblasLower)
            {
                return *RMLPacked(row, column);
            }
            else {
                return *RMUPacked(row, column);
            }
            break;
        case clblasColumnMajor:
            if (fUplo == clblasLower)
            {
                return *CMLPacked(row, column);
            }
            else {
                return *CMUPacked(row, column);
           }
            break;
        default: return FNAN<T>();
    }
}


template <typename T>
static void
printElement(T a)
{
    std::cout << a << "\t";
}

template<>
__template_static void
printElement<FloatComplex>(FloatComplex a)
{
    std::cout << "(" << a.s[0] << ", " << a.s[1] << ")\t";
}

template<>
__template_static void
printElement<DoubleComplex>(DoubleComplex a)
{
    std::cout << "(" << a.s[0] << ", " << a.s[1] << ")\t";
}

template <typename T>
static void
printMatrixBlock(
    clblasOrder order,
    size_t startRow,
    size_t startCol,
    size_t nrRows,
    size_t nrCols,
    size_t lda,
    T *A)
{
    // FIXME : Packed Matrix
    size_t i, j;
    T a;

    for (i = 0; i < nrRows; i++) {
        for (j = 0; j < nrCols; j++) {
            a = getElement(order, clblasNoTrans, startRow + i,
                           startCol + j, A, lda);
            printElement<T>(a);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
}

template <typename T>
static void
reorderMatrix(
    clblasOrder order,
    size_t rowsA,
    size_t columnsA,
    const T *A,
    T *B)
{
    size_t lda = 0, ldb = 0;
    size_t x, y;
    clblasOrder orderB = clblasRowMajor;

    switch (order) {
    case clblasColumnMajor:
        orderB = clblasRowMajor;
        lda = rowsA;
        ldb = columnsA;
        break;
    case clblasRowMajor:
        orderB = clblasColumnMajor;
        lda = columnsA;
        ldb = rowsA;
        break;
    }

    for (y = 0; y < rowsA; y++) {
        for (x = 0; x < columnsA; x++) {
            setElement<T>(orderB, clblasNoTrans, y, x, B, ldb,
                getElement<T>(order, clblasNoTrans, y, x, A, lda));
        }
    }
}

template <typename T>
static void
compareMatrices(
    clblasOrder order,
    size_t M,
    size_t N,
    const T *A,
    const T *B,
    size_t lda,
    const cl_double *absDelta = NULL)
{
    size_t m = 0, n = 0;
    T a, b;
    cl_double delta;

    if( lda > 0 ) // General case
    {
    for (m = 0; m < M; m++) {
        for (n = 0; n < N; n++) {
            a = getElement<T>(order, clblasNoTrans, m, n, A, lda);
            b = getElement<T>(order, clblasNoTrans, m, n, B, lda);
            delta = 0.0;
            if (absDelta != NULL) {
                delta = absDelta[m * N + n];
            }
			if( module(a-b) > delta )		printf("m : %d\t n: %d\n", (int)m, (int)n);
            ASSERT_NEAR(a, b, delta);
        }
    }
    }
    else // Packed case
    {
	if ( order == clblasColumnMajor)
	{
		for ( n = 0; n < N; n++)
		{
			for( m=n; m < M; m++)
			{
            			a = getElement<T>(order, clblasNoTrans, m, n, A, lda);
			        b = getElement<T>(order, clblasNoTrans, m, n, B, lda);
            			delta = 0.0;
            			if (absDelta != NULL) {
                			//delta = absDelta[m * N + n];
            			}
						if( module(a-b) > delta )		printf("m : %d\t n: %d\n", (int)m, (int)n);
            			ASSERT_NEAR(a, b, delta);
			}
		}
	}
	else
	{
		for ( m = 0; m < M; m++)
		{
			for( n = 0; n <= m; n++)
			{
            			a = getElement<T>(order, clblasNoTrans, m, n, A, lda);
			        b = getElement<T>(order, clblasNoTrans, m, n, B, lda);
            			delta = 0.0;
            			if (absDelta != NULL) {
                			//delta = absDelta[m * N + n];
            			}
						if( module(a-b) > delta )		printf("m : %d\t n: %d\n", (int)m, (int)n);
            			ASSERT_NEAR(a, b, delta);
			}
		}
	}
    }
}

template<>
__template_static void
compareMatrices<FloatComplex>(
    clblasOrder order,
    size_t M,
    size_t N,
    const FloatComplex *A,
    const FloatComplex *B,
    size_t lda,
    const cl_double *absDelta)
{
    size_t m = 0, n = 0;
    FloatComplex a, b;
    cl_double delta;

if ( lda > 0 )
{
    for (m = 0; m < M; m++) {
        for (n = 0; n < N; n++) {
            a = getElement<FloatComplex>(order, clblasNoTrans, m, n, A, lda);
            b = getElement<FloatComplex>(order, clblasNoTrans, m, n, B, lda);
            delta = 0.0;
            if (absDelta != NULL) {
                delta = absDelta[m * N + n];
            }
			if( (module(CREAL(a) - CREAL(b)) > delta) || (module(CIMAG(a) - CIMAG(b)) > delta) )
					printf("m : %d\t n: %d\n", (int)m, (int)n);
            ASSERT_NEAR(CREAL(a), CREAL(b), delta);
            ASSERT_NEAR(CIMAG(a), CIMAG(b), delta);
        }
    }
}
    else // Packed case
    {
	if ( order == clblasColumnMajor)
	{
		for ( n = 0; n < N; n++)
		{
			for( m=n; m < M; m++)
			{
            			a = getElement<FloatComplex>(order, clblasNoTrans, m, n, A, lda);
				        b = getElement<FloatComplex>(order, clblasNoTrans, m, n, B, lda);
            			delta = 0.0;
            			if (absDelta != NULL) {
                			//delta = absDelta[m * N + n];
            			}
            			if( (module(CREAL(a) - CREAL(b)) > delta) || (module(CIMAG(a) - CIMAG(b)) > delta) )
							printf("m : %d\t n: %d\n", (int)m, (int)n);
            			ASSERT_NEAR(CREAL(a), CREAL(b), delta);
		            	ASSERT_NEAR(CIMAG(a), CIMAG(b), delta);
			}
		}
	}
	else
	{
		for ( m = 0; m < M; m++)
		{
			for( n = 0; n <= m; n++)
			{
            			a = getElement<FloatComplex>(order, clblasNoTrans, m, n, A, lda);
			        b = getElement<FloatComplex>(order, clblasNoTrans, m, n, B, lda);
            			delta = 0.0;
            			if (absDelta != NULL) {
                			//delta = absDelta[m * N + n];
            			}
						if( (module(CREAL(a) - CREAL(b)) > delta) || (module(CIMAG(a) - CIMAG(b)) > delta) )
							printf("m : %d\t n: %d\n", (int)m, (int)n);
            			ASSERT_NEAR(CREAL(a), CREAL(b), delta);
		            	ASSERT_NEAR(CIMAG(a), CIMAG(b), delta);
			}
		}
	}
    }

}

template<>
__template_static void
compareMatrices<DoubleComplex>(
    clblasOrder order,
    size_t M,
    size_t N,
    const DoubleComplex *A,
    const DoubleComplex *B,
    size_t lda,
    const cl_double *absDelta)
{
    size_t m = 0, n = 0;
    DoubleComplex a, b;
    cl_double delta;
if( lda > 0 )
{
    for (m = 0; m < M; m++) {
        for (n = 0; n < N; n++) {
            a = getElement<DoubleComplex>(order, clblasNoTrans, m, n, A, lda);
            b = getElement<DoubleComplex>(order, clblasNoTrans, m, n, B, lda);
            delta = 0.0;
            if (absDelta != NULL) {
                delta = absDelta[m * N + n];
            }
			if( (module(CREAL(a) - CREAL(b)) > delta) || (module(CIMAG(a) - CIMAG(b)) > delta) )
					printf("m : %d\t n: %d\n", (int)m, (int)n);
            ASSERT_NEAR(CREAL(a), CREAL(b), delta);
            ASSERT_NEAR(CIMAG(a), CIMAG(b), delta);
        }
    }
}
    else // Packed case
    {
	if ( order == clblasColumnMajor)
	{
		for ( n = 0; n < N; n++)
		{
			for( m=n; m < M; m++)
			{
            			a = getElement<DoubleComplex>(order, clblasNoTrans, m, n, A, lda);
			        b = getElement<DoubleComplex>(order, clblasNoTrans, m, n, B, lda);
            			delta = 0.0;
            			if (absDelta != NULL) {
                			//delta = absDelta[m * N + n];
            			}
						if( (module(CREAL(a) - CREAL(b)) > delta) || (module(CIMAG(a) - CIMAG(b)) > delta) )
							printf("m : %d\t n: %d\n", (int)m, (int)n);
            			ASSERT_NEAR(CREAL(a), CREAL(b), delta);
		            	ASSERT_NEAR(CIMAG(a), CIMAG(b), delta);
			}
		}
	}
	else
	{
		for ( m = 0; m < M; m++)
		{
			for( n = 0; n <= m; n++)
			{
            			a = getElement<DoubleComplex>(order, clblasNoTrans, m, n, A, lda);
			        b = getElement<DoubleComplex>(order, clblasNoTrans, m, n, B, lda);
            			delta = 0.0;
            			if (absDelta != NULL) {
                			//delta = absDelta[m * N + n];
            			}
						if( (module(CREAL(a) - CREAL(b)) > delta) || (module(CIMAG(a) - CIMAG(b)) > delta) )
							printf("m : %d\t n: %d\n", (int)m, (int)n);
            			ASSERT_NEAR(CREAL(a), CREAL(b), delta);
		            	ASSERT_NEAR(CIMAG(a), CIMAG(b), delta);
			}
		}
	}
    }

}

template <typename T>
static void
setNans(
    size_t len,
    T *buf)
{
    size_t i;
    for (i = 0; i < len; i++) {
        buf[i] = FNAN<T>();
    }
}

// set to NAN elements of upper or lower triangle of square matrix
template <typename T>
static void
setTriangleNans(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    T *A,
    size_t lda)
{
    size_t i, j;

    // For matrix A
    for (i = 0; i < N; i++) {
        switch (uplo) {
        case clblasUpper:
            for (j = 0; j < i; j++) {
                setElement<T>(order, clblasNoTrans, i, j, A, lda, FNAN<T>());
            }
            break;
        case clblasLower:
            for (j = i + 1; j < N; j++) {
                setElement<T>(order, clblasNoTrans, i, j, A, lda, FNAN<T>());
            }
            break;
        }
    }
}

template <typename T>
static void
setVectorNans(
    size_t offset,
    size_t dx,
    T *B,
    size_t N,
    size_t memLen)
{
    size_t i;
    for (i = 0; i < offset; i++) {
        B[i] = FNAN<T>();
    }
    for (i = offset; i <= offset + dx * (N - 1); i++) {
        if (((i - offset) % dx) != 0) {
            B[i] = FNAN<T>();
        }
    }
    for (; i < memLen; i++) {
        B[i] = FNAN<T>();
    }
}

template <typename T>
static void
compareVectors(
    size_t offset,
    size_t N,
    size_t dy,
    size_t memLen,
    T *blasC,
    T *clblasC)
{
    size_t tailBegin, tailEnd;

    // check the beginning containing NANs
    ASSERT_FALSE(memcmp(blasC, clblasC, offset * sizeof(blasC[0])));

    // check vector values
    compareMatrices<T>(clblasRowMajor, N, 1, blasC + offset,
                       clblasC + offset, dy);
    // check NANs between vector values
    if (dy != 1) {
        size_t i;
        size_t start, end;
        start = offset + 1;
        end = start + dy - 1;
        for (i = 0; i < N - 1; i++) {
            ASSERT_FALSE(memcmp(blasC + start, clblasC + start,
                         (end - start) * sizeof(blasC[0])));
        }
    }
    // check tail containing NANs
    tailBegin = offset;
    if (dy == 1) {
        tailBegin += N;
    }
    else {
        tailBegin += N + (N - 1) * (dy - 1);
    }
    tailEnd = memLen;

    ASSERT_FALSE(memcmp(blasC + tailBegin, clblasC + tailBegin,
                        (tailEnd - tailBegin) * sizeof(blasC[0])));
}



// Works only for NxN matrix
template <typename T>
static T
getElementBanded(
    clblasOrder order,
    clblasUplo uplo,
    size_t row,
    size_t column,
    size_t K,
    const T *A,
    size_t lda)
{
    switch (order)
    {
        case clblasRowMajor:
            A += lda * row;
            return (uplo == clblasLower)? A[ K - (row-column) ]: A[ column-row ];
        break;

        case clblasColumnMajor:
            A += lda * column;
            return (uplo == clblasLower)? A[ row-column ]: A[ K - (column-row) ];
        break;
    }

    /* Unreachable point */
    return FNAN<T>();
}

template <typename T>
static void
setElementBanded(
    clblasOrder order,
    clblasUplo uplo,
    size_t row,
    size_t column,
    size_t K,
    T *A,
    size_t lda,
    T value)
{
    switch (order)
    {
        case clblasRowMajor:
            A += lda * row;
            if (uplo == clblasLower)
            {
                A[ K - (row-column) ] = value;
            }
            else {
                A[ column-row ] = value;
            }
        break;

        case clblasColumnMajor:
            A += lda * column;
            if (uplo == clblasLower)
            {
                A[ row-column ] = value;
            }
            else {
                A[ K - (column-row) ] = value;
            }
        break;
    }
}


//conjugate function to handle rowmajor as columnmajor
// for float and double do nothing
template <typename T>
static void
doConjugate(
    T *A,
    size_t M,
    size_t N,
    size_t lda)
{
	if( M || N || lda|| A){} // Dummy to avoid warnings

	return;
}


template<>
__template_static void
doConjugate<FloatComplex>(
    FloatComplex *A,
    size_t M,
    size_t N,
    size_t lda)
{
    size_t m, n;
    FloatComplex b;

    if ( lda > 0 )
    {
        for (m = 0; m < M; m++)
        {
            for (n = 0; n < N; n++)
            {
			    b = getElement<FloatComplex>(clblasRowMajor, clblasNoTrans, m, n, A, lda);
                CIMAG(b) *= (-1);
                setElement<FloatComplex>(clblasRowMajor, clblasNoTrans, m, n, A, lda, b);
			}
        }
    }
}

template<>
__template_static void
doConjugate<DoubleComplex>(
    DoubleComplex *A,
    size_t M,
    size_t N,
    size_t lda)
{
    size_t m, n;
    DoubleComplex b;

if ( lda > 0 )
{
    for (m = 0; m < M; m++) {
        for (n = 0; n < N; n++) {
            b = getElement<DoubleComplex>(clblasRowMajor, clblasNoTrans, m, n, A, lda);
            CIMAG(b) *= (-1);
            setElement<DoubleComplex>(clblasRowMajor, clblasNoTrans, m, n, A, lda, b);
        }
    }
}
}


template <typename T>
static void compareValues(
    const T *A, const T *B, const cl_double absDelta=0.0 )
{
    T a, b;
    a = *A;
    b = *B;
    ASSERT_NEAR(a, b, absDelta);
}

 template<>
__template_static void
compareValues<FloatComplex> (
    const FloatComplex *A, const FloatComplex *B, const cl_double absDelta )
{
    FloatComplex a, b;

    a = *A;
    b = *B;
    ASSERT_NEAR(CREAL(a), CREAL(b), absDelta);
    ASSERT_NEAR(CIMAG(a), CIMAG(b), absDelta);
}

 template<>
__template_static void
compareValues<DoubleComplex> (
    const DoubleComplex *A, const DoubleComplex *B, const cl_double absDelta )
{
    DoubleComplex a, b;

    a = *A;
    b = *B;
    ASSERT_NEAR(CREAL(a), CREAL(b), absDelta);
    ASSERT_NEAR(CIMAG(a), CIMAG(b), absDelta);
}
#endif  // MATRIX_H_
