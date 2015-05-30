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


#ifndef BLAS_MATH_H_
#define BLAS_MATH_H_

#if defined (_MSC_VER)

#if( _MSC_VER <= 1700 )
static unsigned long long ROW_NAN = 0x7ff0000000000000LL;
#define NAN *(reinterpret_cast<double*>(&ROW_NAN))
#endif

static unsigned int ROW_NANF = 0x7fc00000;
#define NANF *(reinterpret_cast<float*>(&ROW_NANF))

#else   /* _MSC_VER */

#define NANF NAN

#endif  /* !_MSC_VER */

#include <math.h>       // NAN, sqrt, abs
#include <stdlib.h>     // rand()
#include <clBLAS.h>
#include <common.h>

static inline cl_int
module(cl_int a)
{
    return abs(a);
}

static inline cl_float
module(cl_float a)
{
	return fabsf(a);
}

static inline cl_double
module(cl_double a)
{
	return fabs(a);
}

static inline FloatComplex
operator+(FloatComplex a, FloatComplex b)
{
	return floatComplex(CREAL(a) + CREAL(b), CIMAG(b) + CIMAG(b));
}

static inline FloatComplex
operator-(FloatComplex a, FloatComplex b)
{
	return floatComplex(CREAL(a) - CREAL(b), CIMAG(b) - CIMAG(b));
}

static inline FloatComplex
operator*(FloatComplex a, FloatComplex b)
{
	return floatComplex(
		CREAL(a) * CREAL(b) - CIMAG(a) * CIMAG(b),
		CREAL(a) * CIMAG(b) + CREAL(b) * CIMAG(a));
}

static inline FloatComplex
operator*(FloatComplex a, cl_float b)
{
	return floatComplex(CREAL(a) * b, CIMAG(a) * b);
}

static inline FloatComplex
operator/(FloatComplex a, FloatComplex b)
{
	cl_float div = CREAL(b) * CREAL(b) + CIMAG(b) * CIMAG(b);

	return floatComplex(
		(CREAL(a) * CREAL(b) + CIMAG(a) * CIMAG(b)) / div,
		(CREAL(b) * CIMAG(a) - CREAL(a) * CIMAG(b)) / div);
}

static inline FloatComplex
operator/(FloatComplex a, cl_float b)
{
	return floatComplex(CREAL(a) / b, CIMAG(a) / b);
}

static inline cl_float
module(FloatComplex a)
{
    if ((CREAL(a) == 0.0) && (CIMAG(a) == 0.0))
        return 0.0;
	return sqrtf(CREAL(a) * CREAL(a) + CIMAG(a) * CIMAG(a));
}

static inline DoubleComplex
operator+(DoubleComplex a, DoubleComplex b)
{
	return doubleComplex(CREAL(a) + CREAL(b), CIMAG(b) + CIMAG(b));
}

static inline DoubleComplex
operator-(DoubleComplex a, DoubleComplex b)
{
	return doubleComplex(CREAL(a) - CREAL(b), CIMAG(b) - CIMAG(b));
}

static inline DoubleComplex
operator*(DoubleComplex a, DoubleComplex b)
{
	return doubleComplex(
	    CREAL(a) * CREAL(b) - CIMAG(a) * CIMAG(b),
	    CREAL(a) * CIMAG(b) + CREAL(b) * CIMAG(a));
}

static inline DoubleComplex
operator*(DoubleComplex a, cl_double b)
{
	return doubleComplex(CREAL(a) * b, CIMAG(a) * b);
}

static inline DoubleComplex
operator/(DoubleComplex a, DoubleComplex b)
{
	cl_double div = CREAL(b) * CREAL(b) + CIMAG(b) * CIMAG(b);

	return doubleComplex(
		(CREAL(a) * CREAL(b) + CIMAG(a) * CIMAG(b)) / div,
		(CREAL(b) * CIMAG(a) - CREAL(a) * CIMAG(b)) / div);
}

static inline DoubleComplex
operator/(DoubleComplex a, cl_double b)
{
	return doubleComplex(CREAL(a) / b, CIMAG(a) / b);
}

static inline cl_double
module(DoubleComplex a)
{
    if ((CREAL(a) == 0.0) && (CIMAG(a) == 0.0))
        return 0.0;
	return sqrt(CREAL(a) * CREAL(a) + CIMAG(a) * CIMAG(a));
}

// Random generator

template<class T>
static T
randomTrsv(cl_double limit)
{
    T v;
    T temp;
    temp = ((T)rand() / (T)(RAND_MAX));
    temp = temp * (T)limit;
    if(temp == 0)
    {
        if ((rand() % 2) == 1)
        {
            temp = ((T)rand() / (T)(RAND_MAX));
            temp = temp * (T)limit;
        }
    }
    v = static_cast<float>(temp);
    if ((rand() % 2) == 1)
        v = -v;
    return v;
}

template<>
__template_static FloatComplex
randomTrsv<FloatComplex>(cl_double limit)
{
    return floatComplex(randomTrsv<cl_float>(limit), randomTrsv<cl_float>(limit));
}

template<>
__template_static DoubleComplex
randomTrsv<DoubleComplex>(cl_double limit)
{
    return doubleComplex(randomTrsv<cl_double>(limit), randomTrsv<cl_double>(limit));
}

template<typename T>
static T
randomTrsv(cl_double left, cl_double right)
{
    T v;
    T l = static_cast<T>(left);

    v = randomTrsv<T>(right - left);
    if (v < 0) {
        v -= l;
    }
    else {
        v += l;
    }
    return v;
}

template<>
__template_static FloatComplex
randomTrsv<FloatComplex>(cl_double left, cl_double right)
{
    return floatComplex(randomTrsv<cl_float>(left, right),
        randomTrsv<cl_float>(left, right));
}

template<>
__template_static DoubleComplex
randomTrsv<DoubleComplex>(cl_double left, cl_double right)
{
    return doubleComplex(randomTrsv<cl_double>(left, right),
        randomTrsv<cl_double>(left, right));
}


template<class T>
static T
random(cl_double limit)
{
	T v;
    cl_ulong l = static_cast<cl_ulong>(limit);
    if (l == 0) {
        return 0;
    }
	v = static_cast<float>(rand() % l);
	if ((rand() % 2) == 1)
		v = -v;
	return v;
}

template<>
__template_static FloatComplex
random<FloatComplex>(cl_double limit)
{
	return floatComplex(random<cl_float>(limit), random<cl_float>(limit));
}

template<>
__template_static DoubleComplex
random<DoubleComplex>(cl_double limit)
{
	return doubleComplex(random<cl_double>(limit), random<cl_double>(limit));
}

template<typename T>
static T
random(cl_double left, cl_double right)
{
    T v;
    T l = static_cast<T>(left);

    v = random<T>(right - left);
    if (v < 0) {
        v -= l;
    }
    else {
        v += l;
    }
    return v;
}

template<>
__template_static FloatComplex
random<FloatComplex>(cl_double left, cl_double right)
{
	return floatComplex(random<cl_float>(left, right),
        random<cl_float>(left, right));
}

template<>
__template_static DoubleComplex
random<DoubleComplex>(cl_double left, cl_double right)
{
	return doubleComplex(random<cl_double>(left, right),
        random<cl_double>(left, right));
}

// Type-dependant constants

template<class T>
static T
ZERO()
{
	return static_cast<T>(0.0);
}

template<>
__template_static FloatComplex
ZERO<FloatComplex>()
{
	return floatComplex(0.0, 0.0);
}

template<>
__template_static DoubleComplex
ZERO<DoubleComplex>()
{
	return doubleComplex(0.0, 0.0);
}


template<class T>
static T
ONE()
{
	return static_cast<T>(1.0);
}

template<>
__template_static FloatComplex
ONE<FloatComplex>()
{
	return floatComplex(1.0, 0.0);
}

template<>
__template_static DoubleComplex
ONE<DoubleComplex>()
{
	return doubleComplex(1.0, 0.0);
}

template<class T>
static T
FNAN();

template<>
__template_static float
FNAN<float>()
{
    return NANF;
}

template<>
__template_static double
FNAN<double>()
{
    return NAN;
}

template<>
__template_static FloatComplex
FNAN<FloatComplex>()
{
    return floatComplex(NANF, NANF);
}

template<>
__template_static DoubleComplex
FNAN<DoubleComplex>()
{
    return doubleComplex(NAN, NAN);
}

#endif	/* BLAS_MATH_H_ */
