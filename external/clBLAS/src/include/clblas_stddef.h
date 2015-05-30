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


#ifndef CLBLAS_STDDEF_H_
#define CLBLAS_STDDEF_H_

static __inline size_t
szmin(size_t a, size_t b)
{
    return (a <= b ? a : b);
}

static __inline size_t
szmax(size_t a, size_t b)
{
    return (a >= b ? a : b);
}

static __inline unsigned int
umin(unsigned int a, unsigned int b)
{
    return (a <= b ? a : b);
}

static __inline unsigned int
umax(unsigned int a, unsigned int b)
{
    return (a >= b ? a : b);
}

static __inline void
uswap(unsigned int *a, unsigned int *b)
{
    unsigned int tmp;

    tmp = *a;
    *a = *b;
    *b = tmp;
}

static __inline size_t
roundDown(size_t a, size_t b)
{
    return (a / b * b);
}

static __inline size_t
roundUp(size_t a, size_t b)
{
    return (a + b - 1) / b * b;
}

static __inline size_t
divRoundUp(size_t a, size_t b)
{
    return (a / b) + (a % b != 0);
}

static __inline int
isRoundedPow2(size_t a)
{
    return ((a & (a - 1)) == 0);
}

/*
 * Return zero based sequential number of the highest set bit the
 * number. If the number is 0, then the function returns -1.
 */
static __inline int
findHighestSetBit(size_t a)
{
    int n = (sizeof(size_t) * 8 - 1);
    size_t s = (size_t)1 << n;

    for (; (s != 0) && !(s & a); s >>= 1) {
        n--;
    }

    return (s == 0) ? -1 : n;
}

static __inline size_t
roundDownPow2(size_t a)
{
    size_t s;

    if (isRoundedPow2(a)) {
        return a;
    }

    s = (size_t)1 << (sizeof(size_t) * 8 - 1);

    // find the highest non zero bit
    for (; (s != 0) && !(s & a); s >>= 1);

    return s;
}

/*
 * With BLAS we never deal with so large number sufficient for overflowing.
 * So, it's safe
 */
static __inline size_t
roundUpPow2(size_t a)
{
    size_t s;

    if (isRoundedPow2(a)) {
        return a;
    }

    s = roundDownPow2(a);

    return (s << 1);
}

#endif /* CLBLAS_STDDEF_H_ */
