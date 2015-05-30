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


#include <blas-cblas.h>


complex
compose_complex(float x, float y)
{
    complex z = { x, y };
    return z;
}

float
complex_real(complex z)
{
    return z.real;
}

float
complex_imag(complex z)
{
    return z.imag;
}

doublecomplex
compose_doublecomplex(double x, double y)
{
    doublecomplex z = { x, y };
    return z;
}

double
doublecomplex_real(doublecomplex z)
{
    return z.real;
}

double
doublecomplex_imag(doublecomplex z)
{
    return z.imag;
}
