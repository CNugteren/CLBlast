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

#ifndef DELTA_H_
#define DELTA_H_

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

#endif      // DELTA_H

