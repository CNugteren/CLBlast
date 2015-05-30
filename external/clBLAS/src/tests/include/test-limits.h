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


#ifndef TEST_LIMITS_H_
#define TEST_LIMITS_H_

#define FLOAT_UPPER_BOUND   pow(2.0, 23)
#define DOUBLE_UPPER_BOUND  pow(2.0, 52)

#define TRSM_FLOAT_LIMIT_A  pow(2.0, 7)
#define TRSM_DOUBLE_LIMIT_A pow(2.0, 5)
#define TRSM_FLOAT_LIMIT_B  pow(2.0, 16)
#define TRSM_DOUBLE_LIMIT_B pow(2.0, 47)

// Type-dependant constants
template <class T>
static cl_double UPPER_BOUND();
template<>
__template_static cl_double UPPER_BOUND<cl_float>() { return FLOAT_UPPER_BOUND; }
template<>
__template_static cl_double UPPER_BOUND<cl_double>() { return DOUBLE_UPPER_BOUND;}
template<>
__template_static cl_double UPPER_BOUND<FloatComplex>() { return FLOAT_UPPER_BOUND; }
template<>
__template_static cl_double UPPER_BOUND<DoubleComplex>()  { return DOUBLE_UPPER_BOUND; }

template <class T>
static cl_double TRSM_LIMIT_A();
template<>
__template_static cl_double TRSM_LIMIT_A<cl_float>() { return TRSM_FLOAT_LIMIT_A; }
template<>
__template_static cl_double TRSM_LIMIT_A<cl_double>() { return TRSM_DOUBLE_LIMIT_A; }
template<>
__template_static cl_double TRSM_LIMIT_A<FloatComplex>() { return TRSM_FLOAT_LIMIT_A; }
template<>
__template_static cl_double TRSM_LIMIT_A<DoubleComplex>() { return TRSM_DOUBLE_LIMIT_A; }

template <class T>
static cl_double TRSM_LIMIT_B();
template<>
__template_static cl_double TRSM_LIMIT_B<cl_float>() { return TRSM_FLOAT_LIMIT_B; }
template<>
__template_static cl_double TRSM_LIMIT_B<cl_double>() { return TRSM_DOUBLE_LIMIT_B; }
template<>
__template_static cl_double TRSM_LIMIT_B<FloatComplex>() { return TRSM_FLOAT_LIMIT_B; }
template<>
__template_static cl_double TRSM_LIMIT_B<DoubleComplex>() { return TRSM_DOUBLE_LIMIT_B; }

#endif /* TEST_LIMITS_H_ */
