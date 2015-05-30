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


#ifndef CLTYPES_H_
#define CLTYPES_H_

#include <defbool.h>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/**
 * @internal
 * @defgroup DTYPES Data types
 */
/*@{*/

/**
 * @brief OpenCL type identifiers
 */
typedef enum DataType {
    TYPE_FLOAT,             /**< single float precision type */
    TYPE_DOUBLE,            /**< double float precision type */
    TYPE_COMPLEX_FLOAT,     /**< single float precision complex type */
    TYPE_COMPLEX_DOUBLE,    /**< double float precision complex type */
    TYPE_UNSIGNED_INT       /**< Unsigned int, for output buffer for iAMAX routine */
} DataType;

/*@}*/

enum {
    FLOAT4_VECLEN = sizeof(cl_float4) / sizeof(cl_float)
};

/*
 * return size of a BLAS related data type
 */
#ifdef __cplusplus
extern "C"
#endif
unsigned int
dtypeSize(DataType type);

/*
 * width of the matrix (block) in float4 words
 */
size_t
fl4RowWidth(size_t width, size_t typeSize);

static __inline bool
isDoubleBasedType(DataType dtype)
{
    return (dtype == TYPE_DOUBLE || dtype == TYPE_COMPLEX_DOUBLE);
}

static __inline bool
isComplexType(DataType dtype)
{
    return (dtype == TYPE_COMPLEX_FLOAT || dtype == TYPE_COMPLEX_DOUBLE);
}

#endif /* CLTYPES_H_ */
