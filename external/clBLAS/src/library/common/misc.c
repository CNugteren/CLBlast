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


#include <cltypes.h>

unsigned int
dtypeSize(DataType type)
{
    size_t ret;

    switch (type) {
    case TYPE_FLOAT:
        ret = sizeof(cl_float);
        break;
    case TYPE_DOUBLE:
        ret = sizeof(cl_double);
        break;
    case TYPE_COMPLEX_FLOAT:
        ret = sizeof(cl_float2);
        break;
    case TYPE_COMPLEX_DOUBLE:
        ret = sizeof(cl_double2);
        break;
    case TYPE_UNSIGNED_INT:// For iAMAX
        ret = sizeof(cl_uint);
        break;
    default:
        ret = (size_t)-1;
        break;
    }

    return (unsigned int)ret;
}

size_t
fl4RowWidth(size_t width, size_t typeSize)
{
    size_t s;

    s = width / (sizeof(cl_float4) / typeSize);
    if (s * (sizeof(cl_float4) / typeSize) != width) {
        s++;
    }

    return s;
}

