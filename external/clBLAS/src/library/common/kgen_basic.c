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


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <assert.h>

#include <kerngen.h>
#include <mempat.h>

const char *uptrsFullDeclaration =
    "#ifdef cl_khr_fp64\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "#else\n"
    "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
    "#endif\n"
    "\n"
    "typedef union GPtr {\n"
    "    __global float *f;\n"
    "    __global double *d;\n"
    "    __global float2 *f2v;\n"
    "    __global double2 *d2v;\n"
    "    __global float4 *f4v;\n"
    "    __global double4 *d4v;\n"
    "    __global float8 *f8v;\n"
    "    __global double8 *d8v;\n"
    "    __global float16 *f16v;\n"
    "    __global double16 *d16v;\n"
    "} GPtr;\n"
    "\n"
    "typedef union LPtr {\n"
    "    __local float *f;\n"
    "    __local double *d;\n"
    "    __local float2 *f2v;\n"
    "    __local double2 *d2v;\n"
    "    __local float4 *f4v;\n"
    "    __local double4 *d4v;\n"
    "    __local float8 *f8v;\n"
    "    __local double8 *d8v;\n"
    "    __local float16 *f16v;\n"
    "    __local double16 *d16v;\n"
    "} LPtr;\n"
    "\n"
    "typedef union PPtr {\n"
    "    float *f;\n"
    "    double *d;\n"
    "    float2 *f2v;\n"
    "    double2 *d2v;\n"
    "    float4 *f4v;\n"
    "    double4 *d4v;\n"
    "    float8 *f8v;\n"
    "    double8 *d8v;\n"
    "    float16 *f16v;\n"
    "    double16 *d16v;\n"
    "} PPtr;\n\n";

const char *uptrsSingleDeclaration =
    "typedef union GPtr {\n"
    "    __global float *f;\n"
    "    __global float2 *f2v;\n"
    "    __global float4 *f4v;\n"
    "    __global float8 *f8v;\n"
    "    __global float16 *f16v;\n"
    "} GPtr;\n"
    "\n"
    "typedef union LPtr {\n"
    "    __local float *f;\n"
    "    __local float2 *f2v;\n"
    "    __local float4 *f4v;\n"
    "    __local float8 *f8v;\n"
    "    __local float16 *f16v;\n"
    "} LPtr;\n"
    "\n"
    "typedef union PPtr {\n"
    "    float *f;\n"
    "    float2 *f2v;\n"
    "    float4 *f4v;\n"
    "    float8 *f8v;\n"
    "    float16 *f16v;\n"
    "} PPtr;\n\n";

const char
*uptrTypeName(UptrType type)
{
    const char *s = NULL;

    switch(type) {
    case UPTR_GLOBAL:
        s = "GPtr";
        break;
    case UPTR_LOCAL:
        s = "LPtr";
        break;
    case UPTR_PRIVATE:
        s = "PPtr";
        break;
    }

    return s;
}

char
dtypeToPrefix(DataType type)
{
    char c;

    switch (type) {
    case TYPE_FLOAT:
        c = 'f';
        break;
    case TYPE_DOUBLE:
        c = 'd';
        break;
    case TYPE_COMPLEX_FLOAT:
        c = 'c';
        break;
    case TYPE_COMPLEX_DOUBLE:
        c = 'z';
        break;
    default:
        c = 0;
        break;
    }

    return c;
}

const char
*dtypeBuiltinType(DataType dtype)
{
    const char *s;

    switch (dtype) {
    case TYPE_FLOAT:
        s = "float";
        break;
    case TYPE_DOUBLE:
        s = "double";
        break;
    case TYPE_COMPLEX_FLOAT:
        s = "float2";
        break;
    case TYPE_COMPLEX_DOUBLE:
        s = "double2";
        break;
    default:
        s = NULL;
        break;
    }

    return s;
}

const char
*dtypeUPtrField(DataType dtype)
{
    const char *s;

    switch (dtype) {
    case TYPE_FLOAT:
        s = "f";
        break;
    case TYPE_DOUBLE:
        s = "d";
        break;
    case TYPE_COMPLEX_FLOAT:
        s = "f2v";
        break;
    case TYPE_COMPLEX_DOUBLE:
        s = "d2v";
        break;
    default:
        s = NULL;
        break;
    }

    return s;
}

const char
*strOne(DataType dtype)
{
    const char *s;

    if (isComplexType(dtype)) {
        if (isDoubleBasedType(dtype)) {
            s = "(double2)(1, 0)";
        }
        else {
            s = "(float2)(1, 0)";
        }
    }
    else {
        s = "1";
    }

    return s;
}

void
getVectorTypeName(
    DataType dtype,
    unsigned int vecLen,
    const char **typeName,
    const char **typePtrName)
{
    char *tn = "";
    char *tpn = "";

    if (isDoubleBasedType(dtype)) {
        switch (vecLen * dtypeSize(dtype)) {
        case sizeof(cl_double):
            tn = "double";
            tpn = "d";
            break;
        case sizeof(cl_double2):
            tn = "double2";
            tpn = "d2v";
            break;
        case sizeof(cl_double4):
            tn = "double4";
            tpn = "d4v";
            break;
        case sizeof(cl_double8):
            tn = "double8";
            tpn = "d8v";
            break;
        case sizeof(cl_double16):
            tn = "double16";
            tpn = "d16v";
            break;
        };
    }
    else {
        switch (vecLen * dtypeSize(dtype)) {
        case sizeof(cl_float):
            tn = "float";
            tpn = "f";
            break;
        case sizeof(cl_float2):
            tn = "float2";
            tpn = "f2v";
            break;
        case sizeof(cl_float4):
            tn = "float4";
            tpn = "f4v";
            break;
        case sizeof(cl_float8):
            tn = "float8";
            tpn = "f8v";
            break;
        case sizeof(cl_float16):
            tn = "float16";
            tpn = "f16v";
            break;
        };
    }
    if (typeName != NULL) {
        *typeName = tn;
    }
    if (typePtrName != NULL) {
        *typePtrName = tpn;
    }
}

int
kgenAddBarrier(
    struct KgenContext *ctx,
    CLMemFence fence)
{
    int ret;

    if (fence == CLK_LOCAL_MEM_FENCE) {
        ret = kgenAddStmt(ctx, "barrier(CLK_LOCAL_MEM_FENCE);\n");
    }
    else {
        ret = kgenAddStmt(ctx, "barrier(CLK_GLOBAL_MEM_FENCE);\n");
    }
    if (ret) {
        ret = -EOVERFLOW;
    }

    return ret;
}

int
kgenAddMemFence(
    struct KgenContext *ctx,
    CLMemFence fence)
{
    int ret;

    if (fence == CLK_LOCAL_MEM_FENCE) {
        ret = kgenAddStmt(ctx, "mem_fence(CLK_LOCAL_MEM_FENCE);\n");
    }
    else {
        ret = kgenAddStmt(ctx, "mem_fence(CLK_GLOBAL_MEM_FENCE);\n");
    }
    if (ret) {
        ret = -EOVERFLOW;
    }

    return ret;
}

int
kgenDeclareLocalID(
    struct KgenContext *ctx,
    const char *lidName,
    const PGranularity *pgran)
{
    char tmp[128];
    int r;

    if (pgran->wgDim == 1) {
        sprintf(tmp, "const int %s = get_local_id(0);\n", lidName);
    }
    else {
        sprintf(tmp, "const int %s = get_local_id(1) * %u + "
                     "get_local_id(0);\n",
                lidName, pgran->wgSize[0]);
    }

    r = kgenAddStmt(ctx, tmp);

    return (r) ? -EOVERFLOW : 0;
}

int
kgenDeclareGroupID(
    struct KgenContext *ctx,
    const char *gidName,
    const PGranularity *pgran)
{
    char tmp[128];
    int r;

    if (pgran->wgDim == 1) {
        sprintf(tmp, "const int %s = get_global_id(0) / %u;\n",
                gidName, pgran->wgSize[0]);
    }
    else {
        sprintf(tmp, "const int %s = (get_global_id(1) / %u) * "
                     "(get_global_size(0) / %u) + "
                     "get_global_id(0) / %u;\n",
                     gidName, pgran->wgSize[1], pgran->wgSize[0],
                     pgran->wgSize[0]);
    }

    r = kgenAddStmt(ctx, tmp);

    return (r) ? -EOVERFLOW : 0;
}

int
kgenDeclareUptrs(struct KgenContext *ctx, bool withDouble)
{
    int ret;
    const char *s;

    s = (withDouble) ? uptrsFullDeclaration : uptrsSingleDeclaration;
    ret = kgenAddStmt(ctx, s);

    return ret ? -EOVERFLOW: 0;
}

void
kstrcpy(Kstring *kstr, const char *str)
{
    const int lastByte = sizeof(kstr->buf) - 1;

    kstr->buf[lastByte] = '\0';
    strncpy(kstr->buf, str, sizeof(kstr->buf));
    assert(kstr->buf[lastByte] == '\0');
}

void
ksprintf(Kstring *kstr, const char *fmt,...)
{
    va_list ap;
    int len;

    va_start(ap, fmt);
    len = vsnprintf(kstr->buf, sizeof(kstr->buf), fmt, ap);
    va_end(ap);

    // to mute GCC with its warning regarding set but unused variables
#ifdef NDEBUG
    (void)len;
#endif

    assert((size_t)len < sizeof(kstr->buf));
}

void
kstrcatf(Kstring *kstr, const char *fmt,...)
{
    va_list ap;
    int len, maxlen;

    va_start(ap, fmt);
    len = (int)strlen(kstr->buf);
    maxlen = sizeof(kstr->buf) - len;
    len = vsnprintf(kstr->buf + len, maxlen, fmt, ap);
    va_end(ap);

    assert(len < maxlen);
}


