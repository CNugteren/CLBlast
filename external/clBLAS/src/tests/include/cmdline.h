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


#ifndef CMDLINE_H_
#define CMDLINE_H_

#include <clBLAS.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ComplexLong {
    long re;
    long imag;
} ComplexLong;

// flags showing wheter an option was set through the command line
typedef enum SetoptFlags {
    NO_FLAGS  = 0,
    SET_SEED  = (1 << 0),
    SET_ALPHA = (1 << 1),
    SET_BETA  = (1 << 2),
    SET_M     = (1 << 3),
    SET_N     = (1 << 4),
    SET_K     = (1 << 5),
    SET_USE_IMAGES = (1 << 6),
    SET_DEVICE_TYPE = (1 << 7),
    SET_INCX  = (1 << 8),
    SET_INCY  = (1 << 9),
    SET_NUM_COMMAND_QUEUES = (1 << 10)
} SetoptFlags;

typedef struct TestParams {
    clblasOrder order;
    clblasTranspose transA;
    clblasTranspose transB;
    clblasTranspose transC;
    size_t M;
    size_t N;
    size_t K;
    size_t KL;
    size_t KU;
    int incx;
    int incy;
    size_t offA;
    size_t offBX;
    size_t offCY;
    size_t rowsA;
    size_t columnsA;
    size_t rowsB;
    size_t columnsB;
    size_t rowsC;
    size_t columnsC;
	size_t offa;
	size_t offb;
	size_t offc;
    // reminded alpha value set through the command line
    ComplexLong alpha;
    size_t lda;
    size_t ldb;
    // reminded beta value set through the command line
    ComplexLong beta;
    size_t ldc;
    clblasSide side;
    clblasUplo uplo;
    clblasDiag diag;
    unsigned int seed;
    int useImages;
    cl_device_type devType;
    const char*    devName;
    cl_uint numCommandQueues;
    SetoptFlags optFlags;
} TestParams;

int
parseBlasCmdLineArgs(
    int argc,
    char *argv[],
    TestParams *params);

void
printUsage(const char *appName);

void parseEnv(TestParams *params);

#ifdef __cplusplus
}       /* extern "C" { */
#endif

#endif  /* CMDLINE_H_ */
