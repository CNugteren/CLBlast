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


#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <kerngen.h>
#include <blas_kgen.h>
#include "../blas_kgen_legacy.h"

enum {
    ITEM_WORK_M = 8,
    ITEM_WORK_N = 8,
    GROUP_SIZE = ITEM_WORK_M * ITEM_WORK_N,
    BLOCKS_K = 2,
    PACK_RATE = 4,
    RAND_BOUND = 10
};

// float types based unified pointer
typedef union FPtr {
  void *v;
  cl_float *f;
  cl_double *d;
  cl_float2 *f2;
  cl_double2 *d2;
} FPtr;

// float type based unified data type
typedef union FType {
    unsigned char u[sizeof(cl_double)];
    cl_float f;
    cl_float2 f2;
    cl_double d;
    cl_double2 d2;
} FType;

static void
usage(void)
{
    printf("USAGE: blkmul_test type <M N K> [--iter i] [--imA] [--imB] [--alpha] a "
           "--[img-packed]\n"
           "type argument can be a value from the following list:s, d, c, z\n"
           "iter - number of iterations\n"
           "imA, imB - image usage for matrix\n"
           "img-packed - store elements of matrix A or (and) B "
           "to an image in the packed way\n");
}

static void
imageSizes(
    int *height,
    int *width,
    int blockHeight,
    int blockWidth,
    int AB,
    int typeSize,
    int packed)
{
    *width = blockWidth * typeSize / 16;
    *height = blockHeight;
    if (packed) {
        int smallHeight = (AB) ? (blockHeight / ITEM_WORK_M) :
                                 (blockHeight / ITEM_WORK_N);

        *width *= smallHeight * PACK_RATE;
        *height /= smallHeight * PACK_RATE;
    }
}

void
addTestPrefix(struct KgenContext *ctx, bool isDouble)
{
    kgenDeclareUptrs(ctx, isDouble);
}

void
addTestSuffix(
    struct KgenContext *ctx,
    const SubproblemDim subdims[2],
    DataType type,
    BlkMulOpts *mulOpts)
{
    char c;
    char s[300];
    bool isImageA, isImageB;
    char *tName;
    size_t m, n, k;
    size_t blockWidth;
    char imgXA[64], imgYA[64], imgXB[64], imgYB[64];
    unsigned int vecLen = sizeof(cl_float4) / dtypeSize(type);

    isImageA = (mulOpts->aMobj == CLMEM_IMAGE);
    isImageB = (mulOpts->bMobj == CLMEM_IMAGE);

    m = subdims[1].y;
    n = subdims[1].x;
    k = subdims[1].bwidth;
    blockWidth = k * BLOCKS_K;

    switch (type) {
    case TYPE_FLOAT:
        c = 's';
        tName = "float";
        break;
    case TYPE_DOUBLE:
        c = 'd';
        tName = "double";
        break;
    case TYPE_COMPLEX_FLOAT:
        c = 'c';
        tName = "float2";
        break;
    case TYPE_COMPLEX_DOUBLE:
        c = 'z';
        tName = "double2";
        break;
    default:
        return;
    }

    kgenAddBlankLine(ctx);
    kgenAddStmt(ctx, "__kernel void\n");
    kgenAddStmt(ctx, "blkmul_test(\n");
    sprintf(s,"    %s alpha,\n", tName);
    kgenAddStmt(ctx, s);
    if (isImageA) {
        kgenAddStmt(ctx, "    __read_only image2d_t A,\n");
    }
    else {
        sprintf(s,"    __global %s *A,\n", tName);
        kgenAddStmt(ctx, s);
    }
    if (isImageB) {
        kgenAddStmt(ctx, "    __read_only image2d_t B,\n");
    }
    else {
        sprintf(s,"    __global %s *B,\n", tName);
        kgenAddStmt(ctx, s);
    }
    kgenAddStmt(ctx, "    size_t M,\n"
                     "    size_t N,\n"
                     "    size_t K,\n");
    sprintf(s,"    __global %s *C,\n", tName);
    kgenAddStmt(ctx, s);
    kgenAddStmt(ctx, "    size_t iter)\n");
    kgenBeginFuncBody(ctx);
    kgenAddStmt(ctx, "size_t i, j, it, m0, n0;\n");
    if (!isImageA) {
        sprintf(s,"__local %s LA[%lu];\n", tName, m * ITEM_WORK_M * blockWidth);
        kgenAddStmt(ctx, s);
    }
    else {
        if (mulOpts->flags & BLKMUL_IMAGE_PACKED) {
            sprintf(imgXA, "(m0 / %lu) %% %d * %lu", m, PACK_RATE,
                    m * blockWidth / vecLen);
            sprintf(imgYA, "m0 / %lu", m * PACK_RATE);
        }
        else {
            strcpy(imgXA, "0");
            strcpy(imgYA, "m0");
        }
    }
    if (!isImageB) {
        sprintf(s,"__local %s LB[%lu];\n", tName, n * ITEM_WORK_N * blockWidth);
        kgenAddStmt(ctx, s);
    }
    else {
        if (mulOpts->flags & BLKMUL_IMAGE_PACKED) {
            sprintf(imgXB, "(n0 / %lu) %% %d * %lu", n, PACK_RATE,
                    n * blockWidth / vecLen);
            sprintf(imgYB, "n0 / %lu", n * PACK_RATE);
        }
        else {
            strcpy(imgXB, "0");
            strcpy(imgYB, "n0");
        }
    }

    sprintf(s,"__local %s LC[%lu];\n", tName, n * m * GROUP_SIZE);
    kgenAddStmt(ctx, s);

    sprintf(s, "m0 = %lu * (get_global_id(0) / %d);\n"
               "n0 = %lu * (get_global_id(0) %% %d);\n",
            m, ITEM_WORK_N, n, ITEM_WORK_N);
    kgenAddStmt(ctx, s);

    if (!isImageA) {
        kgenAddBlankLine(ctx);
        sprintf(s, "for (i = m0; i < m0 + %lu; i++)", m);
        kgenBeginBranch(ctx, s);
        kgenBeginBranch(ctx, "for (j = 0; j < K; j++)");
        kgenAddStmt(ctx,"LA[i * K + j] = A[i * K  + j];\n");
        kgenEndBranch(ctx, NULL);
        kgenEndBranch(ctx, NULL);
    }

    if (!isImageB) {
        kgenAddBlankLine(ctx);
        sprintf(s, "for (i = n0; i < n0 + %lu; i++)", n);
        kgenBeginBranch(ctx, s);
        kgenBeginBranch(ctx,"for (j = 0; j < K; j++)");
        kgenAddStmt(ctx,"LB[i * K + j] = B[i * K  + j];\n");
        kgenEndBranch(ctx, NULL);
        kgenEndBranch(ctx, NULL);
    }

    kgenAddBlankLine(ctx);

    kgenAddBlankLine(ctx);
    kgenBeginBranch(ctx,"for (it = 0; it < iter; it++)");
    sprintf(s, "for (i = m0; i < m0 + %lu; i++)", m);
    kgenBeginBranch(ctx, s);
    sprintf(s, "for (j = n0; j < n0 + %lu; j++)", n);
    kgenBeginBranch(ctx, s);
    kgenAddStmt(ctx,"LC[i * N + j] = 0;\n");
    kgenEndBranch(ctx, NULL);
    kgenEndBranch(ctx, NULL);

    if (isImageA) {
        if (isImageB) {
            sprintf(s, "%cgemmBlock_%lu_%lu(alpha, A, (int2)(%s, %s), B, "
                       "(int2)(%s, %s), (LPtr)(LC + m0 * %lu + n0));\n",
                    c, m, n, imgXA, imgYA, imgXB, imgYB, subdims[0].x);
        }
        else {
            sprintf(s, "%cgemmBlock_%lu_%lu(alpha, A, (int2)(%s, %s), "
                       "(LPtr)(LB + n0 * %lu), (LPtr)(LC + m0 * %lu + n0));\n",
                    c, m, n, imgXA, imgYA, subdims[0].bwidth, subdims[0].x);
        }
    }
    else {
        if (isImageB) {
            sprintf(s, "%cgemmBlock_%lu_%lu(alpha, (LPtr)(LA + m0 * %lu), B, "
                       "(int2)(%s, %s), (LPtr)(LC + m0 * %lu + n0));\n",
                    c, m, n, subdims[0].bwidth, imgXB, imgYB, subdims[0].x);
        }
        else {
            sprintf(s, "%cgemmBlock_%lu_%lu(alpha, (LPtr)(LA + m0 * %lu), "
                       "(LPtr)(LB + n0 * %lu), (LPtr)(LC + m0 * %lu + n0));\n",
                    c, m, n, subdims[0].bwidth, subdims[0].bwidth,
                    subdims[0].x);
        }
    }
    kgenAddStmt(ctx, s);
    kgenEndBranch(ctx, NULL);

    kgenAddBlankLine(ctx);
    sprintf(s, "for (i = m0; i < m0 + %lu; i++)", m);
    kgenBeginBranch(ctx, s);
    sprintf(s, "for (j = n0; j < n0 + %lu; j++)", n);
    kgenBeginBranch(ctx, s);
    kgenAddStmt(ctx,"C[i * N  + j] = LC[i * N + j];\n");
    kgenEndBranch(ctx, NULL);
    kgenEndBranch(ctx, NULL);

    kgenEndFuncBody(ctx);
}

cl_int
run (char *ker, cl_uint M, cl_uint N, cl_uint K, FType alpha, DataType type, BlkMulOpts *mulOpts, cl_uint iter)
{
    cl_int err;
    cl_platform_id platform;
    cl_context ctx;
    cl_device_id device;
    cl_command_queue queue;
    cl_event evt;
    FType tmp;

    cl_mem imA, imB, bufC;
    FPtr A, B, C, C_naive;
    bool is_complex = type == TYPE_COMPLEX_FLOAT || type == TYPE_COMPLEX_DOUBLE;
    bool is_double = type == TYPE_DOUBLE || type == TYPE_COMPLEX_DOUBLE;
    cl_uint nwords = (is_complex) ? 2 : 1;
    unsigned int tsize = dtypeSize(type);
    cl_kernel kernel;
    const cl_image_format image_format = {CL_RGBA, CL_FLOAT};
    size_t i, j, k;
    size_t globalWorkSize[1] = {GROUP_SIZE};
    size_t localWorkSize[1] = {GROUP_SIZE};
    char log[100000]; size_t logSize;
    cl_long sTime, fTime;
    cl_program program = NULL;
    const char *kernelName = "blkmul_test";
    int imgWidth, imgHeight;
    bool packed = (mulOpts->flags & BLKMUL_IMAGE_PACKED);

    clGetPlatformIDs(1, &platform, NULL);

    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        return err;
    }

    queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        return err;
    }

    /* Prepare OpenCL kernel and its arguments */

    program = clCreateProgramWithSource(ctx, 1, (const char**)&ker, NULL, NULL);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS){
        clGetProgramBuildInfo (program,
            device,
            CL_PROGRAM_BUILD_LOG,
            100000,
            log,
            &logSize);
        printf("%s", log);
        clReleaseProgram(program);
        return err;
    }

    kernel = clCreateKernel(program, kernelName, &err);
    if (err != CL_SUCCESS){
        clReleaseProgram(program);
        return err;
    }

    /* Memory allocation */

    A.v = malloc(M * K * tsize);
    B.v = malloc(K * N * tsize);
    C.v = malloc(M * N * tsize);
    C_naive.v = malloc(M * N * tsize);

    srand(0);
    if (is_double) {
        for(i = 0; i < M * K * nwords; i++){
            A.d[i] = (double)(rand() % RAND_BOUND);
        }
        for(i = 0; i < N * K * nwords; i++){
            B.d[i] = (double)(rand() % RAND_BOUND);
        }
        for(i = 0; i < M * N * nwords; i++){
            C.d[i] = 0.0;
            C_naive.d[i] = 0.0;
        }
    }
    else {
        for(i = 0; i < M * K * nwords; i++){
            A.f[i] = (float)(rand() % RAND_BOUND);
        }
        for(i = 0; i < N * K * nwords; i++){
            B.f[i] = (float)(rand() % RAND_BOUND);
        }
        for(i = 0; i < M * N * nwords; i++){
            C.f[i] = 0.0;
            C_naive.f[i] = 0.0;
        }
    }

    if (mulOpts->aMobj == CLMEM_IMAGE) {
        imageSizes(&imgHeight, &imgWidth, M, K, 0, tsize, packed);
        imA = clCreateImage2D (ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            &image_format, imgWidth, imgHeight, 0, A.v, &err);
    }
    else {
        imA = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
            K * M * tsize, A.v, &err);
    }
    if (err != CL_SUCCESS) {
        clReleaseKernel(kernel);
        return err;
    }
    if (mulOpts->bMobj == CLMEM_IMAGE) {
        imageSizes(&imgHeight, &imgWidth, N, K, 0, tsize, packed);
        imB = clCreateImage2D (ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            &image_format, imgWidth, imgHeight, 0, B.v, &err);
    }
    else {
        imB = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
            K * N * tsize, B.v, &err);
    }
    if (err != CL_SUCCESS) {
        clReleaseMemObject(imA);
        clReleaseKernel(kernel);
        return err;
    }

    bufC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        M * N * tsize, C.v, &err);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(imB);
        clReleaseMemObject(imA);
        clReleaseKernel(kernel);
        return err;
    }

    err = clEnqueueWriteBuffer (queue,
        bufC,
        CL_TRUE,
        0,
        M * N * tsize,
        C.v,
        0,
        NULL,
        NULL);

    /* Argument setting and kernel execution */
    err = clSetKernelArg(kernel, 0, tsize, alpha.u);
    err |= clSetKernelArg(kernel, 1, sizeof(imA), &imA);
    err |= clSetKernelArg(kernel, 2, sizeof(imB), &imB);
    err |= clSetKernelArg(kernel, 3, sizeof(M), &M);
    err |= clSetKernelArg(kernel, 4, sizeof(N), &N);
    err |= clSetKernelArg(kernel, 5, sizeof(K), &K);
    err |= clSetKernelArg(kernel, 6, sizeof(bufC), &bufC);
    err |= clSetKernelArg(kernel, 7, sizeof(iter), &iter);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(bufC);
        clReleaseMemObject(imB);
        clReleaseMemObject(imA);
        clReleaseKernel(kernel);
        return err;
    }

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        globalWorkSize, localWorkSize, 0,
        NULL, &evt);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(bufC);
        clReleaseMemObject(imB);
        clReleaseMemObject(imA);
        clReleaseKernel(kernel);
        return err;
    }

    err = clFinish(queue);

    err = clEnqueueReadBuffer (queue,
        bufC,
        CL_TRUE,
        0,
        M * N * tsize,
        C.v,
        0,
        NULL,
        NULL);

    /* Naive CPU multiplication */
    if (is_double) {
        if (is_complex) {
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    for (k = 0; k < K; k++) {
                        C_naive.d[(i * N + j) * 2] +=
                            A.d[(i * K + k) * 2] * B.d[(j * K + k) * 2] -
                            A.d[(i * K + k) * 2 + 1] * B.d[(j * K + k) * 2 + 1];

                        C_naive.d[(i * N + j) * 2 + 1] +=
                            A.d[(i * K + k) * 2] * B.d[(j * K + k) * 2 + 1] +
                            A.d[(i * K + k) * 2 + 1] * B.d[(j * K + k) * 2];
                    }

                    tmp.d2.s[0] = C_naive.d[(i * N + j) * 2] * alpha.d2.s[0] -
                                  C_naive.d[(i * N + j) * 2 + 1] * alpha.d2.s[1];
                    tmp.d2.s[1] = C_naive.d[(i * N + j) * 2] * alpha.d2.s[1] +
                                  C_naive.d[(i * N + j) * 2 + 1] * alpha.d2.s[0];
                    C_naive.d[(i * N + j) * 2] = tmp.d2.s[0];
                    C_naive.d[(i * N + j) * 2 + 1] = tmp.d2.s[1];
                }
            }

            for (i = 0; i < M * N; i++) {
                if ((C.d[i * 2] != C_naive.d[i * 2])  ||
                    (C.d[i * 2 + 1] !=  C_naive.d[i * 2 + 1])) {
                    printf("Differ at (%lu, %lu): (%lf; %lf) != (%lf; %lf)\n",
                           i / N, i % N, C.d[i * 2], C.d[i * 2 + 1],
                           C_naive.d[i * 2], C_naive.d[i * 2 + 1]);
                    break;
                }
            }
            if (i == M * N) {
                printf("Match\n");
            }
        }
        else {
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    for (k = 0; k < K; k++) {
                        C_naive.d[i * N + j] += A.d[i * K + k] * B.d[j * K + k];
                    }
                    C_naive.d[i * N + j] *= alpha.d;
                }
            }

            for (i = 0; i < M * N; i++) {
                if (C.d[i] != C_naive.d[i]) {
                    printf("Differ at (%lu, %lu): %lf != %lf\n", i / N, i % N,
                           C.d[i], C_naive.d[i]);
                    break;
                }
            }
            if (i == M * N) {
                printf("Match\n");
            }
        }
    }
    else {
        if (is_complex) {
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    for (k = 0; k < K; k++) {
                        C_naive.f[(i * N + j) * 2] +=
                            A.f[(i * K + k) * 2] * B.f[(j * K + k) * 2] -
                            A.f[(i * K + k) * 2 + 1] * B.f[(j * K + k) * 2 + 1];

                        C_naive.f[(i * N + j) * 2 + 1] +=
                            A.f[(i * K + k) * 2] * B.f[(j * K + k) * 2 + 1] +
                            A.f[(i * K + k) * 2 + 1] * B.f[(j * K + k) * 2];
                    }

                    tmp.f2.s[0] = C_naive.f[(i * N + j) * 2] * alpha.f2.s[0] -
                                  C_naive.f[(i * N + j) * 2 + 1] * alpha.f2.s[1];
                    tmp.f2.s[1] = C_naive.f[(i * N + j) * 2] * alpha.f2.s[1] +
                                  C_naive.f[(i * N + j) * 2 + 1] * alpha.f2.s[0];
                    C_naive.f[(i * N + j) * 2] = tmp.f2.s[0];
                    C_naive.f[(i * N + j) * 2 + 1] = tmp.f2.s[1];
                }
            }

            for (i = 0; i < M * N; i++) {
                if ((C.f[i * 2] != C_naive.f[i * 2]) ||
                    (C.f[i * 2 + 1] != C_naive.f[i * 2 + 1])) {
                    printf("Differ at (%lu, %lu): (%lf; %lf) != (%lf; %lf)\n",
                           i / N, i % N, C.f[i * 2], C.f[i * 2 + 1],
                           C_naive.f[i * 2], C_naive.f[i * 2 + 1]);
                    break;
                }
            }
            if (i == M * N) {
                printf("Match\n");
            }
        }
        else {
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    for (k = 0; k < K; k++) {
                        C_naive.f[i * N + j] += A.f[i * K + k] * B.f[j * K + k];
                    }
                    C_naive.f[i * N + j] *= alpha.f;
                }
            }

            for (i = 0; i < M * N; i++) {
                if (C.f[i] != C_naive.f[i]) {
                    printf("Differ at (%lu, %lu): %lf != %lf\n",
                           i / N, i % N, C.f[i], C_naive.f[i]);
                    break;
                }
            }
            if (i == M * N) {
                printf("Match\n");
            }
        }
    }
    /* End of naive CPU multiplication */

    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &sTime, NULL);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &fTime, NULL);

    printf("Total multiplication time: %d ms\nTime per iteration: %d ns\n",
        (int)((fTime-sTime)/1000000), (int)((fTime-sTime)/iter));

    clReleaseMemObject(bufC);
    clReleaseMemObject(imB);
    clReleaseMemObject(imA);
    clReleaseKernel(kernel);
    return CL_SUCCESS;
}

int main(int argc, char *argv[])
{
    char out[65535];
    SubproblemDim subdims[2];
    BlkMulOpts mulOpts;
    DataType dtype;
    int i;
    cl_uint iter = 1, blockM = 4, blockN = 4, blockK = 8;
    struct KgenContext *ctx = createKgenContext(out, 65535, 1);
    FType alpha;
    int cmdAlpha = 0;

    mulOpts.aMobj = CLMEM_BUFFER;
    mulOpts.bMobj = CLMEM_BUFFER;
    mulOpts.flags = BLKMUL_NO_FLAGS;

   // parse command line

    if (argc < 2) {
        usage();
        return 1;
    }

    if (!strcmp(argv[1], "s")) {
        dtype = TYPE_FLOAT;
        alpha.f = 1;
    }
    else if (!strcmp(argv[1], "d")) {
        dtype = TYPE_DOUBLE;
        alpha.d = 1;
    }
    else if (!strcmp(argv[1], "c")) {
        dtype = TYPE_COMPLEX_FLOAT;
        alpha.f2.s[0] = 1;
        alpha.f2.s[1] = 0;
    }
    else if (!strcmp(argv[1], "z")) {
        dtype = TYPE_COMPLEX_DOUBLE;
        alpha.d2.s[0] = 1;
        alpha.d2.s[1] = 0;
    }
    else {
        printf("Wrong type specified: %s\n", argv[1]);
        return 1;
    }

    for (i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--imA") == 0) {
            mulOpts.aMobj = CLMEM_IMAGE;
            continue;
        }
        if (strcmp(argv[i], "--imB") == 0) {
            mulOpts.bMobj = CLMEM_IMAGE;
            continue;
        }
        if (strcmp(argv[i], "--img-packed") == 0) {
            mulOpts.flags |= BLKMUL_IMAGE_PACKED;
            continue;
        }

        if (strcmp(argv[i], "--iter") == 0) {
            if (i + 1 == argc) {
                printf("Error: 'iter' argument is not specified\n");
                usage();
                return 1;
            }
            iter = atoi(argv[i + 1]);
            i++;
            continue;
        }

        if (strcmp(argv[i], "--alpha") == 0) {
            if (i + 1 == argc) {
                printf("Error: 'alpha' argument is not specified\n");
                usage();
                return 1;
            }
            cmdAlpha = atoi(argv[i + 1]);
            i++;
            continue;
        }

        if (i + 2 >= argc) {
            printf("Error: Not all sizes are specified\n");
            usage();
            return 1;
        }
        blockM = atoi(argv[i]);
        blockN = atoi(argv[i + 1]);
        blockK = atoi(argv[i + 2]);
        i += 2;
    }

    if (cmdAlpha) {
        switch (dtype) {
        case TYPE_FLOAT:
            alpha.f = cmdAlpha;
            break;
        case TYPE_DOUBLE:
            alpha.d = cmdAlpha;
            break;
        case TYPE_COMPLEX_FLOAT:
            alpha.f2.s[0] = cmdAlpha;
            alpha.f2.s[1] = -cmdAlpha / 2;
            break;
        case TYPE_COMPLEX_DOUBLE:
            alpha.d2.s[0] = cmdAlpha;
            alpha.d2.s[1] = -cmdAlpha / 2;
            break;
        default:
            break;
        }
    }

    subdims[0].y = blockM * ITEM_WORK_M;
    subdims[0].x = blockN * ITEM_WORK_N;
    subdims[0].bwidth = blockK * BLOCKS_K;
    subdims[1].y = blockM;
    subdims[1].x = blockN;
    subdims[1].bwidth = blockK;

    memset(out, 0, sizeof(out));

    i = isDoubleBasedType(dtype);
    addTestPrefix(ctx, i);

    blkMulGen(ctx, subdims, dtype, &mulOpts);

    addTestSuffix(ctx, subdims, dtype, &mulOpts);

    run(out, subdims[0].y, subdims[0].x, subdims[0].bwidth, alpha,
        dtype, &mulOpts, iter);

    destroyKgenContext(ctx);

	return 0;
}
