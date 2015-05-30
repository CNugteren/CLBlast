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
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <getopt.h>
#include <kerngen.h>
#include <blas_kgen.h>
#include <clblas_stddef.h>

#define JUST_MULTIPLICATION 0

#if JUST_MULTIPLICATION
enum {
    ITEM_WORK_M = 1,
    ITEM_WORK_N = 1,
    ITEM_BLOCKS_K = 1,
};
#else
enum {
    ITEM_WORK_M = 4,
    ITEM_WORK_N = 4,
    ITEM_BLOCKS_K = 3,
    RAND_BOUND = 10
};
#endif

const char *kernelName = "tilemul_test";


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
printUsage(const char *programName, int exitCode)
{
    printf( "USAGE: %s [options] <M N K>\n"
            "  --help, -h                Print this help message.\n"
            "  --device, -d <device>     OpenCL device used. <device> can "
            "be \"gpu\" or \"cpu\". Default is \"gpu\".\n"
            "  --type, -t <type>         Type can be s, d, c or z. Default "
            "is s.\n"
            "  --fetch, -f <vector size> Size of used fetch vectors, in used "
            "types. Default is 1.\n"
            "  --local, -l <matrix>      If matrix is local or global. Matrix "
            "can be A or B. By default, both are global.\n"
            "  --verbose, -v             Turn on verbose mode.\n"
            "  --a, -a <order>\n"
            "  --b, -b <order>\n         Set order for tiles a and b fetching. "
            "Order can be are \"r\" for row major and \"c\" for "
            "column major. Default values are \"r\" for A and \"c\" for B.\n"
            "  --skew, -s <skew_value>   Set skews for tiles along M, N, and K "
            "directions. skew_value can be \"a\" for tile A skew along M, \"b\""
            " for tile B skew along N and \"k\" for both tiles skew along K. "
            "There is no skews by default.\n"
            "  -g, --globalcycling <global_cycling_value>\n"
            "                            Set global cycling for tiles along M, "
            "N and K directions. global_cycling_value can be \"a\" for tile A "
            "global cycling along M, \"b\" for tile B global cycling along N "
            "and \"k\" for both tiles global cycling along K. There is no "
            "global cycling enabled by default.\n"
            "  --iter, -i <num>          Number of iterations.\n"
            "  --core, -c <mulcore>      Multiplier core. <mulcore> can "
            "be \"muladd\", \"mad\" or \"dot\". Default is \"mad\".\n"
            "  --old, -o                 Use old tilemul generator interface "
            "with one generator function call for both fetching and "
            "multiplication. Separate generators functions are used by "
            "default.\n"
            "  M N K                     Size of block.\n",
           programName);
    exit(exitCode);
}

void
genFillTileWithNAN(struct KgenContext *ctx, const Tile *tile)
{
    char tmp[1024];
    Kstring elem;
    unsigned int incRows, incCols;
    unsigned int i, j, v;

    if (!tile->trans) {
        incRows = 1;
        v = incCols = umin(tile->vecLen, tile->nrCols);
    }
    else {
        v = incRows = umin(tile->vecLen, tile->nrRows);
        incCols = 1;
    }

    for (i = 0; i < tile->nrRows; i += incRows) {
        for (j = 0; j < tile->nrCols; j += incCols) {
            sprintfTileElement(&elem, tile, i, j, v);
            sprintf(tmp, "%s = NAN;\n", elem.buf);
            kgenAddStmt(ctx, tmp);
        }
    }

    kgenAddBlankLine(ctx);
}

void
addTestPrefix(struct KgenContext *ctx, bool isDouble)
{
    kgenDeclareUptrs(ctx, isDouble);
}

static void checkRet(int ret, const char *genName)
{
    if (ret != 0) {
        printf("%s generator failed: %s\n", genName, strerror(-ret));
        exit(EXIT_FAILURE);
    }
}

void
genTest(
    struct KgenContext *ctx,
    BlasGenSettings *gset,
    TileMulOpts *mulOpts,
    bool separateFetch)
{
    char s[1024];
    Kstring kstr;
    char *tName, tVect[64], *ptrName;
    KernelVarNames *vnames = &gset->varNames;
    DataType dtype = gset->kextra->dtype;
    const SubproblemDim *subdims = gset->subdims;
    unsigned int vecLen = gset->kextra->vecLen;
    size_t m, n, k;
    unsigned int i, j;
    bool tra, trb, localA, localB, vecCoords;
    int ret;
    TileMulFlags flags = mulOpts->flags;
    FetchOpts fetchOpts;

    m = gset->subdims[1].y;
    n = gset->subdims[1].x;
    k = gset->subdims[1].bwidth;

    tra = ((flags & TILEMUL_TRA) != 0);
    trb = ((flags & TILEMUL_TRB) != 0);
    localA = (mulOpts->memA == CLMEM_LOCAL_MEMORY);
    localB = (mulOpts->memB == CLMEM_LOCAL_MEMORY);

    vecCoords = ((flags & TILEMUL_OPTIMIZE_VEC_COORDS) != 0);

    tVect[0] = '\0';

    if (vecCoords && vecLen != 1) {
        sprintf(tVect, "%u", vecLen);
    }

    switch (dtype) {
    case TYPE_FLOAT:
        tName = "float";
        ptrName = "f";
        break;
    case TYPE_DOUBLE:
        tName = "double";
        ptrName = "d";
        break;
    case TYPE_COMPLEX_FLOAT:
        tName = "float2";
        ptrName = "f2v";
        break;
    case TYPE_COMPLEX_DOUBLE:
        tName = "double2";
        ptrName = "d2v";
        break;
    default:
        return;
    }

    if (vecCoords) {
        //Do not use GPtrs in fetching
        vnames->A = "A";
        vnames->B = "B";
    }
    else {
        vnames->A = localA ? "LAptr" : "((GPtr)A)";
        vnames->B = localB ? "LBptr" : "((GPtr)B)";
    }
    if (!localA) {
        vnames->lda = "lda";

    }
    if (!localB) {
        vnames->ldb = "ldb";
    }
    vnames->sizeM = "M";
    vnames->sizeN = "N";
    vnames->sizeK = "K";
    vnames->skewA = "skewA";
    vnames->skewB = "skewB";
    vnames->skewK = "skewK";
    vnames->coordA = "workItemM";
    vnames->coordB = "workItemN";
    vnames->k = "k";

    kgenAddBlankLine(ctx);
    sprintf(s, "__attribute__((reqd_work_group_size(%i, %i, 1)))\n",
            ITEM_WORK_M, ITEM_WORK_N);
    kgenAddStmt(ctx, s);
    kgenAddStmt(ctx, "__kernel void\n");
    sprintf(s, "%s(\n", kernelName);
    kgenAddStmt(ctx, s);
    sprintf(s,"    %s alpha,\n", tName);
    kgenAddStmt(ctx, s);
    sprintf(s,"    __global %s%s *A,\n", tName, tVect);
    kgenAddStmt(ctx, s);
    sprintf(s,"    __global %s%s *B,\n", tName, tVect);
    kgenAddStmt(ctx, s);
    kgenAddStmt(ctx, "    uint M,\n"
                     "    uint N,\n"
                     "    uint K,\n");
    sprintf(s,
            "    __global %s *C,\n"
            "    const uint iter)\n", tName);
    kgenAddStmt(ctx, s);
    kgenBeginFuncBody(ctx);
    sprintf(s, "uint workItemM = %lu * get_global_id(0);\n"
               "uint workItemN = %lu * get_global_id(1);\n",
            m, n);
    kgenAddStmt(ctx, s);
    if ((flags & TILEMUL_SKEW_A) != 0) {
        kgenAddStmt(ctx, "uint skewA = 0u;\n");
    }
    if ((flags & TILEMUL_SKEW_B) != 0) {
        kgenAddStmt(ctx, "uint skewB = 0u;\n");
    }
    if ((flags & TILEMUL_SKEW_K) != 0) {
        kgenAddStmt(ctx, "uint skewK = 0u;\n");
    }

    if (localA) {
        sprintf(s, "__local %s LA[%lu];\n",
                tName, subdims[0].bwidth * subdims[0].y);
        kgenAddStmt(ctx, s);
    }
    else { //global A
        sprintf(s, "uint lda = %s;\n", tra ? "M" : "K");
        kgenAddStmt(ctx, s);
    }
    if (localB) {
        sprintf(s, "__local %s LB[%lu];\n",
                tName, subdims[0].bwidth * subdims[0].x);
        kgenAddStmt(ctx, s);
    }
    else { //global B
        sprintf(s, "uint ldb = %s;\n", trb ? "K" : "N");
        kgenAddStmt(ctx, s);
    }

    initDefaultTiles(gset, CLBLAS_GEMM, TILE_PACKED, PRIV_STORAGE_ARRAY);
    declareTileStorages(ctx, gset);

    if (vecCoords) {
        size_t ha, hb;
        char *str;

        ha = tra ? k : m;
        hb = trb ? n : k;

        if (ha > 1) {
            str = s;
            str += sprintf(str, "uint%lu ca = {0", ha);
            for (i = 1; i < ha; i++) {
                str += sprintf(str, ", %s * %u / %u", vnames->lda, i, vecLen);
            }
            str += sprintf(str, "};\n");
            kgenAddStmt(ctx, s);
        }
        else {
            kgenAddStmt(ctx, "uint ca = 0;\n");
        }
        vnames->vectCoordA = "ca";

        if (hb > 1) {
            str = s;
            str += sprintf(str, "uint%lu cb = {0", hb);
            for (i = 1; i < hb; i++) {
                str += sprintf(str, ", %s * %u / %u", vnames->ldb, i, vecLen);
            }
            str += sprintf(str, "};\n");
            kgenAddStmt(ctx, s);
        }
        else {
            kgenAddStmt(ctx, "uint cb = 0;\n");
        }
        vnames->vectCoordB = "cb";

//        uint4 ca = {0, vecLDA, vecLDA * 2, vecLDA * 3};
//        uint4 cb = {0, vecLDB, vecLDB * 2, vecLDB * 3};
    }

    kgenAddBlankLine(ctx);

    sprintf(s, "for (int it = 0; it < iter; it++)");
    kgenBeginBranch(ctx, s);

    if (!(localA && localB)) {
        kgenAddStmt(ctx, "uint k = 0;\n");
    }

    genZeroTile(ctx, &gset->tileCY);

    if (vecCoords) {
        char *coordsA[2] = {"workItemM", "k"};
        char *coordsB[2] = {"k", "workItemN"};
        sprintf(s, "A += %s * (lda / %u) + %s / %u;\n",
                coordsA[tra], vecLen, coordsA[1 - tra], vecLen);
        kgenAddStmt(ctx, s);
        sprintf(s, "B += %s * (ldb / %u) + %s / %u;\n",
                coordsB[trb], vecLen, coordsB[1 - trb], vecLen);
        kgenAddStmt(ctx, s);
    }

    sprintf(s, "for (int k0 = 0; k0 < K; k0 += %lu)", subdims[0].bwidth);
    kgenBeginBranch(ctx, s);

    /* Copy data to local memory. We know that the size of matrix is the same
     * that the size of one block and use that.
     */
    if (localA) {
        sprintf(s,
                "event_t evA = async_work_group_copy(LA, A, %lu, 0);\n"
                "wait_group_events(1, &evA);\n"
                "barrier(CLK_LOCAL_MEM_FENCE);\n",
                subdims[0].y * subdims[0].bwidth);
        kgenAddStmt(ctx, s);
        kgenAddStmt(ctx, "LPtr LAptr;\n");
        if (tra) {
            sprintf(s,
                    "LAptr.%s = LA + workItemM;\n", ptrName);
        }
        else {
            sprintf(s,
                    "LAptr.%s = LA + workItemM * %lu;\n",
                    ptrName, subdims[0].bwidth);
        }
        kgenAddStmt(ctx, s);
    }
    if (localB) {
        sprintf(s,
                "event_t evB = async_work_group_copy(LB, B, %lu, 0);\n"
                "wait_group_events(1, &evB);\n"
                "barrier(CLK_LOCAL_MEM_FENCE);\n",
                subdims[0].x * subdims[0].bwidth);
        kgenAddStmt(ctx, s);
        kgenAddStmt(ctx, "LPtr LBptr;\n");
        if (trb) {
            sprintf(s, "LBptr.%s = LB + workItemN * %lu;\n",
                    ptrName, subdims[0].bwidth);
        }
        else {
            sprintf(s, "LBptr.%s = LB + workItemN;\n", ptrName);
        }
        kgenAddStmt(ctx, s);
    }

    if (!separateFetch) {
        ret = tileMulGen(ctx, gset, mulOpts);
        checkRet(ret, "Multiplier");
    }
    else {
        Tile *tileA = &gset->tileA;
        Tile *tileB = &gset->tileBX;

        memset(&fetchOpts, 0, sizeof(fetchOpts));
        if (localA) {
            fetchOpts.memA = CLMEM_LOCAL_MEMORY;
        }
        if (localB) {
            fetchOpts.memB = CLMEM_LOCAL_MEMORY;
        }

        genFillTileWithNAN(ctx, tileA);
        genFillTileWithNAN(ctx, tileB);

        if (subdims[0].bwidth != subdims[1].bwidth) {
            sprintf(s, "for (int k1 = 0; k1 < %lu; k1 += %lu)",
                    subdims[0].bwidth, k);
            kgenBeginBranch(ctx, s);
        }

#if JUST_MULTIPLICATION
        for (i = 0; i < tileA->nrRows; i++) {
            for(j = 0; j < tileA->nrCols; j++) {
                sprintfTileElement(&kstr, tileA, i, j, 1);
                sprintf(s, "%s = %u;\n", kstr.buf, i * tileA->nrCols + j);
                kgenAddStmt(ctx, s);
            }
        }

        for (i = 0; i < tileB->nrRows; i++) {
            for(j = 0; j < tileB->nrCols; j++) {
                sprintfTileElement(&kstr, tileB, i, j, 1);
                sprintf(s, "%s = %u;\n", kstr.buf, i * tileB->nrCols + j);
                kgenAddStmt(ctx, s);
            }
        }
#else
        fetchOpts.mrole = MATRIX_B;
        fetchOpts.lineOffset = 0;
        fetchOpts.linesNum = (tileB->trans) ? tileB->nrCols : tileB->nrRows;
        ret = genFetchInputTile(ctx, NULL, gset, &fetchOpts);
        checkRet(ret, "Fetching tile b");

        fetchOpts.mrole = MATRIX_A;
        fetchOpts.linesNum = (tileA->trans) ? tileA->nrCols : tileA->nrRows;
        kgenAddBlankLine(ctx);
        fetchOpts.lineOffset = 0;
        ret = genFetchInputTile(ctx, NULL, gset, &fetchOpts);
        checkRet(ret, "Fetching tile a");
#endif
        ret = genMulTiles(ctx, gset, mulOpts);
        checkRet(ret, "Multiplier");
#if ! JUST_MULTIPLICATION
        sprintf(s, "k += %lu;\n", k);
        kgenAddStmt(ctx, s);
#endif
        if (subdims[0].bwidth != subdims[1].bwidth) {
            kgenEndBranch(ctx, NULL);
        }
    }
    kgenEndBranch(ctx, NULL); // K loop
    kgenEndBranch(ctx, NULL); // iterations loop

    kgenAddBlankLine(ctx);

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            sprintfTileElement(&kstr, &gset->tileCY, i, j, 1);
                sprintf(s,
                        "((GPtr)C).%s"
                    "[(%d + workItemM) * N  + %d + workItemN] = %s;\n",
                    ptrName, i, j, kstr.buf);
                kgenAddStmt(ctx, s);
            }
                }

    kgenEndFuncBody(ctx);
}

cl_int
run (
        const char *ker,
        cl_uint M,
        cl_uint N,
        cl_uint K,
        FType alpha,
        BlasGenSettings *gset,
        TileMulFlags flags,
        cl_device_type deviceType,
        bool verbose,
        unsigned int iterNum)
{
    cl_int err;
    cl_platform_id platform;
    cl_context ctx;
    cl_device_id device;
    cl_command_queue queue;
    cl_event evt;
    DataType dtype = gset->kextra->dtype;

    cl_mem bufA, bufB, bufC;
    FPtr A, B, C, C_naive;
    bool isComplex = isComplexType(dtype);
    bool isDouble = isDoubleBasedType(dtype);
    cl_uint nwords = (isComplex) ? 2 : 1;
    unsigned int tsize = dtypeSize(dtype);
    cl_kernel kernel;
    size_t i, j, k;
    size_t globalWorkSize[2] = {ITEM_WORK_M, ITEM_WORK_N};
    size_t localWorkSize[2] = {ITEM_WORK_M, ITEM_WORK_N};
    char log[100000];
    size_t logSize;
    cl_long sTime, fTime;
    cl_program program = NULL;

    clGetPlatformIDs(1, &platform, NULL);

    clGetDeviceIDs(platform, deviceType, 1, &device, NULL);

    ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        return err;
    }

    queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        return err;
    }

    /* Prepare OpenCL kernel and its arguments */

    program = clCreateProgramWithSource(ctx, 1, &ker, NULL, NULL);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    clGetProgramBuildInfo (program,
            device,
            CL_PROGRAM_BUILD_LOG,
            sizeof(log),
            log,
            &logSize);
    printf("%s", log);
    if (err != CL_SUCCESS){
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

#if JUST_MULTIPLICATION
    srand(0);
    if (isDouble) {
        for(i = 0; i < M * K * nwords; i++){
            A.d[i] = i;
        }
        for(i = 0; i < N * K * nwords; i++){
            B.d[i] = i + 7;
        }
        for(i = 0; i < M * N * nwords; i++){
            C.d[i] = 0.0;
            C_naive.d[i] = 0.0;
        }
    }
    else {
        for(i = 0; i < M * K * nwords; i++){
            A.f[i] = i;
        }
        for(i = 0; i < N * K * nwords; i++){
            B.f[i] = i + 7;
        }
        for(i = 0; i < M * N * nwords; i++){
            C.f[i] = 0.0;
            C_naive.f[i] = 0.0;
        }
    }

#else
    srand(0);
    if (isDouble) {
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
#endif

    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            K * M * tsize, A.v, &err);
    if (err != CL_SUCCESS) {
        clReleaseKernel(kernel);
        return err;
    }

    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            K * N * tsize, B.v, &err);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(bufA);
        clReleaseKernel(kernel);
        return err;
    }

    bufC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        M * N * tsize, C.v, &err);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufA);
        clReleaseKernel(kernel);
        return err;
    }

    /* Argument setting and kernel execution */
    err = clSetKernelArg(kernel, 0, tsize, alpha.u);
    err |= clSetKernelArg(kernel, 1, sizeof(bufA), &bufA);
    err |= clSetKernelArg(kernel, 2, sizeof(bufB), &bufB);
    err |= clSetKernelArg(kernel, 3, sizeof(M), &M);
    err |= clSetKernelArg(kernel, 4, sizeof(N), &N);
    err |= clSetKernelArg(kernel, 5, sizeof(K), &K);
    err |= clSetKernelArg(kernel, 6, sizeof(bufC), &bufC);
    err |= clSetKernelArg(kernel, 7, sizeof(iterNum), &iterNum);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(bufC);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufA);
        clReleaseKernel(kernel);
        return err;
    }

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
        globalWorkSize, localWorkSize, 0,
        NULL, &evt);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(bufC);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufA);
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
    if (isDouble) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                if (isComplex) {
                    cl_double2 val;
                    for (k = 0; k < K; k++) {
                        cl_double2 bkj = flags & TILEMUL_TRB ?
                                B.d2[j * K + k] : B.d2[k * N + j];
                        cl_double2 aik = flags & TILEMUL_TRA ?
                                A.d2[k * M + i] : A.d2[i * K + k];
                        val.s[0] = aik.s[0] * bkj.s[0] - aik.s[1] * bkj.s[1];
                        val.s[1] = aik.s[0] * bkj.s[1] + aik.s[1] * bkj.s[0];
                        C_naive.d2[i * N + j].s[0] += val.s[0];
                        C_naive.d2[i * N + j].s[1] += val.s[1];
                    }
                    val.s[0] = C_naive.d2[i * N + j].s[0] * alpha.d2.s[0] -
                            C_naive.d2[i * N + j].s[1] * alpha.d2.s[1];
                    val.s[1] = C_naive.d2[i * N + j].s[0] * alpha.d2.s[1] +
                            C_naive.d2[i * N + j].s[1] * alpha.d2.s[0];
                    C_naive.d2[i * N + j] = val;
                }
                else {
                    for (k = 0; k < K; k++) {
                        double bkj = flags & TILEMUL_TRB ?
                                B.d[j * K + k] : B.d[k * N + j];
                        double aik = flags & TILEMUL_TRA ?
                                A.d[k * M + i] : A.d[i * K + k];
                        C_naive.d[i * N + j] += aik * bkj;
                    }
                    C_naive.d[i * N + j] *= alpha.d;
                }
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
    else {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                if (isComplex) {
                    cl_float2 val;
                    for (k = 0; k < K; k++) {
                        cl_float2 bkj = flags & TILEMUL_TRB ?
                                B.f2[j * K + k] : B.f2[k * N + j];
                        cl_float2 aik = flags & TILEMUL_TRA ?
                                A.f2[k * M + i] : A.f2[i * K + k];
                        val.s[0] = aik.s[0] * bkj.s[0] - aik.s[1] * bkj.s[1];
                        val.s[1] = aik.s[0] * bkj.s[1] + aik.s[1] * bkj.s[0];
                        C_naive.f2[i * N + j].s[0] += val.s[0];
                        C_naive.f2[i * N + j].s[1] += val.s[1];
                    }
                    val.s[0] = C_naive.f2[i * N + j].s[0] * alpha.f2.s[0] -
                            C_naive.f2[i * N + j].s[1] * alpha.f2.s[1];
                    val.s[1] = C_naive.f2[i * N + j].s[0] * alpha.f2.s[1] +
                            C_naive.f2[i * N + j].s[1] * alpha.f2.s[0];
                    C_naive.f2[i * N + j] = val;
                }
                else {
                    for (k = 0; k < K; k++) {
                        float bkj = flags & TILEMUL_TRB ?
                                B.f[j * K + k] : B.f[k * N + j];
                        float aik = flags & TILEMUL_TRA ?
                                A.f[k * M + i] : A.f[i * K + k];
                        C_naive.f[i * N + j] += aik * bkj;
                    }
                    C_naive.f[i * N + j] *= alpha.f;
                }
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

    /* End of naive CPU multiplication */
    if (verbose) {
        if (!isDouble) {
            printf("Matrix A:\n");
            for (i = 0; i < M; i++) {
                for (k = 0; k < K; k++) {
                    if (isComplex) {
                        cl_float2 aik = flags & TILEMUL_TRA ?
                                A.f2[k * M + i] : A.f2[i * K + k];
                        printf("(%4.1f, %4.1f) ", aik.s[0], aik.s[1]);
                    }
                    else {
                        float aik = flags & TILEMUL_TRA ?
                                A.f[k * M + i] : A.f[i * K + k];
                        printf("%4.1f ", aik);
                    }
                }
                printf("\n");
            }

            printf("Matrix B:\n");
            for (k = 0; k < K; k++) {
                for (j = 0; j < N; j++) {
                    if (isComplex) {
                        cl_float2 bkj = flags & TILEMUL_TRB ?
                                B.f2[j * K + k] : B.f2[k * N + j];
                        printf("(%4.1f, %4.1f) ", bkj.s[0], bkj.s[1]);
                    }
                    else {
                        float bkj = flags & TILEMUL_TRB ?
                                B.f[j * K + k] : B.f[k * N + j];
                        printf("%4.1f ", bkj);
                    }
                }
                printf("\n");
            }

            printf("CPU calculated matrix:\n");
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    if (isComplex) {
                        printf("(%4.1f, %4.1f) ",
                                C_naive.f2[i * N + j].s[0],
                                C_naive.f2[i * N + j].s[1]);
                    }
                    else {
                        printf("%4.1f ", C_naive.f[i * N + j]);
                    }
                }
                printf("\n");
            }

            printf("GPU calculated matrix:\n");
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    if (isComplex) {
                        printf("(%4.1f, %4.1f) ",
                                C.f2[i * N + j].s[0], C.f2[i * N + j].s[1]);
                    }
                    else {
                        printf("%4.1f ", C.f[i * N + j]);
                    }
                }
                printf("\n");
            }
        }
    }

    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
            &sTime, NULL);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
            &fTime, NULL);

    printf("Total multiplication time: %d ms\nTime per iteration: %d ns\n",
            (int)((fTime-sTime)/1000000), (int)((fTime-sTime)/iterNum));

    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);
    clReleaseKernel(kernel);
    return CL_SUCCESS;
}

int main(int argc, char *argv[])
{
    char out[1024*1024];
    CLBLASKernExtra kextra;
    BlasGenSettings gset;
    TileMulOpts mulOpts;
    int i;
    cl_uint blockM = 4, blockN = 4, blockK = 8;
    struct KgenContext *ctx = createKgenContext(out, sizeof(out), 1);
    FType alpha;
    cl_int err;
    unsigned int iterNum = 1;
    const char* const shortOptions = "hd:f:l:t:a:b:s:g:i:c:ov";
    const struct option longOptions[] = {
            {"help", no_argument, NULL, 'h'},
            {"device", required_argument, NULL, 'd'},
            {"fetch", required_argument, NULL, 'f'},
            {"local", required_argument, NULL, 'l'},
            {"type", required_argument, NULL, 't'},
            {"a", required_argument, NULL, 'a'},
            {"b", required_argument, NULL, 'b'},
            {"skew", required_argument, NULL, 's'},
            {"globalcycling", required_argument, NULL, 'g'},
            {"iter", required_argument, NULL, 'i'},
            {"core", required_argument, NULL, 'c'},
            {"old", no_argument, NULL, 'o'},
            {"verbose", no_argument, NULL, 'v'},
            {NULL, 0, NULL, 0}
    };
    int nextOption;
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
    bool verbose = false;
    SubproblemDim *subdims = gset.subdims;
    bool separateFetch = false;

    memset(&gset, 0, sizeof(gset));
    memset(&mulOpts, 0, sizeof(mulOpts));
    memset(&kextra, 0, sizeof(kextra));
    gset.kextra = &kextra;
    gset.flags |= BGF_WHOLE_A;
    mulOpts.core = TILEMUL_MAD;
    mulOpts.flags = TILEMUL_FORCE_VECTORIZATION;
    kextra.vecLen = 1;
    kextra.dtype = TYPE_FLOAT;

    alpha.f = 1;

    // parse command line
    do {
        nextOption = getopt_long(argc, argv, shortOptions, longOptions, NULL);
        switch (nextOption) {
        case 'h':
            printUsage(argv[0], EXIT_SUCCESS);
            break;
        case 'd':
            if (!strcmp("cpu", optarg)) {
                deviceType = CL_DEVICE_TYPE_CPU;
            }
            else if (!strcmp("gpu", optarg)) {
                deviceType = CL_DEVICE_TYPE_GPU;
            }
            else {
                printf("Unknown device type %s. Supported values are \"cpu\" "
                        "and \"gpu\".\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 'f':
            kextra.vecLen = atoi(optarg);
            break;
        case 'l':
            if (!strcmp(optarg, "A")) {
                mulOpts.memA = CLMEM_LOCAL_MEMORY;
            }
            else if (!strcmp(optarg, "B")) {
                mulOpts.memB = CLMEM_LOCAL_MEMORY;
            }
            else {
                printf("Wrong matrix specified: %s. Supported values are "
                        "A, B.\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 't':
            if (!strcmp(optarg, "s")) {
                kextra.dtype = TYPE_FLOAT;
                alpha.f = 1;
            }
            else if (!strcmp(optarg, "d")) {
                kextra.dtype = TYPE_DOUBLE;
                alpha.d = 1;
            }
            else if (!strcmp(optarg, "c")) {
                kextra.dtype = TYPE_COMPLEX_FLOAT;
                alpha.f2.s[0] = 1;
                alpha.f2.s[1] = 0;
            }
            else if (!strcmp(optarg, "z")) {
                kextra.dtype = TYPE_COMPLEX_DOUBLE;
                alpha.d2.s[0] = 1;
                alpha.d2.s[1] = 0;
            }
            else {
                printf("Wrong type specified: %s. Supported values are "
                        "s, d, c, z.\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 'a':
            if (!strcmp(optarg, "r")) {
                mulOpts.flags &= ~TILEMUL_TRA;
            }
            else if (!strcmp(optarg, "c")) {
                mulOpts.flags |= TILEMUL_TRA;
            }
            else {
                printf("Wrong tile a parameter specified: %s. Supported values "
                        "are \"r\", \"c\".\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 'b':
            if (!strcmp(optarg, "r")) {
                mulOpts.flags &= ~TILEMUL_TRB;
            }
            else if (!strcmp(optarg, "c")) {
                mulOpts.flags |= TILEMUL_TRB;
            }
            else {
                printf("Wrong tile b order specified: %s. Supported values "
                        "are \"r\", \"c\".\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 's':
            if (!strcmp(optarg, "a")) {
                mulOpts.flags |= TILEMUL_SKEW_A;
            }
            else if (!strcmp(optarg, "b")) {
                mulOpts.flags |= TILEMUL_SKEW_B;
            }
            else if (!strcmp(optarg, "k")) {
                mulOpts.flags |= TILEMUL_SKEW_K;
            }
            else {
                printf("Wrong skew parameter specified: %s. Supported values "
                        "are \"a\", \"b\", \"k\"\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 'g':
            if (!strcmp(optarg, "a")) {
                mulOpts.flags |= TILEMUL_GLOBAL_CYCLIC_A;
            }
            else if (!strcmp(optarg, "b")) {
                mulOpts.flags |= TILEMUL_GLOBAL_CYCLIC_B;
            }
            else if (!strcmp(optarg, "k")) {
                mulOpts.flags |= TILEMUL_GLOBAL_CYCLIC_K;
            }
            else {
                printf("Wrong global cycling parameter specified: %s. "
                        "Supported values are \"a\", \"b\", \"k\"\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 'i':
            iterNum = atoi(optarg);
            break;
        case 'c':
            if (!strcmp("muladd", optarg)) {
                mulOpts.core = TILEMUL_MULADD;
            }
            else if (!strcmp("mad", optarg)) {
                mulOpts.core = TILEMUL_MAD;
            }
            else if (!strcmp("dot", optarg)) {
                mulOpts.core = TILEMUL_DOT;
            }
            else {
                printf("Unknown multiplier core %s. Supported values"
                        " are \"muladd\", \"mad\" and \"dot\".\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 'o':
            separateFetch = false;
            break;
        case 'v':
            verbose = true;
            break;
        case -1:
            break;
        default:
            printUsage(argv[0], EXIT_FAILURE);
            break;
        }
    } while (nextOption != -1);

    if (optind + 2 >= argc) {
        printf("Error: Not all sizes are specified\n");
        printUsage(argv[0], EXIT_FAILURE);
    }
    blockM = atoi(argv[optind]);
    blockN = atoi(argv[optind + 1]);
    blockK = atoi(argv[optind + 2]);

    if ((mulOpts.memA == CLMEM_LOCAL_MEMORY ||
            mulOpts.memB == CLMEM_LOCAL_MEMORY) &&
            ((mulOpts.flags & TILEMUL_GLOBAL_CYCLIC) != 0)) {
        printf("One of matrixes is in local memory, "
                "disabling global cycling\n");
        mulOpts.flags &= ~TILEMUL_GLOBAL_CYCLIC;
    }

    if (mulOpts.flags & TILEMUL_TRA) {
        kextra.flags |= KEXTRA_TRANS_A;
    }
    if (mulOpts.flags & TILEMUL_TRB) {
        kextra.flags |= KEXTRA_TRANS_B;
    }

    subdims[0].y = blockM * ITEM_WORK_M;
    subdims[0].x = blockN * ITEM_WORK_N;
    subdims[0].bwidth = blockK * ITEM_BLOCKS_K;
    subdims[1].y = blockM;
    subdims[1].x = blockN;
    subdims[1].bwidth = blockK;

    memset(out, 0, sizeof(out));

    i = isDoubleBasedType(kextra.dtype);
    kgenDeclareUptrs(ctx, i);
    genTest(ctx, &gset, &mulOpts, separateFetch);
    destroyKgenContext(ctx);

    printf("Kernel code: \n\"%s\"\n", out);
    err = run(out, subdims[0].y, subdims[0].x, subdims[0].bwidth, alpha,
              &gset, mulOpts.flags, deviceType, verbose, iterNum);
    if (err != CL_SUCCESS) {
        printf("Test run failed, error %d\n", err);
        return EXIT_FAILURE;
    }
	return EXIT_SUCCESS;
}
