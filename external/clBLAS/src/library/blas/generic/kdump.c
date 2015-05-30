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


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cltypes.h>
#include <clblas-internal.h>

#include "solution_seq.h"

#ifdef DUMP_CLBLAS_KERNELS

enum {
    SRC_BUFSIZE = 512244
};

static void
getFuncName(char *name, BlasFunctionID funcID, DataType dtype)
{
    switch (funcID) {
    case CLBLAS_GEMV:
        strcpy(name + 1, "GEMV");
        break;
    case CLBLAS_SYMV:
        strcpy(name + 1, "SYMV");
        break;
    case CLBLAS_GEMM:
        strcpy(name + 1, "GEMM");
        break;
    case CLBLAS_TRMM:
        strcpy(name + 1, "TRMM");
        break;
    case CLBLAS_TRSM:
        strcpy(name + 1, "TRSM");
        break;
    case CLBLAS_SYRK:
        strcpy(name + 1, "SYRK");
        break;
    case CLBLAS_SYR2K:
        strcpy(name + 1, "SYR2K");
        break;
    default:
        break;
    }

    if (dtype == TYPE_FLOAT) {
        name[0] = 's';
    }
    else {
        name[0] = dtypeToPrefix(dtype);
    }
}

static void
addTranspSuffix(char *buf, clblasTranspose flag)
{
    const char *s;

    if (flag == clblasNoTrans) {
        return;
    }

    s = (clblasTrans) ? "t" : "tc";
    strcat(buf, s);
}

static void
fileNameFromSolution(
    char *name,
    BlasFunctionID funcID,
    const SolutionStep *step)
{
    const char *s;
    const CLBlasKargs *kargs = (const CLBlasKargs*)&step->args;
    bool isTriangFn;

    isTriangFn = (funcID == CLBLAS_TRMM || funcID == CLBLAS_TRSM);
    strcpy(name, "./");
    name += strlen(name);
    getFuncName(name, funcID, kargs->dtype);
    s = (kargs->order == clblasRowMajor) ? "_row_" : "_col_";
    strcat(name, s);
    addTranspSuffix(name, kargs->transA);
    if (isTriangFn) {
        s = (kargs->uplo == clblasUpper) ? "_upper" : "_lower";
        strcat(name, s);
        s = (kargs->side == clblasRight) ? "_right" : "_left";
        strcat(name, s);
    }
    else {
        addTranspSuffix(name, kargs->transB);
    }

    name += strlen(name);
    sprintf(name, "_%lu_%lu", kargs->M, kargs->N);
    if (!isTriangFn) {
        name += strlen(name);
        sprintf(name, "_%lu", kargs->K);
    }
    strcat(name, ".kdump");
}

void
dumpKernel(
    const SolutionStep *step,
    CLBlasKernelType ktype)
{
    FILE *file;
    char tmp[1024];
    MemoryPattern *pattern;
    const char *s;
    const CLBlasKargs *kargs = (const CLBlasKargs*)&step->args;
    char *srcBuf;
    unsigned int i;

    fileNameFromSolution(tmp, step->funcID, step);
    file = fopen((const char*)tmp, "a+");
    pattern = &clblasSolvers[step->funcID].memPatterns[step->patternID];

    // now, dump the info
    sprintf(tmp, "offset M = %lu, offset N = %lu, offset A = %lu,"
                 "offset BX = %lu, offset CY = %lu\n",
            kargs->offsetM, kargs->offsetN, kargs->offA, kargs->offBX,
            kargs->offCY);
    fputs(tmp, file);

    sprintf(tmp, "Memory pattern = %s\n", pattern->name);
    fputs(tmp, file);

    s = kernelTypeString(ktype);
    sprintf(tmp, "Kernel type = %s\n", s);
    fputs(tmp, file);

    // data parallelism granularity
    if (step->pgran.wgDim == 1) {
        sprintf(tmp, "work group size = %u\n", step->pgran.wgSize[0]);
    }
    else {
        sprintf(tmp, "work group size = %u x %u\n", step->pgran.wgSize[0],
                step->pgran.wgSize[1]);
    }
    fputs(tmp, file);

    fputs("Problem granulation\n", file);
    for (i = 0; i < pattern->nrLevels; i++) {
        sprintf(tmp, "[%u]: ", i);
        fputs(tmp, file);
        sprintfGranulation(tmp, step->subdims, i);
        fputs(tmp, file);
        fputs("\n", file);
    }

    srcBuf = malloc(SRC_BUFSIZE);
    if (srcBuf != NULL) {
        clGetProgramInfo(step->kernels[ktype]->program,
                         CL_PROGRAM_SOURCE, SRC_BUFSIZE, srcBuf, NULL);
        fputs("Kernel source:\n\n", file);
        fputs(srcBuf, file);
    }
    else {
        fputs("Kernel source: not available\n", file);
    }
    free(srcBuf);

    fputs("--------------------------------------------------------------"
          "------------------------------------------------------------\n",
          file);

    fclose(file);
}

#endif      /* DUMP_CLBLAS_KERNELS */
