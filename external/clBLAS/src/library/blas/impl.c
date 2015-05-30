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
#include <defbool.h>
#include <clBLAS.h>
#include <clblas-internal.h>

clblasStatus
clblasSelectImplementation(
    clblasImplementation impl)
{
    switch (impl) {
    case clblasDefaultGemm:
    case clblasLdsBlockGemm:
    case clblasImageBlockGemm:
    case clblasBlockGemmWithCaching:
        clblasSolvers[CLBLAS_GEMM].defaultPattern =
            getGemmMemPatternIndex(impl);
        break;
    case clblasDefaultTrmm:
    case clblasLdsBlockTrmm:
    case clblasImageBlockTrmm:
    case clblasBlockTrmmWithCaching:
        clblasSolvers[CLBLAS_TRMM].defaultPattern =
            getTrmmMemPatternIndex(impl);
        break;
    case clblasDefaultTrsm:
    case clblasLdsBlockTrsm:
    case clblasImageBlockTrsm:
    case clblasBlockTrsmWithCaching:
    case clblasBlockTrsmWithoutLds:
        clblasSolvers[CLBLAS_TRSM].defaultPattern =
            getTrsmMemPatternIndex(impl);
        break;
    default:
        return clblasInvalidValue;
    }

    return clblasSuccess;
}

int
scratchImagesEnabled(void)
{
    int enable = 0;
    const char *envImpl;

    envImpl = getenv("AMD_CLBLAS_GEMM_IMPLEMENTATION");
    if ((envImpl != NULL) && (strcmp(envImpl, "1") == 0)) {
        enable = 1;
    };
    envImpl = getenv("AMD_CLBLAS_TRMM_IMPLEMENTATION");
    if ((envImpl != NULL) && (strcmp(envImpl, "1") == 0)) {
        enable = 1;
    };
    envImpl = getenv("AMD_CLBLAS_TRSM_IMPLEMENTATION");
    if ((envImpl != NULL) && (strcmp(envImpl, "1") == 0)) {
        enable = 1;
    };

    return enable;
}

void
parseEnvImplementation(void)
{
    const char *envImpl;

    envImpl = getenv("AMD_CLBLAS_GEMM_IMPLEMENTATION");
    clblasSelectImplementation(clblasDefaultGemm);
    if (envImpl != NULL) {
        if (strcmp(envImpl, "0") == 0) {
            clblasSelectImplementation(clblasLdsBlockGemm);
        }
        else if (strcmp(envImpl, "1") == 0) {
            clblasSelectImplementation(clblasImageBlockGemm);
        }
        else if (strcmp(envImpl, "2") == 0) {
            clblasSelectImplementation(clblasBlockGemmWithCaching);
        }
    }

    envImpl = getenv("AMD_CLBLAS_TRMM_IMPLEMENTATION");
    clblasSelectImplementation(clblasDefaultTrmm);
    if (envImpl != NULL) {
        if (strcmp(envImpl, "0") == 0) {
            clblasSelectImplementation(clblasLdsBlockTrmm);
        }
        else if (strcmp(envImpl, "1") == 0) {
            clblasSelectImplementation(clblasImageBlockTrmm);
        }
        else if (strcmp(envImpl, "2") == 0) {
            clblasSelectImplementation(clblasBlockTrmmWithCaching);
        }
    }

    envImpl = getenv("AMD_CLBLAS_TRSM_IMPLEMENTATION");
    clblasSelectImplementation(clblasDefaultTrsm);
    if (envImpl != NULL) {
        if (strcmp(envImpl, "0") == 0) {
            clblasSelectImplementation(clblasLdsBlockTrsm);
        }
        else if (strcmp(envImpl, "1") == 0) {
            clblasSelectImplementation(clblasImageBlockTrsm);
        }
        else if (strcmp(envImpl, "2") == 0) {
            clblasSelectImplementation(clblasBlockTrsmWithoutLds);
        }
        else if (strcmp(envImpl, "3") == 0) {
            clblasSelectImplementation(clblasBlockTrsmWithCaching);
        }
    }
}
