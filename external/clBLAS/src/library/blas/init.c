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


#include <clBLAS.h>
#include <toolslib.h>
#include <kern_cache.h>
#include <clBLAS.version.h>
#include <trace_malloc.h>

#include "clblas-internal.h"
#include <events.h>
#include <stdlib.h>
#include <stdio.h>

clblasStatus
clblasGetVersion(cl_uint* major, cl_uint* minor, cl_uint* patch)
{
    *major = clblasVersionMajor;
    *minor = clblasVersionMinor;
    *patch = clblasVersionPatch;

    return clblasSuccess;
}

clblasStatus
clblasSetup(void)
{
    solver_id_t sidsNum;
	char* tmp			= NULL;

	//	Made the cache unlimited by default
	size_t kCacheLimit = 0;

    if (clblasInitialized) {
        return clblasSuccess;
    }

    // printf("\n%s, line %d\n", __func__, __LINE__);
    initMallocTrace();

    clblasSolvers[CLBLAS_GEMM].nrPatterns =
        initGemmMemPatterns(clblasSolvers[CLBLAS_GEMM].memPatterns);
    clblasSolvers[CLBLAS_GEMM].defaultPattern = -1;

    clblasSolvers[CLBLAS_TRMM].nrPatterns =
        initTrmmMemPatterns(clblasSolvers[CLBLAS_TRMM].memPatterns);
    clblasSolvers[CLBLAS_TRMM].defaultPattern = -1;

    clblasSolvers[CLBLAS_TRSM].nrPatterns =
        initTrsmMemPatterns(clblasSolvers[CLBLAS_TRSM].memPatterns);
    clblasSolvers[CLBLAS_TRSM].defaultPattern = -1;

    clblasSolvers[CLBLAS_GEMV].nrPatterns =
        initGemvMemPatterns(clblasSolvers[CLBLAS_GEMV].memPatterns);
    clblasSolvers[CLBLAS_GEMV].defaultPattern = -1;

    clblasSolvers[CLBLAS_SYMV].nrPatterns =
        initSymvMemPatterns(clblasSolvers[CLBLAS_SYMV].memPatterns);
    clblasSolvers[CLBLAS_SYMV].defaultPattern = -1;

    clblasSolvers[CLBLAS_SYR2K].nrPatterns =
        initSyr2kMemPatterns(clblasSolvers[CLBLAS_SYR2K].memPatterns);
    clblasSolvers[CLBLAS_SYR2K].defaultPattern = -1;

    clblasSolvers[CLBLAS_SYRK].nrPatterns =
        initSyrkMemPatterns(clblasSolvers[CLBLAS_SYRK].memPatterns);
    clblasSolvers[CLBLAS_SYRK].defaultPattern = -1;

	clblasSolvers[CLBLAS_TRMV].nrPatterns =
		initTrmvMemPatterns(clblasSolvers[CLBLAS_TRMV].memPatterns);
	clblasSolvers[CLBLAS_TRMV].defaultPattern = -1;

	// HEMV uses the same memory pattern as TRMV.
	clblasSolvers[CLBLAS_HEMV].nrPatterns =
		initTrmvMemPatterns(clblasSolvers[CLBLAS_HEMV].memPatterns);
	clblasSolvers[CLBLAS_HEMV].defaultPattern = -1;

	clblasSolvers[CLBLAS_TRSV].nrPatterns =
		initTrsvMemPatterns(clblasSolvers[CLBLAS_TRSV].memPatterns);
	clblasSolvers[CLBLAS_TRSV].defaultPattern = -1;

	clblasSolvers[CLBLAS_TRSV_GEMV].nrPatterns =
		initTrsvGemvMemPatterns(clblasSolvers[CLBLAS_TRSV_GEMV].memPatterns);
	clblasSolvers[CLBLAS_TRSV_GEMV].defaultPattern = -1;

	clblasSolvers[CLBLAS_SYMM].nrPatterns =
		initSymmMemPatterns(clblasSolvers[CLBLAS_SYMM].memPatterns);
	clblasSolvers[CLBLAS_SYMM].defaultPattern = -1;

	clblasSolvers[CLBLAS_GEMM2].nrPatterns =
		initGemmV2MemPatterns(clblasSolvers[CLBLAS_GEMM2].memPatterns);
	clblasSolvers[CLBLAS_GEMM2].defaultPattern = -1;

	clblasSolvers[CLBLAS_GEMM_TAIL].nrPatterns =
		initGemmV2TailMemPatterns(clblasSolvers[CLBLAS_GEMM_TAIL].memPatterns);
	clblasSolvers[CLBLAS_GEMM_TAIL].defaultPattern = -1;

	clblasSolvers[CLBLAS_SYR].nrPatterns =
        initSyrMemPatterns(clblasSolvers[CLBLAS_SYR].memPatterns);
 	clblasSolvers[CLBLAS_SYR].defaultPattern = -1;

	clblasSolvers[CLBLAS_SYR2].nrPatterns =
        initSyr2MemPatterns(clblasSolvers[CLBLAS_SYR2].memPatterns);
    clblasSolvers[CLBLAS_SYR2].defaultPattern = -1;

	clblasSolvers[CLBLAS_GER].nrPatterns =
		initGerMemPatterns(clblasSolvers[CLBLAS_GER].memPatterns);
	clblasSolvers[CLBLAS_GER].defaultPattern = -1;

	clblasSolvers[CLBLAS_HER].nrPatterns =
        initHerMemPatterns(clblasSolvers[CLBLAS_HER].memPatterns);
 	clblasSolvers[CLBLAS_HER].defaultPattern = -1;

	clblasSolvers[CLBLAS_HER2].nrPatterns =
        initHer2MemPatterns(clblasSolvers[CLBLAS_HER2].memPatterns);
    clblasSolvers[CLBLAS_HER2].defaultPattern = -1;

    clblasSolvers[CLBLAS_GBMV].nrPatterns =
		initGbmvMemPatterns(clblasSolvers[CLBLAS_GBMV].memPatterns);
	clblasSolvers[CLBLAS_GBMV].defaultPattern = -1;

	clblasSolvers[CLBLAS_SWAP].nrPatterns =
        initSwapMemPatterns(clblasSolvers[CLBLAS_SWAP].memPatterns);
    clblasSolvers[CLBLAS_SWAP].defaultPattern = -1;

    clblasSolvers[CLBLAS_SCAL].nrPatterns =
        initScalMemPatterns(clblasSolvers[CLBLAS_SCAL].memPatterns);
    clblasSolvers[CLBLAS_SCAL].defaultPattern = -1;

    clblasSolvers[CLBLAS_COPY].nrPatterns =
        initCopyMemPatterns(clblasSolvers[CLBLAS_COPY].memPatterns);
    clblasSolvers[CLBLAS_COPY].defaultPattern = -1;

     clblasSolvers[CLBLAS_AXPY].nrPatterns =
        initAxpyMemPatterns(clblasSolvers[CLBLAS_AXPY].memPatterns);
    clblasSolvers[CLBLAS_AXPY].defaultPattern = -1;

    clblasSolvers[CLBLAS_DOT].nrPatterns =
       initDotMemPatterns(clblasSolvers[CLBLAS_DOT].memPatterns);
    clblasSolvers[CLBLAS_DOT].defaultPattern = -1;

    clblasSolvers[CLBLAS_REDUCTION_EPILOGUE].nrPatterns =
       initReductionMemPatterns(clblasSolvers[CLBLAS_REDUCTION_EPILOGUE].memPatterns);
    clblasSolvers[CLBLAS_REDUCTION_EPILOGUE].defaultPattern = -1;

    clblasSolvers[CLBLAS_ROTG].nrPatterns =
       initRotgMemPatterns(clblasSolvers[CLBLAS_ROTG].memPatterns);
    clblasSolvers[CLBLAS_ROTG].defaultPattern = -1;

    clblasSolvers[CLBLAS_ROTMG].nrPatterns =
       initRotmgMemPatterns(clblasSolvers[CLBLAS_ROTMG].memPatterns);
    clblasSolvers[CLBLAS_ROTMG].defaultPattern = -1;

    clblasSolvers[CLBLAS_ROTM].nrPatterns =
       initRotmMemPatterns(clblasSolvers[CLBLAS_ROTM].memPatterns);
    clblasSolvers[CLBLAS_ROTM].defaultPattern = -1;

    clblasSolvers[CLBLAS_iAMAX].nrPatterns =
       initiAmaxMemPatterns(clblasSolvers[CLBLAS_iAMAX].memPatterns);
    clblasSolvers[CLBLAS_iAMAX].defaultPattern = -1;

    clblasSolvers[CLBLAS_NRM2].nrPatterns =
       initNrm2MemPatterns(clblasSolvers[CLBLAS_NRM2].memPatterns);
    clblasSolvers[CLBLAS_NRM2].defaultPattern = -1;

    clblasSolvers[CLBLAS_ASUM].nrPatterns =
       initAsumMemPatterns(clblasSolvers[CLBLAS_ASUM].memPatterns);
    clblasSolvers[CLBLAS_ASUM].defaultPattern = -1;

    sidsNum = makeSolverID(BLAS_FUNCTIONS_NUMBER, 0);

	//	Read environmental variable to limit or disable ( 0 ) the size of the kernel cache in memory
	tmp = getenv( "AMD_CLBLAS_KCACHE_LIMIT_MB" );
	if( tmp != NULL )
	{
		kCacheLimit = atol( tmp );
#if defined( _WIN32 )
		printf( "Kernel Cache limit: %Iu MB\n", kCacheLimit );
#else
		printf( "Kernel Cache limit: %zu MB\n", kCacheLimit );
#endif
		kCacheLimit *= (1024 * 1024);
	}

    if (kCacheLimit || (tmp == NULL)) {
        clblasKernelCache = createKernelCache(sidsNum, kCacheLimit);
    	if (clblasKernelCache == NULL) {
        	return clblasOutOfHostMemory;
        }
    }
    if (initSCImages()) {
        destroyKernelCache(clblasKernelCache);
        return clblasOutOfHostMemory;
    }

    decomposeEventsSetup();

    initStorageCache();

    clblasInitialized = 1;
    return clblasSuccess;
}

void
clblasTeardown(void)
{
    if (!clblasInitialized) {
        return;
    }

    printMallocStatistics();

    if (clblasKernelCache != NULL) {
        printKernelCacheSize(clblasKernelCache);
        destroyKernelCache(clblasKernelCache);
        clblasKernelCache = NULL;
    }
    releaseSCImages();
    decomposeEventsTeardown();

    // win32 - crashes
    destroyStorageCache();

    printMemLeaksInfo();
    releaseMallocTrace();

    clblasInitialized = 0;
}
