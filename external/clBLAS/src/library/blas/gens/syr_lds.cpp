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

/*
 * SYR Generator
 */

#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <clblas_stddef.h>
#include <clBLAS.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>
#include "blas_kgen.h"

#include <kprintf.hpp>
#include <syr_her.clT>
#include <solution_seq.h>
//#define DEBUG_SYR

extern "C"
unsigned int dtypeSize(DataType type);


static char Prefix[4];

static SolverFlags
solverFlags(void)
{
	#ifdef DEBUG_SYR
	printf("solverFlags called......\n");
	#endif

    return (SolverFlags)(SF_WSPACE_1D | SF_TOP_INPUT_SQUARE_BLOCKS);
}

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra);

static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra);


static void
assignKargs(KernelArg *args, const void *params, const void*);

extern "C"
void initSyrDefaultPattern(MemoryPattern *mempat);

static  KernelExtraFlags
selectVectorization(
    void *kargs,
    unsigned int vlen );

static void
setBuildOpts(
    char * buildOptStr,
    const void *kArgs);

static bool
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs);

static SolverOps syrOps = {
    generator,
    assignKargs,
    isFitToLDS,
    NULL, // Prepare Translate Dims
    NULL, // Inner Decomposition Axis
    calcNrThreads,
    NULL,
    solverFlags,
	NULL,
	NULL,
	NULL,
	setBuildOpts,
	selectVectorization
};

static  KernelExtraFlags
selectVectorization(
	void *args,
	unsigned int vlen )
{
	KernelExtraFlags kflags = KEXTRA_NO_FLAGS;
	CLBlasKargs *kargs  = (CLBlasKargs *)args;

	if(kargs->uplo == clblasUpper)
	{
		if( (kargs->N) % vlen)
        {
			kflags = KEXTRA_NO_COPY_VEC_A;
		}
	}

    if( kargs->pigFuncID == CLBLAS_SPR )
    {
        kflags = KEXTRA_NO_COPY_VEC_A;      // Packed-case never do aligned access
    }

#ifdef DEBUG_SYR
	printf("SYR: selectVectorization being called\n");
#endif
	return kflags;
}


static void
setBuildOpts(
    char * buildOptStr,
    const void *args)
{
	const SolutionStep *step = (const SolutionStep *)args;
    const CLBlasKargs *kargs = (const CLBlasKargs *)(&step->args);
	if ( kargs->dtype == TYPE_DOUBLE )
	{
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DDOUBLE_PRECISION");
		#ifdef DEBUG_SYR
		printf("Setting build options ... Double... for DOUBLE PRECISION support\n");
		#endif
	}
    if( kargs->pigFuncID == CLBLAS_SPR )
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DPACKED");
    }

	return;
}

static CLBLASMpatExtra mpatExtra;

extern "C"
void initSyrDefaultPattern(MemoryPattern *mempat)
{
	#ifdef DEBUG_SYR
	printf("initSyrDefaultPattern called with mempat = 0x%p\n", (void *)mempat);
	fflush(stdout);
	#endif

    mempat->name = "LDS based syr";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &syrOps;

    mpatExtra.aMset = 0;
    mpatExtra.bMset = CLMEM_LEVEL_LDS; // For "x" vector
    mpatExtra.mobjA = CLMEM_GLOBAL_MEMORY;
    mpatExtra.mobjB = CLMEM_GLOBAL_MEMORY;
    mempat->extra = &mpatExtra;

	Prefix[TYPE_FLOAT] = 'S';
	Prefix[TYPE_DOUBLE] = 'D';
}

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *_extra)
{
	int BLOCKSIZE = pgran->wgSize[0]; // 1D Block
	#ifdef DEBUG_SYR
	printf("calcNrThreads called from syr_reg.cpp\n");
	#endif

    const CLBlasKargs *kargs = (const CLBlasKargs *)args;
	const CLBLASKernExtra *extra = ( CLBLASKernExtra *)_extra;

	clblasOrder order = ( extra->flags & KEXTRA_COLUMN_MAJOR) ? clblasColumnMajor: clblasRowMajor;

    if ( order == clblasRowMajor )
    {
        order = clblasColumnMajor;
    }
	#ifdef DEBUG_SYR
	printf("subdims->y : %d, subdims->x : %d\n", (int)subdims->y, (int)subdims->x);
	#endif
	size_t TARGETROWS = subdims->y ;

	#ifdef DEBUG_SYR
	printf("kargs-> N : %d, TARGETROWS: %d\n", (int)kargs->N, TARGETROWS);
	#endif

	size_t blocks = ((kargs->N - 1)/ TARGETROWS) + 1;
	#ifdef DEBUG_SYR
	printf("blocks : %d\n", blocks);
	#endif

	threads[0] = ((blocks * (blocks + 1)) / 2) * BLOCKSIZE;
	#ifdef DEBUG_SYR
	printf("pgran-wgSize[0] : %d, globalthreads[0] : %d\n", (int)pgran->wgSize[0], (int)threads[0]);
	#endif
	threads[1] = 1;
}


//
// FIXME: Report correct return value - Needs change in KPRINTF
//
static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra)
{
	int BLOCKSIZE  = pgran->wgSize[0];
	char tempTemplate[32*1024];
	char targetRows[10], blockSize[10];

	if ( buf == NULL) // return buffer size
	{
		buflen = (64 * 1024 * sizeof(char));
		return (ssize_t)buflen;
	}
	CLBLASKernExtra *extraFlags = ( CLBLASKernExtra *)extra;

	#ifdef DEBUG_SYR
 	printf("SYR GENERATOR called....\n");
	#endif

	clblasUplo uplo   = ( extraFlags->flags & KEXTRA_UPPER_TRIANG) ? clblasUpper : clblasLower;
	clblasOrder order = ( extraFlags->flags & KEXTRA_COLUMN_MAJOR) ? clblasColumnMajor: clblasRowMajor;


	if ((subdims->y % extraFlags->vecLenA) != 0)
	{
		printf("WARNING: SYR: generator: TARGETROWS must be divisible by Vector Length\n");
		return 0;
	}

	size_t TARGETROWS = 0;
	if(order == clblasColumnMajor)
	{
		( uplo == clblasLower )?
			     (strcpy(tempTemplate, (char*)syr_her_CL_kernel)) : (strcpy(tempTemplate, (char*)syr_her_CU_kernel));
	}
	else
	{
		printf("WARNING: SYR: Rowmajor order is implemented in columnMajor. This part should never get executed.\n");
		return 0;
	}

	TARGETROWS = subdims->y;
	if ((BLOCKSIZE % TARGETROWS) != 0)
	{
		printf("WARNING: SYR: generator: Invalid Block Size\n");
		return 0;
	}

	#ifdef DEBUG_SYR
	printf("dataType : %c\n", Prefix[extraFlags->dtype]);
	#endif

	// FIXME: VECTORSIZE HARD CODED
	// FIXME : SetKernelArgs.. sends offa, offx, and lda should be received as uint
    unsigned int vecLenA = extraFlags->vecLenA;

	#ifdef DEBUG_SYR
	printf("Vector length used : %d\n\n", vecLenA);
	#endif

	bool doVLOAD = false;
	if( extraFlags->flags &  KEXTRA_NO_COPY_VEC_A )
	{
		doVLOAD = true;
		#ifdef DEBUG_SYR
			printf("DOing VLOAD as Aligned Data Pointer not Availabe\n");
		#endif
	}
	else
	{
		#ifdef DEBUG_SYR
			printf("Using Aligned Data Pointer .........................\n");
		#endif
	}
	kprintf kobj( Prefix[extraFlags->dtype], vecLenA, doVLOAD, doVLOAD);

	sprintf( targetRows, "%" SPREFIX "u", TARGETROWS );
	sprintf( blockSize, "%d", BLOCKSIZE );

	#ifdef DEBUG_SYR
    printf("TARGET ROWS = %s\n", targetRows);
    printf("BLOCK SIZE = %s\n", blockSize);
	#endif

    kobj.put("%TARGET_ROWS", (const char *)targetRows);
    kobj.put("%BLOCKSIZE", (const char *) blockSize);
    kobj.spit((char*)buf, tempTemplate);

	return (64 * 1024 * sizeof(char));
    // return 0;//(ret < 0) ? -EOVERFLOW : ret;
}

/*
	 __global %TYPE* _A, __global const %TYPE* _X, int N, int offx, int incx, int offa, int lda, %PTYPE alpha
*/
static void
assignKargs(KernelArg *args, const void *params, const void*)
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;
    cl_int inc;

    INIT_KARG(&args[0], blasArgs->A); 	//A - input/output matrix - argument
    INIT_KARG(&args[1], blasArgs->B); 	//x - x vector
    initSizeKarg(&args[2], blasArgs->N);
	initSizeKarg(&args[3], blasArgs->offBX);
    inc = blasArgs->ldb.vector;
    INIT_KARG(&args[4], inc);
    initSizeKarg(&args[5], blasArgs->offA);
	initSizeKarg(&args[6], blasArgs->lda.matrix);
    assignScalarKarg(&args[7], &(blasArgs->alpha), blasArgs->dtype);
	return;
}

static bool
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs)
{
    cl_ulong maxSize;
    CLBlasKargs *blasArgs;

	blasArgs = (CLBlasKargs *)kernelArgs;

	// 2 buffers for xShared and yShared and 2 integers for the values of iShared and jShared.

	maxSize = (dim->y * 2 * sizeof(dtype)) + (2 * sizeof(int));

    return ((maxSize) <= ldsSize);

}
//#undef DEBUG_SYR
