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
 * amax generator
 */
//#define DEBUG_AMAX

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
#include <iamax.clT>
#include <solution_seq.h>

extern "C"
unsigned int dtypeSize(DataType type);


static char Prefix[4];

static SolverFlags
solverFlags(void)
{
	#ifdef DEBUG_AMAX
	printf("solverFlags called...\n");
	#endif

    return (SF_WSPACE_1D);
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
    fixupArgs(void *args, SubproblemDim *subdims, void *extra);

static void
assignKargs(KernelArg *args, const void *params, const void* extra );

extern "C"
void initAmaxRegisterPattern(MemoryPattern *mempat);

static  KernelExtraFlags
selectVectorization(
    void *kargs,
    unsigned int vlen );

static void
setBuildOpts(
    char * buildOptStr,
    const void *kArgs);

static SolverOps amaxOps = {
    generator,
    assignKargs,
    NULL,
    NULL, // Prepare Translate Dims
    NULL, // Inner Decomposition Axis
    calcNrThreads,
    NULL,
    solverFlags,
	fixupArgs,
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

    if( (((kargs->offa) % vlen) != 0))
    {
        kflags = KEXTRA_NO_COPY_VEC_A;
    }
	return kflags;
}

static void
setBuildOpts(
    char * buildOptStr,
    const void *args)
{
	const SolutionStep *step = (const SolutionStep *)args;
    const CLBlasKargs *kargs = (const CLBlasKargs *)(&step->args);
	if ( kargs->dtype == TYPE_DOUBLE || kargs->dtype == TYPE_COMPLEX_DOUBLE)
	{
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DDOUBLE_PRECISION");
		#ifdef DEBUG_AMAX
		printf("Setting build options ... Double... for DOUBLE PRECISION support\n");
		#endif
    }

    if( (kargs->ldb.vector) != 1)
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DINCX_NONUNITY");
    }

    if( (kargs->ldb.vector) < 1)
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DRETURN_ON_INVALID");
    }

    if( (kargs->redctnType == REDUCE_MAX_WITH_INDEX_ATOMICS))
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DREDUCE_MAX_WITH_INDEX_ATOMICS");
    }

	return;
}


static CLBLASMpatExtra mpatExtra;

extern "C"
void initiAmaxRegisterPattern(MemoryPattern *mempat)
{
	#ifdef DEBUG_AMAX
	printf("initRegPattern called with mempat = 0x%p\n", mempat);
	#endif

	fflush(stdout);
    mempat->name = "Register AMAX";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &amaxOps;

    mpatExtra.aMset = CLMEM_LEVEL_L2;
    mpatExtra.bMset = CLMEM_LEVEL_L2;
    mpatExtra.mobjA = CLMEM_GLOBAL_MEMORY;
    mpatExtra.mobjB = CLMEM_GLOBAL_MEMORY;
    mempat->extra = &mpatExtra;

	Prefix[TYPE_FLOAT] = 'S';
	Prefix[TYPE_DOUBLE] = 'D';
	Prefix[TYPE_COMPLEX_FLOAT] = 'C';
	Prefix[TYPE_COMPLEX_DOUBLE] = 'Z';
}

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *_extra)
{
    DUMMY_ARG_USAGE(subdims);
	int BLOCKSIZE = pgran->wgSize[0] * pgran->wgSize[1]; // 1D Block
    const CLBLASKernExtra *extra = ( CLBLASKernExtra *)_extra;
    unsigned int VEC_LEN = extra->vecLenA;
	#ifdef DEBUG_AMAX
	printf("calcNrThreads called from amax.cpp\n");
	#endif

    const CLBlasKargs *kargs = (CLBlasKargs *)args;

	size_t blocks = ((kargs->N - 1)/ (BLOCKSIZE*VEC_LEN)) + 1;

	#ifdef DEBUG_AMAX
	printf("blocks : %d\n", blocks);
	#endif

	threads[0] = blocks * BLOCKSIZE;
	#ifdef DEBUG_AMAX
	printf("pgran-wgSize[0] : %d, globalthreads[0]  : %d\n", pgran->wgSize[0], threads[0]);
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

	DUMMY_ARG_USAGE(subdims);
	size_t BLOCKSIZE  = pgran->wgSize[0];
	char tempTemplate[32*1024];

	if ( buf == NULL) // return buffer size
	{
		buflen = (32 * 1024 * sizeof(char));
        return (ssize_t)buflen;
	}
	CLBLASKernExtra *extraFlags = ( CLBLASKernExtra *)extra;

	#ifdef DEBUG_AMAX
 	printf("AMAX GENERATOR called....\n");
	printf("dataType : %c\n", Prefix[extraFlags->dtype]);
	#endif

    unsigned int vecLenA = extraFlags->vecLenA;

	#ifdef DEBUG_AMAX
	printf("Vector length used : %d\n\n", vecLenA);
	#endif

	bool doVLOAD = false;
	if( extraFlags->flags &  KEXTRA_NO_COPY_VEC_A )
	{
		doVLOAD = true;
		#ifdef DEBUG_AMAX
		printf("DOing VLOAD as Aligned Data Pointer not Availabe\n");
		#endif
	}
	else
	{
		#ifdef DEBUG_AMAX
		printf("Using Aligned Data Pointer .........................\n");
		#endif
	}
    strcpy( tempTemplate, (char*)iamax_kernel );
	kprintf kobj( Prefix[extraFlags->dtype], vecLenA, doVLOAD, doVLOAD, BLOCKSIZE);
    kobj.spit((char*)buf, tempTemplate);

    return (32 * 1024 * sizeof(char));
}

/*
__kernel void %PREFIXiamax_kernel( __global %TYPE *_X, __global %TYPE _scratchBuf, __global %TYPE *_iMax,
                                        uint N, uint offx, int incx, uint offiMax )
*/
static void
assignKargs(KernelArg *args, const void *params, const void* )
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;
	cl_int incx;

    INIT_KARG(&args[0], blasArgs->B);
	INIT_KARG(&args[1], blasArgs->D);
    initSizeKarg(&args[2], blasArgs->N);
    initSizeKarg(&args[3], blasArgs->offb);
    incx = blasArgs->ldb.vector;
    INIT_KARG(&args[4], incx);

	return;
}

/** The purpose of this function is to add an work-group size indicator in
    kernelKey, so that a different kernel is generated when work-group size is changed.
    Reduction loop is unrolled in kprintf based on work-group size.

    Member of SubproblemDim- bwidth, will be used to store work-group size of the current kernel
    this will become a kernelKey, and kernel cache will be accordingly managed.
    Note -- SubproblemDim is a member of kernelKey
**/
static void
fixupArgs(void *args, SubproblemDim *subdims, void *extra)
{
    DUMMY_ARG_USAGE(extra);
    CLBlasKargs *kargs = (CLBlasKargs*)args;
    SolutionStep *step = container_of(kargs, args, SolutionStep);

    subdims->bwidth = (step->pgran.wgSize[0]) * (step->pgran.wgSize[1]);
}

