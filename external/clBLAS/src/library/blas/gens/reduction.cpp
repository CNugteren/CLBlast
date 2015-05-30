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
 * reduction generator
 */
//#define DEBUG_REDUCTION

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
#include <reduction.clT>
#include <solution_seq.h>

extern "C"
unsigned int dtypeSize(DataType type);


static char Prefix[4];

static SolverFlags
solverFlags(void)
{
	#ifdef DEBUG_REDUCTION
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
void initReductionRegisterPattern(MemoryPattern *mempat);

static  KernelExtraFlags
selectVectorization(
    void *kargs,
    unsigned int vlen );

static void
setBuildOpts(
    char * buildOptStr,
    const void *kArgs);

static SolverOps reductionOps = {
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

    if( (((kargs->offBX) % vlen) != 0) || (((kargs->offCY) % vlen) != 0) )
    {
        kflags = KEXTRA_NO_COPY_VEC_A;
    }
    // Since ssq will be vector-loaded from Nth location of scratch buffer i.e scratchBuff[N]
    // If N is not a multiple of vlen, then use vload
    if( (kargs->redctnType == REDUCE_BY_SSQ) && (((kargs->N) % vlen) != 0) )
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
	}
    switch(kargs->redctnType)
    {
        case REDUCE_BY_SUM:                 addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DREDUCE_BY_SUM");
                                            break;

        case REDUCE_BY_MAX:                 addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DREDUCE_BY_MAX");
                                            break;

        case REDUCE_BY_MIN:                 addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DREDUCE_BY_MIN");
                                            break;

        case REDUCE_MAX_WITH_INDEX:         addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DREDUCE_MAX_WITH_INDEX");
                                            break;

        case REDUCE_BY_HYPOT:               addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DREDUCE_BY_HYPOT");
                                            break;

        case REDUCE_BY_SSQ:                 addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DREDUCE_BY_SSQ");
                                            break;

        case REDUCE_MAX_WITH_INDEX_ATOMICS: addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DREDUCE_MAX_WITH_INDEX_ATOMICS");
                                            break;

        default:                            printf("Invalid reduction type!!\n");
                                            break;
    }

	return;
}


static CLBLASMpatExtra mpatExtra;

extern "C"
void initReductionRegisterPattern(MemoryPattern *mempat)
{
	#ifdef DEBUG_REDUCTION
	printf("initRegPattern called with mempat = 0x%p\n", mempat);
	#endif

	fflush(stdout);
    mempat->name = "Register accumulation based swap";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &reductionOps;

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
	DUMMY_ARGS_USAGE_3(subdims, args, _extra);
    int BLOCKSIZE = pgran->wgSize[0] * pgran->wgSize[1]; // 1D Block

	size_t blocks = 1;          // Reduction will use only 1 block
	#ifdef DEBUG_REDUCTION
	printf("blocks : %d\n", blocks);
	#endif

	threads[0] = blocks * BLOCKSIZE;
	#ifdef DEBUG_REDUCTION
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
    SolutionStep *step = container_of( pgran , pgran, SolutionStep);
    CLBlasKargs* kargs = (CLBlasKargs*) &(step->args);
    char const *kernName;

    if(kargs->redctnType == REDUCE_BY_SUM) {
            kernName = red_sum_kernel;
    } else if(kargs->redctnType == REDUCE_BY_MAX) {
            kernName = red_max_kernel;
    } else if(kargs->redctnType == REDUCE_BY_MIN) {
            kernName = red_min_kernel;
    } else if(kargs->redctnType == REDUCE_MAX_WITH_INDEX) {
            kernName = red_with_index_kernel;
    } else if(kargs->redctnType == REDUCE_BY_HYPOT) {
            kernName = red_hypot_kernel;
    } else if(kargs->redctnType == REDUCE_BY_SSQ) {
            kernName = red_ssq_kernel;
    }

	#ifdef DEBUG_REDUCTION
 	printf("REDUCTION GENERATOR called....\n");
	printf("dataType : %c\n", Prefix[extraFlags->dtype]);
	printf("Vector length used : %d\n\n", vecLenA);
	#endif

    unsigned int vecLenA = extraFlags->vecLenA;
	bool doVLOAD = false;
	if( extraFlags->flags &  KEXTRA_NO_COPY_VEC_A )
	{
		doVLOAD = true;
	}
    strcpy( tempTemplate, kernName );
	kprintf kobj( Prefix[extraFlags->dtype], vecLenA, doVLOAD, doVLOAD, BLOCKSIZE);
    kobj.spit((char*)buf, tempTemplate);

    return (32 * 1024 * sizeof(char));
}

/*
 __kernel void %PREFIXred_sum_kernel( __global %TYPE *_X, __global %TYPE *_res,
                                                      uint N, uint offx, uint offRes )
*/
static void
assignKargs(KernelArg *args, const void *params, const void* _extra)
{
    DUMMY_ARG_USAGE(_extra);
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;

    INIT_KARG(&args[0], blasArgs->D);
	INIT_KARG(&args[1], blasArgs->A);
    initSizeKarg(&args[2], blasArgs->N);
    size_t offScratch = 0;
    initSizeKarg(&args[3], offScratch);
    initSizeKarg(&args[4], blasArgs->offA);

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
