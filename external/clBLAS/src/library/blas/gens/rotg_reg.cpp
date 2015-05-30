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
 * rotg generator
 */
//#define DEBUG_ROTG

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
#include <rotg.clT>
#include <solution_seq.h>
#include "blas_subgroup.h"
#include "gen_helper.h"

extern "C"
unsigned int dtypeSize(DataType type);


static char Prefix[4];

static SolverFlags
solverFlags(void)
{
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
assignKargs(KernelArg *args, const void *params, const void* extra );

extern "C"
void initRotgRegisterPattern(MemoryPattern *mempat);

static void
setBuildOpts(
    char * buildOptStr,
    const void *kArgs);

static SolverOps rotgOps = {
    generator,
    assignKargs,
    NULL,
    NULL, // Prepare Translate Dims
    NULL, // Inner Decomposition Axis
    calcNrThreads,
    NULL,
    solverFlags,
	NULL,
	NULL,
	NULL,
	setBuildOpts,
	NULL
};

static void
setBuildOpts(
    char * buildOptStr,
    const void *args)
{
	const SolutionStep *step = (const SolutionStep *)args;
    const CLBlasKargs *kargs = (const CLBlasKargs *)(&step->args);
	if ( (kargs->dtype == TYPE_DOUBLE) || (kargs->dtype == TYPE_COMPLEX_DOUBLE) ) {
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DDOUBLE_PRECISION");
	}
	if( (kargs->dtype == TYPE_COMPLEX_FLOAT) || (kargs->dtype == TYPE_COMPLEX_DOUBLE) ) {
	    addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DCOMPLEX");
	}

	return;
}


static CLBLASMpatExtra mpatExtra;

extern "C"
void initRotgRegisterPattern(MemoryPattern *mempat)
{
	#ifdef DEBUG_ROTG
	printf("initRegPattern called with mempat = 0x%p\n", mempat);
	#endif

	fflush(stdout);
    mempat->name = "Register accumulation based swap";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &rotgOps;

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
	int BLOCKSIZE = pgran->wgSize[0] * pgran->wgSize[1]; // 1D Block
	DUMMY_ARGS_USAGE_3(subdims, _extra, args);

	size_t blocks = 1;  // Only 1 work-group is enough
	#ifdef DEBUG_ROTG
	printf("blocks : %d\n", blocks);
	#endif

	threads[0] = blocks * BLOCKSIZE;
	#ifdef DEBUG_ROTG
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

	CLBLASKernExtra *extraFlags = ( CLBLASKernExtra *)extra;
	DUMMY_ARGS_USAGE_2(subdims, pgran);
	char tempTemplate[32*1024];

	if ( buf == NULL) // return buffer size
	{
		buflen = (32 * 1024 * sizeof(char));
        return (ssize_t)buflen;
	}

	#ifdef DEBUG_ROTG
	printf("dataType : %c\n", Prefix[extraFlags->dtype]);
	#endif

    strcpy( tempTemplate, (char*)rotg_kernel );

	kprintf kobj( Prefix[extraFlags->dtype], 1, false, false);
    kobj.spit((char*)buf, tempTemplate);

    return (32 * 1024 * sizeof(char));
}

/*
__kernel void %PREFIXrotg_kernel( __global %TYPE *_A, __global %TYPE *_B, __global %PTYPE *_C,
                                __global %TYPE *_S, uint offa, uint offb, uint offc, uint offs )

*/
static void
assignKargs(KernelArg *args, const void *params, const void* )
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;

    INIT_KARG(&args[0], blasArgs->A);
	INIT_KARG(&args[1], blasArgs->B);
	INIT_KARG(&args[2], blasArgs->C);
    INIT_KARG(&args[3], blasArgs->D);
    initSizeKarg(&args[4], blasArgs->offa);
    initSizeKarg(&args[5], blasArgs->offb);
    initSizeKarg(&args[6], blasArgs->offc);
    initSizeKarg(&args[7], blasArgs->offd);

	return;
}
