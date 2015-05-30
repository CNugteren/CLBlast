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
 * rotmg generator
 */
//#define DEBUG_ROTMG

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
#include <rotmg.clT>
#include <solution_seq.h>

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
void initRotmgRegisterPattern(MemoryPattern *mempat);

static void
setBuildOpts(
    char * buildOptStr,
    const void *kArgs);

static SolverOps rotmgOps = {
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
	if ( kargs->dtype == TYPE_DOUBLE || kargs->dtype == TYPE_COMPLEX_DOUBLE)
	{
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DDOUBLE_PRECISION");
	}

	return;
}


static CLBLASMpatExtra mpatExtra;

extern "C"
void initRotmgRegisterPattern(MemoryPattern *mempat)
{
	#ifdef DEBUG_ROTMG
	printf("initRegPattern called with mempat = 0x%p\n", mempat);
	#endif

	fflush(stdout);
    mempat->name = "Register accumulation based swap";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &rotmgOps;

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
	DUMMY_ARGS_USAGE_3(subdims, _extra, args);
	int BLOCKSIZE = pgran->wgSize[0] * pgran->wgSize[1]; // 1D Block

	size_t blocks = 1;  // Only 1 work-group is enough
	#ifdef DEBUG_ROTMG
	printf("blocks : %d\n", blocks);
	#endif

	threads[0] = blocks * BLOCKSIZE;
	#ifdef DEBUG_ROTMG
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

	DUMMY_ARGS_USAGE_2(subdims, pgran);
	CLBLASKernExtra *extraFlags = ( CLBLASKernExtra *)extra;
	char tempTemplate[32*1024];

	if ( buf == NULL) // return buffer size
	{
		buflen = (32 * 1024 * sizeof(char));
        return (ssize_t)buflen;
	}

	#ifdef DEBUG_ROTMG
	printf("dataType : %c\n", Prefix[extraFlags->dtype]);
	#endif

    strcpy( tempTemplate, (char*)rotmg_kernel );

	kprintf kobj( Prefix[extraFlags->dtype], 1, false, false);
    kobj.spit((char*)buf, tempTemplate);

    return (32 * 1024 * sizeof(char));
}

/*
__kernel void %PREFIXrotmg_kernel( __global %TYPE *_D1, __global %TYPE *_D2, __global %TYPE *_X1,
                                __global %TYPE *_Y1, __global %TYPE *_param,
                                uint offD1, uint offD2, uint offX1, uint offY1, uint offParam )

*/
static void
assignKargs(KernelArg *args, const void *params, const void* )
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;

    INIT_KARG(&args[0], blasArgs->A);
	INIT_KARG(&args[1], blasArgs->B);
	INIT_KARG(&args[2], blasArgs->C);
    INIT_KARG(&args[3], blasArgs->D);
    INIT_KARG(&args[4], blasArgs->E);
    initSizeKarg(&args[5], blasArgs->offa);
    initSizeKarg(&args[6], blasArgs->offb);
    initSizeKarg(&args[7], blasArgs->offc);
    initSizeKarg(&args[8], blasArgs->offd);
    initSizeKarg(&args[9], blasArgs->offe);

	return;
}
