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
 * rotm generator
 */
//#define DEBUG_ROTM

#define WORKGROUPS_PER_CU  32

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
#include <rotm.clT>
#include <solution_seq.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

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
void initRotmRegisterPattern(MemoryPattern *mempat);

static  KernelExtraFlags
selectVectorization(
    void *kargs,
    unsigned int vlen );

static void
setBuildOpts(
    char * buildOptStr,
    const void *kArgs);

static SolverOps rotmOps = {
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
	if(kargs->pigFuncID == CLBLAS_ROT)
	{
	    addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DDO_ROT");
	}
	if( (kargs->ldb.vector) != 1) {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DINCX_NONUNITY");
    }
    if( (kargs->ldc.vector) != 1) {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DINCY_NONUNITY");
    }

	return;
}


static CLBLASMpatExtra mpatExtra;

extern "C"
void initRotmRegisterPattern(MemoryPattern *mempat)
{
	#ifdef DEBUG_ROTM
	printf("initRegPattern called with mempat = 0x%p\n", mempat);
	#endif

	fflush(stdout);
    mempat->name = "Register accumulation based swap";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &rotmOps;

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
    const CLBLASKernExtra *extra = ( CLBLASKernExtra *)_extra;
    CLBlasKargs *kargs = (CLBlasKargs *)args;
    SolutionStep *step = container_of(kargs, args, SolutionStep);
    TargetDevice *kDevice = &(step->device);

    cl_int err;
    unsigned int numComputeUnits = deviceComputeUnits( (kDevice->id), &err );
    if(err != CL_SUCCESS) {
        numComputeUnits = 1;
    }

    unsigned int vecLen = extra->vecLenA;
	unsigned int blockSize = pgran->wgSize[0] * pgran->wgSize[1];

	unsigned int wgToSpawn = ((kargs->N - 1)/ (blockSize*vecLen)) + 1;
    wgToSpawn = min( wgToSpawn, (numComputeUnits * WORKGROUPS_PER_CU) );

	threads[0] = wgToSpawn * blockSize;
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

	DUMMY_ARGS_USAGE_2(pgran, subdims);
	CLBLASKernExtra *extraFlags = ( CLBLASKernExtra *)extra;
	char tempTemplate[32*1024];

	if ( buf == NULL) // return buffer size
	{
		buflen = (32 * 1024 * sizeof(char));
        return (ssize_t)buflen;
	}

	#ifdef DEBUG_ROTM
	printf("dataType : %c\n", Prefix[extraFlags->dtype]);
	#endif

    unsigned int vecLenA = extraFlags->vecLenA;

	#ifdef DEBUG_ROTM
	printf("Vector length used : %d\n\n", vecLenA);
	#endif

	bool doVLOAD = false;
	if( extraFlags->flags &  KEXTRA_NO_COPY_VEC_A )
	{
		doVLOAD = true;
		#ifdef DEBUG_ROTM
			printf("DOing VLOAD as Aligned Data Pointer not Availabe\n");
		#endif
	}
	else
	{
		#ifdef DEBUG_ROTM
			printf("Using Aligned Data Pointer .......\n");
		#endif
	}
    strcpy( tempTemplate, (char*)rotm_kernel );
	kprintf kobj( Prefix[extraFlags->dtype], vecLenA, doVLOAD, doVLOAD);
    kobj.spit((char*)buf, tempTemplate);

    return (32 * 1024 * sizeof(char));
}

/*
__kernel void %PREFIXrotm_kernel( __global %TYPE *_X, __global %TYPE *_Y, uint N,
                                uint offx, int incx, uint offy, int incy
#ifndef DO_ROT
                                , __global %TYPE *_param, uint offParam             // Rotm parameters
#else
                                , %PTYPE C,  %PTYPE S                               // Rot parameters
#endif
*/
static void
assignKargs(KernelArg *args, const void *params, const void* )
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;
	cl_int incx, incy;

    INIT_KARG(&args[0], blasArgs->A);
	INIT_KARG(&args[1], blasArgs->B);
    initSizeKarg(&args[2], blasArgs->N);
    initSizeKarg(&args[3], blasArgs->offBX);
    incx = blasArgs->ldb.vector;
    INIT_KARG(&args[4], incx);
    initSizeKarg(&args[5], blasArgs->offCY);
    incy = blasArgs->ldc.vector;
    INIT_KARG(&args[6], incy);

    if(blasArgs->pigFuncID == CLBLAS_ROT)
    {
        DataType alphaBetaType = (blasArgs->dtype == TYPE_COMPLEX_FLOAT)? TYPE_FLOAT:
                                ((blasArgs->dtype == TYPE_COMPLEX_DOUBLE)? TYPE_DOUBLE:  blasArgs->dtype);
        assignScalarKarg(&args[7], &(blasArgs->alpha), alphaBetaType);
	    assignScalarKarg(&args[8], &(blasArgs->beta), alphaBetaType);
	}
	else if(blasArgs->pigFuncID == CLBLAS_ROTM)
	{
        INIT_KARG(&args[7], blasArgs->D);
        initSizeKarg(&args[8], blasArgs->offd);
    }

	return;
}
