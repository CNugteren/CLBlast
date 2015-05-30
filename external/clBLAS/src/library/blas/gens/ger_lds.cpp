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
 * ger generator
 */
//#define DEBUG_GER

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
#include <ger.clT>
#include <solution_seq.h>

extern "C"
unsigned int dtypeSize(DataType type);


static char Prefix[4];

static int
getDefaultDecomposition(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void *pArgs);

static SolverFlags
solverFlags(void)
{
	#ifdef DEBUG_GER
	printf("solverFlags callen......\n");
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
assignKargs(KernelArg *args, const void *params, const void* );

extern "C"
void initGerRegisterPattern(MemoryPattern *mempat);

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

static SolverOps gerOps = {
    generator,
    assignKargs,
    isFitToLDS,
    NULL, // Prepare Translate Dims
    NULL, // Inner Decomposition Axis
    calcNrThreads,
    NULL,	// Related to images
    solverFlags,
	NULL,
    getDefaultDecomposition,
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

    if(((kargs->lda.matrix) % vlen) != 0)
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

		#ifdef DEBUG_GER
		printf("Setting build options ... Double... for DOUBLE PRECISION support\n");
		#endif
	}
	return;
}

static CLBLASMpatExtra mpatExtra;

extern "C"
void initGerRegisterPattern(MemoryPattern *mempat)
{
    mempat->name = "Register accumulation based ger";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &gerOps;

    //CHECK THIS
	mpatExtra.aMset = CLMEM_LEVEL_L2;
    mpatExtra.bMset = CLMEM_LEVEL_L1 | CLMEM_LEVEL_LDS; // For "x" vector
    mpatExtra.mobjA = CLMEM_BUFFER;
    mpatExtra.mobjB = CLMEM_BUFFER;
    mempat->extra = &mpatExtra;

	Prefix[TYPE_FLOAT] = 'S';
	Prefix[TYPE_DOUBLE] = 'D';
	Prefix[TYPE_COMPLEX_FLOAT] = 'C';
	Prefix[TYPE_COMPLEX_DOUBLE] = 'Z';

	#ifdef DEBUG_GER
	printf("initGerRegPattern called with mempat = 0x%p\n", mempat);
	fflush(stdout);
	#endif
}

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *_extra)
{
	const CLBlasKargs *kargs = (const CLBlasKargs *)args;
	const CLBLASKernExtra *extra = ( CLBLASKernExtra *)_extra;
	size_t BLOCKSIZE = pgran->wgSize[0] * pgran->wgSize[1]; // 1D Block
	size_t BH, BW;
    unsigned int VEC_LEN = extra->vecLenA;

	clblasOrder order = ( extra->flags & KEXTRA_COLUMN_MAJOR) ? clblasColumnMajor: clblasRowMajor;

	size_t nBlocksY;                //number of blocks in Y dir ( Although we say 1D block to opencl )
    size_t nBlocksX;                //number of blocks in X dir

    BH = subdims->y;
	BW = subdims->x;

    if ( order == clblasColumnMajor )
    {
		nBlocksY = ( kargs->M + BH*VEC_LEN - 1 ) / (BH*VEC_LEN);
        nBlocksX = ( kargs->N + BW - 1) / BW;
    }
    else
    {
		nBlocksY = ( kargs->M + BH - 1) / BH;
        nBlocksX = ( kargs->N + BW*VEC_LEN - 1) / (BW*VEC_LEN);
    }
	size_t blocks = nBlocksX * nBlocksY;
	threads[0] = blocks * BLOCKSIZE;
	threads[1] = 1;

	#ifdef DEBUG_GER
	printf("calcNrThreads called from GER_Reg.cpp.. wgSize[0]: %u\twgSize[1]: %u\n", pgran->wgSize[0], pgran->wgSize[1]);
	printf("subdim->y :%u\t subdim->x : %u\n", subdims->y, subdims->x);
	printf("kargs-> M : %d,  kargs-> N: %d,  BH: %d,  BW: %d\n", kargs->M, kargs->N, BH, BW);
	printf("blocks : %d\tglobalthreads[0]  : %u\t VecLen :%d\n", blocks, threads[0], VEC_LEN);
	#endif

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
	size_t BH, BW;//BLOCKSIZE  = pgran->wgSize[0]; // Because we are using 1D block
    unsigned int VEC_LEN;
	char tempTemplate[32*1024];
	char bhStr[10], bwStr[10];


	pgran = pgran; // To remove warnings
	if ( buf == NULL) // return buffer size
    {
          buflen = (64 * 1024 * sizeof(char));
          return (ssize_t)buflen;
    }
	CLBLASKernExtra *extraFlags = ( CLBLASKernExtra *)extra;
	clblasOrder order = ( extraFlags->flags & KEXTRA_COLUMN_MAJOR) ? clblasColumnMajor: clblasRowMajor;
	VEC_LEN = extraFlags->vecLenA;

	#ifdef DEBUG_GER
	printf("GER GENERATOR called.... with %s order,  DataType %c  & Vector-Length: %d\n",
				((order == clblasColumnMajor)? "ColumnMajor": "RowMajor"), Prefix[extraFlags->dtype], VEC_LEN );
	#endif

	if( order == clblasColumnMajor )
	{
		strcpy( tempTemplate, (char*)ger_C_kernel );
	}
	else
	{
		strcpy( tempTemplate, (char*)ger_R_kernel );
	}

	// FIXME: VECTORSIZE HARD CODED
	// FIXME: SetKernelArgs.. sends offa, offx, and lda should be received as uint

	bool doVLOAD = false;
	if( extraFlags->flags &  KEXTRA_NO_COPY_VEC_A )
	{
		doVLOAD = true;
		#ifdef DEBUG_GER
			printf("DOing VLOAD as Aligned Data Pointer not Availabe\n");
		#endif
	}
	else
	{
		#ifdef DEBUG_GER
			printf("Using Aligned Data Pointer .........................\n");
		#endif
	}
	kprintf kobj( Prefix[extraFlags->dtype], VEC_LEN, doVLOAD, doVLOAD);

	BH = subdims->y;
	BW = subdims->x;
	sprintf( bhStr, "%" SPREFIX "u", BH );
	sprintf( bwStr, "%" SPREFIX "u", BW );

	#ifdef DEBUG_GER
    printf("BH = %s\n", bhStr);
    printf("BW = %s\n", bwStr);
	#endif

    kobj.put("%BH_DEF", (const char *)bhStr);
    kobj.put("%BW_DEF", (const char *)bwStr);
    kobj.spit((char*)buf, tempTemplate);


	return (64 * 1024 * sizeof(char));
    // return 0;//(ret < 0) ? -EOVERFLOW : ret;
}

/*
		( __global const %TYPE* X, __global const %TYPE* Y, __global %TYPE* A,
				uint M, uint N, uint offx, int incx, uint offy, int incy, uint offa, uint lda,
				%TYPE alpha, int doConj )
*/

static void
assignKargs(KernelArg *args, const void *params, const void*)
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;
    cl_int incx, incy, doConj;

    INIT_KARG(&args[0], blasArgs->B); 	//  B - our X vector
    INIT_KARG(&args[1], blasArgs->C); 	//  C - our Y vector
    INIT_KARG(&args[2], blasArgs->A); 	//  A - matrix A
    initSizeKarg(&args[3], blasArgs->M);
	initSizeKarg(&args[4], blasArgs->N);

	incx = blasArgs->ldb.vector;
	incy = blasArgs->ldc.vector;
	initSizeKarg(&args[5], blasArgs->offBX);
    INIT_KARG(&args[6], incx);
	initSizeKarg(&args[7], blasArgs->offCY);
   	INIT_KARG(&args[8], incy);
	initSizeKarg(&args[9], blasArgs->offa);
    initSizeKarg(&args[10], blasArgs->lda.matrix);

   	assignScalarKarg(&args[11], &(blasArgs->alpha), blasArgs->dtype);
	doConj = (cl_int)(blasArgs->K);
	INIT_KARG(&args[12], doConj);	// K was used as doConj

	#ifdef DEBUG_GER
	printf("doConj = %d\n", doConj );
	#endif

	return;
}

static bool
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs)
{
    const CLBlasKargs *kargs = (const CLBlasKargs*)kernelArgs;
    SolutionStep *step = container_of(kargs, args, SolutionStep);
    unsigned int vecLen;
    vecLen = ((CLBLASKernExtra*)(step->kernels[CLBLAS_COMPUTING_KERNEL]->extra))->vecLenA;

    cl_ulong maxSize;

    if( kargs->order == clblasColumnMajor ) {
        maxSize = ( dim[0].x + (dim[0].y * vecLen) ) * sizeof(dtype);
    } else {
        maxSize = ( (dim[0].x * vecLen) + dim[0].y ) * sizeof(dtype);
    }
    return ( maxSize <= ldsSize );
}

static int
getDefaultDecomposition(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void *pArgs)
{
    SolutionStep *step = container_of( pgran , pgran, SolutionStep);
    size_t maxWorkGroupSize;
    cl_device_id devID = step->device.id;
    size_t wgX, wgY;
    pArgs = pArgs;

    clGetDeviceInfo(devID, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(size_t), &maxWorkGroupSize, NULL);

    if( step->args.order == clblasColumnMajor )
	{
		wgY = 16;						// BH preferably 16(quarter wave-front)
		subdims[0].y = wgY;
		wgX = maxWorkGroupSize / wgY;	// BW is left upto maxWorkGroupSize of the device
		wgX = szmin( wgX, 16 );
		subdims[0].x = wgX;
	}
	else {
		wgX = 16;
		subdims[0].x = wgX;
		wgY = maxWorkGroupSize / wgX;
		wgY = szmin( wgY, 16 );
		subdims[0].y = wgY;
	}

    pgran->wgDim = 1; //1D blocking
    pgran->wgSize[0] = (unsigned int)(wgX * wgY);
    pgran->wgSize[1] = 1;

    if(subdimsNum > 0)
    {
        subdims[0].itemX = subdims[0].x;
        subdims[0].itemY = subdims[0].y;
        subdims[0].bwidth = 1;
    }
    if(subdimsNum > 1)
    {
        subdims[1].itemY = 1;
        subdims[1].itemX = 1;
        subdims[1].y = subdims[1].itemY;
        subdims[1].x = subdims[1].itemX;
        subdims[1].bwidth = 1;
    }

    return 0;
}
