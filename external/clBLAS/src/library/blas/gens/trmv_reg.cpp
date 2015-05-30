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
 * trmv generator
 */
//#define DEBUG_TRMV

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
#include <trmv.clT>
#include <solution_seq.h>

extern "C"
unsigned int dtypeSize(DataType type);


static char Prefix[4];

static SolverFlags
solverFlags(void)
{
	#ifdef DEBUG_TRMV
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
assignKargs(KernelArg *args, const void *params, const void* extra );

extern "C"
void initTrmvRegisterPattern(MemoryPattern *mempat);

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

static SolverOps trmvOps = {
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

    if( ( (kargs->uplo == clblasLower) && (kargs->order == clblasColumnMajor) ) ||
          ( (kargs->uplo == clblasUpper) && (kargs->order == clblasRowMajor) )   )
	    {
			if( (kargs->N) % vlen)
			{
				kflags = KEXTRA_NO_COPY_VEC_A;
			}
		}
    if( kargs->pigFuncID == CLBLAS_TPMV || kargs->pigFuncID == CLBLAS_HPMV || kargs->pigFuncID == CLBLAS_SPMV )
    {
        kflags = KEXTRA_NO_COPY_VEC_A;     // Packed-case never do aligned access
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
		#ifdef DEBUG_TRMV
		printf("Setting build options ... Double... for DOUBLE PRECISION support\n");
		#endif
	}
    if( (step->funcID == CLBLAS_HEMV) || (kargs->pigFuncID == CLBLAS_HPMV) || (kargs->pigFuncID == CLBLAS_SPMV) )
	{
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DHEMV_ONLY");
		/*
		if(kargs->diag == clblasUnit)
		{
			addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DHEMV_ZERO_DIAG");
		}
		*/
	}
    if ( kargs->pigFuncID == CLBLAS_SPMV )
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DSPMV_ONLY");
    }
    if( (kargs->pigFuncID == CLBLAS_TPMV) || (kargs->pigFuncID == CLBLAS_HPMV) || (kargs->pigFuncID == CLBLAS_SPMV) )
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DPACKED");
    }

	return;
}


static CLBLASMpatExtra mpatExtra;

extern "C"
void initTrmvRegisterPattern(MemoryPattern *mempat)
{
	#ifdef DEBUG_TRMV
	printf("initTRMVREgPattern called with mempat = 0x%p\n", mempat);
	#endif

	fflush(stdout);
    mempat->name = "Register accumulation based trmv";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &trmvOps;

    mpatExtra.aMset = CLMEM_LEVEL_L2;
    mpatExtra.bMset = CLMEM_LEVEL_L1 | CLMEM_LEVEL_LDS; // For "x" vector
    mpatExtra.mobjA = CLMEM_BUFFER;
    mpatExtra.mobjB = CLMEM_BUFFER;
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
	#ifdef DEBUG_TRMV
	printf("calcNrThreads called from TRMV_Reg.c\n");
	#endif

    const CLBlasKargs *kargs = (const CLBlasKargs *)args;
	const CLBLASKernExtra *extra = ( CLBLASKernExtra *)_extra;

	clblasOrder order = ( extra->flags & KEXTRA_COLUMN_MAJOR) ? clblasColumnMajor: clblasRowMajor;
	clblasTranspose trans = ( extra->flags & KEXTRA_TRANS_A) ? clblasTrans :
								(( extra->flags & KEXTRA_CONJUGATE_A) ? clblasConjTrans: clblasNoTrans);

	// unity and doConj handled in setKernelArgs
    if ( order == clblasRowMajor )
    {
        order = clblasColumnMajor;
        if ( trans == clblasNoTrans)
        {
            trans = clblasTrans;
        }
        else if ( trans == clblasTrans )
        {
            trans = clblasNoTrans;
        }
        else // clblasConjTrans
        {
            trans = clblasNoTrans;
        }
    }

	size_t TARGETROWS =  (trans == clblasNoTrans) ? subdims->y : BLOCKSIZE/(subdims->y/extra->vecLenA);

	#ifdef DEBUG_TRMV
	printf("kargs-> N : %d, TARGETROWS: %d\n", kargs->N, TARGETROWS);
	#endif

	size_t blocks = ((kargs->N - 1)/ TARGETROWS) + 1;
	#ifdef DEBUG_TRMV
	printf("blocks : %d\n", blocks);
	#endif

	threads[0] = blocks * BLOCKSIZE;
	#ifdef DEBUG_TRMV
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

	size_t BLOCKSIZE  = pgran->wgSize[0];
	char tempTemplate[32*1024];
	char targetRows[10], blockSize[10];

	if ( buf == NULL) // return buffer size
	{
		buflen = (64 * 1024 * sizeof(char));
        return (ssize_t)buflen;
	}
	CLBLASKernExtra *extraFlags = ( CLBLASKernExtra *)extra;

	#ifdef DEBUG_TRMV
 	printf("TRMV GENERATOR called....\n");
	#endif

	if((( extraFlags->flags &  KEXTRA_TRANS_A) || ( extraFlags ->flags & KEXTRA_CONJUGATE_A )))
	{
		#ifdef DEBUG_TRMV
		printf("A is trans or CONJ-TRANS\n");
		#endif
	}
	else
	{
		#ifdef DEBUG_TRMV
		printf("A is noTrans...\n");
		#endif
	}

	clblasUplo uplo   = ( extraFlags->flags & KEXTRA_UPPER_TRIANG) ? clblasUpper : clblasLower;
	clblasOrder order = ( extraFlags->flags & KEXTRA_COLUMN_MAJOR) ? clblasColumnMajor: clblasRowMajor;
	clblasTranspose trans = ( extraFlags->flags & KEXTRA_TRANS_A) ? clblasTrans : (( extraFlags->flags & KEXTRA_CONJUGATE_A) ? clblasConjTrans: clblasNoTrans);

	// unity and doConj handled in setKernelArgs
    if ( order == clblasRowMajor )
    {
        order = clblasColumnMajor;
        if ( trans == clblasNoTrans)
        {
            trans = clblasTrans;
        }
        else if ( trans == clblasTrans )
        {
            trans = clblasNoTrans;
        }
        else // clblasConjTrans
        {
            trans = clblasNoTrans;
        }

		uplo = ( uplo == clblasUpper)? clblasLower : clblasUpper;
    }


	if ((subdims->y % extraFlags->vecLenA) != 0)
	{
		printf("WARNING: TRMV: generator: TARGETROWS must be divisible by Vector Length\n");
		return 0;
	}

	size_t TARGETROWS = 0;
	if ( trans == clblasNoTrans)
	{
		#ifdef DEBUG_TRMV
		printf("clblasNoTrans....%s\n",	( uplo == clblasLower )?"LOWER":"UPPER");
		#endif

		( uplo == clblasLower )?
		    		(strcpy(tempTemplate, (char*)trmv_CL_kernel)) : (strcpy(tempTemplate, (char*)trmv_CU_kernel));

		TARGETROWS = subdims->y;
		if ((BLOCKSIZE % TARGETROWS) != 0)
		{
			printf("WARNING: TRMV: generator: Invalid Block Size\n");
			return 0;
		}
	}
	else // Transpose cases...
	{
		#ifdef DEBUG_TRMV
		printf("clblasTrans....%s\n",	( uplo == clblasLower )?"LOWER":"UPPER");
		#endif

		( uplo == clblasLower )?
		    		(strcpy(tempTemplate, (char*)trmv_CLT_kernel)) : (strcpy(tempTemplate, (char*)trmv_CUT_kernel));

		if ((BLOCKSIZE % (subdims->y / extraFlags->vecLenA)) != 0)
		{
			printf("WARNING: TRMV: generator: Invalid Block Size\n");
			return 0;
		}
		TARGETROWS = BLOCKSIZE/(subdims->y / extraFlags->vecLenA);
	}

	#ifdef DEBUG_TRMV
	printf("dataType : %c\n", Prefix[extraFlags->dtype]);
	#endif

	// FIXME: VECTORSIZE HARD CODED
	// FIXME : SetKernelArgs.. sends offa, offx, and lda should be received as uint
    unsigned int vecLenA = extraFlags->vecLenA;

	#ifdef DEBUG_TRMV
	printf("Vector length used : %d\n\n", vecLenA);
	#endif

	bool doVLOAD = false;
	if( extraFlags->flags &  KEXTRA_NO_COPY_VEC_A )
	{
		doVLOAD = true;
		#ifdef DEBUG_TRMV
			printf("DOing VLOAD as Aligned Data Pointer not Availabe\n");
		#endif
	}
	else
	{
		#ifdef DEBUG_TRMV
			printf("Using Aligned Data Pointer .........................\n");
		#endif
	}
	kprintf kobj( Prefix[extraFlags->dtype], vecLenA, doVLOAD);

    sprintf( targetRows, "%" SPREFIX "u", TARGETROWS );
	sprintf( blockSize, "%" SPREFIX "u", BLOCKSIZE );

	#ifdef DEBUG_TRMV
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
		(__global %TYPE const* restrict A, __global %TYPE * _xnew, __global %TYPE const* restrict _x_vector, uint N,
		int incx, int isUnity, uint lda, int doConj, uint offa, uint offx)
*/
static void
assignKargs(KernelArg *args, const void *params, const void* )
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;

	//NOTE: This will not work if SolutionStep->args is not passed in const void *params.
	SolutionStep *step = container_of(blasArgs, args, SolutionStep);

	cl_int inc;
	cl_int unity, doConj;
    //bool incxOne = (blasArgs->ldb.vector == 1);
    //bool incyOne = (blasArgs->ldc.vector == 1);

    INIT_KARG(&args[0], blasArgs->A); 	//A - input matrix - argument
    if( (step->funcID == CLBLAS_HEMV) || (blasArgs->pigFuncID == CLBLAS_HPMV) || (blasArgs->pigFuncID == CLBLAS_SPMV) )
	{
		INIT_KARG(&args[1], blasArgs->C);   //y - since the 2nd argument is the result buffer, we should send y for HEMV
        INIT_KARG(&args[2], blasArgs->B);   //x - actual x vector argument
	}
	else
	{
		INIT_KARG(&args[1], blasArgs->B); 	//x - result buffer = _xnew argument
    	INIT_KARG(&args[2], blasArgs->C); 	//y - scratch == _x_vector argument
    }
	initSizeKarg(&args[3], blasArgs->N);
    inc = blasArgs->ldb.vector;
    INIT_KARG(&args[4], inc);
	unity = (blasArgs->diag == clblasUnit);
   	INIT_KARG(&args[5], unity);
    initSizeKarg(&args[6], blasArgs->lda.matrix);
	doConj = (blasArgs->transA == clblasConjTrans);

	#ifdef DEBUG_TRMV
	printf("doConj is : %d, unity is : %d, incx is : %d\n", doConj, unity, inc);
	#endif

   	INIT_KARG(&args[7], doConj);
	initSizeKarg(&args[8], blasArgs->offa);
	initSizeKarg(&args[9], blasArgs->offBX);

	// For HEMV both alpha and beta has to be passed.
	if( (step->funcID == CLBLAS_HEMV) || (blasArgs->pigFuncID == CLBLAS_HPMV) || (blasArgs->pigFuncID == CLBLAS_SPMV) )
	{
		inc = blasArgs->ldc.vector;
		INIT_KARG(&args[10], inc);
		initSizeKarg(&args[11], blasArgs->offCY);
		assignScalarKarg(&args[12], &(blasArgs->alpha), blasArgs->dtype);
		assignScalarKarg(&args[13], &(blasArgs->beta), blasArgs->dtype);
	}

	return;
}

static bool
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs)
{
	size_t x, y;
    cl_ulong maxSize;
    CLBlasKargs *blasArgs = (CLBlasKargs *)kernelArgs;
	//size_t tile;
	size_t maxBlockSize = 256; // PENDING: Query MAX_WORKGROUP_SIZE from OpenCL
	size_t extra;
	int naturalVecLength = sizeof(cl_float4)/sizeof(dtype);
    dim = dim; // Dummy- to remove warnings
	//extra = (blasArgs->transA == clblasNoTrans) ? dim[0].bwidth : dim[0].y;
	//extra =  (extra > maxBlockSize) ? maxBlockSize : extra;

	//
	// TRMV is colMajor always...
	//
	y = 16; // Optimized for 16 float4 type reads by a quarter wavefront
	x = maxBlockSize / y;

	maxSize = x*y*sizeof(cl_float4); // PENDING: Implementing %REDUCE_SUM can bring this down to sizeof(cl_float) for non-transpose cases
	extra = ((blasArgs->transA == clblasNoTrans) ? x : (y*naturalVecLength)) * sizeof(dtype);
    return ((maxSize + extra) <= ldsSize);
/*
	tile = dim[0].y * dim[0].bwidth;
	tile = (tile > maxBlockSize) ?  (maxBlockSize) : tile;
	tile += extra;
	maxSize = tile * dtypeSize(dtype);
*/
}
