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
 * gbmv generator
 */
//#define DEBUG_GBMV

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
#include <gbmv.clT>
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
	#ifdef DEBUG_GBMV
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
void initGbmvRegisterPattern(MemoryPattern *mempat);

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

static SolverOps gbmvOps = {
    generator,
    assignKargs,
    isFitToLDS,
    NULL, // Prepare Translate Dims
    NULL, // Inner Decomposition Axis
    calcNrThreads,
    NULL,
    solverFlags,
	NULL,
    getDefaultDecomposition,
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

	if ( (kargs->dtype == TYPE_DOUBLE) || (kargs->dtype == TYPE_COMPLEX_DOUBLE) )
	{
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DDOUBLE_PRECISION");
		#ifdef DEBUG_GBMV
		printf("Setting build options ... Double... for DOUBLE PRECISION support\n");
		#endif
	}

    if( kargs->pigFuncID == CLBLAS_TBMV )
	{
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DTBMV_ONLY");
		if( kargs->diag == clblasUnit )
		{
		    addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DUNIT_DIAG");
		}
	}
	if( ((kargs->pigFuncID == CLBLAS_GBMV) || (kargs->pigFuncID == CLBLAS_TBMV)) && (kargs->transA == clblasConjTrans) )
	{
	    addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DDO_CONJ");
	}

	if( (kargs->pigFuncID == CLBLAS_SBMV) || (kargs->pigFuncID == CLBLAS_HBMV) )
	{
	    bool isUpper = ( kargs->uplo == clblasUpper )? true: false;
	    isUpper = ( kargs->order == clblasColumnMajor )? !isUpper : isUpper;

	    if( isUpper )
	            addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DGIVEN_SHBMV_UPPER");
	    else    addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DGIVEN_SHBMV_LOWER");

        if(kargs->pigFuncID == CLBLAS_HBMV)
        {
            addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DHBMV_ONLY");
            if( kargs->order == clblasColumnMajor )  // Since routine calls Row-major, the whole matrix has to be conjugated while loading
            {
                addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DDO_CONJ");
            }
        }
	}

	return;
}


static CLBLASMpatExtra mpatExtra;

extern "C"
void initGbmvRegisterPattern(MemoryPattern *mempat)
{
	#ifdef DEBUG_GBMV
	printf("initGBMVREgPattern called with mempat = 0x%p\n", mempat);
	#endif

	fflush(stdout);
    mempat->name = "Register accumulation based gbmv";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &gbmvOps;

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
	size_t fM, fN;

    const CLBlasKargs *kargs = (const CLBlasKargs *)args;
	const CLBLASKernExtra *extra = ( CLBLASKernExtra *)_extra;

	clblasOrder order = ( extra->flags & KEXTRA_COLUMN_MAJOR) ? clblasColumnMajor: clblasRowMajor;
	clblasTranspose trans = ( extra->flags & KEXTRA_TRANS_A) ? clblasTrans :
								(( extra->flags & KEXTRA_CONJUGATE_A) ? clblasConjTrans: clblasNoTrans);

    fM = kargs->M;
    fN = kargs->N;
    if ( order == clblasColumnMajor )
    {
        order = clblasRowMajor;
        fM = kargs->N;
        fN = kargs->M;
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
    if( (kargs->pigFuncID == CLBLAS_SBMV) || (kargs->pigFuncID == CLBLAS_HBMV) )    // Only NT kernel is used
    {
        trans = clblasNoTrans;
    }

	size_t blocks;
	size_t H = subdims->x;
	size_t TARGET_ROWS =  BLOCKSIZE / H;

	if( trans == clblasNoTrans )
	{
    	blocks = ((fM - 1)/ TARGET_ROWS) + 1;
    }
    else {
        blocks = ((fN - 1)/ H) + 1;
    }

	threads[0] = blocks * BLOCKSIZE;
	threads[1] = 1;

	#ifdef DEBUG_GBMV
	    printf("calcNrThreads called from gbmv.cpp\n");
	    printf("BLOCKSIZE : %d, subdims->x : %d\n", BLOCKSIZE, H);
	    printf("blocks : %d\n", blocks);
	    printf("pgran-wgSize[0] : %d, globalthreads[0]  : %d\n", pgran->wgSize[0], threads[0]);
	#endif

}

static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra)
{

	size_t BLOCKSIZE  = pgran->wgSize[0];
	size_t H = subdims->x;
	char tempTemplate[64*1024];
	char def_target_rows[10], def_h[10];

    SolutionStep *step = container_of( pgran , pgran, SolutionStep);    // NOTE: using container_of() to get pigFuncID
    CLBlasKargs* kargs = (CLBlasKargs*) &(step->args);


	if ( buf == NULL) // return buffer size
	{
		buflen = (64 * 1024 * sizeof(char));
        return (ssize_t)buflen;
	}

	CLBLASKernExtra *extraFlags = ( CLBLASKernExtra *)extra;

	//clblasUplo uplo   = ( extraFlags->flags & KEXTRA_UPPER_TRIANG) ? clblasUpper : clblasLower;
	clblasOrder order = ( extraFlags->flags & KEXTRA_COLUMN_MAJOR) ? clblasColumnMajor: clblasRowMajor;
	clblasTranspose trans = ( extraFlags->flags & KEXTRA_TRANS_A) ? clblasTrans : (( extraFlags->flags & KEXTRA_CONJUGATE_A) ? clblasConjTrans: clblasNoTrans);

    if ( order == clblasColumnMajor )
    {
        order = clblasRowMajor;
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
    if( (kargs->pigFuncID == CLBLAS_SBMV) || (kargs->pigFuncID == CLBLAS_HBMV) )    // Only NT kernel is used
    {
        trans = clblasNoTrans;
    }

	if ((BLOCKSIZE % H) != 0)
    {
		printf("WARNING: GBMV: generator: Invalid Block Size\n");
		return 0;
	}
	size_t TARGET_ROWS =  BLOCKSIZE / H;

	if ( trans == clblasNoTrans)
	{
		strcpy(tempTemplate, (char*)gbmv_RNT_kernel);
	}
	else // Transpose cases...
	{
        strcpy(tempTemplate, (char*)gbmv_RT_kernel);;
	}

    unsigned int vecLenA = extraFlags->vecLenA;

	bool doVLOAD = false;       // Always scalar load for banded matrices
	kprintf kobj( Prefix[extraFlags->dtype], vecLenA, doVLOAD);

    sprintf( def_target_rows, "%d", (int)TARGET_ROWS );
	sprintf( def_h, "%d", (int)H );

	#ifdef DEBUG_GBMV
	    printf("GBMV GENERATOR called....\n");
	    if((( extraFlags->flags &  KEXTRA_TRANS_A) || ( extraFlags ->flags & KEXTRA_CONJUGATE_A )))
	    {
	        printf("A is trans or CONJ-TRANS\n");
	    }
	    else
	    {
	        printf("A is noTrans...\n");
	    }
        printf("TARGET ROWS = %s\n", def_target_rows);
        printf("H = %s\n", def_h);
        printf("dataType : %c\n", Prefix[extraFlags->dtype]);
	#endif

    kobj.put("%DEF_H", (const char *)def_h);
    kobj.put("%DEF_TARGET_ROWS", (const char *)def_target_rows);
    kobj.spit((char*)buf, tempTemplate);

	return (64 * 1024 * sizeof(char));
}

/*
__kernel void %PREFIXgbmv_RNT_kernel( __global const %TYPE * _A, __global %TYPE * _y_vector, __global %TYPE const* restrict _x_vector,
                                        uint M, uint N, uint KL, uint KU, uint lda, int incx, int incy, uint offa, uint offx, uint offy
ifndef TBMV_ONLY
                                    ,%TYPE alpha, %TYPE beta
endif
*/
static void
assignKargs(KernelArg *args, const void *params, const void* )
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;

	size_t fM, fN, fKL, fKU;
	cl_int inc;

	if( blasArgs->order == clblasColumnMajor )       // M, N, KL, KU gets swapped
	{
	    fM = blasArgs->N;
	    fN = blasArgs->M;
	    fKL = blasArgs->KU;
	    fKU = blasArgs->KL;
	}
	else    {
	    fM = blasArgs->M;
	    fN = blasArgs->N;
	    fKL = blasArgs->KL;
	    fKU = blasArgs->KU;
	}

    INIT_KARG(&args[0], blasArgs->A); 	    //A - input matrix - argument
    INIT_KARG(&args[1], blasArgs->C);       //y - y vector
    INIT_KARG(&args[2], blasArgs->B);       //x - actual x vector argument

	initSizeKarg(&args[3], fM);
    initSizeKarg(&args[4], fN);
    initSizeKarg(&args[5], fKL);
    initSizeKarg(&args[6], fKU);

    initSizeKarg(&args[7], blasArgs->lda.matrix);
    inc = blasArgs->ldb.vector;
    INIT_KARG(&args[8], inc);
    inc = blasArgs->ldc.vector;
    INIT_KARG(&args[9], inc);

	initSizeKarg(&args[10], blasArgs->offa);
	initSizeKarg(&args[11], blasArgs->offBX);
	initSizeKarg(&args[12], blasArgs->offCY);

	// For GBMV, SBMV, HBMV both alpha and beta has to be passed.
	if( (blasArgs->pigFuncID == CLBLAS_GBMV) || (blasArgs->pigFuncID == CLBLAS_SBMV) || (blasArgs->pigFuncID == CLBLAS_HBMV) )
	{
		assignScalarKarg(&args[13], &(blasArgs->alpha), blasArgs->dtype);
		assignScalarKarg(&args[14], &(blasArgs->beta), blasArgs->dtype);
	}

	#ifdef DEBUG_GBMV
    printf("KL %d\tKU %d\n", fKL, fKU);
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
    kernelArgs = kernelArgs; // To remove warnings
    cl_ulong maxSize = ( (dim[0].x+1) * dim[0].y ) * sizeof(dtype);

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

    if (maxWorkGroupSize >= 256)
    {
        wgX = 32;
        wgY = 8;
    } else if (maxWorkGroupSize >= 128)
    {
        wgX = 32;
        wgY = 4;
    } else {
        //
        // PENDING: What if maxWorkGroupSize < 64 ????
        //
        wgX = 32;
        wgY = 2;
    }

    pgran->wgDim = 1; //1D blocking
    pgran->wgSize[0] = (unsigned int)(wgX * wgY);
    pgran->wgSize[1] = 1;

    if(subdimsNum > 0)
    {
        subdims[0].y = wgY ;
        subdims[0].x = wgX ;
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
