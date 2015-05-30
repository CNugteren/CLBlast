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
 * trsv gemv generator -
 *
 * This generator generates code for the GEMV portion of TRSV.
 * The idea is to call this routine after solving a subset of coefficients.
 * This generator will help to update the RHS of remaining equations using the
 * currently solved variables.
 * The current clBLAS implementation of GEMV does not have support complex types.
 * Hence, Need to write this kludge.
 * One day, this should go away and be completely replaced by existing GEMV
 *
 * NOTE:
 * This generator is highly tied to TRSV and is not a replacement for GEMV.
 * In some cases, this generator generates code not only for updating the RHS
 * but also for solving the next triangle (trtri based solve) as well.
 * We have seen marginal performance increases (1GB/s) by doing so.
 * If this is not important, one can replace this with GEMV when GEMV becomes
 * feature-complete.
 */

#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <clblas_stddef.h>
#include <clBLAS.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>
#include <trsv_gemv.clT>
#include <kprintf.hpp>
#include <solution_seq.h>

//#define DEBUG_TRSV_GEMV

extern "C"
unsigned int dtypeSize(DataType type);

static char Prefix[4]; // PENDING: Magic "4" == Number of data types supported (float, double, cl_float2, cl_double2)

static SolverFlags
solverFlags(void)
{
	#ifdef DEBUG_TRSV_GEMV
	printf("TRSV GEMV solverFlags(): solverFlags called......\n");
	#endif

    return (SF_WSPACE_1D);
}

static bool isTransposeFeasible(size_t triangle, size_t blockSize, size_t vecLen, size_t &TARGETHEIGHT);

static bool isNoTransposeFeasible(size_t triangle, size_t blockSize, size_t vecLen,
									size_t & TARGETROWS, size_t & TARGETWIDTH, size_t &NLOOPS);

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
void initTrsvGemvDefaultPattern(MemoryPattern *mempat);

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

static SolverOps trsvGemvOps = {
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
		#ifdef DEBUG_TRSV_GEMV
		printf("TRSV GEMV: Setting build options ... Double... for DOUBLE PRECISION support\n");
		#endif
	}
    if( kargs->pigFuncID == CLBLAS_TPSV)
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DPACKED");
        #ifdef DEBUG_TRSV_GEMV
            printf("TPSV GEMV: Setting build options ... PACKED\n");
        #endif
    }
	return;
}

static CLBLASMpatExtra mpatExtra;

extern "C"
void initTrsvGemvDefaultPattern(MemoryPattern *mempat)
{
	#ifdef DEBUG_TRSV_GEMV
	printf("TRSV GEMV: initTrsvGemvDefaultPattern called with mempat = 0x%p\n", (void*)mempat);
	#endif

    mempat->name = "TRSV - GEMV Update Kernel";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &trsvGemvOps;

    mpatExtra.aMset = CLMEM_LEVEL_L2;
    mpatExtra.bMset = CLMEM_LEVEL_L1 | CLMEM_LEVEL_LDS;
    mpatExtra.mobjA = CLMEM_BUFFER; // == No images
    mpatExtra.mobjB = CLMEM_BUFFER; // == No images
    mempat->extra = &mpatExtra;

	Prefix[TYPE_FLOAT] = 'S';
	Prefix[TYPE_DOUBLE] = 'D';
	Prefix[TYPE_COMPLEX_FLOAT] = 'C';
	Prefix[TYPE_COMPLEX_DOUBLE] = 'Z';
}

/*
 * Helper function that helps in calculating the "TARGET WIDTH" of
 * a block with Block Size needed for the case where
 * "theight" number of variables have been solved.
 * This is applicable only to NON-TRANSPOSE cases.
 */
static cl_ulong getTargetWidth(size_t theight, size_t blk_size, size_t vwidth)
{
	cl_ulong nLoops_v, nLoops;
	//
	// NOTE: This function should be called only for Non-Transpose cases
	// NOTE: Does not check if the block size is suitable for our purposes
	// NOTE:
	nLoops_v = (theight * theight) / blk_size;
	nLoops = nLoops_v / vwidth;
	if (nLoops == 0)
	{
		return 0;
	}
	return theight/nLoops;
}

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *_extra)
{
	size_t BLOCKSIZE = pgran->wgSize[0] * pgran->wgSize[1]; // 1D Block
	CLBlasKargs *kargs = (CLBlasKargs *)args;
	CLBLASKernExtra *extra = (CLBLASKernExtra*) _extra;
	size_t blocks;
	size_t vecLenA = extra->vecLenA;

	#ifdef DEBUG_TRSV_GEMV
	printf("TRSV GEMV: calcNrThreads() called \n");
	#endif

	if (((kargs->order == clblasColumnMajor) && (kargs->transA == clblasNoTrans)) ||
	   ((kargs->order == clblasRowMajor) && (kargs->transA != clblasNoTrans)))
	{
		size_t rowsLeft, TARGETROWS;

		//CL, CU
		TARGETROWS = subdims->y;
		rowsLeft = kargs->endRow;
		blocks = ((rowsLeft-1)/TARGETROWS) + 1;
	} else {
		size_t TARGETHEIGHT;
		if (isTransposeFeasible(subdims->y, BLOCKSIZE, vecLenA, TARGETHEIGHT) == false)
		{
			threads[0] =0; threads[1] = 0;
			#ifdef DEBUG_TRSV_GEMV
			printf("TRSV GEMV: calcNrThreads() WARNING: Returning 0\n");
			#endif
			return;
		}
		if (
			((kargs->uplo == clblasUpper) && (kargs->order == clblasColumnMajor)) ||
		   	((kargs->uplo == clblasLower) && (kargs->order == clblasRowMajor))
		   )
		{
			blocks = ((kargs->N - kargs->endRow -1) / (BLOCKSIZE / TARGETHEIGHT)) + 1;
		} else {
			blocks = (kargs->startRow)/(BLOCKSIZE/TARGETHEIGHT) + 1;
		}
	}

	#ifdef DEBUG_TRSV_GEMV
	printf("blocks : %lu\n", blocks);
	#endif
	threads[0] = blocks * BLOCKSIZE;
	threads[1] = 1;
	#ifdef DEBUG_TRSV_GEMV
	printf("pgran-wgSize[0] : %d, globalthreads[0]  : %lu\n", pgran->wgSize[0], threads[0]);
	#endif
	return;
}

static bool isTransposeFeasible(size_t triangle, size_t blockSize, size_t vecLen, size_t &TARGETHEIGHT)
{
	size_t maxHeight;

	if (triangle % vecLen)
	{
		#ifdef DEBUG_TRSV_GEMV
		printf("TRSV GEMV: isTransposeFeasible(): triangle not multiple of vectorLength\n");
		#endif
		return false;
	}
	maxHeight = triangle/vecLen;
	while (blockSize % maxHeight)
	{
		maxHeight--;
	}
	// maxHeight at minimum will be 1
	#ifdef DEBUG_TRSV_GEMV
	printf("TRSV GEMV: isTransposeFeasible(): Target Height  chosen = %lu\n", maxHeight);
	#endif
	TARGETHEIGHT = maxHeight;
	return true;
}

/*
 * NOTE:
 * No-Transpose case - The code iterates along the X direction. Vectoring is along Y Direction.
 * Since we dont iterate on Y direction (triangle height), this fixes the "blocky" component of the blocksize.
 * The blockSize then determines how much width the block has on X direction and thus the number of loops
 * can be calculated from that information.
 */
static bool isNoTransposeFeasible(size_t triangle, size_t blockSize, size_t vecLen,
									size_t & TARGETROWS, size_t & TARGETWIDTH, size_t &NLOOPS)
{
	size_t blockx, blocky, nLoops;

	if ( ((triangle*triangle) % blockSize) != 0)
	{
		#ifdef DEBUG_TRSV_GEMV
		printf("TRSV GEMV: isNoTransposeFeasible(): triangle*triangle not multiple of blockSize\n");
		#endif
		return false;
	}

	if (triangle % vecLen)
	{
		#ifdef DEBUG_TRSV_GEMV
		printf("TRSV GEMV: isNoTransposeFeasible(): triangle not multiple of vectorLength\n");
		#endif
		return false;
	}

	blocky = triangle/vecLen;
	if (blockSize % blocky)
	{
		#ifdef DEBUG_TRSV_GEMV
		printf("TRSV GEMV: isNoTransposeFeasible(): blockSize not multiple of blocky\n");
		#endif
		return false;
	}
	blockx = blockSize / blocky;
	if (triangle % blockx)
	{
		#ifdef DEBUG_TRSV_GEMV
		printf("TRSV GEMV: isNoTransposeFeasible(): blockSize not multiple of blocky\n");
		#endif
		return false;
	}
	nLoops = triangle/blockx;

	TARGETROWS = triangle;
	TARGETWIDTH = blockx;
	NLOOPS = nLoops;
	return true;
}

//
// FIXME: Report correct return value when "buf" is NULL - Needs change in KPRINTF
// FIXME: Return correct return value when "buf" is NON NULL - Needs change in KPRINTF
// FIXME: "buflen" check needs to be more accurate. Relies on above changes to KPRINTF
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
    unsigned int vecLenA = extraFlags->vecLenA;
	char tempTemplate[32*1024];
	char TARGETROWS_S[10], NLOOPS_S[10], TARGETWIDTH_S[10];
	size_t TARGETROWS, NLOOPS, TARGETWIDTH;
	char TARGETHEIGHT_S[10], BLOCKSIZE_S[10], TRIANGLE_HEIGHT_S[10];
	size_t TARGETHEIGHT;
	bool doVLOAD = false;
	int BLOCKSIZE = pgran->wgSize[0] * pgran->wgSize[1];  // [1] will always be 1 since we are a 1D implementation

	if (buf == NULL) // PENDING: Return correct buffer size
	{
		return (32 * 1024 * sizeof(char));
	}
	if (buflen > 32*1024)
	{
		#ifdef DEBUG_TRSV_GEMV
		printf("TRSV GEMV: generator(): WARNING: Returning 0 as buflen is > 32K\n");
		#endif
		return 0;
	}

	if( extraFlags->flags &  KEXTRA_NO_COPY_VEC_A )
	{
		doVLOAD = true;
		#ifdef DEBUG_TRSV_GEMV
		printf("DOing VLOAD as Aligned Data Pointer not Availabe\n");
		#endif
	}
	else
	{
		#ifdef DEBUG_TRSV_GEMV
			printf("Using Aligned Data Pointer .........................\n");
		#endif
	}
	kprintf kobj( Prefix[extraFlags->dtype], vecLenA, doVLOAD);

	#ifdef DEBUG_TRSV_GEMV
 	printf("TRSV GEMV GENERATOR called....\n");
	#endif

	clblasUplo uplo   = ( extraFlags->flags & KEXTRA_UPPER_TRIANG) ? clblasUpper : clblasLower;
	clblasOrder order = ( extraFlags->flags & KEXTRA_COLUMN_MAJOR) ? clblasColumnMajor: clblasRowMajor;
	clblasTranspose trans =
	(extraFlags->flags & KEXTRA_TRANS_A) ? clblasTrans : (( extraFlags->flags & KEXTRA_CONJUGATE_A) ? clblasConjTrans: clblasNoTrans);
	bool unit = (((extraFlags->flags) & KEXTRA_UNIT_DIAGONAL) != 0);

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

	//
	// Check Feasibility and then generate the code.
	//
	if ( trans != clblasNoTrans)
	{
		if (isTransposeFeasible(subdims->y, BLOCKSIZE, vecLenA, TARGETHEIGHT) == false)
		{
			return 0;
		}
        sprintf( TARGETHEIGHT_S, "%" SPREFIX "u", TARGETHEIGHT );
	    sprintf( BLOCKSIZE_S, "%d", BLOCKSIZE );
        sprintf( TRIANGLE_HEIGHT_S, "%" SPREFIX "u", subdims->y );

		kobj.put("%TARGET_HEIGHT", TARGETHEIGHT_S);
		kobj.put("%BLOCKSIZE", BLOCKSIZE_S);
		kobj.put("%TRIANGLE_HEIGHT", TRIANGLE_HEIGHT_S);
		( uplo == clblasLower )?
		    		(strcpy(tempTemplate, (char*)trsv_CLT_ComputeRectangle_kernel)) :
					(strcpy(tempTemplate, (char*)trsv_CUT_ComputeRectangle_kernel));

	}
	else // No-Transpose cases...
	{
		if (isNoTransposeFeasible(subdims->y, BLOCKSIZE, vecLenA, TARGETROWS, TARGETWIDTH, NLOOPS) == false)
		{
			return 0;
		}
        sprintf( TARGETROWS_S, "%" SPREFIX "u", TARGETROWS );
	    sprintf( TARGETWIDTH_S, "%" SPREFIX "u", TARGETWIDTH );
        sprintf( NLOOPS_S, "%" SPREFIX "u", NLOOPS );
		kobj.put("%TARGET_ROWS", TARGETROWS_S);
		kobj.put("%TARGET_WIDTH", TARGETWIDTH_S);
		kobj.put("%NLOOPS", NLOOPS_S);
		if (unit)
		{
			( uplo == clblasLower )?
		    (strcpy(tempTemplate, (char*)trsv_CL_ComputeRectangle_kernel)) : (strcpy(tempTemplate, (char*)trsv_CU_ComputeRectangle_kernel));
		} else {
			( uplo == clblasLower )?
		    (strcpy(tempTemplate, (char*)trsv_CL_ComputeRectangle_NonUnity_kernel)) : (strcpy(tempTemplate, (char*)trsv_CU_ComputeRectangle_NonUnity_kernel));
		}
	}

	#ifdef DEBUG_TRSV_GEMV
	printf("dataType : %c\n", Prefix[extraFlags->dtype]);
	#endif

	// FIXME: VECTORSIZE HARD CODED
	// FIXME : SetKernelArgs.. sends offa, offx, and lda should be received as uint

	#ifdef DEBUG_TRSV_GEMV
	printf("Vector length used : %d\n\n", vecLenA);
	#endif

    kobj.spit((char*)buf, tempTemplate);
	return (32 * 1024 * sizeof(char));
}

static void
assignKargs(KernelArg *args, const void *params, const void*)
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;
    cl_int inc;
	cl_int unity, doConj;

    INIT_KARG(&args[0], blasArgs->A); 	//A - input matrix - argument
    INIT_KARG(&args[1], blasArgs->B); 	//x - result buffer = _xnew argument
    initSizeKarg(&args[2], blasArgs->N);
    inc = blasArgs->ldb.vector;
    INIT_KARG(&args[3], inc);
	unity = (blasArgs->diag == clblasUnit);
   	INIT_KARG(&args[4], unity);
    initSizeKarg(&args[5], blasArgs->lda.matrix);
	doConj = (blasArgs->transA == clblasConjTrans);
	#ifdef DEBUG_TRSV_GEMV
	printf("TRMV GEMV: assignKargs: doConj is : %d, unity is : %d, incx is : %d\n", doConj, unity, inc);
	printf("TRMV GEMV: startRow, startCol set to %d, %d\n", blasArgs->startRow, blasArgs->endRow);
	#endif
   	INIT_KARG(&args[6], doConj);
	INIT_KARG(&args[7], blasArgs->startRow);
	INIT_KARG(&args[8], blasArgs->endRow);
	initSizeKarg(&args[9], blasArgs->offa);
	initSizeKarg(&args[10], blasArgs->offBX);
	return;
}

/*
 * isFitToLDS()
 *
 * 1. We will assume "dim[0].y" as the TRIANGLE_HEIGHT oiow - The number of variables solved
 *    by the corresponding TRTRI kernel
 *
 * NOTE:
 * 1. It is Possible that this function can cause "dim[0].y" to change from what was used in
 *    the "trtri" counterpart.
 *    In such a case, we will detect this in "xtrsv.c" and abort the TRSV call.
 * 2. We may need to mellow down the bloated numbers we are returning down here.
 */
static bool
isFitToLDS(
    SubproblemDim *dim,
    DataType dtype,
    cl_ulong ldsSize,
    const void *kernelArgs)
{
    CLBlasKargs *blasArgs = (CLBlasKargs *)kernelArgs;
    size_t MAXBLOCKSIZE = 256;
    cl_ulong maxSize;

    if  (
            ((blasArgs->transA == clblasNoTrans) && (blasArgs->order == clblasColumnMajor)) ||
            ((blasArgs->transA != clblasNoTrans) && (blasArgs->order == clblasRowMajor))
        )
    {
        //
        // Estimate worst case Local Memory needed - Vector Width of 4 irrespective of data-type?
        //
        cl_ulong tw;

        tw = getTargetWidth(dim[0].y, MAXBLOCKSIZE, 4);
        if (tw == 0)
        {
            do {
                MAXBLOCKSIZE /= 2;
                tw = getTargetWidth(dim[0].y, MAXBLOCKSIZE, 4);
            } while((MAXBLOCKSIZE > 1) && (tw == 0));
        }
        #ifdef DEBUG_TRSV_GEMV
        printf("TRSV GEMV: isFitLDS() tw = %lu\n", tw);
        #endif
        maxSize = (1+4+tw)*dtypeSize(dtype) + MAXBLOCKSIZE*dtypeSize(dtype)*4;
        #ifdef DEBUG_TRSV_GEMV
        printf("TRSV GEMV: isFitLDS() maxSize = %lu, ldsSize = %lu, Y = %lu\n", maxSize, ldsSize, dim[0].y);
        #endif
        return (maxSize < ldsSize);
    }

    //
    // The remaining kernels use "TriangleWidth" amount of local memory for storing the RHS.
    // We will assume "dim[0].y" to be the "TriangleWidth"
    //
	MAXBLOCKSIZE = (dim[0].y)*(dim[0].y) > 256 ? 256 : dim[0].y*dim[0].y;
    maxSize = (dim[0].y + MAXBLOCKSIZE)*dtypeSize(dtype);
    return (maxSize < ldsSize);
}
