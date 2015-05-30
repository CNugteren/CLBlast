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
 * Cached global buffers based gemm generator
 */

#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <clblas_stddef.h>
#include <clBLAS.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>
#include <kprintf.hpp>
#include <gemm.clT>
#include <symm_helper.clT>
#include <solution_seq.h>

extern "C" int
gemmHasNTail(size_t N, int vecLen, clblasOrder order, clblasTranspose transA, clblasTranspose transB);

extern "C" int
gemmHasMTail(size_t M, int vecLen, clblasOrder order, clblasTranspose transA, clblasTranspose transB);


//#define DEBUG_GEMM_TAIL
static CLBLASMpatExtra mpatExtra;

static char Prefix[4];

static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra);

static void
assignKargs(KernelArg *args, const void *params, const void *extra);

static SolverFlags
solverFlags(void);

static void
setBuildOpts(
    char * buildOptStr,
    const void *args);

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra);

static SolverOps gemmSops = {
    generator,
    assignKargs,
    NULL,
    NULL,
   	NULL,
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
    KernelExtraFlags kflags = step->extraFlags;

	addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DTAIL_RUN -DM_TAIL_PRESENT -DN_TAIL_PRESENT");
    if ( kargs->dtype == TYPE_DOUBLE || kargs->dtype == TYPE_COMPLEX_DOUBLE)
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DDOUBLE_PRECISION");
        #ifdef DEBUG_GEMM_TAIL
        printf("Setting build options ... Double... for DOUBLE PRECISION support\n");
        #endif
    }

    if (isComplexType(kargs->dtype))
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DCOMPLEX");
    }

    if (kflags & KEXTRA_CONJUGATE_A)
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DCONJUGATE_A");
    }
    if (kflags & KEXTRA_CONJUGATE_B)
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DCONJUGATE_B");
    }


    switch(kargs->pigFuncID)
    {
        case CLBLAS_GEMM2:
        case CLBLAS_GEMM_TAIL:
            break;

        case CLBLAS_HERK:
            addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DHERK");
            if(kargs->uplo == clblasLower)
            {
                addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DHERK_LOWER_TRIANGLE");
            }
            else if(kargs->uplo == clblasUpper)
            {
                addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DHERK_UPPER_TRIANGLE");
            }
            break;

        case CLBLAS_HEMM:
        case CLBLAS_SYMM_DIAGONAL:
        case CLBLAS_HEMM_DIAGONAL:
        case CLBLAS_SYMM:
            #ifdef DEBUG_GEMM_2
            printf("GEMM2: setBuildOpts: Setting options for SYMM\n");
            #endif
            if (kargs->side == clblasLeft)
            {
                addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__SYMM_LEFT__");
            }
            if (kargs->side == clblasRight)
            {
                addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__SYMM_RIGHT__");
            }
            if (kargs->uplo == clblasLower)
            {
                addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__SYMM_LOWER__");
            }
            if (kargs->uplo == clblasUpper)
            {
                addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__SYMM_UPPER__");
            }
            // Define the order for Legacy sake.
            if (kargs->order == clblasColumnMajor)
            {
                addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__SYMM_COLMAJOR__");
            } else {
                addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__SYMM_ROWMAJOR__");
            }
            if ((kargs->pigFuncID == CLBLAS_SYMM_DIAGONAL) || (kargs->pigFuncID == CLBLAS_HEMM_DIAGONAL))
            {
                addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__SYMM_DIAGONAL__");
            }
            if (kargs->pigFuncID == CLBLAS_HEMM_DIAGONAL)
            {
                addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__HEMM__");
            }
            break;

        default:
            printf("GEMM TAIL: Unknown pigFuncID\n");
            break;
    }
    #ifdef DEBUG_GEMM_TAIL
    printf("GEMMTAIL: Build options = %s\n", buildOptStr);
    #endif
}

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra)
{
    int BLOCKSIZE = pgran->wgSize[0]; // 1D Block
	size_t tailM, tailN, M, N;
	size_t Y, X;
	size_t nWorkGroupsAY, nWorkGroupsAX, nWorkGroupsA;
	size_t nWorkGroupsBY, nWorkGroupsBX, nWorkGroupsB;
	size_t totalWorkGroups;
    #ifdef DEBUG_GEMM_TAIL
    printf("calcNrThreads called from gemm_tail.cpp\n");
    #endif
    const CLBlasKargs *kargs = (const CLBlasKargs *)args;
    const CLBLASKernExtra *kextra = ( CLBLASKernExtra *)extra;
	KernelExtraFlags kflags = kextra->flags;

	//
	// RowMajor GEMM can be expressed in terms of Column Major GEMM
	//
    if ((kflags & KEXTRA_COLUMN_MAJOR) == 0)
    {
    	printf("calcNrThreads: FIXME: RowMajor is NOT supported \n");
        return;
    }

	if (kextra->vecLenA != 1)
	{
    	printf("GEMM_TAIL: calcNrThreads(): Vector Length must be 1 for TAIL. Non-one Vector Length Requested\n");
		return;
	}

	tailM = kargs->tailStartM;
	tailN = kargs->tailStartN;
	M = kargs->M;
	N = kargs->N;

    Y = 8;
    if (Y != subdims->y)
	{
		Y = subdims->y;
	}
    X = BLOCKSIZE/Y;
    /*
    LEGACY CODE: Outdated now. TAIL can handle this condition now using MTAIL_PRESENT and NTAIL_PRESENT
	if (tailN % X)
	{
		printf("GEMM_TAIL: calcNrThreads(): WARNING: tailN is not divisible by X. Will produce Wrong results!\n");
	}
    */

	//
	// A Tail Workgroup will process YxX panel
	//
	/*
			 ______________
			|			|  |
			|			|  |
			|			|  | B Tail panel
			|___________|  |
			|___________|__|
		    <---  A   -->
	 */
	if(tailM != M)
	{
		#ifdef DEBUG_GEMM_TAIL
		printf("GEMM_TAIL: M has TAIL\n");
		#endif
		nWorkGroupsAY = ((M - tailM -1)/Y + 1);
		nWorkGroupsAX = ((tailN - 1)/X + 1);
		nWorkGroupsA = nWorkGroupsAY * nWorkGroupsAX;
	} else {
		nWorkGroupsA = 0;
	}

	if (tailN != N)
	{
		#ifdef DEBUG_GEMM_TAIL
		printf("GEMM_TAIL: N has TAIL\n");
		#endif
		nWorkGroupsBY = ((M-1)/Y) + 1;
		nWorkGroupsBX = ((N-tailN-1)/X) + 1;
		nWorkGroupsB = nWorkGroupsBY * nWorkGroupsBX;
	} else {
		nWorkGroupsB = 0;
	}

	totalWorkGroups = nWorkGroupsA + nWorkGroupsB;

	threads[0] = totalWorkGroups * BLOCKSIZE;
	threads[1] = 1;
	#ifdef DEBUG_GEMM_TAIL
	printf("GEMM_TAIL: calcNrThreads(): vlen:%d, <tailM:%lu, M:%lu>, <tailN:%lu, N:%lu, nWorkGroupsA<%lu,%lu>, nWorkGroupsB<%lu,%lu>\n",
			kextra->vecLenA, tailM, M, tailN, N, nWorkGroupsAY, nWorkGroupsAX, nWorkGroupsBY, nWorkGroupsBX);
	printf("GEMM_TAIL: calcNrThreads(): globalThreads0=%lu, globalThreads1=%lu\n", threads[0], threads[1]);
	#endif
	return;
}

static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra)
{
    CLBLASKernExtra *kextra = (CLBLASKernExtra*)extra;
    KernelExtraFlags kflags = kextra->flags;
    DataType dtype = kextra->dtype;
    char tempTemplate[32*1024];
    char itemx[10], itemy[10], width[10], itemy_by_width[10], itemx_by_width[10];
    size_t Y, X, BLOCKSIZE, ITEMX, ITEMY;

    if (buf == NULL)
    {
        buflen = 32*1024*sizeof(char);
        return (ssize_t)buflen;
    }

    //
    // PENDING: Add Support for Row Major at the xAPI.c level
	// Row major calcs can be expressed in terms of column major
    //
    if ((kflags & KEXTRA_COLUMN_MAJOR) == 0)
    {
        return 0;
    }

    kprintf kobj(Prefix[dtype], 1, false, false); // Only Scalar Access

    BLOCKSIZE = pgran->wgSize[0];
    #ifdef DEBUG_GEMM_TAIL
    printf("GEMM- generator(): Blocksize passed = %lu, subdimy = %lu, subdimx = %lu, veclen = %d \n", BLOCKSIZE, subdims->y, subdims->x, kextra->vecLenA);
    #endif

    Y = 8;
    if (Y != subdims->y)
	{
		//printf("GEMM_TAIL: generator(): WARNING: subdims->y is un-suitable.\n");
		Y = subdims->y;
	}
    X = BLOCKSIZE/Y;
    ITEMY = (subdims->y) / Y;
    ITEMX = (subdims->x) / X;
    if (ITEMX == 0)
    {
        ITEMX = 1;
    }

    if ((BLOCKSIZE % Y) || ((subdims->y) % Y) || ((subdims->x)%X) || (ITEMY % kextra->vecLenA) || ((X*ITEMX) % kextra->vecLenA))
    {
        printf("WARNING: GEMM TAIL - generator: subdim and blocksize in-compatible. This code should never execute!\n");
    }

    sprintf(width, "%lu", Y);
    sprintf(itemy, "%lu", ITEMY);
    sprintf(itemx, "%lu", ITEMX);
    sprintf(itemy_by_width, "%lu", (size_t) ITEMY/kextra->vecLenA);
    sprintf(itemx_by_width, "%lu", (size_t) ITEMX/kextra->vecLenA);

    kobj.put("%WIDTH", width);
    kobj.put("%ITEMX", itemx);
    kobj.put("%ITEMY", itemy);
    kobj.put("%ITEMY_BY_V", itemy_by_width);
    kobj.put("%ITEMX_BY_V", itemx_by_width);
    kobj.put("%PANEL", "1");
    kobj.put("%PANEL_BY_V", "1");
    #ifdef DEBUG_GEMM_TAIL
    printf("ColMajor GEMM - WIDTH = %s, ITEMX = %s, ITEMY = %s\n", width, itemx, itemy);
    #endif

    strcpy(tempTemplate, SYMM_HEMM_HELPER);
    if ((kflags & KEXTRA_TRANS_A) == 0)
    {
        if (kflags & KEXTRA_TRANS_B)
        {
			#ifdef DEBUG_GEMM_TAIL
			printf("GEMM_TAIL: Using GEMM_NT_KERNEL\n");
			#endif
            strcat(tempTemplate, GEMM_NT_KERNEL);
        } else {
			#ifdef DEBUG_GEMM_TAIL
			printf("GEMM_TAIL: Using GEMM_NN_KERNEL\n");
			#endif
            strcat(tempTemplate, GEMM_NN_KERNEL);
		}
    } else {
        //
        // NOTE: A^T * B Never leaves any tails. This should NEVER be called.
        // PENDING: A^T * B^T support is PENDING
        tempTemplate[0] = 0;
    }

    kobj.spit(buf, tempTemplate);
    //#ifdef DEBUG_GEMM_TAIL
    //printf("Kernel = \n%s\n", buf);
    //#endif
    size_t tail = strlen(buf) + 1;
    while(tail < 32*1024)
    {
        buf[tail++] = 0;
    }
    return 32*1024*sizeof(char);
}

static void
assignKargs(KernelArg *args, const void *params, const void*)
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;

    #ifdef DEBUG_GEMM_TAIL
    printf("SAlpha=%f, DAlpha=%f, CAlpha =<%f, %f>, DAlpha=<%f, %f>\n",
            blasArgs->alpha.argFloat, blasArgs->alpha.argDouble, CREAL(blasArgs->alpha.argFloatComplex), CIMAG(blasArgs->alpha.argFloatComplex),
            CREAL(blasArgs->alpha.argDoubleComplex) , CIMAG(blasArgs->alpha.argDoubleComplex));
    printf("SBeta=%f, DBeta=%f, CBeta=<%f, %f>, DBeta=<%f, %f>\n",
            blasArgs->beta.argFloat, blasArgs->beta.argDouble, CREAL(blasArgs->beta.argFloatComplex), CIMAG(blasArgs->beta.argFloatComplex),
            CREAL(blasArgs->beta.argDoubleComplex) , CIMAG(blasArgs->beta.argDoubleComplex));
	printf("TailStartM = %lu, TailStartN = %lu\n", blasArgs->tailStartM, blasArgs->tailStartN);
    #endif

    INIT_KARG(&args[0], blasArgs->A);   //A - input matrix - argument
    INIT_KARG(&args[1], blasArgs->B);   //x - result buffer = _xnew argument
    INIT_KARG(&args[2], blasArgs->C);   //y - scratch == _x_vector argument
    initSizeKarg(&args[3], blasArgs->M);
    initSizeKarg(&args[4], blasArgs->N);
    initSizeKarg(&args[5], blasArgs->K);
    initSizeKarg(&args[6], blasArgs->lda.matrix);
    initSizeKarg(&args[7], blasArgs->ldb.matrix);
    initSizeKarg(&args[8], blasArgs->ldc.matrix);
    initSizeKarg(&args[9], blasArgs->offA);
    initSizeKarg(&args[10], blasArgs->offBX);
    initSizeKarg(&args[11], blasArgs->offCY);
    assignScalarKarg(&args[12], &(blasArgs->alpha), blasArgs->dtype);
    assignScalarKarg(&args[13], &(blasArgs->beta), blasArgs->dtype);
    initSizeKarg(&args[14], blasArgs->tailStartM);
    initSizeKarg(&args[15], blasArgs->tailStartN);
    return;
}

static SolverFlags
solverFlags(void)
{
    return (SF_WSPACE_1D);
}

extern "C"
void
initGemmV2TailCachedPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based gemm tail";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &gemmSops;

    mpatExtra.aMset = CLMEM_LEVEL_L1;
    mpatExtra.bMset = CLMEM_LEVEL_L1;
    mpatExtra.mobjA = CLMEM_BUFFER;
    mpatExtra.mobjB = CLMEM_BUFFER;
    mempat->extra = &mpatExtra;


    Prefix[TYPE_FLOAT] = 'S';
    Prefix[TYPE_DOUBLE] = 'D';
    Prefix[TYPE_COMPLEX_FLOAT] = 'C';
    Prefix[TYPE_COMPLEX_DOUBLE] = 'Z';
}

