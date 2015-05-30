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
#include <gemm_helper.clT>
#include <symm_helper.clT>
#include <solution_seq.h>
#include "tuned_numbers.h"

//#define DEBUG_GEMM_2
static CLBLASMpatExtra mpatExtra;

static char Prefix[4];

/* Function, finding default decomposition */
static int
getDefaultDecomposition(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void *pArgs);

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
	getDefaultDecomposition,
	NULL,
	setBuildOpts,
	NULL
};

static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra)
{
    const CLBlasKargs *kargs = (const CLBlasKargs *)args;
    //const CLBLASKernExtra *kextra = ( CLBLASKernExtra *)extra;
    //KernelExtraFlags kflags = kextra->flags;
    size_t M, N;

    M = kargs->M;
    N = kargs->N;

    threads[1] = 1;

    if ((subdims->x != SUBDIM_UNUSED) &&
        (subdims->y != SUBDIM_UNUSED)) {

        size_t groupWorkX, groupWorkY;
        size_t nrGroupsX, nrGroupsY;
        int nrDims;

        groupWorkX = subdims->x;
        groupWorkY = subdims->y;

        nrGroupsX = N / groupWorkX;
        if (N % groupWorkX) {
            nrGroupsX++;
        }

        nrGroupsY = M / groupWorkY;
        if (M % groupWorkY) {
            nrGroupsY++;
        }
        nrDims = (pgran == NULL) ? 1 : pgran->wgDim;
        threads[0] = nrGroupsX * nrGroupsY;

        if(kargs->pigFuncID == CLBLAS_HERK)
        {
            threads[0] = (nrGroupsY * (nrGroupsY + 1)) / 2;
        }

    }

    if (pgran != NULL) {
        threads[0] *= pgran->wgSize[0];
        threads[1] *= pgran->wgSize[1];
    }
}

static void
setBuildOpts(
    char * buildOptStr,
    const void *args)
{
	SolutionStep *step = (SolutionStep *)args;
    const CLBlasKargs *kargs = (const CLBlasKargs *)(&step->args);
	const SubproblemDim *dims = step->subdims;
	//size_t vecLen = sizeof(cl_float4)/dtypeSize(kargs->dtype);
    KernelExtraFlags kflags = step->extraFlags;

    blockSizes bestSize = bestBlockSizeForDevice( step );

    if ( kargs->dtype == TYPE_DOUBLE || kargs->dtype == TYPE_COMPLEX_DOUBLE)
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DDOUBLE_PRECISION");
    }

    if (isComplexType(kargs->dtype))
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DCOMPLEX");
    }

    if ((bestSize.useBarrier) == 1)
    {
	    addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DGEMM_NEEDS_BARRIER");
    }

    if (kargs->M % dims->y)
	{
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DM_TAIL_PRESENT");
    }

	if (kargs->N % dims->x)
	{
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DN_TAIL_PRESENT");
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
        case CLBLAS_HEMM:
        case CLBLAS_SYMM:
        case CLBLAS_SYMM_DIAGONAL:
        case CLBLAS_HEMM_DIAGONAL:
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

         default:
            break;
    }

    #ifdef DEBUG_GEMM_2
	printf("buildStr: %s\n", buildOptStr);
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
    char tempTemplate[64*1024]; //PENDING: Is it safe to have 64K in stack for threadSafety?
    char itemx[10], itemy[10], width[10], itemy_by_width[10], itemx_by_width[10];
    char bwidth[10], panel_by_v[10];
    size_t Y, X, BLOCKSIZE, ITEMX, ITEMY;
	bool doVLOAD = false;
	unsigned int veclen;

    if (buf == NULL)
    {
        buflen = 64*1024*sizeof(char);
        return (ssize_t)buflen;
    }

    //
    // PENDING: Add Support for Row Major
    //
    if ((kflags & KEXTRA_COLUMN_MAJOR) == 0)
    {
        return 0;
    }

	if ((kflags & KEXTRA_NO_COPY_VEC_A) || (kflags & KEXTRA_NO_COPY_VEC_B) || (kflags  & KEXTRA_NO_COPY_VEC_C))
	{
		#ifdef DEBUG_GEMM_2
		printf("GEMM2: Doing un-aligned access\n");
		#endif
		doVLOAD= true;
	} else {
		#ifdef DEBUG_GEMM_2
		printf("GEMM2: Doing Aligned access\n");
		#endif
	}


    BLOCKSIZE = pgran->wgSize[0];
    #ifdef DEBUG_GEMM_2
    printf("GEMM2- generator(): Blocksize passed = %lu, subdimy = %lu, subdimx = %lu, veclen = %d \n",
                                BLOCKSIZE, subdims->y, subdims->x, kextra->vecLen);
    #endif

	veclen = kextra->vecLen;

    ITEMY = subdims->itemY;
    ITEMX = subdims->itemX;
    Y = subdims->y / ITEMY;
    X = subdims->x / ITEMX;

	//
	// Handle in-compatible subdims and workgroup sizes
	// We will use "veclen" of 1 as our shield against these in-compatible
    // geometries.
	//
    if ( (ITEMY % kextra->vecLen) || ((ITEMX % kextra->vecLen) && (kflags & KEXTRA_TRANS_B)) )
    {
        //
        // FIXME:
        // This kernel must be stored against vecLen of 1 in Kernel Cache.
        // This needs change in EXTRA structure. However, this is against the API.
        // We are going against the API by changing fields in EXTRA structure.
        // One alternate FIX is to return an error.
        //
        kextra->vecLen = kextra->vecLenA = kextra->vecLenB = kextra->vecLenC = 1;

       	doVLOAD = true;
		veclen = 1;
    }

	//
	// PENDING: Selective Vectorization for A, B and C access has to be added
	// 			in KPRINTF module (VLOADA, VLOADB, VLOADC, VSTOREC)
	//
    kprintf kobj(Prefix[dtype], veclen, doVLOAD, doVLOAD); // Only Vectored Access
    sprintf(width, "%lu", Y);
    sprintf(itemy, "%lu", ITEMY);
    sprintf(itemx, "%lu", ITEMX);
    sprintf(itemy_by_width, "%lu", (size_t) ITEMY/veclen);
    sprintf(itemx_by_width, "%lu", (size_t) ITEMX/veclen);
    //sprintf(bwidth, "%lu", subdims->bwidth);
    //sprintf(panel_by_v, "%lu", (subdims->bwidth / veclen));
    sprintf(bwidth, "%lu", (size_t) veclen);
    sprintf(panel_by_v, "%lu", (size_t) 1);

    kobj.put("%WIDTH", width);
    kobj.put("%ITEMX", itemx);
    kobj.put("%ITEMY", itemy);
    kobj.put("%ITEMY_BY_V", itemy_by_width);
    kobj.put("%ITEMX_BY_V", itemx_by_width);
    kobj.put("%PANEL", bwidth);
    kobj.put("%PANEL_BY_V", panel_by_v);
    #ifdef DEBUG_GEMM_2
    printf("ColMajor GEMM - WIDTH = %s, PANEL = %lu, ITEMX = %s, ITEMY = %s, Veclen = %lu\n", width, subdims->bwidth, itemx, itemy, veclen);
    #endif

    strcpy(tempTemplate, SYMM_HEMM_HELPER);
	if ((kflags & KEXTRA_TRANS_A) == 0)
	{
		if (kflags & KEXTRA_TRANS_B)
		{
			#ifdef DEBUG_GEMM_2
			printf("Using GEMM_NT_KERNEL\n");
			#endif
    		strcat(tempTemplate, GEMM_HELPER);
            strcat(tempTemplate, GEMM_NT_KERNEL);
		} else {
			#ifdef DEBUG_GEMM_2
			printf("Using GEMM_NN_KERNEL\n");
			#endif
    		strcat(tempTemplate, GEMM_HELPER);
    		strcat(tempTemplate, GEMM_NN_KERNEL);
		}
	} else {
		// PENDING:
		if (kflags & KEXTRA_TRANS_B)
		{
		    tempTemplate[0] = 0;
		} else {
			#ifdef DEBUG_GEMM_2
			printf("Using GEMM_TN_KERNEL\n");
			#endif
    		strcat(tempTemplate, GEMM_HELPER);
    		strcat(tempTemplate, GEMM_TN_KERNEL);
	    }
	}
    kobj.spit(buf, tempTemplate);
    #ifdef DEBUG_GEMM_KPRINTF
    printf("Kernel = \n%s\n", buf);
    #endif
    size_t tail = strlen(buf) + 1;
    while(tail < 64*1024)
    {
        buf[tail++] = 0;
    }
    return 64*1024*sizeof(char);
}

static void
assignKargs(KernelArg *args, const void *params, const void*)
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;

    #ifdef DEBUG_GEMM_2
    printf("SAlpha=%f, DAlpha=%f, CAlpha =<%f, %f>, DAlpha=<%f, %f>\n",
            blasArgs->alpha.argFloat, blasArgs->alpha.argDouble, CREAL(blasArgs->alpha.argFloatComplex), CIMAG(blasArgs->alpha.argFloatComplex),
            CREAL(blasArgs->alpha.argDoubleComplex) , CIMAG(blasArgs->alpha.argDoubleComplex));
    printf("SBeta=%f, DBeta=%f, CBeta=<%f, %f>, DBeta=<%f, %f>\n",
            blasArgs->beta.argFloat, blasArgs->beta.argDouble, CREAL(blasArgs->beta.argFloatComplex), CIMAG(blasArgs->beta.argFloatComplex),
            CREAL(blasArgs->beta.argDoubleComplex) , CIMAG(blasArgs->beta.argDoubleComplex));
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
    return;
}

static SolverFlags
solverFlags(void)
{
    return (SF_WSPACE_1D);
}

extern "C"
void
initGemmV2CachedPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based block gemm";
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

static int
getDefaultDecomposition(
    PGranularity *pgran,
    SubproblemDim *subdims,
    unsigned int subdimsNum,
    void *pArgs)
{

    DUMMY_ARG_USAGE(pArgs);
    //
    // FIXME:  container_of() - Counts on the fact that "getDefaultDecomposition" is called
    //          with step->pgran, step->subdims
    //
    SolutionStep *step = container_of( pgran , pgran, SolutionStep);

    blockSizes bestSize = bestBlockSizeForDevice( step );

    pgran->wgSize[0] = bestSize.TY * bestSize.TX;
    pgran->wgSize[1] = 1;
    pgran->wgDim = 1;

    if (subdimsNum >= 1)
    {
        subdims[0].y = bestSize.TY * bestSize.ITEMY;
        subdims[0].x = bestSize.TX * bestSize.ITEMX;
        subdims[0].itemY = bestSize.ITEMY;
        subdims[0].itemX = bestSize.ITEMX;
        subdims[0].bwidth = 4;
    }
    if (subdimsNum >= 2)
    {
        subdims[1].y = bestSize.TY * bestSize.ITEMY;
        subdims[1].x = bestSize.TX * bestSize.ITEMX;
        subdims[1].itemY = bestSize.ITEMY;
        subdims[1].itemX = bestSize.ITEMX;
        subdims[1].bwidth = 4;
    }

    return 0;
}

