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
 * Cached global buffers based symm generator
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
#include <symm.clT>
#include <solution_seq.h>

//#define DEBUG_SYMM

extern "C"
unsigned int dtypeSize(DataType type);

static char Prefix[4];

static CLBLASMpatExtra mpatExtra;

extern "C"
unsigned int dtypeSize(DataType type);

static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra);

static void
assignKargs(KernelArg *args, const void *params, const void*);

/*
static void
calcNrThreads(
    size_t threads[2],
    const SubproblemDim *subdims,
    const PGranularity *pgran,
    const void *args,
    const void *extra);
	*/

static SolverFlags
solverFlags(void);

static void
setBuildOpts(
    char * buildOptStr,
	const void *kArgs);


static SolverOps symmSops = {
    generator,
    assignKargs,
    NULL, 				//isFitLDS?
    NULL,				//prepareTranslateDims?
    NULL,				//DecomAxis
    NULL, 				// calcNrThreads,
    NULL,				//ImagePackMode
    solverFlags, 		//SolverFlags
	NULL,
	NULL,
	NULL,
	setBuildOpts, 		//Set Build Options
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
        #ifdef DEBUG_TRMV
        printf("Setting build options ... Double... for DOUBLE PRECISION support\n");
        #endif
    }

	if (kargs->side == clblasLeft)
	{
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__SYMM_LEFT__ ");
	} else {
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__SYMM_RIGHT__");
	}

	if (kargs->uplo == clblasUpper)
	{
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__SYMM_UPPER__");
	} else {
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__SYMM_LOWER__");
	}

	if (kargs->order == clblasColumnMajor)
	{
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__SYMM_COLMAJOR__");
	} else {
		addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-D__SYMM_ROWMAJOR__");
	}

	strcat(buildOptStr, " -cl-mad-enable ");
	#ifdef DEBUG_SYMM
	printf("setBuildOptions: Setting to %s\n", buildOptStr);
	#endif
    return;
}

static SolverFlags
solverFlags(void)
{
	return (SF_WSPACE_1D);
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
	char itemx[10], itemy[10], width[10], itemy_by_width[10];
	size_t Y, X, BLOCKSIZE, ITEMX, ITEMY;

	if (buf == NULL)
	{
		buflen = 32*1024*sizeof(char);
        return (ssize_t)buflen;
	}

	//
	// Row-major is implemented in terms of column major routines
	//
	if ((kflags & KEXTRA_COLUMN_MAJOR) == 0)
	{
		return 0;
	}
	kprintf kobj(Prefix[dtype], kextra->vecLenA, true, true);

	BLOCKSIZE = pgran->wgSize[0];
	#ifdef DEBUG_SYMM
	printf("SYMM- generator(): Blocksize passed = %lu, subdimy = %lu, subdimx = %lu, veclen = %lu \n", BLOCKSIZE, subdims->y, subdims->x, kextra->vecLenA);
	#endif

	Y = 16;
	while (Y*(kextra->vecLenA) > subdims->y)
	{
		Y /= 2;
	}

	X = BLOCKSIZE/Y;
	ITEMY = (subdims->y) / Y;
	ITEMX = (subdims->x) / X;
	if (ITEMX == 0)
   	{
   		ITEMX = 1;
	}

	if ((BLOCKSIZE % Y) || ((subdims->y) % Y) || ((subdims->x)%X) || (ITEMY % kextra->vecLenA))
	{
		printf("WARNING: SYMM- generator: subdim and blocksize in-compatible.\n");
	}

	sprintf(width, "%" SPREFIX "u", Y);
	sprintf(itemy, "%" SPREFIX "u", ITEMY);
	sprintf(itemx, "%" SPREFIX "u", ITEMX);
	sprintf(itemy_by_width, "%" SPREFIX "u", (size_t) ITEMY/kextra->vecLenA);

	kobj.put("%WIDTH", width);
	kobj.put("%ITEMX", itemx);
	kobj.put("%ITEMY", itemy);
	kobj.put("%ITEMY_BY_V", itemy_by_width);
	#ifdef DEBUG_SYMM
	printf("ColMajor SYMM - WIDTH = %s, ITEMX = %s, ITEMY = %s\n", width, itemx, itemy);
	#endif

	strcpy(tempTemplate, SYMM_C_KERNEL);
	kobj.spit(buf, tempTemplate);
	#ifdef DEBUG_SYMM
   	printf("Kernel = \n%s\n", buf);
   	#endif
   	size_t tail = strlen(buf) + 1;
   	while(tail < 32*1024)
   	{
   		buf[tail++] = 0;
  	}
	return 32*1024*sizeof(char);
}

/*
__kernel void symm_C_kernel( __global %TYPE const * restrict _A, __global %TYPE const * restrict _B, __global %TYPE *C,\n\
							uint M, uint N, uint _lda, uint _ldb, int ldc, %TYPE alpha, %TYPE beta)
*/

static void
assignKargs(KernelArg *args, const void *params, const void*)
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;

	#ifdef DEBUG_SYMM
	printf("SAlpha=%f, DAlpha=%f, CAlpha =<%f, %f>, DAlpha=<%f, %f>\n",
			blasArgs->alpha.argFloat, blasArgs->alpha.argDouble, CREAL(blasArgs->alpha.argFloatComplex), CIMAG(blasArgs->alpha.argFloatComplex),
			CREAL(blasArgs->alpha.argDoubleComplex) , CIMAG(blasArgs->alpha.argDoubleComplex));
	printf("SBeta=%f, DBeta=%f, CBeta=<%f, %f>, DBeta=<%f, %f>\n",
			blasArgs->beta.argFloat, blasArgs->beta.argDouble, CREAL(blasArgs->beta.argFloatComplex), CIMAG(blasArgs->beta.argFloatComplex),
			CREAL(blasArgs->beta.argDoubleComplex) , CIMAG(blasArgs->beta.argDoubleComplex));
	#endif

    INIT_KARG(&args[0], blasArgs->A);   //A - input matrix - argument
	INIT_KARG(&args[1], blasArgs->B);
	INIT_KARG(&args[2], blasArgs->C);
	initSizeKarg(&args[3], blasArgs->M);
	initSizeKarg(&args[4], blasArgs->N);
	initSizeKarg(&args[5], blasArgs->lda.matrix);
	initSizeKarg(&args[6], blasArgs->ldb.matrix);
	initSizeKarg(&args[7], blasArgs->ldc.matrix);
	initSizeKarg(&args[8], blasArgs->offa); //PENDING: offA or offa ??
	initSizeKarg(&args[9], blasArgs->offBX);
	initSizeKarg(&args[10], blasArgs->offCY);
	assignScalarKarg(&args[11], &(blasArgs->alpha), blasArgs->dtype);
	assignScalarKarg(&args[12], &(blasArgs->beta), blasArgs->dtype);
	return;
}

extern "C"
void
initSymmDefaultPattern(MemoryPattern *mempat)
{
    mempat->name = "Cached global memory based block Symm";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &symmSops;

    mpatExtra.aMset = CLMEM_LEVEL_L1;
    mpatExtra.bMset = CLMEM_LEVEL_L1;
    mpatExtra.mobjA = CLMEM_BUFFER;
    mpatExtra.mobjB = CLMEM_BUFFER;
    mempat->extra = &mpatExtra;

    Prefix[TYPE_FLOAT] = 'S';
    Prefix[TYPE_DOUBLE] = 'D';
    Prefix[TYPE_COMPLEX_FLOAT] = 'C';
    Prefix[TYPE_COMPLEX_DOUBLE] = 'Z';
	return;
}

