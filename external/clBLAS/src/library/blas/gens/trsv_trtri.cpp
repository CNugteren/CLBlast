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
 * trsv trtri generator -
 *
 * This kernel solves the triangular system of equations with only 1 work-group.
 * This is terribly slow and forms the weakest link in the chain.
 * It solves 1 variable per work-item. So, the size of the triangle that can be solved
 * is limited by the hardware's MAX_WORKGROUP_SIZE.
 * The "chain" for solving larger systems of equations involve a "gemv" operation
 * which can be exploited by "xtrsv.c". However, the current "gemv" implementation
 * does NOT support "single complex" and "double complex" data types.
 * So, to give complete support, another "trsv_gemv" generator will be used.
 */

#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <clblas_stddef.h>
#include <clBLAS.h>
#include <blas_mempat.h>
#include <clkern.h>
#include <clblas-internal.h>
#include <trsv.clT>
#include <solution_seq.h>
//#include "blas_kgen.h"

#include <kprintf.hpp>

//#define DEBUG_TRSV_TRTRI

extern "C"
unsigned int dtypeSize(DataType type);


static char Prefix[4]; // PENDING: Magic "4" == Number of data types supported (float, double, cl_float2, cl_double2)


static SolverFlags
solverFlags(void)
{
    #ifdef DEBUG_TRSV_TRTRI
    printf("TRSV TRTRI solverFlags(): solverFlags callen......\n");
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
assignKargs(KernelArg *args, const void *params, const void*);

extern "C"
void initTrsvDefaultPattern(MemoryPattern *mempat);

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

static ssize_t
generator_tbsv(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra);

static SolverOps trsvOps = {
    generator,
    assignKargs,
    isFitToLDS,
    NULL, // Prepare Translate Dims
    NULL, // Inner Decomposition Axis
    calcNrThreads,
    NULL, // Image related
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
        #ifdef DEBUG_TRSV_TRTRI
        printf("TRSV TRTRI: Setting build options ... Double... for DOUBLE PRECISION support\n");
        #endif
    }
    if( kargs->pigFuncID == CLBLAS_TPSV)
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DPACKED");
        #ifdef DEBUG_TRSV_TRTRI
            printf("TPSV TRTRI: Setting build options ... PACKED\n");
        #endif
    }
    if( kargs->pigFuncID == CLBLAS_TBSV)
    {
        addBuildOpt( buildOptStr, BUILD_OPTS_MAXLEN, "-DBANDED");
        #ifdef DEBUG_TRSV_TRTRI
        printf("TBSV TRTRI: Setting build options .. BANDED\n");
        #endif
    }
    return;
}

static CLBLASMpatExtra mpatExtra;

extern "C"
void initTrsvDefaultPattern(MemoryPattern *mempat)
{
    #ifdef DEBUG_TRSV_TRTRI
    printf("TRSV TRTRI: initTRSVDefaultPattern called with mempat = 0x%p\n", (void*)mempat);
    #endif

    mempat->name = "Triangular matrix solver - Only 1 workgroup";
    mempat->nrLevels = 2;
    mempat->cuLevel = 0;
    mempat->thLevel = 1;
    mempat->sops = &trsvOps;

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

//
// Read comments atop "isFitToLDS()"
// This function is required by "isFitLDS()"
//
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
    #ifdef DEBUG_TRSV_TRTRI
    printf("TRSV TRTRI: calcNrThreads() called \n");
    #endif
    int blocks = 1;

    _extra = _extra; // Dummy- to avoid warnings

    #ifdef DEBUG_TRSV_TRTRI
    printf("blocks : %d\n", blocks);
    #endif

    if (((kargs->order == clblasColumnMajor) && (kargs->transA == clblasNoTrans)) ||
       ((kargs->order == clblasRowMajor) && (kargs->transA != clblasNoTrans)))
     {
        if (subdims->y > BLOCKSIZE)
        {
            // These little kernels cannot handle arbitrary numbers
            printf("TRSV calcNrThreads(): Warning. TRTRI Cannot handle subproblemdim of size %lu\n", subdims->y);
            threads[0] = 0;
            threads[1] = 0;
            return;
        }
    } else {
        if (subdims->y > 1024)
        {
            // These little kernels cannot handle arbitrary numbers
            printf("TRSV calcNrThreads(): Warning. TRTRI Cannot handle subproblemdim of size %lu\n", subdims->y);
            threads[0] = 0;
            threads[1] = 0;
            return;
        }
    }

    threads[0] = blocks * BLOCKSIZE;
    threads[1] = 1;
    #ifdef DEBUG_TRSV_TRTRI
    printf("pgran-wgSize[0] : %d, globalthreads[0]  : %lu\n", pgran->wgSize[0], threads[0]);
    #endif
    return;
}

//
// FIXME: Report correct return value when "buf" is NULL - Needs change in KPRINTF
// FIXME: Return correct return value - Needs change in KPRINTF
//
static ssize_t
generator(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra)
{
    char tempTemplate[32*1024];
    char vector_size_trans[10], triangle_height[10];

    pgran = pgran; // Dummy- to avoid warnings

    if (buf == NULL) // PENDING: Return correct buffer size
    {
        buflen = (32 * 1024 * sizeof(char));
        return (ssize_t)buflen;
    }

    CLBLASKernExtra *extraFlags = ( CLBLASKernExtra *)extra;
    SolutionStep *step = container_of( pgran , pgran, SolutionStep);    // NOTE: using container_of() to get pigFuncID
    CLBlasKargs* kargs = (CLBlasKargs*) &(step->args);

    if(kargs->pigFuncID == CLBLAS_TBSV)
    {
        return generator_tbsv(buf, buflen, subdims, pgran, extra);
    }

    #ifdef DEBUG_TRSV_TRTRI
     printf("TRSV GENERATOR called....\n");

    if((( extraFlags->flags &  KEXTRA_TRANS_A) || ( extraFlags ->flags & KEXTRA_CONJUGATE_A )))
    {
        printf("A is trans or CONJ-TRANS\n");
    }
    else
    {
        printf("A is noTrans...\n");
    }
    #endif

    clblasUplo uplo   = ( extraFlags->flags & KEXTRA_UPPER_TRIANG) ? clblasUpper : clblasLower;
    clblasOrder order = ( extraFlags->flags & KEXTRA_COLUMN_MAJOR) ? clblasColumnMajor: clblasRowMajor;
    clblasTranspose trans = ( extraFlags->flags & KEXTRA_TRANS_A) ? clblasTrans : (( extraFlags->flags & KEXTRA_CONJUGATE_A) ? clblasConjTrans: clblasNoTrans);
    //bool unit = (((extraFlags->flags) & KEXTRA_UNIT_DIAGONAL) != 0);

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

    if ( trans == clblasNoTrans)
    {
        ( uplo == clblasLower )?
                    (strcpy(tempTemplate, (char*)trsv_CL_SolveTriangle_kernel)) :
                    (strcpy(tempTemplate, (char*)trsv_CU_SolveTriangle_kernel));
    }
    else // Transpose cases...
    {
        ( uplo == clblasLower )?
                    (strcpy(tempTemplate, (char*)trsv_CLT_SolveTriangle_kernel)) :
                    (strcpy(tempTemplate, (char*)trsv_CUT_SolveTriangle_kernel));
    }

    #ifdef DEBUG_TRSV_TRTRI
    printf("dataType : %c\n", Prefix[extraFlags->dtype]);
    #endif

    // FIXME: VECTORSIZE HARD CODED
    // FIXME : SetKernelArgs.. sends offa, offx, and lda should be received as uint
    unsigned int vecLenA = extraFlags->vecLenA;

    #ifdef DEBUG_TRSV_TRTRI
    printf("Vector length used : %d\n\n", vecLenA);
    #endif

    bool doVLOAD = false;
    if( extraFlags->flags &  KEXTRA_NO_COPY_VEC_A )
    {
        doVLOAD = true;
        #ifdef DEBUG_TRSV_TRTRI
            printf("DOing VLOAD as Aligned Data Pointer not Availabe\n");
        #endif
    }
    else
    {
        #ifdef DEBUG_TRSV_TRTRI
            printf("Using Aligned Data Pointer .........................\n");
        #endif
    }
    kprintf kobj( Prefix[extraFlags->dtype], vecLenA, doVLOAD);

    if (trans != clblasNoTrans)
    {
        sprintf( vector_size_trans, "%u", vecLenA );
        sprintf( triangle_height, "%ld", subdims[0].y );
        #ifdef DEBUG_TRSV_TRTRI
        printf("vector size trans = %s\n", vector_size_trans);
        #endif
        kobj.put("%PREFIXVECTOR_SIZE_TRANS", (const char *)vector_size_trans);
        kobj.put("%TRIANGLE_HEIGHT", triangle_height);
    }
    kobj.spit((char*)buf, tempTemplate);
    return (32 * 1024 * sizeof(char));
}

static void
assignKargs(KernelArg *args, const void *params, const void*)
{
    CLBlasKargs *blasArgs = (CLBlasKargs*)params;
    cl_int inc;
    cl_int unity, doConj;

    INIT_KARG(&args[0], blasArgs->A);     //A - input matrix - argument
    INIT_KARG(&args[1], blasArgs->B);     //x - result buffer = _xnew argument
    initSizeKarg(&args[2], blasArgs->N);
    inc = blasArgs->ldb.vector;
    INIT_KARG(&args[3], inc);
    unity = (blasArgs->diag == clblasUnit);
    INIT_KARG(&args[4], unity);
    initSizeKarg(&args[5], blasArgs->lda.matrix);
    doConj = (blasArgs->transA == clblasConjTrans);
    #ifdef DEBUG_TRSV_TRTRI
    printf("TRMV TRTRI: assignKargs: doConj is : %d, unity is : %d, incx is : %d\n", doConj, unity, inc);
    printf("TRMV TRTRI: startRow, startCol set to %d, %d\n", blasArgs->startRow, blasArgs->endRow);
    #endif
    INIT_KARG(&args[6], doConj);
    INIT_KARG(&args[7], blasArgs->startRow);
    INIT_KARG(&args[8], blasArgs->endRow);
    initSizeKarg(&args[9], blasArgs->offa);
    initSizeKarg(&args[10], blasArgs->offBX);

    if( blasArgs->pigFuncID == CLBLAS_TBSV)
    {
        initSizeKarg(&args[11], blasArgs->K);
    }
    return;
}

/*
 * isFitToLDS() is based on the "trsv_gemv" counterpart than the kernel corresponding to TRTRI
 * The Kernels corersponding to TRTRI are run with only 1 Workgroup.
 * So, it really does not matter at all.
 * But, if dim[0].y selected by the library changes between TRTRI and TRSV_GEMV, results will go
 * wrong. So, by using the same "isFitToLDS" function, we will indirectly force the library to
 * choose the same "SubproblemDim" for both cases.
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
        #ifdef DEBUG_TRSV_TRTRI
        printf("TRSV TRTRI: isFitLDS() tw = %lu\n", tw);
        #endif
        maxSize = (1+4+tw)*dtypeSize(dtype) + MAXBLOCKSIZE*dtypeSize(dtype)*4;
        #ifdef DEBUG_TRSV_TRTRI
        printf("TRSV TRTRI: isFitLDS() maxSize = %lu, ldsSize = %lu, Y=%lu\n", maxSize, ldsSize, dim[0].y);
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

static ssize_t
generator_tbsv(
   char *buf,
   size_t buflen,
   const struct SubproblemDim *subdims,
   const struct PGranularity *pgran,
   void *extra)
{
    char tempTemplate[32*1024];
    char vector_size_trans[10], triangle_height[10];

    pgran = pgran; // Dummy- to avoid warnings

    if (buf == NULL) // PENDING: Return correct buffer size
    {
        buflen = (32 * 1024 * sizeof(char));
        return (ssize_t)buflen;
    }

    CLBLASKernExtra *extraFlags = ( CLBLASKernExtra *)extra;

    clblasUplo uplo   = ( extraFlags->flags & KEXTRA_UPPER_TRIANG) ? clblasUpper : clblasLower;
    clblasOrder order = ( extraFlags->flags & KEXTRA_COLUMN_MAJOR) ? clblasColumnMajor: clblasRowMajor;
    clblasTranspose trans = ( extraFlags->flags & KEXTRA_TRANS_A) ? clblasTrans : (( extraFlags->flags & KEXTRA_CONJUGATE_A) ? clblasConjTrans: clblasNoTrans);

    // unity and doConj handled in setKernelArgs
    if ( order == clblasColumnMajor )
    {
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

    if ( trans == clblasNoTrans)
    {
        ( uplo == clblasLower )?
                    (strcpy(tempTemplate, (char*)trsv_CL_SolveTriangle_kernel)) :
                    (strcpy(tempTemplate, (char*)trsv_CU_SolveTriangle_kernel));
    }
    else // Transpose cases...
    {
        ( uplo == clblasLower )?
                    (strcpy(tempTemplate, (char*)trsv_CLT_SolveTriangle_kernel)) :
                    (strcpy(tempTemplate, (char*)trsv_CUT_SolveTriangle_kernel));
    }

    unsigned int vecLenA = extraFlags->vecLenA;

    bool doVLOAD = false;
    if( extraFlags->flags &  KEXTRA_NO_COPY_VEC_A )
    {
        doVLOAD = true;
        #ifdef DEBUG_TRSV_TRTRI
            printf("DOing VLOAD as Aligned Data Pointer not Availabe\n");
        #endif
    }
    else
    {
        #ifdef DEBUG_TRSV_TRTRI
            printf("Using Aligned Data Pointer .........................\n");
        #endif
    }
    kprintf kobj( Prefix[extraFlags->dtype], vecLenA, doVLOAD);

    if (trans != clblasNoTrans)
    {
        sprintf( vector_size_trans, "%u", vecLenA );
        sprintf( triangle_height, "%ld", subdims[0].y );
        kobj.put("%PREFIXVECTOR_SIZE_TRANS", (const char *)vector_size_trans);
        kobj.put("%TRIANGLE_HEIGHT", triangle_height);
    }
    kobj.spit((char*)buf, tempTemplate);
    return (32 * 1024 * sizeof(char));
}

