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


#define DO_THEIRS
#define DO_TRMV
#define DO_TRSV
#define DO_SYMM
#define DO_SYR
#define DO_SPR
#define DO_GER
#define DO_GERC
#define DO_HER
#define DO_SYR2
#define DO_HER2
#define DO_HER
#define DO_SYR2
#define DO_HER2
#define DO_HEMV
#define DO_HEMM
#define DO_HERK
#define DO_TPMV
#define DO_HPMV
#define DO_SPMV
#define DO_TPSV
#define DO_HPR
#define DO_SBMV
#define DO_HPR2
#define DO_SPR2
#define DO_GBMV
#define DO_HBMV
#define DO_TBMV
#define DO_TBSV
#define DO_HER2K
#define DO_SCAL
#define DO_SSCAL
#define DO_DOT
#define DO_DOTC
#define DO_SWAP
#define DO_COPY
#define DO_AXPY
#define DO_ROTG
#define DO_ROTM
#define DO_ROTMG
#define DO_ROT
#define DO_NRM2
#define DO_ASUM
#define DO_iAMAX

#ifndef FUNC_H_
#define FUNC_H_

//#define _USE_GEMV_COMPLEX
#include <typeinfo>
#include <her.h>
#include <her2.h>
#include <spmv.h>
#include <gbmv.h>
#include <tbmv.h>
#include <tbsv.h>
#include <rotmg.h>

// Functions of BaseMetod Modified
// <typeinfo> included : As using typeid()
// testDG.h contains common definitions and enumerations used for populate()
/*  bool prepareDataToRun();
    void copyData(baseMetod<T> & source);
    void initDefault(size_t s, unsigned int q, USE_BUFFER ub);
    void destroy();
    void compareData(baseMetod<T> & source);
    cl_int getResult();

    matrix.h
    // Added support for Packed Matrix
    getElement();
    setElement();

// New stuff added
   populate() : Can generate data for general, packed, symmetric, lower-upper triangle
   //
   // Set diagonal elements to unity, random, zero
   // Row-Major, Col-Major support
   // TODO: Hermition Matrix, Banded Matrix support
*/
enum USE_BUFFER
{
   	USE_ABC,
   	USE_AB,
	USE_AC,
	USE_AXY, 	//For TRMV and friends
	USE_APXY,	//For TPMV and friends
	USE_AX, 	//For TRSV and friends
	USE_APX,	//For TPSV and friends
	USE_X,      //For blas-1 routines
	USE_XY,
    USE_ABXY,
    USE_ABCXY,  // for xROTMG
	USE_NOTHING	// Don't Care: Memory Allocation handled by derived Metod (xxxMetod class)
};

typedef enum BUFFER
{
	Aresult,
	APresult,
	Bresult,
	Cresult,
	Xresult,
	Yresult
} BUFFER_KIND;

template<typename T>
class baseMetod
{
protected:
    clMath::BlasBase *base;
public:
    typedef T TYPE;
    T t;

    USE_BUFFER 	   inputBuffers;
    BUFFER_KIND	   resultBuffer;
    BUFFER_KIND    resultBuffer_additional;

    clblasOrder order;

    cl_command_queue* queues;
    cl_uint           qnum;

    cl_context       context;

    cl_event* outEvent;
    cl_event* inEvent;
    cl_uint   inEventCount;


    int seed;
    const char* env;

    size_t size;
    //size_t size2;

	BUFFER_KIND resultLocation;

    TYPE alpha, beta;
    cl_mem bufA, bufB, bufC, bufX, bufY, bufAP;
    TYPE *A, *AP, *B, *C, *X, *Y;
    size_t ASize, BSize, CSize, XSize, YSize;

    void initOutEvent();
    bool prepareDataToRun();
    void copyData(baseMetod<T> & source);
    void initDefault(size_t s, unsigned int q, USE_BUFFER ub);
    void destroy();
    void compareData(baseMetod<T> & source);
    cl_int getResult();

};

template <typename T> bool
baseMetod<T>::prepareDataToRun()
{
    if (A != NULL) {
        bufA = base->createEnqueueBuffer(A, size * size * sizeof (TYPE), 0, ((resultBuffer == Aresult )? CL_MEM_READ_WRITE:CL_MEM_READ_ONLY));
	if ( bufA == NULL){ return false; }
    }

    if (B != NULL) {
        bufB = base->createEnqueueBuffer(B, size * size * sizeof (TYPE), 0, ((resultBuffer == Bresult )? CL_MEM_READ_WRITE:CL_MEM_READ_ONLY));
	if ( bufB == NULL){ return false; }
    }

    if (C != NULL) {
        bufC = base->createEnqueueBuffer(C, size * size * sizeof (TYPE), 0, ((resultBuffer == Cresult )? CL_MEM_READ_WRITE:CL_MEM_READ_ONLY));
	if ( bufC == NULL){ return false; }
    }

    if (AP != NULL) {
        bufAP = base->createEnqueueBuffer(AP, ((size * (size + 1)) / 2) * sizeof (TYPE), 0, ((resultBuffer == APresult )? CL_MEM_READ_WRITE:CL_MEM_READ_ONLY));
	if ( bufAP == NULL){ return false; }
    }

    if (X != NULL) {
        bufX = base->createEnqueueBuffer(X, size * sizeof (TYPE), 0, ((resultBuffer == Xresult )? CL_MEM_READ_WRITE:CL_MEM_READ_ONLY));
	if ( bufX == NULL){ return false; }
    }

    if (Y != NULL) {
        bufY = base->createEnqueueBuffer(Y, size * sizeof (TYPE), 0, ((resultBuffer == Yresult )? CL_MEM_READ_WRITE:CL_MEM_READ_ONLY));
	if ( bufY == NULL){ return false; }
    }

    return true;
}
template <typename T> void
baseMetod<T>::initOutEvent()
{
    outEvent = new cl_event[qnum];
    for (unsigned int i = 0; i < qnum; ++i) {
        outEvent[i] = NULL;
    }
}
template <typename T> void
baseMetod<T>::copyData(baseMetod<T> & source)
{
    if (source.A != NULL) {
        //A = new TYPE[size * size];
        memcpy(A, source.A, size * size * sizeof(TYPE));
    }
    if (source.B != NULL) {
        //B = new TYPE[size * size];
        memcpy(B, source.B, size * size * sizeof(TYPE));
    }
    if (source.C != NULL) {
        //C = new TYPE[size * size];
        memcpy(C, source.C, size * size * sizeof(TYPE));
    }

    if (source.AP != NULL) {
        //A = new TYPE[size * size];
        memcpy(AP, source.AP, ((size * (size+1)) /2 )* sizeof(TYPE));
    }
    if (source.X != NULL) {
        //A = new TYPE[size * size];
        memcpy(X, source.X, size * sizeof(TYPE));
    }
    if (source.Y != NULL) {
        //A = new TYPE[size * size];
        memcpy(Y, source.Y, size * sizeof(TYPE));
    }

    alpha = source.alpha;
    beta = source.beta;
}

template <typename T> void
baseMetod<T>::initDefault(size_t s, unsigned int q, USE_BUFFER ub)
{
    size = s;

    order = clblasColumnMajor;

    seed = 12345;
    base = clMath::BlasBase::getInstance();

    if (q  > 0) {
        base->setNumCommandQueues(q);
    }

    queues = base->commandQueues();
    qnum = base->numCommandQueues();

    context = base->context();
    alpha = convertMultiplier<TYPE>(base->alpha());
    beta = convertMultiplier<TYPE>(base->beta());

    outEvent= NULL;
    inEvent = NULL;

    inEventCount = 0;

    switch (ub) {
    case USE_ABC:
        A = new TYPE[size * size];
        B = new TYPE[size * size];
        C = new TYPE[size * size];
		AP = NULL;
		X  = NULL;
		Y  = NULL;
        break;

    case USE_AB:
        A = new TYPE[size * size];
        B = new TYPE[size * size];
        AP = NULL;
		C  = NULL;
		X  = NULL;
		Y  = NULL;
        break;

	case USE_AC:
        A = new TYPE[size * size];
        C = new TYPE[size * size];
        AP = NULL;
        B  = NULL;
        X  = NULL;
        Y  = NULL;
        break;


    case USE_AX:

		A = new TYPE[size * size];
		X = new TYPE[size];
		AP = NULL;
		B  = NULL;
		C  = NULL;
		Y  = NULL;
		break;

    case USE_AXY:
		A = new TYPE[size * size];
		X = new TYPE[size];
		Y = new TYPE[size];
		AP = NULL;
		B  = NULL;
		C  = NULL;
		break;

    case USE_APXY:
		AP = new TYPE[(size * (size + 1)) /2];
		X = new TYPE[size];
		Y = new TYPE[size];
		A = NULL;
		B  = NULL;
		C  = NULL;
		break;

    case USE_APX:
		AP = new TYPE[(size * (size + 1)) /2];
		X = new TYPE[size];
		A  = NULL;
		B  = NULL;
		C  = NULL;
		Y  = NULL;
		break;

	case USE_ABXY:
		X = new TYPE[size];
		Y = new TYPE[size];
		AP = NULL;
		A = new TYPE[size * size];
		B  = new TYPE[size * size];
		C  = NULL;
		break;

    // Currently used only for xROTMG requiring 5 buffers
    // change if it is reusable for more tests
    case USE_ABCXY:
        X = new TYPE[size];
        Y = new TYPE[size];
        AP = NULL;
        A = new TYPE[size*size];//for D1
        B = new TYPE[size*size];// for D2
        C = new TYPE[size*size];//for SPARAM
        break;

	case USE_X:
		X = new TYPE[size];
		Y = NULL;
		AP = NULL;
		A = NULL;
		B  = NULL;
		C  = NULL;
		break;

    case USE_XY: // suitable for BLAS-1 routines: copy & swap
		X = new TYPE[size];
		Y = new TYPE[size];
		AP = NULL;
		A = NULL;
		B  = NULL;
		C  = NULL;
		break;

    default:
        AP = NULL;
		A  = NULL;
		B  = NULL;
		C  = NULL;
		X  = NULL;
		Y  = NULL;
    }

    bufA = NULL;
    bufB = NULL;
    bufC = NULL;
    bufX = NULL;
    bufY = NULL;
    bufAP = NULL;

    srand(seed);
    //std::cerr << "init = " << size << std::endl;

    env = NULL;
}

template <typename T> void
baseMetod<T>::destroy()
{
    if (outEvent != NULL) {
        for (unsigned int i = 0; i < qnum; ++i) {
            outEvent[i] = NULL;
        }
        delete[](outEvent);
    }

    //std::cerr << "destroy "<< std::endl;

    delete[] this->A;
    delete[] this->B;
    delete[] this->C;
    delete[] this->AP;
    delete[] this->X;
    delete[] this->Y;

    clReleaseMemObject(this->bufA);
    clReleaseMemObject(this->bufB);
    clReleaseMemObject(this->bufC);
    clReleaseMemObject(this->bufAP);
    clReleaseMemObject(this->bufX);
    clReleaseMemObject(this->bufY);

    A = NULL;
    B = NULL;
    C = NULL;
    AP = NULL;
    X = NULL;
    Y = NULL;

    bufA = NULL;
    bufB = NULL;
    bufC = NULL;
    bufAP = NULL;
    bufX = NULL;
    bufY = NULL;
}

template <typename T> void
baseMetod<T>::compareData(baseMetod<T> & source)
{
/*    if (C == NULL) {
        compareMatrices<T>(order, size, size, B, source.B, size);
    }
    else {
        compareMatrices<T>(order, size, size, C, source.C, size);
    }
*/

/*
	if (C == NULL && ( X == NULL)) {
		 resultBuffer = Bresult;
	}
	else
	{
		 resultBuffer = Cresult;
	}
*/

	T* s1 = NULL;
	T* s2 = NULL;

	s1 = ( resultBuffer == Aresult)? A: ( resultBuffer == Bresult) ? B: ( resultBuffer == Cresult)? C:( resultBuffer == Xresult)? X:( resultBuffer == Yresult)? Y: AP;
	s2 = ( resultBuffer == Aresult)? source.A: ( resultBuffer == Bresult) ? source.B: ( resultBuffer == Cresult)? source.C:( resultBuffer == Xresult)? source.X:( resultBuffer == Yresult)? source.Y: source.AP;

	clblasOrder fOrder;

	size_t m,n,lda;

    if ( resultBuffer == Aresult || resultBuffer == Bresult || resultBuffer == Cresult )
    {
		m = size;
		n = size;
		lda = size;
		fOrder = order;
    }
    else if ( resultBuffer == Xresult || resultBuffer == Yresult )
    {
		m = size;
		n = 1;
		lda = size;
		fOrder = clblasColumnMajor;
    }
    else if ( resultBuffer == APresult)
    {
		m = size;
		n = size;
		lda = 0; // compareMatrix expects lda = 0 for Packed Matrix
		fOrder = order;
    }

	compareMatrices<T>( fOrder, m, n, s1, s2, lda);
}

template <typename T> cl_int
baseMetod<T>::getResult()
{
    cl_int err;
/*
    if (C == NULL) {
        err = clEnqueueReadBuffer(queues[0], bufB, CL_TRUE, 0, size * size * sizeof(TYPE),
                B, 0, NULL, NULL);
    }
    else {
        err = clEnqueueReadBuffer(queues[0], bufC, CL_TRUE, 0, size * size * sizeof(TYPE),
                C, 0, NULL, NULL);
    }
*/
/*
	if (C == NULL) {
		 resultBuffer = Bresult;
	}
	else
	{
		 resultBuffer = Cresult;
	}
*/

    T* s =  NULL;
    s = ( resultBuffer == Aresult)? A: ( resultBuffer == Bresult) ? B: ( resultBuffer == Cresult)? C:( resultBuffer == Xresult)? X:( resultBuffer == Yresult)? Y: AP;

   cl_mem bufs = ( resultBuffer == Aresult)? bufA: ( resultBuffer == Bresult) ? bufB: ( resultBuffer == Cresult)? bufC:( resultBuffer == Xresult)? bufX:( resultBuffer == Yresult)? bufY: bufAP;

    size_t transferSize = 0;
    if ( resultBuffer == Aresult || resultBuffer == Bresult || resultBuffer == Cresult )
    {
	transferSize = size * size;
    }
    else if ( resultBuffer == Xresult || resultBuffer == Yresult )
    {
	transferSize = size;
    }
    else if ( resultBuffer == APresult)
    {
	transferSize = (size * (size + 1))/2;
    }

    transferSize *= sizeof(TYPE);
	err = CL_SUCCESS;

    err = clEnqueueReadBuffer(queues[0], bufs, CL_TRUE, 0, transferSize,
                s, 0, NULL, NULL);
    return err;

}
///////
template<typename T>
class GemmMetod : public baseMetod<T>
{
private:
    typedef T TYPE;

    clblasTranspose transA;
    clblasTranspose transB;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
GemmMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_ABC);
    transA = clblasNoTrans;
    transB = clblasNoTrans;
	this->resultBuffer = Cresult;

    baseMetod<T>::env = "AMD_CLBLAS_GEMM_IMPLEMENTATION";
}

template <typename T> void
GemmMetod<T>::generateData()
{
    bool useAlpha = this->base->useAlpha();
    bool useBeta = this->base->useBeta();

    randomGemmMatrices<TYPE>(this->order, transA, transB,
            this->size, this->size, this->size, useAlpha,
            &this->alpha, this->A, this->size, this->B,
            this->size, useBeta, &this->beta, this->C, this->size);

}

template <typename T> cl_int
GemmMetod<T>::run()
{
    return (cl_int)::clMath::clblas::gemm(this->order, transA, transB,
        this->size, this->size, this->size, this->alpha, this->bufA, 0,
        this->size, this->bufB, 0, this->size, this->beta, this->bufC, 0,
        this->size, this->qnum, this->queues, this->inEventCount,
        this->inEvent, this->outEvent);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class TrmmMetod  : public baseMetod<T>
{
    typedef T TYPE;

    clblasTranspose transA;
    clblasTranspose transB;
    clblasUplo uplo;
    clblasSide side;
    clblasDiag diag;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
TrmmMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AB);

    transA = clblasNoTrans;
    transB = clblasNoTrans;
    side = clblasLeft;
    uplo = clblasUpper;
    diag = clblasUnit;
	this->resultBuffer = Bresult;

    baseMetod<T>::env = "AMD_CLBLAS_TRMM_IMPLEMENTATION";
}

template <typename T> void
TrmmMetod<T>::generateData()
{
    bool useAlpha = this->base->useAlpha();
    randomTrmmMatrices<TYPE>(this->order, side, uplo, diag,
            this->size, this->size, useAlpha, &this->alpha, this->A,
            this->size, this->B, this->size);
}

template <typename T> cl_int
TrmmMetod<T>::run()
{
    return (cl_int)::clMath::clblas::trmm(this->order, this->side, this->uplo,
          this->transA, clblasUnit, this->size, this->size, this->alpha,
          this->bufA, 0, this->size, this->bufB, 0, this->size,
          this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class TrsmMetod  : public baseMetod<T>
{
    typedef T TYPE;

    clblasTranspose transA;
    clblasUplo uplo;
    clblasSide side;
    clblasDiag diag;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
TrsmMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AB);

    transA = clblasNoTrans;
    side = clblasLeft;
    uplo = clblasUpper;
    diag = clblasUnit;
	this->resultBuffer = Bresult;

    baseMetod<T>::env = "AMD_CLBLAS_TRSM_IMPLEMENTATION";
}

template <typename T> void
TrsmMetod<T>::generateData()
{
    bool useAlpha = this->base->useAlpha();

    randomTrsmMatrices<T>(this->order, side, uplo, diag,
            this->size, this->size, useAlpha, &this->alpha,
            this->A, this->size, this->B, this->size);

}
template <typename T> cl_int
TrsmMetod<T>::run()
{
    return (cl_int)::clMath::clblas::trsm(this->order, side, uplo,
        transA, diag, this->size, this->size, this->alpha, this->bufA, 0,
        this->size, this->bufB, 0, this->size, this->qnum, this->queues,
        this->inEventCount, this->inEvent, this->outEvent);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class GemvMetod : public baseMetod<T>
{
    typedef T TYPE;

    clblasTranspose transA;
    clblasTranspose transB;
    clblasTranspose transC;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
GemvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_ABC);

    transA = clblasNoTrans;
    transB = clblasNoTrans;
    transC = clblasNoTrans;
	this->resultBuffer = Cresult;
}

template <typename T> void
GemvMetod<T>::generateData()
{
    bool useAlpha = this->base->useAlpha();
    bool useBeta = this->base->useBeta();

    randomGemmxMatrices<T>(this->order, transA, transB, transC,
            this->size, this->size, this->size, useAlpha,
            &this->alpha, this->A, this->size, this->B, this->size, useBeta,
            &this->beta, this->C, this->size);

}

template <typename T> cl_int
GemvMetod<T>::run()
{
    return (cl_int)::clMath::clblas::gemv(this->order, transA,
            this->size, this->size, this->alpha, this->bufA, 0, this->size,
            this->bufB, 0, 1, this->beta, this->bufC, 0, 1,
            this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class SymvMetod : public baseMetod<T>
{
    typedef T TYPE;

    clblasTranspose transA;
    clblasTranspose transB;
    clblasTranspose transC;
    clblasUplo uplo;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
SymvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_ABC);

    transA = clblasNoTrans;
    transB = clblasNoTrans;
    transC = clblasNoTrans;
    uplo = clblasUpper;
	this->resultBuffer = Cresult;
}

template <typename T> void
SymvMetod<T>::generateData()
{
    bool useAlpha = this->base->useAlpha();
    bool useBeta = this->base->useBeta();

    randomGemmxMatrices<T>(this->order, transA, transB, transC,
            this->size, this->size, this->size, useAlpha,
            &this->alpha, this->A, this->size, this->B, this->size, useBeta,
            &this->beta, this->C, this->size);

}


template <typename T> cl_int
SymvMetod<T>::run()
{
    return (cl_int)::clMath::clblas::symv(this->order, uplo, this->size, this->alpha,
            this->bufA, 0, this->size, this->bufB, 0, 1, this->beta, this->bufC, 0, 1,
            this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class Syr2kMetod : public baseMetod<T>
{
public:
    typedef T TYPE;

    clblasTranspose transA;
    clblasTranspose transB;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
Syr2kMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_ABC);
    transA = clblasNoTrans;
    transB = clblasNoTrans;
    uplo = clblasUpper;
	this->resultBuffer = Cresult;
}

template <typename T> void
Syr2kMetod<T>::generateData()
{
    bool useBeta = this->base->useBeta();

    randomGemmMatrices<T>(this->order, transA, transB, this->size, this->size,
        this->size, true, &this->alpha, this->A, this->size, this->B,
        this->size, useBeta, &this->beta, this->C, this->size);
}

template <typename T> cl_int
Syr2kMetod<T>::run()
{
    return (cl_int)::clMath::clblas::syr2k(this->order, uplo, transA,
        this->size, this->size, this->alpha, this->bufA, 0,
        this->size, this->bufB, 0, this->size, this->beta, this->bufC, 0,
        this->size, this->qnum, this->queues,
        this->inEventCount, this->inEvent, this->outEvent);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class TrmvMetod : public baseMetod<T>
{
public:
    typedef T TYPE;

   	clblasTranspose transA;
	clblasUplo uplo;
	clblasDiag diagA;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Packed Matrix : lda must be set to zero
// 2. Always test with RowMajor Lower in case of Packed matrix
// 3. NoTrans case only supported for Packed matrix
//
template <typename T> void
TrmvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->order = clblasRowMajor;
    transA = clblasNoTrans;
    uplo   = clblasLower;
    diagA  = clblasUnit;
    this->resultBuffer = Xresult;
}

template <typename T> void
TrmvMetod<T>::generateData()
{
/*
enum RealMatrixCreationFlags {
		NO_FLAGS			= 0,
		ROW_MAJOR_ORDER 		= 1,
		PACKED_MATRIX 			= 2,
		SYMMETRIC_MATRIX		= 4,
		UPPER_HALF_ONLY			= 8,
		LOWER_HALF_ONLY			= 16,
		NO_ALIGNMENT			= 32,
		UNIT_DIAGONAL			= 64,
		RANDOM_INIT			= 128,
		ZERO_DIAGONAL			= 256
*/

//bool useBeta = this->base->useBeta();
/*
    randomGemmMatrices<T>(this->order, transA, transB,this->size, this->size, this->size,
        true, &this->alpha, this->A, this->size, this->B, this->size, useBeta, &this->beta, this->C, this->size);
*/

    // Set flags...
    int creationFlags = 0;
    creationFlags =  creationFlags | RANDOM_INIT;

    // Default is Column-Major
    creationFlags = ( (this-> order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
    // Setting uplo
    creationFlags = ( (this-> uplo) == clblasLower)? (creationFlags | LOWER_HALF_ONLY) : (creationFlags | UPPER_HALF_ONLY);
    BlasRoutineID BlasFn = CLBLAS_TRMV;

    // Matrix A
	 populate( this->A, this->size, this->size, this->size, BlasFn, creationFlags);
     populate( this->X, this->size, 1, this->size, BlasFn);
}

template <typename T> cl_int
TrmvMetod<T>::run()
{


	DataType type;

	type = ( typeid(T) == typeid(float))? TYPE_FLOAT:( typeid(T) == typeid(double))? TYPE_DOUBLE:
																( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

 	return (cl_int)::clMath::clblas::trmv(type, this->order, uplo, transA, diagA, this->size, this->bufA, 0, this->size, this->bufX,
						0, 1, this->bufY/* as Xcopy */, this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);


}

/////////////////////////////////////////////////////////////////////

template<typename T>
class TrsvMetod : public baseMetod<T>
{
public:
    typedef T TYPE;

        clblasTranspose transA;
        clblasUplo uplo;
        clblasDiag diagA;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Packed Matrix : lda must be set to zero
// 2. Always test with RowMajor Lower in case of Packed matrix
// 3. NoTrans case only supported for Packed matrix
//
template <typename T> void
TrsvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AX);
    this->order = clblasRowMajor;
    transA = clblasNoTrans;
    uplo   = clblasLower;
    diagA  = clblasUnit;
    this->resultBuffer = Xresult;
}

template <typename T> void
TrsvMetod<T>::generateData()
{
/*
enum RealMatrixCreationFlags {
                NO_FLAGS                        = 0,
                ROW_MAJOR_ORDER                 = 1,
                PACKED_MATRIX                   = 2,
                SYMMETRIC_MATRIX                = 4,
                UPPER_HALF_ONLY                 = 8,
                LOWER_HALF_ONLY                 = 16,
                NO_ALIGNMENT                    = 32,
                UNIT_DIAGONAL                   = 64,
                RANDOM_INIT                     = 128,
                ZERO_DIAGONAL                   = 256
*/

	int creationFlags = 0;
    creationFlags =  creationFlags | RANDOM_INIT;

    // Default is Column-Major
    creationFlags = ( (this-> order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
    // Setting uplo
    creationFlags = ( (this-> uplo) == clblasLower)? (creationFlags | LOWER_HALF_ONLY) : (creationFlags | UPPER_HALF_ONLY);


    // Matrix A
    //populate( this->A, this->size, this->size, this->size, creationFlags);
    //populate( this->X, this->size, 1, this->size);

    randomTrsvMatrices(this->order, this->uplo, this->diagA, this->size, this->A, this->size, this->X, 1);
}

template <typename T> cl_int
TrsvMetod<T>::run()
{


        DataType type;
		type = ( typeid(T) == typeid(float))? TYPE_FLOAT:( typeid(T) == typeid(double))? TYPE_DOUBLE:
                                                                ( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;


        return (cl_int)::clMath::clblas::trsv(type, this->order, uplo, transA, diagA, this->size, this->bufA, 0, this->size, this->bufX,
                                                0, 1, this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);


}

/////////////////////////////////////////////////////////////////////

template<typename T>
class TpsvMetod : public baseMetod<T>
{
public:
    typedef T TYPE;

        clblasTranspose transA;
        clblasUplo uplo;
        clblasDiag diagA;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Packed Matrix : lda must be set to zero
// 2. Always test with RowMajor Lower in case of Packed matrix
// 3. NoTrans case only supported for Packed matrix
//
template <typename T> void
TpsvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AX);
    this->order = clblasRowMajor;
    transA = clblasNoTrans;
    uplo   = clblasLower;
    diagA  = clblasUnit;
    this->resultBuffer = Xresult;
}

template <typename T> void
TpsvMetod<T>::generateData()
{
    randomTrsvMatrices(this->order, this->uplo, this->diagA, this->size, this->A, 0, this->X, 1);
}

template <typename T> cl_int
TpsvMetod<T>::run()
{


        DataType type;
        type = ( typeid(T) == typeid(float))? TYPE_FLOAT:( typeid(T) == typeid(double))? TYPE_DOUBLE:
                                                                ( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;


        return (cl_int)::clMath::clblas::tpsv(type, this->order, uplo, transA, diagA, this->size, this->bufA, 0, this->bufX,
                                                0, 1, this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);


}


/////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class SymmMetod : public baseMetod<T>
{
public:
    typedef T TYPE;

    clblasSide side;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Packed Matrix : lda must be set to zero
// 2. Always test with RowMajor Lower in case of Packed matrix
// 3. NoTrans case only supported for Packed matrix
//
template <typename T> void
SymmMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_ABC);
    this->order = clblasRowMajor;
    uplo   = clblasLower;
    side   = clblasLeft;
    this->resultBuffer = Cresult;

}

template <typename T> void
SymmMetod<T>::generateData()
{
/*
enum RealMatrixCreationFlags {
        NO_FLAGS            = 0,
        ROW_MAJOR_ORDER         = 1,
        PACKED_MATRIX           = 2,
        SYMMETRIC_MATRIX        = 4,
        UPPER_HALF_ONLY         = 8,
        LOWER_HALF_ONLY         = 16,
        NO_ALIGNMENT            = 32,
        UNIT_DIAGONAL           = 64,
        RANDOM_INIT         = 128,
        ZERO_DIAGONAL           = 256
*/

//bool useBeta = this->base->useBeta();
/*
    randomGemmMatrices<T>(this->order, transA, transB,this->size, this->size, this->size,
        true, &this->alpha, this->A, this->size, this->B, this->size, useBeta, &this->beta, this->C, this->size);
*/


    // Set flags...
    int creationFlags = 0, creationFlagsA;
    creationFlags =  creationFlags | RANDOM_INIT;

    // Default is Column-Major
    creationFlags = ( (this-> order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
    // Setting uplo
	//In this case only A matrix is either upper or lower triangular
	creationFlagsA = creationFlags;
    creationFlagsA = ( (this-> uplo) == clblasLower)? (creationFlagsA | LOWER_HALF_ONLY) : (creationFlagsA | UPPER_HALF_ONLY);
	BlasRoutineID BlasFn = CLBLAS_SYMM;
    // Matrix A
        populate(this->A, this->size, this->size, this->size, BlasFn, creationFlagsA );
		populate(this->B, this->size, this->size, this->size, BlasFn, creationFlags);
		populate(this->C, this->size, this->size, this->size, BlasFn, creationFlags);
}

template <typename T> cl_int
SymmMetod<T>::run()
{

    return (cl_int)::clMath::clblas::symm(this->order, side, uplo, this->size, this->size, this->alpha, this->bufA, 0, this->size,
											this->bufB, 0, this->size, this->beta, this->bufC, 0, this->size,
                        					this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class SyrMetod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Testing for Row Major order.
// 2. Lower triangle
//
template <typename T> void
SyrMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AX);
    this->order = clblasRowMajor;
    uplo   = clblasLower;
    this->resultBuffer = Aresult;
}

template <typename T> void
SyrMetod<T>::generateData()
{
    /*
	// Set flags...
    int creationFlags = 0;
    creationFlags =  creationFlags | RANDOM_INIT;

    // Default is Column-Major
    creationFlags = ( (this-> order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
    creationFlags = ( (this-> uplo) == clblasLower)? (creationFlags | LOWER_HALF_ONLY) : (creationFlags | UPPER_HALF_ONLY);
	BlasRoutineID BlasFn = CLBLAS_GER;

    // Matrix A
    populate( this->A, this->size, this->size, this->size, BlasFn, creationFlags);
	//Vector X
    populate( this->X, this->size, 1, this->size, BlasFn);
	*/
	randomSyrMatrices( this->order, uplo, this->size, false, &(this->alpha), this->A, this->size, this->X, 1);
}

template <typename T> cl_int
SyrMetod<T>::run()
{

    return (cl_int)::clMath::clblas::syr(this->order, uplo, this->size, this->alpha, this->bufX, 0, 1,
										 this->bufA, 0, this->size, this->qnum, this->queues,
										 this->inEventCount, this->inEvent, this->outEvent);


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Syr2Metod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Testing for Row Major order.
// 2. Lower triangle
//
template <typename T> void
Syr2Metod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->order = clblasRowMajor;
    uplo   = clblasLower;
    this->resultBuffer = Aresult;
}

template <typename T> void
Syr2Metod<T>::generateData()
{
    randomSyr2Matrices( this->order, uplo, this->size, false, &(this->alpha), this->A, this->size, this->X, 1, this->Y, 1);
}

template <typename T> cl_int
Syr2Metod<T>::run()
{

    return (cl_int)::clMath::clblas::syr2(this->order, uplo, this->size, this->alpha, this->bufX, 0, 1, this->bufY, 0, 1,
										 this->bufA, 0, this->size, this->qnum, this->queues,
										 this->inEventCount, this->inEvent, this->outEvent);


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class GerMetod : public baseMetod<T>
{
public:
    typedef T TYPE;
	int incx, incy;
        int m;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Packed Matrix : lda must be set to zero
// 2. Always test with RowMajor Lower in case of Packed matrix
// 3. NoTrans case only supported for Packed matrix
//
template <typename T> void
GerMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->order = clblasRowMajor;
    this->resultBuffer = Aresult;

}

template <typename T> void
GerMetod<T>::generateData()
{
/*
enum RealMatrixCreationFlags {
        NO_FLAGS            = 0,
        ROW_MAJOR_ORDER         = 1,
        PACKED_MATRIX           = 2,
        SYMMETRIC_MATRIX        = 4,
        UPPER_HALF_ONLY         = 8,
        LOWER_HALF_ONLY         = 16,
        NO_ALIGNMENT            = 32,
        UNIT_DIAGONAL           = 64,
        RANDOM_INIT             = 128,
        ZERO_DIAGONAL           = 256
*/

//bool useBeta = this->base->useBeta();
/*
    randomGemmMatrices<T>(this->order, transA, transB,this->size, this->size, this->size,
        true, &this->alpha, this->A, this->size, this->B, this->size, useBeta, &this->beta, this->C, this->size);
*/


    // Set flags...
    int creationFlags = 0;
    creationFlags =  creationFlags | RANDOM_INIT;

    // Default is Column-Major
    creationFlags = ( (this-> order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);

	BlasRoutineID BlasFn = CLBLAS_GER;
    // Matrix A
    populate(this->A, this->size, this->size, this->size,  BlasFn, creationFlags);
    populate(this->X, this->size, 1, (1 + (m - 1) * abs(incx)), BlasFn, 0);
    populate(this->Y, this->size, 1, (1 + (m - 1) * abs(incy)), BlasFn, 0);
}

template <typename T> cl_int
GerMetod<T>::run()
{

    return (cl_int)::clMath::clblas::ger(this->order, this->size, this->size, this->alpha,
						this->bufX, 0, 1, this->bufY, 0, 1, this->bufA, 0, this->size,
						this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);


}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename T>
class GercMetod : public baseMetod<T>
{
public:
    typedef T TYPE;
	int incx, incy;
        int m;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Packed Matrix : lda must be set to zero
// 2. Always test with RowMajor Lower in case of Packed matrix
// 3. NoTrans case only supported for Packed matrix
//
template <typename T> void
GercMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->order = clblasRowMajor;
    this->resultBuffer = Aresult;

}

template <typename T> void
GercMetod<T>::generateData()
{
    // Set flags...
    int creationFlags = 0;
    creationFlags =  creationFlags | RANDOM_INIT;

    // Default is Column-Major
    creationFlags = ( (this-> order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);

	BlasRoutineID BlasFn = CLBLAS_GER;
    // Matrix A
    populate(this->A, this->size, this->size, this->size,  BlasFn, creationFlags);
    populate(this->X, this->size, 1, this->size, BlasFn, 0);
    populate(this->Y, this->size, 1, this->size, BlasFn, 0);
}

template <typename T> cl_int
GercMetod<T>::run()
{

    return (cl_int)::clMath::clblas::gerc(this->order, this->size, this->size, this->alpha,
						this->bufX, 0, 1, this->bufY, 0, 1, this->bufA, 0, this->size,
						this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);


}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename T>
class HerMetod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Testing for Row Major order.
// 2. Lower triangle
//
template <typename T> void
HerMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AX);
    this->order = clblasRowMajor;
    uplo = clblasLower;
    this->resultBuffer = Aresult;
}

template <typename T> void
HerMetod<T>::generateData()
{
	randomHerMatrices( this->order, uplo, this->size, &(this->alpha), this->A, this->size, this->X, 1 );
}


template <typename T> cl_int
HerMetod<T>::run()
{

    return (cl_int)::clMath::clblas::her(this->order, this->uplo, this->size, CREAL(this->alpha), this->bufX, 0, 1,
                                                                                 this->bufA, 0, this->size, this->qnum, this->queues,
                                                                                 this->inEventCount, this->inEvent, this->outEvent);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Her2Metod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Testing for Row Major order.
// 2. Lower triangle
//
template <typename T> void
Her2Metod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->order = clblasRowMajor;
    uplo = clblasLower;
    this->resultBuffer = Aresult;
}

template <typename T> void
Her2Metod<T>::generateData()
{
	randomHer2Matrices<T>(this->order, uplo, this->size, &(this->alpha), this->A, this->size,
								this->X, 1, this->Y, 1);
}

template <typename T> cl_int
Her2Metod<T>::run()
{

    return (cl_int)::clMath::clblas::her2(this->order, this->uplo, this->size, this->alpha, this->bufX, 0, 1, this->bufY, 0, 1,
                                                                                 this->bufA, 0, this->size, this->qnum, this->queues,
                                                                                 this->inEventCount, this->inEvent, this->outEvent);


}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class HemmMetod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasSide side;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Packed Matrix : lda must be set to zero
// 2. Always test with RowMajor Lower in case of Packed matrix
// 3. NoTrans case only supported for Packed matrix
//
template <typename T> void
HemmMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_ABC);
    this->order = clblasRowMajor;
    uplo   = clblasLower;
    side   = clblasLeft;
    this->resultBuffer = Cresult;

}

template <typename T> void
HemmMetod<T>::generateData()
{
/*
    int creationFlags = 0, creationFlagsA;
    creationFlags =  creationFlags | RANDOM_INIT;

    creationFlags = ( (this-> order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
    creationFlagsA = creationFlags;
    creationFlagsA = ( (this-> uplo) == clblasLower)? (creationFlagsA | LOWER_HALF_ONLY) : (creationFlagsA | UPPER_HALF_ONLY);
    BlasRoutineID BlasFn = CLBLAS_HEMM;

        populate(this->A, this->size, this->size, this->size, BlasFn, creationFlagsA );
        populate(this->B, this->size, this->size, this->size, BlasFn, creationFlags);
        populate(this->C, this->size, this->size, this->size, BlasFn, creationFlags);
*/

	randomGemmMatrices<T>(this->order, clblasNoTrans, clblasNoTrans,
            this->size, this->size, this->size, false,
            &this->alpha, this->A, this->size, this->B,
            this->size, false, &this->beta, this->C, this->size);
}

template <typename T> cl_int
HemmMetod<T>::run()
{

    return (cl_int)::clMath::clblas::hemm(this->order, side, uplo, this->size, this->size, this->alpha, this->bufA, 0, this->size,
                                                                                        this->bufB, 0, this->size, this->beta, this->bufC, 0, this->size,
                                                                this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);


}

// HEMV

template<typename T>
class HemvMetod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
HemvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->order = clblasRowMajor;
    uplo   = clblasLower;
    this->resultBuffer = Yresult;
}

template <typename T> void
HemvMetod<T>::generateData()
{
	randomHemvMatrices(this->order, uplo, this->size, false, &(this->alpha), this->A, this->size,
							this->X, 1, false, &(this->beta), this->Y, 1);
}

template <typename T> cl_int
HemvMetod<T>::run()
{

    return (cl_int)::clMath::clblas::hemv(this->order, uplo, this->size, this->alpha, this->bufA, 0, this->size,
                                                   this->bufX, 0, 1, this->beta, this->bufY, 0, 1,
                                                   this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);

}
/////////////////////////////////////////////////////
template<typename T>
class HerkMetod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasUplo uplo;
    clblasTranspose transA;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Packed Matrix : lda must be set to zero
// 2. Always test with RowMajor Lower in case of Packed matrix
// 3. NoTrans case only supported for Packed matrix
//
template <typename T> void
HerkMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AC);
    this->order = clblasRowMajor;
    uplo   = clblasLower;
    transA = clblasNoTrans;
    this->resultBuffer = Cresult;

}

template <typename T> void
HerkMetod<T>::generateData()
{

	randomGemmMatrices<T>(this->order, this->transA, clblasNoTrans,
        this->size, this->size, this->size, false, &this->alpha, this->A, this->size,
        NULL, 0, false, &this->beta, this->C, this->size);
}

template <typename T> cl_int
HerkMetod<T>::run()
{

    return (cl_int)::clMath::clblas::herk(this->order, uplo, transA, this->size, this->size, CREAL(this->alpha), this->bufA, 0, this->size,
                                                                                         CREAL(this->beta), this->bufC, 0, this->size,
                                                                this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);


}

////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class TpmvMetod : public baseMetod<T>
{
public:
    typedef T TYPE;

    clblasTranspose trans;
    clblasUplo uplo;
    clblasDiag diag;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Packed Matrix : lda must be set to zero
// 2. Always test with RowMajor Lower in case of Packed matrix
// 3. NoTrans case only supported for Packed matrix
//
template <typename T> void
TpmvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_APXY);
    this->order = clblasRowMajor;
    trans = clblasNoTrans;
    uplo   = clblasLower;
    diag  = clblasUnit;
    this->resultBuffer = Xresult;
}

template <typename T> void
TpmvMetod<T>::generateData()
{

    // Set flags...
    int creationFlags = 0;
    creationFlags =  creationFlags | RANDOM_INIT | PACKED_MATRIX;

    // Default is Column-Major
    creationFlags = ( (this-> order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
    // Setting uplo
    creationFlags = ( (this-> uplo) == clblasLower)? (creationFlags | LOWER_HALF_ONLY) : (creationFlags | UPPER_HALF_ONLY);
    BlasRoutineID BlasFn = CLBLAS_TRMV;

    // Matrix A
     populate( this->AP, this->size, this->size, 0, BlasFn, creationFlags);
     populate( this->X, this->size, 1, this->size, BlasFn);
}

template <typename T> cl_int
TpmvMetod<T>::run()
{


    DataType type;

    type = ( typeid(T) == typeid(float))? TYPE_FLOAT:( typeid(T) == typeid(double))? TYPE_DOUBLE:
                                                                ( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    return (cl_int)::clMath::clblas::tpmv(type, this->order, uplo, trans, diag, this->size, this->bufAP, 0, this->bufX,
                        0, 1, this->bufY/* as Xcopy */, this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);


}
///////////////////////////////////////////////////////////////////////////
template<typename T>
class SpmvMetod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
SpmvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_APXY);
    this->order = clblasRowMajor;
    uplo   = clblasLower;
    this->resultBuffer = Yresult;
}

template <typename T> void
SpmvMetod<T>::generateData()
{
	randomSpmvMatrices(this->order, uplo, this->size, false, &(this->alpha), this->AP,
							this->X, 1, false, &(this->beta), this->Y, 1);
}

template <typename T> cl_int
SpmvMetod<T>::run()
{

    return (cl_int)::clMath::clblas::spmv(this->order, uplo, this->size, this->alpha, this->bufAP, 0,
                                                   this->bufX, 0, 1, this->beta, this->bufY, 0, 1,
                                                   this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);

}

///////////////////////////////////////////////////////////////////////////
template<typename T>
class HpmvMetod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
HpmvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_APXY);
    this->order = clblasRowMajor;
    uplo   = clblasLower;
    this->resultBuffer = Yresult;
}

template <typename T> void
HpmvMetod<T>::generateData()
{
	randomHemvMatrices(this->order, uplo, this->size, false, &(this->alpha), this->AP, 0,
							this->X, 1, false, &(this->beta), this->Y, 1);
}

template <typename T> cl_int
HpmvMetod<T>::run()
{

    return (cl_int)::clMath::clblas::hpmv(this->order, uplo, this->size, this->alpha, this->bufAP, 0,
                                                   this->bufX, 0, 1, this->beta, this->bufY, 0, 1,
                                                   this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);

}

//////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class SprMetod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Testing for Row Major order.
// 2. Lower triangle
//
template <typename T> void
SprMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_APX);
    this->order = clblasRowMajor;
    uplo   = clblasLower;
    this->resultBuffer = APresult;
}

template <typename T> void
SprMetod<T>::generateData()
{
    randomSyrMatrices( this->order,uplo, this->size, false, &(this->alpha), this->AP, 0, this->X, 1);
}

template <typename T> cl_int
SprMetod<T>::run()
{

    return (cl_int)::clMath::clblas::spr(this->order, uplo, this->size, this->alpha, this->bufX, 0, 1,
                                         this->bufAP, 0, this->qnum, this->queues,
                                         this->inEventCount, this->inEvent, this->outEvent);


}

///////////////////////////////////////////////////////////

template<typename T>
class HprMetod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
HprMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_APX);
    this->order = clblasRowMajor;
    uplo = clblasLower;
    this->resultBuffer = APresult;
}

template <typename T> void
HprMetod<T>::generateData()
{
	randomHerMatrices( this->order, uplo, this->size, &(this->alpha), this->AP, 0, this->X, 1 );
}


template <typename T> cl_int
HprMetod<T>::run()
{

    return (cl_int)::clMath::clblas::hpr(this->order, this->uplo, this->size, CREAL(this->alpha), this->bufX, 0, 1,
                                                                                 this->bufAP, 0, this->qnum, this->queues,
                                                                                 this->inEventCount, this->inEvent, this->outEvent);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Hpr2Metod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Testing for Row Major order.
// 2. Lower triangle
//
template <typename T> void
Hpr2Metod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_APXY);
    this->order = clblasRowMajor;
    uplo = clblasLower;
    this->resultBuffer = APresult;
}

template <typename T> void
Hpr2Metod<T>::generateData()
{
	randomHer2Matrices<T>(this->order, uplo, this->size, &(this->alpha), this->AP, 0,
								this->X, 1, this->Y, 1);
}

template <typename T> cl_int
Hpr2Metod<T>::run()
{

    return (cl_int)::clMath::clblas::hpr2(this->order, this->uplo, this->size, this->alpha, this->bufX, 0, 1, this->bufY, 0, 1,
                                                                                 this->bufAP, 0, this->qnum, this->queues,
                                                                                 this->inEventCount, this->inEvent, this->outEvent);


}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Spr2Metod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasUplo uplo;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Testing for Row Major order.
// 2. Lower triangle
//
template <typename T> void
Spr2Metod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_APXY);
    this->order = clblasRowMajor;
    uplo   = clblasLower;
    this->resultBuffer = APresult;
}

template <typename T> void
Spr2Metod<T>::generateData()
{
    randomSyr2Matrices( this->order, uplo, this->size, false, &(this->alpha), this->AP, 0, this->X, 1, this->Y, 1);
}

template <typename T> cl_int
Spr2Metod<T>::run()
{

    return (cl_int)::clMath::clblas::spr2(this->order, uplo, this->size, this->alpha, this->bufX, 0, 1, this->bufY, 0, 1,
										 this->bufAP, 0, this->qnum, this->queues,
										 this->inEventCount, this->inEvent, this->outEvent);


}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class GbmvMetod : public baseMetod<T>
{
    typedef T TYPE;

    clblasTranspose transA;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Testing for Row Major order.
// 2. Non Transpose
//
template <typename T> void
GbmvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->order = clblasRowMajor;
    transA = clblasNoTrans;
    this->resultBuffer = Yresult;
}

template <typename T> void
GbmvMetod<T>::generateData()
{
   randomGbmvMatrices(this->order, this->transA, this->size, this->size, &(this->alpha), &(this->beta),
                        this->A, this->size, this->X, 1, this->Y, 1);
}

template <typename T> cl_int
GbmvMetod<T>::run()
{
    return (cl_int)clMath::clblas::gbmv(this->order, this->transA, this->size, this->size, (1), (1),
                                        this->alpha, this->bufA, 0, this->size, this->bufX, 0, 1, this->beta, this->bufY, 0, 1,
                                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class TbmvMetod : public baseMetod<T>
{
    typedef T TYPE;

    clblasTranspose transA;
    clblasUplo uplo;
    clblasDiag diag;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Testing for Row Major order.
// 2. Non Transpose
//
template <typename T> void
TbmvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->order = clblasRowMajor;
    transA = clblasNoTrans;
    uplo = clblasUpper;
    diag = clblasNonUnit;
    this->resultBuffer = Yresult;
}

template <typename T> void
TbmvMetod<T>::generateData()
{
   randomTbmvMatrices(this->size, this->A, this->size, this->X, 1);
}

template <typename T> cl_int
TbmvMetod<T>::run()
{
    DataType type;

    type = ( typeid(T) == typeid(float))? TYPE_FLOAT:( typeid(T) == typeid(double))? TYPE_DOUBLE:
                                                     ( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    return (cl_int)clMath::clblas::tbmv(type, this->order, this->uplo, this->transA, this->diag, this->size, (1),
                                        this->bufA, 0, this->size, this->bufX, 0, 1, this->bufY,
                                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class SbmvMetod : public baseMetod<T>
{
    typedef T TYPE;

    clblasUplo uplo;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Testing for Row Major order.
// 2. Non Transpose
//
template <typename T> void
SbmvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->order = clblasRowMajor;
    uplo = clblasUpper;
    this->resultBuffer = Yresult;
}

template <typename T> void
SbmvMetod<T>::generateData()
{
   randomGbmvMatrices(this->order, clblasNoTrans, this->size, this->size, &(this->alpha), &(this->beta),
                        this->A, this->size, this->X, 1, this->Y, 1);
}

template <typename T> cl_int
SbmvMetod<T>::run()
{
    return (cl_int)clMath::clblas::sbmv(this->order, this->uplo, this->size, 1,
                                        this->alpha, this->bufA, 0, this->size, this->bufX, 0, 1, this->beta, this->bufY, 0, 1,
                                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//HBMV

template<typename T>
class HbmvMetod : public baseMetod<T>
{
    typedef T TYPE;

    clblasUplo uplo;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

// Assumptions
// 1. Testing for Row Major order.
// 2. Non Transpose
//
template <typename T> void
HbmvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->order = clblasRowMajor;
    uplo = clblasUpper;
    this->resultBuffer = Yresult;
}

template <typename T> void
HbmvMetod<T>::generateData()
{
   randomGbmvMatrices(this->order, clblasNoTrans, this->size, this->size, &(this->alpha), &(this->beta),
                        this->A, this->size, this->X, 1, this->Y, 1);
}

template <typename T> cl_int
HbmvMetod<T>::run()
{
    return (cl_int)clMath::clblas::hbmv(this->order, this->uplo, this->size, 1,
                                        this->alpha, this->bufA, 0, this->size, this->bufX, 0, 1, this->beta, this->bufY, 0, 1,
                                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class TbsvMetod : public baseMetod<T>
{
    typedef T TYPE;

    clblasTranspose transA;
    clblasUplo uplo;
    clblasDiag diag;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
TbsvMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AX);
    this->order = clblasRowMajor;
    transA = clblasNoTrans;
    uplo = clblasUpper;
    diag = clblasNonUnit;
    this->resultBuffer = Xresult;
}

template <typename T> void
TbsvMetod<T>::generateData()
{
   randomTbsvMatrices(this->order, this->uplo, this->diag, this->size, 1, this->A, 2, this->X, 1);
}

template <typename T> cl_int
TbsvMetod<T>::run()
{
    DataType type;

    type = ( typeid(T) == typeid(float))? TYPE_FLOAT:( typeid(T) == typeid(double))? TYPE_DOUBLE:
                                                     ( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;
    return (cl_int)clMath::clblas::tbsv(type, this->order, this->uplo, this->transA, this->diag, this->size, 1,
                                        this->bufA, 0, 2, this->bufX, 0, 1,
                                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}
///////////////////////////////////////////////


template<typename T>
class Her2kMetod : public baseMetod<T>
{
public:
    typedef T TYPE;
    clblasUplo uplo;
    clblasTranspose transA;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
Her2kMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_ABC);
    this->order = clblasRowMajor;
    uplo   = clblasLower;
    transA = clblasNoTrans;
    this->resultBuffer = Cresult;

}

template <typename T> void
Her2kMetod<T>::generateData()
{

	clblasTranspose ftransB = (this->transA==clblasNoTrans)? clblasConjTrans: clblasNoTrans;

    randomGemmMatrices<T>(this->order, this->transA, ftransB,
                                this->size, this->size, this->size, false, &this->alpha, this->A, this->size,
                                this->B, this->size, false, &this->beta, this->C, this->size);
}

template <typename T> cl_int
Her2kMetod<T>::run()
{

    return (cl_int)::clMath::clblas::her2k(this->order, uplo, this->transA, this->size, this->size, this->alpha,
                                   this->bufA, 0, this->size, this->bufB, 0, this->size, CREAL(this->beta), this->bufC, 0, this->size,
                                   this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);


}

////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class ScalMetod : public baseMetod<T>
{
public:
    typedef T TYPE;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
ScalMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_X);
    this->resultBuffer = Xresult;

}

template <typename T> void
ScalMetod<T>::generateData()
{
    randomVectors(this->size, this->X, 1);
}

template <typename T> cl_int
ScalMetod<T>::run()
{

    return (cl_int)::clMath::clblas::scal(false, this->size, this->alpha, this->bufX, 0,
                        1, this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

////////////////////////////////////////////////////////////////////////////////////////////
// Sscal is for handling the 2 extra cases csscal and zdscal
template<typename T>
class SscalMetod : public baseMetod<T>
{
public:
    typedef T TYPE;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
SscalMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_X);
    this->resultBuffer = Xresult;

}

template <typename T> void
SscalMetod<T>::generateData()
{
    randomVectors(this->size, this->X, 1);
}

template <typename T> cl_int
SscalMetod<T>::run()
{

    return (cl_int)::clMath::clblas::scal(true, this->size, this->alpha, this->bufX, 0,
                        1, this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class SwapMetod : public baseMetod<T>
{
public:
    typedef T TYPE;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
SwapMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_XY);
    this->resultBuffer = Xresult;
    // no need to have 2 buffers as result, as this is not a correctness test.
    // proper correctness testing happens in test-correctness.

}

template <typename T> void
SwapMetod<T>::generateData()
{
    randomVectors(this->size, this->X, 1, this->Y, 1);
}

template <typename T> cl_int
SwapMetod<T>::run()
{
    DataType type;

	type = ( typeid(T) == typeid(float))? TYPE_FLOAT:
            ( typeid(T) == typeid(double))? TYPE_DOUBLE:
			( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT:
             TYPE_COMPLEX_DOUBLE;

    return (cl_int)::clMath::clblas::swap(type, this->size, this->bufX, 0, 1, this->bufY, 0, 1,
                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}
////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class DotMetod : public baseMetod<T>
{
    typedef T TYPE;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
DotMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_ABXY);
    this->resultBuffer = Yresult;
}

template <typename T> void
DotMetod<T>::generateData()
{
    //BlasRoutineID BlasFn = CLBLAS_DOT;

//    populate( this->X, this->size, 1, this->size, BlasFn);
    randomVectors(this->size, this->X, 1, this->Y, 1, true);

}

template <typename T> cl_int
DotMetod<T>::run()
{
    DataType type;

        type = ( typeid(T) == typeid(float))? TYPE_FLOAT:( typeid(T) == typeid(double))? TYPE_DOUBLE:
                                                                        ( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    return (cl_int)clMath::clblas::dot( type, this->size, this->bufA, 0,
                                        this->bufX, 0, 1, this->bufY, 0, 1, this->bufB,
                                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//DOTC
template<typename T>
class DotcMetod : public baseMetod<T>
{
    typedef T TYPE;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
DotcMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_ABXY);
    this->resultBuffer = Yresult;
}

template <typename T> void
DotcMetod<T>::generateData()
{
    randomVectors(this->size, this->X, 1, this->Y, 1, true);

}

template <typename T> cl_int
DotcMetod<T>::run()
{
    DataType type;

        type = ( typeid(T) == typeid(float))? TYPE_FLOAT:( typeid(T) == typeid(double))? TYPE_DOUBLE:
                                                                        ( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    return (cl_int)clMath::clblas::dotc( type, this->size, this->bufA, 0,
                                        this->bufX, 0, 1, this->bufY, 0, 1, this->bufB,
                                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}
////////////////////////////////////////////////////////////////////////////////////////////
//COPY

template<typename T>
class CopyMetod : public baseMetod<T>
{
public:
    typedef T TYPE;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
CopyMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_XY);
    this->resultBuffer = Yresult;
}

template <typename T> void
CopyMetod<T>::generateData()
{
    randomVectors(this->size, this->X, 1, this->Y, 1);
}

template <typename T> cl_int
CopyMetod<T>::run()
{
    DataType type;

        type = ( typeid(T) == typeid(float))? TYPE_FLOAT:
            ( typeid(T) == typeid(double))? TYPE_DOUBLE:
                        ( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT:
             TYPE_COMPLEX_DOUBLE;

    return (cl_int)::clMath::clblas::copy(type, this->size, this->bufX, 0, 1, this->bufY, 0, 1,
                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename T>
class AxpyMetod : public baseMetod<T>
{
public:
    typedef T TYPE;

public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
AxpyMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_XY);
    this->resultBuffer = Yresult;

}

template <typename T> void
AxpyMetod<T>::generateData()
{
    randomVectors(this->size, this->X, 1, this->Y, 1);
}

template <typename T> cl_int
AxpyMetod<T>::run()
{

    return (cl_int)::clMath::clblas::axpy(this->size, this->alpha, this->bufX, 0, 1, this->bufY, 0, 1,
                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class RotgMetod : public baseMetod<T>
{
    typedef T TYPE;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
RotgMetod<T>::initDefault(size_t s, unsigned int q)
{
    //USE_ABXY is actually used to create 2 2-D arrays and 2 vectors
    //But here we use is to create the required 4 vectors. So more than required memory is allocated here.
    //As this is functionality test, this does not affect the purpose of the tests.
    //Here X=SA, Y=SB, A=C and B=S, where RHS's represent the standard netlib variable names
    baseMetod<T>::initDefault(1, q, USE_ABXY);
    this->resultBuffer = Yresult;
    s = s; //Warning
}

template <typename T> void
RotgMetod<T>::generateData()
{
    randomVectors(this->size, this->X, 1, this->Y, 1);

}

template <typename T> cl_int
RotgMetod<T>::run()
{
    DataType type;

    type = ( typeid(T) == typeid(float))? TYPE_FLOAT:( typeid(T) == typeid(double))? TYPE_DOUBLE:
                                                                        ( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    return (cl_int)clMath::clblas::rotg( type, this->bufX, 0, this->bufY, 0, this->bufA, 0, this->bufB, 0,
                                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class RotmMetod : public baseMetod<T>
{
    typedef T TYPE;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
RotmMetod<T>::initDefault(size_t s, unsigned int q)
{
    // USE_AXY allocates space for 1 2-D array A and 2 vectors: X & Y
    // Here are we are allocating more memory for PARAM than required, to reuse code. A corrosponds to PARAM.
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->resultBuffer = Yresult;
}

template <typename T> void
RotmMetod<T>::generateData()
{
    randomVectors(this->size, this->X, 1, this->Y, 1);
    randomVectors(4, this->A + 1, 1);
    *(this->A) = 0; //Only 4 inputs are valid here, which are tested in correctness and performance test
}

template <typename T> cl_int
RotmMetod<T>::run()
{
    DataType type;

    type = ( typeid(T) == typeid(float))? TYPE_FLOAT: TYPE_DOUBLE;

    return (cl_int)clMath::clblas::rotm( type, this->size, this->bufX, 0, 1, this->bufY, 0, 1, this->bufA, 0,
                                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}


////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class RotmgMetod : public baseMetod<T>
{
    typedef T TYPE;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
RotmgMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(5, q, USE_ABCXY);
    this->resultBuffer = Cresult;
    s = s; //Warning
}

template <typename T> void
RotmgMetod<T>::generateData()
{
    randomRotmg(this->A, this->B, this->X, this->Y, this->C);

    *(this->C) = 0; //Only 4 inputs are valid here, which are tested in correctness and performance test
}

template <typename T> cl_int
RotmgMetod<T>::run()
{
    DataType type;

    type = ( typeid(T) == typeid(float))? TYPE_FLOAT: TYPE_DOUBLE;

    return (cl_int)clMath::clblas::rotmg( type, this->bufA, 0, this->bufB, 0, this->bufX, 0, this->bufY, 0,
                                         this->bufC, 0, this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

/////////////////////////////////////////////////////////////////////////////////

template<typename T>
class RotMetod : public baseMetod<T>
{
    typedef T TYPE;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
RotMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_XY);
    this->resultBuffer = Yresult;
}

template <typename T> void
RotMetod<T>::generateData()
{
    randomVectors(this->size, this->X, 1, this->Y, 1);
}

template <typename T> cl_int
RotMetod<T>::run()
{
    //DataType type;

	//type = ( typeid(T) == typeid(float))? TYPE_FLOAT:( typeid(T) == typeid(double))? TYPE_DOUBLE:
      //                                                                  ( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;


    return (cl_int)clMath::clblas::rot( this->size, this->bufX, 0, 1, this->bufY, 0, 1, this->alpha, this->beta,
                                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Nrm2Metod : public baseMetod<T>
{
    typedef T TYPE;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
Nrm2Metod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->resultBuffer = Yresult;
}

template <typename T> void
Nrm2Metod<T>::generateData()
{
    randomVectors(this->size, this->X, 1, this->Y, 1, true);
}

template <typename T> cl_int
Nrm2Metod<T>::run()
{
    DataType type;

        type = ( typeid(T) == typeid(float))? TYPE_FLOAT:( typeid(T) == typeid(double))? TYPE_DOUBLE:
               ( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    return (cl_int)clMath::clblas::nrm2(type, this->size, this->bufY, 0,
                                        this->bufX, 0, 1, this->bufA,
                                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

///////////////////////////////////////////////////////

//ASUM

template<typename T>
class AsumMetod : public baseMetod<T>
{
    typedef T TYPE;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
AsumMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->resultBuffer = Xresult;
}

template <typename T> void
AsumMetod<T>::generateData()
{
    randomVectors(this->size, this->X, 1, (T*)NULL, 0, true);
}

template <typename T> cl_int
AsumMetod<T>::run()
{
    DataType type;

        type = ( typeid(T) == typeid(float))? TYPE_FLOAT:( typeid(T) == typeid(double))? TYPE_DOUBLE:
                                                                        ( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    return (cl_int)clMath::clblas::asum( type, this->size, this->bufA, 0,
                                        this->bufX, 0, 1, this->bufY,
                                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

//////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////

//iAMAX

template<typename T>
class iAmaxMetod : public baseMetod<T>
{
    typedef T TYPE;
public:
    void initDefault(size_t s, unsigned int q);
    cl_int run();
    void generateData();
};

template <typename T> void
iAmaxMetod<T>::initDefault(size_t s, unsigned int q)
{
    baseMetod<T>::initDefault(s, q, USE_AXY);
    this->resultBuffer = Yresult;
}

template <typename T> void
iAmaxMetod<T>::generateData()
{
    randomVectors(this->size, this->X, 1, this->Y, 1);
}

template <typename T> cl_int
iAmaxMetod<T>::run()
{
    DataType type;

        type = ( typeid(T) == typeid(float))? TYPE_FLOAT:( typeid(T) == typeid(double))? TYPE_DOUBLE:
                                ( typeid(T) == typeid(FloatComplex))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    return (cl_int)clMath::clblas::iamax( type, this->size,
                                        this->bufY, 0, this->bufX, 0, 1, this->bufA,
                                        this->qnum, this->queues, this->inEventCount, this->inEvent, this->outEvent);
}

//////////////////////////////////////////////////////////////////////

#define CHECK_DOUBLE \
{ \
    clMath::BlasBase* base = clMath::BlasBase::getInstance();\
    if (!base->isDevSupportDoublePrecision()) {\
        ::std::cerr << ">> Double precision is not supported"\
            << ::std::endl \
            << ">> Test skipped." << ::std::endl;\
        SUCCEED();\
        return;\
    }\
}

#endif  // FUNC_H_
