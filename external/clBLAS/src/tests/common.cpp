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


#include <iostream>
#include <string.h>
#include <clBLAS.h>

#include <common.h>

cl_context
getQueueContext(cl_command_queue commandQueue, cl_int *error)
{
    cl_int err;
    cl_context ctx = NULL;

    err = clGetCommandQueueInfo(commandQueue, CL_QUEUE_CONTEXT,
        sizeof(cl_context), &ctx, NULL);
    if (error != NULL) {
        *error = err;
    }
    return ctx;
}

cl_int
waitForSuccessfulFinish(
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_event *events)
{
    cl_int err = CL_SUCCESS;
    cl_uint i;

    for (i = 0; i < numCommandQueues; i++) {
        cl_int e;
        cl_int status;

        e = clFinish(commandQueues[i]);
        if ((events != NULL) && (events[i] != NULL)) {
            if (e == CL_SUCCESS) {
        status = CL_COMPLETE;
                e = clGetEventInfo(events[i], CL_EVENT_COMMAND_EXECUTION_STATUS,
            sizeof(status), &status, NULL);
                if ((e == CL_SUCCESS) && (status < 0)) {
                    e = -status;
        }
        }
            clReleaseEvent(events[i]);
    }

        if (err == CL_SUCCESS) {
            err = e;
}
    }

    return err;
}

cl_int
flushAll(
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues)
{
    cl_int err;
    cl_uint i;

    for (i = 0; i < numCommandQueues; i++) {
        err = clFlush(commandQueues[i]);
        if (err != CL_SUCCESS) {
            return err;
        }
    }

    return CL_SUCCESS;
}

void
printTestParams(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    bool useAlpha,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    size_t offB,
    size_t ldb,
    bool useBeta,
    ComplexLong beta,
    size_t offC,
    size_t ldc)
{
    ::std::cerr << orderStr(order) << ", " << transStr(transA) << ", " <<
        transStr(transB) << ::std::endl;
    ::std::cerr << "M = " << M << ", N = " << N << ", K = " << K << ::std::endl;
    ::std::cerr << "offA = " << offA << ", offB = " << offB << ", offC = " <<
        offC << ::std::endl;
    ::std::cerr << "lda = " << lda << ", ldb = " << ldb << ", ldc = " <<
        ldc << ::std::endl;
    if (useAlpha) {
        ::std::cerr << "alpha = (" << alpha.re << "," << alpha.imag
            << ")" << ::std::endl;
    }
    if (useBeta) {
        ::std::cerr << "beta = (" << beta.re << "," << beta.imag
            << ")" << ::std::endl;
    }
}

void
printTestParams(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    bool useAlpha,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    size_t offB,
    size_t ldb)
{
    ::std::cerr << orderStr(order) << ", " << sideStr(side) << ", " <<
        uploStr(uplo) << ", " << transStr(transA) << ", " <<
        diagStr(diag) << ::std::endl;
    ::std::cerr << "M = " << M << ", N = " << N << ::std::endl;
    ::std::cerr << "offA = " << offA << ", offB = " << offB << ::std::endl;
    ::std::cerr << "lda = " << lda << ", ldb = " << ldb << ::std::endl;
    if (useAlpha) {
        ::std::cerr << "alpha = (" << alpha.re << "," << alpha.imag
            << ")" << ::std::endl;
    }
}

//SYR
void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
	double alpha,
    size_t offx,
    int incx,
    size_t offa,
    size_t lda)
{
	::std::cerr << orderStr(order) << ", " << uploStr(uplo) << ::std::endl;
	::std::cerr << "N = " << N << ", offx = " << offx << ", incx = " << incx << ::std::endl;
	::std::cerr << "offa = " << offa << ::std::endl;
    if( lda )
        ::std::cerr << ", lda = " << lda << ::std::endl;
	::std::cerr << "alpha = " << alpha << ::std::endl;
}

//SPR
void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    double alpha,
    size_t offx,
    int incx,
    size_t offa)
{
    ::std::cerr << orderStr(order) << ", " << uploStr(uplo) << ::std::endl;
    ::std::cerr << "N = " << N << ", offx = " << offx << ", incx = " << incx << ::std::endl;
    ::std::cerr << "offa = " << offa << ::std::endl;
    ::std::cerr << "alpha = " << alpha << ::std::endl;
}

//SYR2
void
printTestParams(
	clblasOrder order,
	clblasUplo  uplo,
	size_t N,
	double alpha,
	size_t offx,
	int incx,
	size_t offy,
	int incy,
	size_t offa,
	size_t lda)
{
	::std::cerr << orderStr(order) << ", " << uploStr(uplo) << ::std::endl;
	::std::cerr << "N = " << N << ", offx = " << offx << ", incx = " << incx << ::std::endl;
	::std::cerr << "offy = " << offy << ", incy = " << incy << ::std::endl;
	::std::cerr << "offa = " << offa << ::std::endl;
    if( lda )
        ::std::cerr << ", lda = " << lda << ::std::endl;
	::std::cerr << "alpha = " << alpha << ::std::endl;
}

//copy, dot, swap, dotc

void
printTestParams(
    size_t N,
    size_t offx,
    int incx,
    size_t offy,
    int incy)
{

    ::std::cerr << "N = " << N << ", offx = " << offx << ", incx = " << incx << ::std::endl;
    ::std::cerr << "offy = " << offy << ", incy = " << incy << ::std::endl;
}

//HER2
void
printTestParams(
    clblasOrder order,
    clblasUplo  uplo,
    size_t N,
    bool useAlpha,
    cl_float2 alpha,
    size_t offx,
    int incx,
    size_t offy,
    int incy,
    size_t offa,
    size_t lda)
{
    ::std::cerr << orderStr(order) << ", " << uploStr(uplo) << ::std::endl;
    ::std::cerr << "N = " << N << ", offx = " << offx << ", incx = " << incx << ::std::endl;
    ::std::cerr << "offy = " << offy << ", incy = " << incy << ::std::endl;
    ::std::cerr << "offa = " << offa << ::std::endl;
    if( lda )
        ::std::cerr << ", lda = " << lda << ::std::endl;
        if(useAlpha)
    ::std::cerr << "alpha = (" << CREAL(alpha) << ", " << CIMAG(alpha) << ")" << ::std::endl;
}

//HEMV
void
printTestParams(
    clblasOrder order,
    clblasUplo  uplo,
    size_t N,
    ComplexLong alpha,
    size_t offa,
    size_t lda,
    size_t offx,
    int incx,
    ComplexLong beta,
    size_t offy,
    int incy)
{
    ::std::cerr << orderStr(order) << ", " << uploStr(uplo) << ::std::endl;
    ::std::cerr << "N = " << N << ", offx = " << offx << ", incx = " << incx << ::std::endl;
    ::std::cerr << "offy = " << offy << ", incy = " << incy << ::std::endl;
    ::std::cerr << "offa = " << offa;
    if( lda )
        ::std::cerr << ", lda = " << lda;
    ::std::cerr << ::std::endl << "alpha = (" << alpha.re << "," << alpha.imag << ")" << ::std::endl;
    ::std::cerr << "beta = (" << beta.re << "," << beta.imag << ")" << ::std::endl;
}
//SYMM , HEMM
void
printTestParams(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    bool useAlpha,
    ComplexLong alpha,
    bool useBeta,
    ComplexLong beta,
    size_t lda,
    size_t ldb,
    size_t ldc,
    size_t offa,
    size_t offb,
    size_t offc )
{
    ::std::cerr << orderStr(order) << ", " << sideStr(side) << ", " << uploStr(uplo) << ::std::endl;
    ::std::cerr << "M = " << M << ", N = " << N << ::std::endl;
    ::std::cerr << "lda = " << lda << ", ldb = " << ldb << ", ldc = " << ldc<< ::std::endl;
    if (useAlpha) {
        ::std::cerr << "alpha = (" << alpha.re << "," << alpha.imag << ")" << ::std::endl; }
	 if (useBeta) {
        ::std::cerr << "beta = (" << beta.re << "," << beta.imag  << ")" << ::std::endl; }
	::std::cerr << "offa = " << offa << ", offb = " << offb << ", offc = " << offc<< ::std::endl;

}

//xHEMM
void
printTestParams(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    bool useAlpha,
    cl_float2 alpha,
    bool useBeta,
    cl_float2 beta,
    size_t lda,
    size_t ldb,
    size_t ldc,
    size_t offa,
    size_t offb,
    size_t offc )
{
    ::std::cerr << orderStr(order) << ", " << sideStr(side) << ", " << uploStr(uplo) << ::std::endl;
    ::std::cerr << "M = " << M << ", N = " << N << ::std::endl;
    ::std::cerr << "lda = " << lda << ", ldb = " << ldb << ", ldc = " << ldc<< ::std::endl;
    if (useAlpha) {
        ::std::cerr << "alpha = (" << CREAL(alpha) << "," << CIMAG(alpha) << ")" << ::std::endl; }
         if (useBeta) {
        ::std::cerr << "beta = (" << CREAL(beta) << "," << CIMAG(beta)  << ")" << ::std::endl; }
        ::std::cerr << "offa = " << offa << ", offb = " << offb << ", offc = " << offc<< ::std::endl;

}



void
printTestParams(
    clblasOrder order,
    size_t M,
    size_t N,
    bool useAlpha,
    ComplexLong alpha,
    size_t lda,
    int incx,
    int incy,
    size_t offa,
    size_t offx,
    size_t offy )
{
    ::std::cerr << orderStr(order) << ", " << ::std::endl;
    ::std::cerr << "M = " << M << ", N = " << N << ::std::endl;
    ::std::cerr << "lda = " << lda << ", incx = " << incx << ", incy = " << incy<< ::std::endl;
    if (useAlpha) {
        ::std::cerr << "alpha = (" << alpha.re << "," << alpha.imag << ")" << ::std::endl; }
        ::std::cerr << "offa = " << offa << ", offx = " << offx << ", offy = " << offy << ::std::endl;

}

// xGBMV
void
printTestParams(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    size_t KL,
    size_t KU,
    ComplexLong alpha,
    size_t offa,
    size_t lda,
    size_t offx,
    int incx,
    ComplexLong beta,
    size_t offy,
    int incy)
{
    ::std::cerr << orderStr(order) << ", " << transStr(transA) << ", " << ::std::endl;
    ::std::cerr << "M = " << M << ", N = " << N << ", KL = " << KL << ", KU = " << KU << ::std::endl;
    ::std::cerr << "lda = " << lda << ", incx = " << incx << ", incy = " << incy<< ::std::endl;
    ::std::cerr << "offa = " << offa << ", offx = " << offx << ", offy = " << offy << ::std::endl;
    ::std::cerr << "alpha = (" << alpha.re << "," << alpha.imag << ")" << ::std::endl;
    ::std::cerr << "beta = (" << beta.re << "," << beta.imag << ")" << ::std::endl;
}

//HBMV
//SBMV

void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    size_t K,
    ComplexLong alpha,
    size_t offa,
    size_t lda,
    size_t offx,
    int incx,
    ComplexLong beta,
    size_t offy,
    int incy)
{
    ::std::cerr << orderStr(order) << ", " << uploStr(uplo) << ", " << ::std::endl;
    ::std::cerr << ", N = " << N << ", K = " << K << ::std::endl;
    ::std::cerr << "lda = " << lda << ", incx = " << incx << ", incy = " << incy<< ::std::endl;
    ::std::cerr << "offa = " << offa << ", offx = " << offx << ", offy = " << offy << ::std::endl;
    ::std::cerr << "alpha = (" << alpha.re << "," << alpha.imag << ")" << ::std::endl;
    ::std::cerr << "beta = (" << beta.re << "," << beta.imag << ")" << ::std::endl;
}


//xTBMV
void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    size_t KLU,
    size_t offA,
    size_t lda,
    size_t offx,
    int incx,
    size_t offy,
    int incy)
{
    ::std::cerr << orderStr(order) << ", " << uploStr(uplo) << ", " << transStr(transA) << ", " << diagStr(diag) << ::std::endl;
    ::std::cerr << ", N = " << N << ", KL or KU = " << KLU << ::std::endl;
    ::std::cerr << "lda = " << lda << ", incx = " << incx << ", incy = " << incy<< ::std::endl;
    ::std::cerr << "offa = " << offA << ", offx = " << offx << ", offy = " << offy << ::std::endl;
}

//HER
void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    bool useAlpha,
    ComplexLong alpha,
    size_t lda,
    int incx,
    size_t offa,
    size_t offx)
{
    ::std::cerr << orderStr(order) << ", " << ::std::endl;
    ::std::cerr << uploStr(uplo) << ", " << ::std::endl;
    ::std::cerr << " N = " << N << ::std::endl;
    ::std::cerr << "lda = " << lda << ", incx = " << incx << ::std::endl;
    if (useAlpha) {
        ::std::cerr << "alpha = (" << alpha.re << "," << alpha.imag << ")" << ::std::endl; }
        ::std::cerr << "offa = " << offa << ", offx = " << offx << ::std::endl;

}


void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    size_t lda,
    int incx,
	size_t offa,
	size_t offx)
{
    ::std::cerr << orderStr(order) << ", " << uploStr(uplo) << ", " << transStr(transA)
				<< ", " <<diagStr(diag) << ::std::endl;
    ::std::cerr << " N = " << N << ::std::endl;
    ::std::cerr << "lda = " << lda << ", incx = " << incx << ::std::endl;
	::std::cerr << "offa = " << offa << ", offx = " << offx << ::std::endl;
}

//xTPMV
void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t N,
    int incx,
    size_t offa,
    size_t offx)
{
    ::std::cerr << orderStr(order) << ", " << uploStr(uplo) << ", " << transStr(transA)
                << ", " <<diagStr(diag) << ::std::endl;
    ::std::cerr << " N = " << N << ::std::endl;
    ::std::cerr << " incx = " << incx << ::std::endl;
    ::std::cerr << "offa = " << offa << ", offx = " << offx << ::std::endl;
}


void
printTestParams(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    bool useAlpha,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    int incx,
    bool useBeta,
    ComplexLong beta,
    int incy)
{
    ::std::cerr << orderStr(order) << ", " << transStr(transA) << ::std::endl;
    ::std::cerr << "M = " << M << ", N = " << N << ::std::endl;
    ::std::cerr << "offA = " << offA << ::std::endl;
    ::std::cerr << "lda = " << lda << ::std::endl;
    if (useAlpha) {
        ::std::cerr << "alpha = (" << alpha.re << "," << alpha.imag
            << ")" << ::std::endl;
    }
    if (useBeta) {
        ::std::cerr << "beta = (" << beta.re << "," << beta.imag
            << ")" << ::std::endl;
    }
    ::std::cerr << "incx = " << incx << ", incy = " << incy << ::std::endl;
}

void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    bool useAlpha,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    int incx,
    bool useBeta,
    ComplexLong beta,
    int incy)
{
    ::std::cerr << orderStr(order) << ", " << uploStr(uplo) << ::std::endl;
    ::std::cerr << "N = " << N << ::std::endl;
    ::std::cerr << "offA = " << offA << ::std::endl;
    if( lda )
    ::std::cerr << "lda = " << lda << ::std::endl;
    if (useAlpha) {
        ::std::cerr << "alpha = (" << alpha.re << "," << alpha.imag
            << ")" << ::std::endl;
    }
    if (useBeta) {
        ::std::cerr << "beta = (" << beta.re << "," << beta.imag
            << ")" << ::std::endl;
    }
    ::std::cerr << "incx = " << incx << ", incy = " << incy << ::std::endl;
}

void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    bool useAlpha,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    size_t offB,
    size_t ldb,
    bool useBeta,
    ComplexLong beta,
    size_t offC,
    size_t ldc)
{
    ::std::cerr << orderStr(order) << ", " << uploStr(uplo)
        << ", " << transStr(transA) << ::std::endl;
    ::std::cerr << "N = " << N << ", K = " << K << ::std::endl;
    ::std::cerr << "offA = " << offA << ", offB = " << offB
            << ", offC = " << offC << ::std::endl;
    ::std::cerr << "lda = " << lda << ", ldb = " << ldb
        << ", ldc = " << ldc << ::std::endl;
    if (useAlpha) {
        ::std::cerr << "alpha = (" << alpha.re << "," << alpha.imag
            << ")" << ::std::endl;
    }
    if (useBeta) {
        ::std::cerr << "beta = (" << beta.re << "," << beta.imag
            << ")" << ::std::endl;
    }
}

void
printTestParams(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    bool useAlpha,
    ComplexLong alpha,
    size_t offA,
    size_t lda,
    bool useBeta,
    ComplexLong beta,
    size_t offC,
    size_t ldc)
{
    ::std::cerr << orderStr(order) << ", " << uploStr(uplo)
        << ", " << transStr(transA) << ::std::endl;
    ::std::cerr << "N = " << N << ", K = " << K << ::std::endl;
    ::std::cerr << "offA = " << offA << ", offC = " << offC << ::std::endl;
    ::std::cerr << "lda = " << lda << ", ldc = " << ldc << ::std::endl;
    if (useAlpha) {
        ::std::cerr << "alpha = (" << alpha.re << "," << alpha.imag
            << ")" << ::std::endl;
    }
    if (useBeta) {
        ::std::cerr << "beta = (" << beta.re << "," << beta.imag
            << ")" << ::std::endl;
    }
}

//For scal
void
printTestParams(
    size_t N,
    ComplexLong alpha,
    size_t offx,
    int incx)
{
    ::std::cerr << "N = " << N << ", alpha = (" << alpha.re << "," << alpha.imag << ")" << ::std::endl;
    ::std::cerr << "offx = " << offx << ", incx = " << incx << ::std::endl;
}

//For axpy
void
printTestParams(
    size_t N,
    ComplexLong alpha,
    size_t offx,
    int incx,
    size_t offy,
    int incy)
{
    ::std::cerr << "N = " << N << ", alpha = (" << alpha.re << "," << alpha.imag << ")" << ::std::endl;
    ::std::cerr << "offx = " << offx << ", incx = " << incx << ::std::endl;
    ::std::cerr << "offy = " << offy << ", incy = " << incy << ::std::endl;
}


//xROT
void
printTestParams(
    size_t N,
    size_t offx,
    int incx,
	size_t offy,
	int incy,
	ComplexLong C,
	ComplexLong S)
{
    ::std::cerr << "N = " << N << ::std::endl;
	::std::cerr << "C = (" << C.re << "," << C.imag << ")" << ",S = (" << S.re << "," << S.imag << ")" << ::std::endl;
    ::std::cerr << "offx = " << offx << ", incx = " << incx << ", offy = "<< offy << ", incy = " << incy <<  ::std::endl;
}

// xROTG
void
printTestParams(size_t offSA, size_t offSB, size_t offC, size_t offS)
{
    ::std::cerr << "offSA = " << offSA << ", offSB = " << offSB << ", offC = " << offC << ",offS = " << offS << std::endl;
}

//xROTM
void
printTestParams(size_t N, size_t offx, int incx, size_t offy, int incy, size_t offParam, ComplexLong sflagParam)
{
    ::std::cerr << "N = " << N << ", offx = " << offx << ", incx = " << incx << ", offy = " << offy
                << ", incy = " << incy << ", offParam = " << offParam << ", PARAM[0] = " << sflagParam.re << std::endl;
}

//xROTMG
void
printTestParams(int offX, int offY, int offD1, int offD2, int offParam, ComplexLong sflagParam)
{
    ::std::cerr << "offX = " << offX << ", offY = " << offY << ", offD1 = " << offD1 << ", offD2 = " << offD2
                << ", offParam = " << offParam << ", PARAM[0] = " << sflagParam.re << std::endl;
}


// xNRM2, xASUM, iXAMAX
void
printTestParams(
    size_t N,
    size_t offx,
    int incx)
{
    ::std::cerr << "N = " << N << ", offx = " << offx << ", incx = " << incx << ::std::endl;
}

const char*
orderStr(clblasOrder order)
{
    switch (order) {
    case clblasColumnMajor:
        return "clblasColumnMajor";
    case clblasRowMajor:
        return "clblasRowMajor";
    default:
        return NULL;
    }
}

const char*
sideStr(clblasSide side)
{
    switch (side) {
    case clblasLeft:
        return "clblasLeft";
    case clblasRight:
        return "clblasRight";
    default:
        return NULL;
    }
}

const char*
uploStr(clblasUplo uplo)
{
    switch (uplo) {
    case clblasUpper:
        return "clblasUpper";
    case clblasLower:
        return "clblasLower";
    default:
        return NULL;
    }
}

const char*
transStr(clblasTranspose trans)
{
    switch (trans) {
    case clblasNoTrans:
        return "clblasNoTrans";
    case clblasTrans:
        return "clblasTrans";
    case clblasConjTrans:
        return "clblasConjTrans";
    default:
        return NULL;
    }
}

const char*
diagStr(clblasDiag diag)
{
    switch (diag) {
    case clblasNonUnit:
        return "clblasNonUnit";
    case clblasUnit:
        return "clblasUnit";
    default:
        return NULL;
    }
}

char
encodeTranspose(clblasTranspose value)
{
    switch (value) {
    case clblasNoTrans:      return 'N';
    case clblasTrans:        return 'T';
    case clblasConjTrans:    return 'C';
    }
    return '\0';
}

char
encodeUplo(clblasUplo value)
{
    switch (value) {
    case clblasUpper:  return 'U';
    case clblasLower:  return 'L';
    }
    return '\0';
}

char
encodeDiag(clblasDiag value)
{
    switch (value) {
    case clblasUnit:       return 'U';
    case clblasNonUnit:    return 'N';
    }
    return '\0';
}

char
encodeSide(clblasSide value)
{
    switch (value) {
    case clblasLeft:   return 'L';
    case clblasRight:  return 'R';
    }
    return '\0';
}

int
functionBlasLevel(BlasFunctionID funct) {
    switch (funct) {

    case FN_SSCAL:
    case FN_DSCAL:
    case FN_CSCAL:
    case FN_ZSCAL:
    case FN_CSSCAL:
    case FN_ZDSCAL:

    case FN_SSWAP:
    case FN_DSWAP:
    case FN_CSWAP:
    case FN_ZSWAP:

    case FN_SAXPY:
    case FN_DAXPY:
    case FN_CAXPY:
    case FN_ZAXPY:

	case FN_SDOT:
    case FN_DDOT:
    case FN_CDOTU:
    case FN_ZDOTU:
    case FN_CDOTC:
    case FN_ZDOTC:

	case FN_SCOPY:
    case FN_DCOPY:
    case FN_CCOPY:
    case FN_ZCOPY:

    case FN_SROTG:
    case FN_DROTG:
    case FN_CROTG:
    case FN_ZROTG:

    case FN_SROT:
    case FN_DROT:
    case FN_CSROT:
    case FN_ZDROT:

    case FN_SASUM:
    case FN_DASUM:
    case FN_SCASUM:
    case FN_DZASUM:

    case FN_SROTM:
    case FN_DROTM:

    case FN_SROTMG:
    case FN_DROTMG:

    case FN_SNRM2:
    case FN_DNRM2:
    case FN_SCNRM2:
    case FN_DZNRM2:

    case FN_iSAMAX:
    case FN_iDAMAX:
    case FN_iCAMAX:
    case FN_iZAMAX:

    return 1;

    case FN_SGEMV:
    case FN_DGEMV:
    case FN_CGEMV:
    case FN_ZGEMV:

    case FN_SSYMV:
    case FN_DSYMV:
    case FN_SSPMV:
    case FN_DSPMV:

    case FN_STRMV:
    case FN_DTRMV:
    case FN_CTRMV:
    case FN_ZTRMV:

    case FN_STPMV:
    case FN_DTPMV:
    case FN_CTPMV:
    case FN_ZTPMV:

    case FN_STRSV:
    case FN_DTRSV:
    case FN_CTRSV:
    case FN_ZTRSV:

    case FN_STPSV:
    case FN_DTPSV:
    case FN_CTPSV:
    case FN_ZTPSV:

    case FN_SGER:
    case FN_DGER:
    case FN_CGERU:
    case FN_ZGERU:
    case FN_CGERC:
    case FN_ZGERC:

    case FN_CHER:
    case FN_ZHER:
    case FN_CHER2:
    case FN_ZHER2:

    case FN_CHPR:
    case FN_ZHPR:
    case FN_CHPR2:
    case FN_ZHPR2:

	case FN_SSYR:
	case FN_DSYR:
    case FN_SSPR:
	case FN_DSPR:

	case FN_SSYR2:
	case FN_DSYR2:
    case FN_SSPR2:
	case FN_DSPR2:

	case FN_CHEMV:
	case FN_ZHEMV:
    case FN_CHPMV:
    case FN_ZHPMV:

    case FN_SGBMV:
	case FN_DGBMV:
	case FN_CGBMV:
	case FN_ZGBMV:

	case FN_STBMV:
	case FN_DTBMV:
	case FN_CTBMV:
	case FN_ZTBMV:

	case FN_SSBMV:
	case FN_DSBMV:

	case FN_CHBMV:
	case FN_ZHBMV:

	case FN_STBSV:
	case FN_DTBSV:
	case FN_CTBSV:
	case FN_ZTBSV:

    return 2;

	case FN_CHEMM:
    case FN_ZHEMM:

    case FN_SSYMM:
    case FN_DSYMM:
    case FN_CSYMM:
    case FN_ZSYMM:

    case FN_SGEMM:
    case FN_DGEMM:
    case FN_CGEMM:
    case FN_ZGEMM:

    case FN_SGEMM_2:
    case FN_DGEMM_2:
    case FN_CGEMM_2:
    case FN_ZGEMM_2:

    case FN_STRMM:
    case FN_DTRMM:
    case FN_CTRMM:
    case FN_ZTRMM:

    case FN_STRSM:
    case FN_DTRSM:
    case FN_CTRSM:
    case FN_ZTRSM:

    case FN_SSYR2K:
    case FN_DSYR2K:
    case FN_CSYR2K:
    case FN_ZSYR2K:

    case FN_SSYRK:
    case FN_DSYRK:
    case FN_CSYRK:
    case FN_ZSYRK:

	case FN_CHERK:
	case FN_ZHERK:
	case FN_CHER2K:
	case FN_ZHER2K:

        return 3;
    default:
        return 0;
    }
}
