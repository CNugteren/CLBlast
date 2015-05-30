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


//#include <stdlib.h>             // srand()
#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>
//
//#include "common.h"
//#include "blas.h"
#include "blas-wrapper.h"
#include "clBLAS-wrapper.h"
#include "BlasBase.h"
#include "blas-random.h"
#include "timer.h"

#include "func.h"

template <typename M>
class MQueueClass
{
    M metod;
protected:
    void init();
    void run();
    void destroy();
public:
    void testQueue();
};

template <typename M> void
MQueueClass<M>::init()
{
    size_t maxElem = 1024*2;

    metod.initDefault(maxElem, 0);
    metod.generateData();

    metod.outEvent = NULL;

}

template <typename M> void
MQueueClass<M>::run()
{
    cl_int err;
    bool b = metod.prepareDataToRun();
    ASSERT_EQ(b, true);

    int qmax = metod.qnum;

    metod.initOutEvent();
    cl_int ret = CL_SUCCESS;

    err = metod.run();
    ASSERT_EQ(err, CL_SUCCESS);
    //::std::cerr << "queues = " << base->numCommandQueues() << std::endl;


    for (int q = 0; q < qmax; ++q) {
        err = clFinish(metod.queues[q]);
        ASSERT_EQ(err, CL_SUCCESS) << "clFinish()";

        err = clGetEventInfo(metod.outEvent[q], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &ret, NULL);
        //std::cerr << "2: err=" <<  err <<" ret=" <<  ret << std::endl;
        ASSERT_EQ(err, CL_SUCCESS) << "clGetEventInfo()";
        ASSERT_EQ(ret, CL_COMPLETE) << "clGetEventInfo()";
     }
}
template <typename M> void
MQueueClass<M>::destroy()
{
    metod.destroy();
}

template <typename M> void
MQueueClass<M>::testQueue()
{
    init();
    run();
    destroy();
}

#ifdef DO_THEIRS
//******************************************************//
TEST(QUEUE, sgemm) {
    MQueueClass< GemmMetod<float> > ec;
    ec.testQueue();
}

TEST(QUEUE, cgemm) {
    MQueueClass< GemmMetod<FloatComplex> > ec;
    ec.testQueue();
}

TEST(QUEUE, dgemm) {
    CHECK_DOUBLE;
    MQueueClass< GemmMetod<cl_double> > ec;
    ec.testQueue();
}

TEST(QUEUE, zgemm) {
    CHECK_DOUBLE;
    MQueueClass<GemmMetod<DoubleComplex> > ec;
    ec.testQueue();
}
//******************************************************//
TEST(QUEUE, strmm) {
    MQueueClass<TrmmMetod<float> > ec;
    ec.testQueue();
}

TEST(QUEUE, ctrmm) {
    MQueueClass<TrmmMetod<FloatComplex> > ec;
    ec.testQueue();
}

TEST(QUEUE, dtrmm) {
    CHECK_DOUBLE;
    MQueueClass<TrmmMetod<cl_double> > ec;
    ec.testQueue();
}

TEST(QUEUE, ztrmm) {
    CHECK_DOUBLE;
    MQueueClass<TrmmMetod<DoubleComplex> > ec;
    ec.testQueue();
}
//******************************************************//
TEST(QUEUE, strsm) {
    MQueueClass<TrsmMetod<float> > ec;
    ec.testQueue();
}

TEST(QUEUE, ctrsm) {
    MQueueClass<TrsmMetod<FloatComplex> > ec;
    ec.testQueue();
}

TEST(QUEUE, dtrsm) {
    CHECK_DOUBLE;
    MQueueClass<TrsmMetod<cl_double> > ec;
    ec.testQueue();
}

TEST(QUEUE, ztrsm) {
    CHECK_DOUBLE;
    MQueueClass<TrsmMetod<DoubleComplex> > ec;
    ec.testQueue();
}
//******************************************************//
TEST(QUEUE, sgemv) {
    MQueueClass<GemvMetod<float> > ec;
    ec.testQueue();
}
#if defined(_USE_GEMV_COMPLEX)
TEST(QUEUE, cgemv) {
    MQueueClass<GemvMetod<FloatComplex> > ec;
    ec.testQueue();
}
#endif
TEST(QUEUE, dgemv) {
    CHECK_DOUBLE;
    MQueueClass<GemvMetod<cl_double> > ec;
    ec.testQueue();
}
#if defined(_USE_GEMV_COMPLEX)
TEST(QUEUE, zgemv) {
    CHECK_DOUBLE;
    MQueueClass<GemvMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif
//******************************************************//
TEST(QUEUE, ssymv) {
    MQueueClass<SymvMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, dsymv) {
    CHECK_DOUBLE;
    MQueueClass<SymvMetod<cl_double> > ec;
    ec.testQueue();
}
//******************************************************//
TEST(QUEUE, ssyr2k) {
    MQueueClass<Syr2kMetod<float> > ec;
    ec.testQueue();
}

TEST(QUEUE, dsyr2k) {
    CHECK_DOUBLE;
    MQueueClass<Syr2kMetod<cl_double> > ec;
    ec.testQueue();
}
#endif  //DO_THEIRS

//******************************************************
#ifdef DO_TRMV
TEST(QUEUE, strmv) {
    MQueueClass< TrmvMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, dtrmv) {
	CHECK_DOUBLE;
    MQueueClass< TrmvMetod<cl_double> > ec;
    ec.testQueue();
}
TEST(QUEUE, ctrmv) {
    MQueueClass< TrmvMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, ztrmv) {
	CHECK_DOUBLE;
    MQueueClass< TrmvMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif
// ******************************************************/

#ifdef DO_TPMV
TEST(QUEUE, stpmv) {
    MQueueClass< TpmvMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, dtpmv) {
    CHECK_DOUBLE;
    MQueueClass< TpmvMetod<cl_double> > ec;
    ec.testQueue();
}
TEST(QUEUE, ctpmv) {
    MQueueClass< TpmvMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, ztpmv) {
    CHECK_DOUBLE;
    MQueueClass< TpmvMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_TRSV
TEST(QUEUE, strsv) {
    MQueueClass< TrsvMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, dtrsv) {
	CHECK_DOUBLE;
    MQueueClass< TrsvMetod<double> > ec;
    ec.testQueue();
}
TEST(QUEUE, ctrsv) {
    MQueueClass< TrsvMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, ztrsv) {
	CHECK_DOUBLE;
    MQueueClass< TrsvMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_TPSV
TEST(QUEUE, stpsv) {
    MQueueClass< TpsvMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, dtpsv) {
    CHECK_DOUBLE;
    MQueueClass< TpsvMetod<double> > ec;
    ec.testQueue();
}
TEST(QUEUE, ctpsv) {
    MQueueClass< TpsvMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, ztpsv) {
    CHECK_DOUBLE;
    MQueueClass< TpsvMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_SYMM
TEST(QUEUE, Ssymm) {
    MQueueClass< SymmMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Dsymm) {
	CHECK_DOUBLE;
    MQueueClass< SymmMetod<cl_double> > ec;
    ec.testQueue();
}
TEST(QUEUE, Csymm) {
    MQueueClass< SymmMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, Zsymm) {
	CHECK_DOUBLE;
    MQueueClass< SymmMetod<DoubleComplex> > ec;
    ec.testQueue();
}

#endif

#ifdef DO_SYR
TEST(QUEUE, Ssyr) {
    MQueueClass< SyrMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Dsyr) {
    CHECK_DOUBLE;
    MQueueClass< SyrMetod<cl_double> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_SPR
TEST(QUEUE, Sspr) {
    MQueueClass< SprMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Dspr) {
    CHECK_DOUBLE;
    MQueueClass< SprMetod<cl_double> > ec;
    ec.testQueue();
}
#endif


#ifdef DO_SYR2
TEST(QUEUE, Ssyr2) {
    MQueueClass< Syr2Metod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Dsyr2) {
    CHECK_DOUBLE;
    MQueueClass< Syr2Metod<cl_double> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_GER
TEST(QUEUE, sger) {
    MQueueClass< GerMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, dger) {
	CHECK_DOUBLE;
    MQueueClass< GerMetod<cl_double> > ec;
    ec.testQueue();
}
TEST(QUEUE, cger) {
    MQueueClass< GerMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, zger) {
	CHECK_DOUBLE;
    MQueueClass< GerMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_GERC
TEST(QUEUE, cgerc) {
    MQueueClass< GercMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, zgerc) {
	CHECK_DOUBLE;
    MQueueClass< GercMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_HER
TEST(QUEUE, cher) {
    MQueueClass< HerMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, zher) {
        CHECK_DOUBLE;
    MQueueClass<HerMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_HER2
TEST(QUEUE, cher2) {
    MQueueClass< Her2Metod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, zher2) {
        CHECK_DOUBLE;
    MQueueClass<Her2Metod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_HEMM
TEST(QUEUE, chemm) {
    MQueueClass< HemmMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, zhemm) {
        CHECK_DOUBLE;
    MQueueClass<HemmMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif


#ifdef DO_HEMV
TEST(QUEUE, chemv) {
    MQueueClass< HemvMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, zhemv) {
        CHECK_DOUBLE;
    MQueueClass<HemvMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_HERK
TEST(QUEUE, cherk) {
    MQueueClass< HerkMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, zherk) {
    CHECK_DOUBLE;
    MQueueClass<HerkMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif


#ifdef DO_HPMV
TEST(QUEUE, chpmv) {
    MQueueClass< HpmvMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, zhpmv) {
        CHECK_DOUBLE;
    MQueueClass<HpmvMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif


#ifdef DO_SPMV
TEST(QUEUE, sspmv) {
    MQueueClass<SpmvMetod<cl_float> > ec;
    ec.testQueue();
}
TEST(QUEUE, dspmv) {
        CHECK_DOUBLE;
    MQueueClass<SpmvMetod<cl_double> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_SPR2
TEST(QUEUE, Sspr2) {
    MQueueClass< Spr2Metod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Dspr2) {
    CHECK_DOUBLE;
    MQueueClass< Spr2Metod<cl_double> > ec;
    ec.testQueue();
}
#endif


#ifdef DO_HPR
TEST(QUEUE, chpr) {
    MQueueClass< HprMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, zhpr) {
        CHECK_DOUBLE;
    MQueueClass<HprMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_HPR2
TEST(QUEUE, chpr2) {
    MQueueClass< Hpr2Metod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, zhpr2) {
        CHECK_DOUBLE;
    MQueueClass<Hpr2Metod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_GBMV
TEST(QUEUE, SGBMV) {
    MQueueClass< GbmvMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, DGBMV) {
    CHECK_DOUBLE;
    MQueueClass< GbmvMetod<cl_double> > ec;
    ec.testQueue();
}
TEST(QUEUE, CGBMV) {
    MQueueClass< GbmvMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, ZGBMV) {
    CHECK_DOUBLE;
    MQueueClass< GbmvMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_SYR
TEST(QUEUE, Ssbmv) {
    MQueueClass< SbmvMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Dsbmv) {
    CHECK_DOUBLE;
    MQueueClass< SbmvMetod<cl_double> > ec;
    ec.testQueue();
}
#endif

//DOT

#ifdef DO_DOT
TEST(QUEUE, Sdot) {
    MQueueClass< DotMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Ddot) {
    CHECK_DOUBLE;
    MQueueClass< DotMetod<cl_double> > ec;
    ec.testQueue();
}

TEST(QUEUE, Cdotu) {
    MQueueClass< DotMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, Zdotu) {
    CHECK_DOUBLE;
    MQueueClass< DotMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

//ASUM
#ifdef DO_ASUM
TEST(QUEUE, Sasum) {
    MQueueClass< AsumMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Dasum) {
    CHECK_DOUBLE;
    MQueueClass< AsumMetod<cl_double> > ec;
    ec.testQueue();
}

TEST(QUEUE, Scasum) {
    MQueueClass< AsumMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, Dzasum) {
    CHECK_DOUBLE;
    MQueueClass< AsumMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

//iAMAX
#ifdef DO_iAMAX
TEST(QUEUE, iSamax) {
    MQueueClass< iAmaxMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, iDamax) {
    CHECK_DOUBLE;
    MQueueClass< iAmaxMetod<cl_double> > ec;
    ec.testQueue();
}

TEST(QUEUE, iCamax) {
    MQueueClass< iAmaxMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, iZamax) {
    CHECK_DOUBLE;
    MQueueClass< iAmaxMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

//DOTC
#ifdef DO_DOTC
TEST(QUEUE, Cdotc) {
    MQueueClass< DotcMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, Zdotc) {
    CHECK_DOUBLE;
    MQueueClass< DotcMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif


#ifdef DO_SYR
TEST(QUEUE, Chbmv) {
    MQueueClass< HbmvMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, Zhbmv) {
    CHECK_DOUBLE;
    MQueueClass< HbmvMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_TBMV
TEST(QUEUE, STBMV) {
    MQueueClass< TbmvMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, DTBMV) {
    CHECK_DOUBLE;
    MQueueClass< TbmvMetod<cl_double> > ec;
    ec.testQueue();
}
TEST(QUEUE, CTBMV) {
    MQueueClass< TbmvMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, ZTBMV) {
    CHECK_DOUBLE;
    MQueueClass< TbmvMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_TBSV
TEST(QUEUE, STBSV) {
    MQueueClass< TbsvMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, DTBSV) {
    CHECK_DOUBLE;
    MQueueClass< TbsvMetod<cl_double> > ec;
    ec.testQueue();
}
TEST(QUEUE, CTBSV) {
    MQueueClass< TbsvMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, ZTBSV) {
    CHECK_DOUBLE;
    MQueueClass< TbsvMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_HER2K
TEST(QUEUE, cher2k) {
    MQueueClass< Her2kMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, zher2k) {
    CHECK_DOUBLE;
    MQueueClass<Her2kMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif


#ifdef DO_SCAL
TEST(QUEUE, Sscal) {
    MQueueClass< ScalMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Dscal) {
    CHECK_DOUBLE;
    MQueueClass< ScalMetod<cl_double> > ec;
    ec.testQueue();
}
TEST(QUEUE, Cscal) {
    MQueueClass< ScalMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, Zscal) {
    CHECK_DOUBLE;
    MQueueClass< ScalMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_SSCAL
TEST(QUEUE, Csscal) {
    MQueueClass< SscalMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, Zdscal) {
    CHECK_DOUBLE;
    MQueueClass< SscalMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_SWAP
TEST(QUEUE, Sswap) {
    MQueueClass< SwapMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Dswap) {
    CHECK_DOUBLE;
    MQueueClass< SwapMetod<cl_double> > ec;
    ec.testQueue();
}
TEST(QUEUE, Cswap) {
    MQueueClass< SwapMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, Zswap) {
    CHECK_DOUBLE;
    MQueueClass< SwapMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif


#ifdef DO_COPY
TEST(QUEUE, Scopy) {
    MQueueClass< CopyMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Dcopy) {
    CHECK_DOUBLE;
    MQueueClass< CopyMetod<cl_double> > ec;
    ec.testQueue();
}
TEST(QUEUE, Ccopy) {
    MQueueClass< CopyMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, Zcopy) {
    CHECK_DOUBLE;
    MQueueClass< CopyMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_AXPY
TEST(QUEUE, Saxpy) {
    MQueueClass< AxpyMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Daxpy) {
    CHECK_DOUBLE;
    MQueueClass< AxpyMetod<cl_double> > ec;
    ec.testQueue();
}
TEST(QUEUE, Caxpy) {
    MQueueClass< AxpyMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, Zaxpy) {
    CHECK_DOUBLE;
    MQueueClass< AxpyMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_ROTG
TEST(QUEUE, Srotg) {
    MQueueClass< RotgMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Drotg) {
    CHECK_DOUBLE;
    MQueueClass< RotgMetod<cl_double> > ec;
    ec.testQueue();
}
TEST(QUEUE, Crotg) {
    MQueueClass< RotgMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, Zrotg) {
    CHECK_DOUBLE;
    MQueueClass< RotgMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_ROTM
TEST(QUEUE, Srotm) {
    MQueueClass< RotmMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Drotm) {
    CHECK_DOUBLE;
    MQueueClass< RotmMetod<cl_double> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_ROT
TEST(QUEUE, Srot) {
    MQueueClass< RotMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Drot) {
    CHECK_DOUBLE;
    MQueueClass< RotMetod<cl_double> > ec;
    ec.testQueue();
}

TEST(QUEUE, Csrot) {
    MQueueClass< RotMetod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, Zdrot) {
    CHECK_DOUBLE;
    MQueueClass< RotMetod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_ROTMG
TEST(QUEUE, Srotmg) {
    MQueueClass< RotmgMetod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Drotmg) {
    CHECK_DOUBLE;
    MQueueClass< RotmgMetod<cl_double> > ec;
    ec.testQueue();
}
#endif

#ifdef DO_NRM2
TEST(QUEUE, Snrm2) {
    MQueueClass< Nrm2Metod<float> > ec;
    ec.testQueue();
}
TEST(QUEUE, Dnrm2) {
    CHECK_DOUBLE;
    MQueueClass< Nrm2Metod<cl_double> > ec;
    ec.testQueue();
}

TEST(QUEUE, Scnrm2) {
    MQueueClass< Nrm2Metod<FloatComplex> > ec;
    ec.testQueue();
}
TEST(QUEUE, Dznrm2) {
    CHECK_DOUBLE;
    MQueueClass< Nrm2Metod<DoubleComplex> > ec;
    ec.testQueue();
}
#endif
