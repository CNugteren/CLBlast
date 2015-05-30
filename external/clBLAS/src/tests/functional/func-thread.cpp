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
#include <symv.h>
#include "func.h"

// Parallel thread
#define P_TH 5

#if defined(_MSC_VER)
#include "windows.h"
#include "process.h"


#define THREAD_ID HANDLE
#define THREAD_START(ID, DATA) \
     ID = (HANDLE)_beginthreadex(NULL, 0, &phfunc<M>, &DATA, 0, NULL);
#define THREAD_WAIT(ID, RET) \
{ \
    DWORD r;\
    WaitForSingleObject(ID, INFINITE); \
    GetExitCodeThread(ID, &r);\
    RET = (r == 1);\
}

template <typename M>
unsigned __stdcall
phfunc(void* vm)
{
    unsigned ret;
    M* m = (M*) vm;

    cl_uint err = m->run();
    clWaitForEvents(1, m->outEvent);

    err = m->getResult();
    ret = (err == CL_SUCCESS)? 1:0;
    _endthreadex(ret);
    return ret;
}

#else /* defined(_MCS_VER) */
#include "pthread.h"

#define THREAD_ID pthread_t
#define THREAD_START(ID, DATA) \
     pthread_create(&ID, NULL, phfunc<M>, &DATA)
#define THREAD_WAIT(ID, RET) \
{ \
    void* r;\
    int res = pthread_join(pt[i], &r); \
    (void) res; \
    RET =(bool)r;\
}



template <typename M>
void*
phfunc(void* vm)
{
    M* m = (M*) vm;

    cl_uint err = m->run();
    clWaitForEvents(1, m->outEvent);
    sleep(1);

    err = m->getResult();
    return (void *)(err == CL_SUCCESS);
}

#endif


template <typename M>
class MThreadClass
{
    M s_metod;
    M m_metod[P_TH];
protected:
    void init();
    void run();
    void destroy();
public:
    void mthread();
};

template <typename M> void
MThreadClass<M>::init()
{
    //size_t maxElem = 1024; PENDING: Make it back to 1024
    size_t maxElem = 128;

    s_metod.initDefault(maxElem, 1);
    s_metod.generateData();

    for (int i=0; i < P_TH; ++i ) {
        m_metod[i].initDefault(maxElem, 1);
        //m_metod[i].generateData();
        m_metod[i].copyData(s_metod);
    }
}

template <typename M> void
MThreadClass<M>::run()
{
    cl_int err;
    bool b = s_metod.prepareDataToRun();
    ASSERT_EQ(b, true);
    for (int i=0; i < P_TH; ++i ) {
        bool b = m_metod[i].prepareDataToRun();
        m_metod[i].initOutEvent();
        ASSERT_EQ(b, true);
    }

    err = s_metod.run();
    if (err == CL_SUCCESS) {
        err = clFinish(s_metod.queues[0]);
        ASSERT_EQ(err, CL_SUCCESS) << "clFinish()";

        err = s_metod.getResult();
        ASSERT_EQ(err, CL_SUCCESS);

        THREAD_ID pt[P_TH];

        for (int i=0; i < P_TH; ++i ) {
             THREAD_START(pt[i], m_metod[i]);
        }

        for (int i=0; i < P_TH; ++i ) {
            bool ret;
            THREAD_WAIT(pt[i], ret);
            EXPECT_EQ(ret, true);
            s_metod.compareData(m_metod[i]);
        }
    }
    else {
        ::std::cerr << ">> Test skipped." << err <<::std::endl;
        SUCCEED();
        return;
    }
}
template <typename M> void
MThreadClass<M>::destroy()
{
    s_metod.destroy();
    for (int i=0; i < P_TH; ++i ) {
        m_metod[i].destroy();
    }
}

template <typename M> void
MThreadClass<M>::mthread()
{
    init();
    run();
    destroy();
}

#ifdef DO_THEIRS
TEST(THREAD, sgemm) {
    MThreadClass<GemmMetod<cl_float> > ec;
    ec.mthread();
}
TEST(THREAD, cgemm) {
    MThreadClass<GemmMetod<FloatComplex> > ec;
    ec.mthread();
}
TEST(THREAD, dgemm) {
    CHECK_DOUBLE;
    MThreadClass<GemmMetod<cl_double> > ec;
    ec.mthread();
}
TEST(THREAD, zgemm) {
    CHECK_DOUBLE;
    MThreadClass<GemmMetod<DoubleComplex> > ec;
    ec.mthread();
}

TEST(THREAD, strmm) {
    MThreadClass<TrmmMetod<float> > ec;
    ec.mthread();
}
TEST(THREAD, ctrmm) {
    MThreadClass<TrmmMetod<FloatComplex> > ec;
    ec.mthread();
}
TEST(THREAD, dtrmm) {
    CHECK_DOUBLE;
    MThreadClass<TrmmMetod<cl_double> > ec;
    ec.mthread();
}
TEST(THREAD, ztrmm) {
    CHECK_DOUBLE;
    MThreadClass<TrmmMetod<DoubleComplex> > ec;
    ec.mthread();
}
//////////////////////////////////////////////////////////////
TEST(THREAD, strsm) {
    MThreadClass<TrsmMetod<float> > ec;
    ec.mthread();
}
TEST(THREAD, ctrsm) {
    MThreadClass<TrsmMetod<FloatComplex> > ec;
    ec.mthread();
}
TEST(THREAD, dtrsm) {
    CHECK_DOUBLE;
    MThreadClass<TrsmMetod<cl_double> > ec;
    ec.mthread();
}
TEST(THREAD, ztrsm) {
    CHECK_DOUBLE;
    MThreadClass<TrsmMetod<DoubleComplex> > ec;
    ec.mthread();
}
//////////////////////////////////////////////////////////////
TEST(THREAD, sgemv) {
    MThreadClass<GemvMetod<float> > ec;
    ec.mthread();
}
#if defined(_USE_GEMV_COMPLEX)
TEST(THREAD, cgemv) {
    MThreadClass<GemvMetod<FloatComplex> > ec;
    ec.mthread();
}
#endif
TEST(THREAD, dgemv) {
    CHECK_DOUBLE;
    MThreadClass<GemvMetod<cl_double> > ec;
    ec.mthread();
}

#if defined(_USE_GEMV_COMPLEX)
TEST(THREAD, zgemv) {
    CHECK_DOUBLE;
    MThreadClass<GemvMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

//////////////////////////////////////////////////////////////
TEST(THREAD, ssymv) {
    MThreadClass<SymvMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, dsymv) {
    CHECK_DOUBLE;
    MThreadClass<SymvMetod<cl_double> > ec;
    ec.mthread();
}
//******************************************************//
TEST(THREAD, ssyr2k) {
    MThreadClass<Syr2kMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, dsyr2k) {
    CHECK_DOUBLE;
    MThreadClass<Syr2kMetod<cl_double> > ec;
    ec.mthread();
}
#endif //DO_THIERS

#ifdef DO_TRMV
TEST(THREAD, strmv) {
    MThreadClass<TrmvMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, dtrmv) {
	CHECK_DOUBLE;
    MThreadClass<TrmvMetod<cl_double> > ec;
    ec.mthread();
}

TEST(THREAD, ctrmv) {
    MThreadClass<TrmvMetod<FloatComplex> > ec;
    ec.mthread();
}
TEST(THREAD, ztrmv) {
	CHECK_DOUBLE;
    MThreadClass<TrmvMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_TPMV
TEST(THREAD, stpmv) {
    MThreadClass<TpmvMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, dtpmv) {
    CHECK_DOUBLE;
    MThreadClass<TpmvMetod<cl_double> > ec;
    ec.mthread();
}

TEST(THREAD, ctpmv) {
    MThreadClass<TpmvMetod<FloatComplex> > ec;
    ec.mthread();
}
TEST(THREAD, ztpmv) {
    CHECK_DOUBLE;
    MThreadClass<TpmvMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif


#ifdef DO_TRSV
TEST(THREAD, strsv) {
    MThreadClass<TrsvMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, dtrsv) {
	CHECK_DOUBLE;
    MThreadClass<TrsvMetod<double> > ec;
    ec.mthread();
}

TEST(THREAD, ctrsv) {
    MThreadClass<TrsvMetod<FloatComplex> > ec;
    ec.mthread();
}
TEST(THREAD, ztrsv) {
	CHECK_DOUBLE;
    MThreadClass<TrsvMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_TPSV
TEST(THREAD, stpsv) {
    MThreadClass<TpsvMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, dtpsv) {
    CHECK_DOUBLE;
    MThreadClass<TpsvMetod<double> > ec;
    ec.mthread();
}

TEST(THREAD, ctpsv) {
    MThreadClass<TpsvMetod<FloatComplex> > ec;
    ec.mthread();
}
TEST(THREAD, ztpsv) {
    CHECK_DOUBLE;
    MThreadClass<TpsvMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_SYMM
TEST(THREAD, Ssymm) {
    MThreadClass<SymmMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, Dsymm) {
	CHECK_DOUBLE;
    MThreadClass<SymmMetod<cl_double> > ec;
    ec.mthread();
}

TEST(THREAD, Csymm) {
    MThreadClass<SymmMetod<FloatComplex> > ec;
    ec.mthread();
}
TEST(THREAD, Zsymm) {
	CHECK_DOUBLE;
    MThreadClass<SymmMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_SYR
TEST(THREAD, Ssyr) {
    MThreadClass<SyrMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, Dsyr) {
    CHECK_DOUBLE;
    MThreadClass<SyrMetod<cl_double> > ec;
    ec.mthread();
}
#endif

#ifdef DO_SPR
TEST(THREAD, Sspr) {
    MThreadClass<SprMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, Dspr) {
    CHECK_DOUBLE;
    MThreadClass<SprMetod<cl_double> > ec;
    ec.mthread();
}
#endif


#ifdef DO_SYR2
TEST(THREAD, Ssyr2) {
    MThreadClass<Syr2Metod<float> > ec;
    ec.mthread();
}

TEST(THREAD, Dsyr2) {
    CHECK_DOUBLE;
    MThreadClass<Syr2Metod<cl_double> > ec;
    ec.mthread();
}
#endif

#ifdef DO_GER
TEST(THREAD, Sger) {
    MThreadClass<GerMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, Dger) {
	CHECK_DOUBLE;
    MThreadClass<GerMetod<cl_double> > ec;
    ec.mthread();
}

TEST(THREAD, Cger) {
    MThreadClass<GerMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zger) {
	CHECK_DOUBLE;
    MThreadClass<GerMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_GERC
TEST(THREAD, Cgerc) {
    MThreadClass<GercMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zgerc) {
	CHECK_DOUBLE;
    MThreadClass<GercMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_HER
TEST(THREAD, Cher) {
    MThreadClass<HerMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zher) {
    CHECK_DOUBLE;
    MThreadClass<HerMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_HER2
TEST(THREAD, Cher2) {
    MThreadClass<Her2Metod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zher2) {
    CHECK_DOUBLE;
    MThreadClass<Her2Metod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_HEMM
TEST(THREAD, Chemm) {
    MThreadClass<HemmMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zhemm) {
    CHECK_DOUBLE;
    MThreadClass<HemmMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif


#ifdef DO_HEMV
TEST(THREAD, Chemv) {
    MThreadClass<HemvMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zhemv) {
    CHECK_DOUBLE;
    MThreadClass<HemvMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_HERK
TEST(THREAD, Cherk) {
    MThreadClass<HerkMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zherk) {
    CHECK_DOUBLE;
    MThreadClass<HerkMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_HPMV
TEST(THREAD, Chpmv) {
    MThreadClass<HpmvMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zhpmv) {
    CHECK_DOUBLE;
    MThreadClass<HpmvMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif


#ifdef DO_SPMV
TEST(THREAD, Sspmv) {
    MThreadClass<SpmvMetod<cl_float> > ec;
    ec.mthread();
}

TEST(THREAD, Dspmv) {
    CHECK_DOUBLE;
    MThreadClass<SpmvMetod<cl_double> > ec;
    ec.mthread();
}
#endif


#ifdef DO_SPR2
TEST(THREAD, Sspr2) {
    MThreadClass<Spr2Metod<float> > ec;
    ec.mthread();
}

TEST(THREAD, Dspr2) {
    CHECK_DOUBLE;
    MThreadClass<Spr2Metod<cl_double> > ec;
    ec.mthread();
}
#endif

#ifdef DO_HPR
TEST(THREAD, Chpr) {
    MThreadClass<HprMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zhpr) {
    CHECK_DOUBLE;
    MThreadClass<HprMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_HPR2
TEST(THREAD, Chpr2) {
    MThreadClass<Hpr2Metod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zhpr2) {
    CHECK_DOUBLE;
    MThreadClass<Hpr2Metod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_GBMV
TEST(THREAD, SGBMV) {
    MThreadClass<GbmvMetod<cl_float> > ec;
    ec.mthread();
}

TEST(THREAD, ZGBMV) {
    CHECK_DOUBLE;
    MThreadClass<GbmvMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_SBMV
TEST(THREAD, Ssbmv) {
    MThreadClass<SbmvMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, Dsbmv) {
    CHECK_DOUBLE;
    MThreadClass<SbmvMetod<cl_double> > ec;
    ec.mthread();
}
#endif

#ifdef DO_HBMV
TEST(THREAD, Chbmv) {
    MThreadClass<HbmvMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zhbmv) {
    CHECK_DOUBLE;
    MThreadClass<HbmvMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif


#ifdef DO_TBMV
TEST(THREAD, STBMV) {
    MThreadClass<TbmvMetod<cl_float> > ec;
    ec.mthread();
}

TEST(THREAD, ZTBMV) {
    CHECK_DOUBLE;
    MThreadClass<TbmvMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_TBSV
TEST(THREAD, STBSV) {
    MThreadClass<TbsvMetod<cl_float> > ec;
    ec.mthread();
}

TEST(THREAD, ZTBSV) {
    CHECK_DOUBLE;
    MThreadClass<TbsvMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_HER2K
TEST(THREAD, Cher2k) {
    MThreadClass<Her2kMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zher2k) {
    CHECK_DOUBLE;
    MThreadClass<Her2kMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_SCAL
TEST(THREAD, Sscal) {
    MThreadClass<ScalMetod<cl_float> > ec;
    ec.mthread();
}

TEST(THREAD, Zscal) {
    CHECK_DOUBLE;
    MThreadClass<ScalMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_SSCAL
TEST(THREAD, Csscal) {
    MThreadClass<SscalMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zdscal) {
    CHECK_DOUBLE;
    MThreadClass<ScalMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_SWAP
TEST(THREAD, Sswap) {
    MThreadClass<SwapMetod<cl_float> > ec;
    ec.mthread();
}

TEST(THREAD, Zswap) {
    CHECK_DOUBLE;
    MThreadClass<SwapMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_AXPY
TEST(THREAD, Saxpy) {
    MThreadClass<AxpyMetod<cl_float> > ec;
    ec.mthread();
}

TEST(THREAD, Zaxpy) {
    CHECK_DOUBLE;
    MThreadClass<AxpyMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_COPY
TEST(THREAD, Scopy) {
    MThreadClass<CopyMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, Dcopy) {
        CHECK_DOUBLE;
    MThreadClass<CopyMetod<cl_double> > ec;
    ec.mthread();
}

TEST(THREAD, Ccopy) {
    MThreadClass<CopyMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zcopy) {
        CHECK_DOUBLE;
    MThreadClass<CopyMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

//DOT
#ifdef DO_DOT
TEST(THREAD, Sdot) {
    MThreadClass<DotMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, Ddot) {
    CHECK_DOUBLE;
    MThreadClass<DotMetod<cl_double> > ec;
    ec.mthread();
}

TEST(THREAD, Cdotu) {
    MThreadClass<DotMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zdotu) {
    CHECK_DOUBLE;
    MThreadClass<DotMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

//ASUM
#ifdef DO_ASUM
TEST(THREAD, Sasum) {
    MThreadClass<AsumMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, Dasum) {
    CHECK_DOUBLE;
    MThreadClass<AsumMetod<cl_double> > ec;
    ec.mthread();
}

TEST(THREAD, Scasum) {
    MThreadClass<AsumMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Dzasum) {
    CHECK_DOUBLE;
    MThreadClass<AsumMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

//iAMAX
#ifdef DO_iAMAX
TEST(THREAD, iSamax) {
    MThreadClass<iAmaxMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, iDamax) {
    CHECK_DOUBLE;
    MThreadClass<iAmaxMetod<cl_double> > ec;
    ec.mthread();
}

TEST(THREAD, iCamax) {
    MThreadClass<iAmaxMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, iZamax) {
    CHECK_DOUBLE;
    MThreadClass<iAmaxMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

//DOTC
#ifdef DO_DOTC
TEST(THREAD, Cdotc) {
    MThreadClass<DotcMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zdotc) {
    CHECK_DOUBLE;
    MThreadClass<DotcMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif


#ifdef DO_ROTG
TEST(THREAD, Srotg) {
    MThreadClass<RotgMetod<cl_float> > ec;
    ec.mthread();
}

TEST(THREAD, Zrotg) {
    CHECK_DOUBLE;
    MThreadClass<RotgMetod<DoubleComplex> > ec;
    ec.mthread();
}
#endif

#ifdef DO_ROTM
TEST(THREAD, Srotm) {
    MThreadClass<RotmMetod<cl_float> > ec;
    ec.mthread();
}
#endif

#ifdef DO_ROT
TEST(THREAD, Srot) {
    MThreadClass<RotMetod<float> > ec;
    ec.mthread();
}

TEST(THREAD, Drot) {
    CHECK_DOUBLE;
    MThreadClass<RotMetod<cl_double> > ec;
    ec.mthread();
}

TEST(THREAD, Csrot) {
    MThreadClass<RotMetod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Zdrot) {
    CHECK_DOUBLE;
    MThreadClass<RotMetod<DoubleComplex> > ec;
    ec.mthread();
}


#endif

#ifdef DO_ROTMG
TEST(THREAD, Srotmg) {
    MThreadClass<RotmgMetod<cl_float> > ec;
    ec.mthread();
}
#endif

#ifdef DO_NRM2
TEST(THREAD, Snrm2) {
    MThreadClass<Nrm2Metod<float> > ec;
    ec.mthread();
}

TEST(THREAD, Dnrm2) {
    CHECK_DOUBLE;
    MThreadClass<Nrm2Metod<cl_double> > ec;
    ec.mthread();
}

TEST(THREAD, Scnrm2) {
    MThreadClass<Nrm2Metod<FloatComplex> > ec;
    ec.mthread();
}

TEST(THREAD, Dznrm2) {
    CHECK_DOUBLE;
    MThreadClass<Nrm2Metod<DoubleComplex> > ec;
    ec.mthread();
}
#endif
