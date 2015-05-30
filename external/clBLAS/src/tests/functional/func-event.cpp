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
//#include <string.h>             // memcpy()
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
class EventClass
{
    M metod;
protected:
    void eventOutCorrectnessTest();
    void eventInCorrectnessTest();
    bool generateData();
public:
    void runOut();
    void runIn();
};
template <typename T> bool
EventClass<T>::generateData()
{
    metod.generateData();
    bool ret =metod.prepareDataToRun();
    if (!ret) {
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
    }
    return ret;
}

template <typename M> void
EventClass<M>::runOut()
{
    metod.initDefault(512*4, 1);
    eventOutCorrectnessTest();
    metod.destroy();
}

template <typename M> void
EventClass<M>::runIn()
{
    metod.initDefault(256, 1);
    eventInCorrectnessTest();
    metod.destroy();
}


template <typename M> void
EventClass<M>::eventOutCorrectnessTest()
{
    cl_int err;

    if (generateData()) {

        metod.initOutEvent();
        err = metod.run();
        ASSERT_EQ(err, CL_SUCCESS) << "clFinish()";

        //logEvent(events);
        err = clFinish(metod.queues[0]);

        ASSERT_EQ(err, CL_SUCCESS) << "clFinish()";

        cl_int ret = CL_SUCCESS;
        err = clGetEventInfo(*metod.outEvent, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &ret, NULL);
        ASSERT_EQ(err, CL_SUCCESS) << "clGetEventInfo()";
        ASSERT_EQ(ret, CL_COMPLETE) << "clGetEventInfo()";
    }
}

template <typename T> void
EventClass<T>::eventInCorrectnessTest()
{
    cl_int err;
    cl_int ret = CL_SUCCESS;
    int qmax = metod.qnum;
    nano_time_t minSleepTime = 100000000;


    if (generateData()) {

        metod.outEvent = new cl_event[1];
        metod.outEvent[0] = NULL;

        nano_time_t timeFirst = getCurrentTime();
        // First run.
        err = metod.run();
        ASSERT_EQ(err, CL_SUCCESS) << "clFinish()";
        for (int i = 0; i < qmax; ++i) {
            err = clFinish(metod.queues[i]);
        }
        ASSERT_EQ(err, CL_SUCCESS) << "clFinish()";
        timeFirst = getCurrentTime() - timeFirst;

        cl_event event = clCreateUserEvent(metod.context, &err);
        ASSERT_EQ(err, CL_SUCCESS) << "clCreateUserEvent()";

        metod.inEventCount = 1;
        metod.inEvent = &event;

        err = metod.run();
        ASSERT_EQ(err, CL_SUCCESS) << "runClBlasFunction()";

        clFlush(metod.queues[0]);

        //
        sleepTime((timeFirst < minSleepTime)? minSleepTime : timeFirst);

        clSetUserEventStatus(event, CL_COMPLETE);

        err = clFinish(metod.queues[0]);
        err = clGetEventInfo(metod.outEvent[0], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &ret, NULL);
        ASSERT_EQ(err, CL_SUCCESS) << "clGetEventInfo()";
        ASSERT_EQ(ret, CL_COMPLETE) << "clGetEventInfo()";

        clReleaseEvent(event);
        metod.inEventCount = 0;
        metod.inEvent = NULL;

    }
}
#ifdef DO_THEIRS
//******************************************************//
TEST(EVENT_OUT, sgemm) {
    EventClass< GemmMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, cgemm) {
    EventClass< GemmMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, dgemm) {
    CHECK_DOUBLE;
    EventClass< GemmMetod<cl_double> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, zgemm) {
    CHECK_DOUBLE;
    EventClass<GemmMetod<DoubleComplex> > ec;
    ec.runOut();
}
//******************************************************//
TEST(EVENT_OUT, strmm) {
    EventClass<TrmmMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, ctrmm) {
    EventClass<TrmmMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, dtrmm) {
    CHECK_DOUBLE;
    EventClass<TrmmMetod<cl_double> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, ztrmm) {
    CHECK_DOUBLE;
    EventClass<TrmmMetod<DoubleComplex> > ec;
    ec.runOut();
}
//******************************************************//
TEST(EVENT_OUT, strsm) {
    EventClass<TrsmMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, ctrsm) {
    EventClass<TrsmMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, dtrsm) {
    CHECK_DOUBLE;
    EventClass<TrsmMetod<cl_double> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, ztrsm) {
    CHECK_DOUBLE;
    EventClass<TrsmMetod<DoubleComplex> > ec;
    ec.runOut();
}
//******************************************************//
TEST(EVENT_OUT, sgemv) {
    EventClass<GemvMetod<float> > ec;
    ec.runOut();

}
#if defined(_USE_GEMV_COMPLEX)
TEST(EVENT_OUT, cgemv) {
    EventClass<GemvMetod<FloatComplex> > ec;
    ec.runOut();
}
#endif
TEST(EVENT_OUT, dgemv) {
    CHECK_DOUBLE;
    EventClass<GemvMetod<cl_double> > ec;
    ec.runOut();
}
#if defined(_USE_GEMV_COMPLEX)
TEST(EVENT_OUT, zgemv) {
    CHECK_DOUBLE;
    EventClass<GemvMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif
//******************************************************//
TEST(EVENT_OUT, ssymv) {
    EventClass<SymvMetod<float> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, dsymv) {
    CHECK_DOUBLE;
    EventClass<SymvMetod<cl_double> > ec;
    ec.runOut();
}
//******************************************************//
TEST(EVENT_OUT, ssyr2k) {
    EventClass<Syr2kMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, dsyr2k) {
    CHECK_DOUBLE;
    EventClass<Syr2kMetod<cl_double> > ec;
    ec.runOut();
}
//******************************************************//
//******************************************************//
TEST(EVENT_IN, sgemm) {
    EventClass<GemmMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, cgemm) {
    EventClass<GemmMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, dgemm) {
    CHECK_DOUBLE;
    EventClass<GemmMetod<cl_double> > ec;
    ec.runIn();
}
TEST(EVENT_IN, zgemm) {
    CHECK_DOUBLE;
    EventClass<GemmMetod<DoubleComplex> > ec;
    ec.runIn();
}
//******************************************************//
TEST(EVENT_IN, strmm) {
    EventClass<TrmmMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, ctrmm) {
    EventClass<TrsmMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, dgtrmm) {
    CHECK_DOUBLE;
    EventClass<TrmmMetod<cl_double> > ec;
    ec.runIn();
}
TEST(EVENT_IN, ztrmm) {
    CHECK_DOUBLE;
    EventClass<TrmmMetod<DoubleComplex> > ec;
    ec.runIn();
}
//******************************************************//
TEST(EVENT_IN, strsm) {
    EventClass<TrsmMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, ctrsm) {
    EventClass<TrsmMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, dtrsm) {
    CHECK_DOUBLE;
    EventClass<TrsmMetod<cl_double> > ec;
    ec.runIn();
}
TEST(EVENT_IN, ztrsm) {
    CHECK_DOUBLE;
    EventClass<TrsmMetod<DoubleComplex> > ec;
    ec.runIn();
}
//******************************************************//
TEST(EVENT_IN, sgemv) {
    EventClass<GemvMetod<float> > ec;
    ec.runIn();
}
#if defined(_USE_GEMV_COMPLEX)
TEST(EVENT_IN, cgemv) {
    EventClass<GemvMetod<FloatComplex> > ec;
    ec.runIn();
}
#endif
TEST(EVENT_IN, dgemv) {
    CHECK_DOUBLE;
    EventClass<GemvMetod<cl_double> > ec;
    ec.runIn();
}
#if defined(_USE_GEMV_COMPLEX)
TEST(EVENT_IN, zgemv) {
    CHECK_DOUBLE;
    EventClass<GemvMetod<DoubleComplex> > ec;
    ec.runIn();
}
#endif
//******************************************************//
TEST(EVENT_IN, ssymv) {
    EventClass<SymvMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, dsymv) {
    CHECK_DOUBLE;
    EventClass<SymvMetod<cl_double> > ec;
    ec.runIn();
}
//******************************************************//
TEST(EVENT_IN, ssyr2k) {
    EventClass<Syr2kMetod<float> > ec;
    ec.runIn();
}

TEST(EVENT_IN, dsyr2k) {
    CHECK_DOUBLE;
    EventClass<Syr2kMetod<cl_double> > ec;
    ec.runIn();
}
#endif

#ifdef DO_TRMV
// TRMV
//******************************************************//
TEST(EVENT_OUT, strmv) {
    EventClass<TrmvMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, dtrmv) {
	CHECK_DOUBLE;
    EventClass<TrmvMetod<double> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, ctrmv) {
    EventClass<TrmvMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, ztrmv) {
	CHECK_DOUBLE;
    EventClass<TrmvMetod<DoubleComplex> > ec;
    ec.runOut();
}

TEST(EVENT_IN, strmv) {
    EventClass<TrmvMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, dtrmv) {
	CHECK_DOUBLE;
    EventClass<TrmvMetod<double> > ec;
    ec.runIn();
}
TEST(EVENT_IN, ctrmv) {
    EventClass<TrmvMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, ztrmv) {
	CHECK_DOUBLE;
    EventClass<TrmvMetod<DoubleComplex> > ec;
    ec.runIn();
}
#endif

#ifdef DO_TPMV
TEST(EVENT_OUT, stpmv) {
    EventClass<TpmvMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, dtpmv) {
    CHECK_DOUBLE;
    EventClass<TpmvMetod<double> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, ctpmv) {
    EventClass<TpmvMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, ztpmv) {
    CHECK_DOUBLE;
    EventClass<TpmvMetod<DoubleComplex> > ec;
    ec.runOut();
}

TEST(EVENT_IN, stpmv) {
    EventClass<TpmvMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, dtpmv) {
    CHECK_DOUBLE;
    EventClass<TpmvMetod<double> > ec;
    ec.runIn();
}
TEST(EVENT_IN, ctpmv) {
    EventClass<TpmvMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, ztpmv) {
    CHECK_DOUBLE;
    EventClass<TpmvMetod<DoubleComplex> > ec;
    ec.runIn();
}
#endif

#ifdef DO_TRSV
//******************************************************//
// TRSV

TEST(EVENT_OUT, strsv) {
    EventClass<TrsvMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, dtrsv) {
	CHECK_DOUBLE;
    EventClass<TrsvMetod<double> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, ctrsv) {
    EventClass<TrsvMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, ztrsv) {
	CHECK_DOUBLE;
    EventClass<TrsvMetod<DoubleComplex> > ec;
    ec.runOut();
}

TEST(EVENT_IN, strsv) {
    EventClass<TrsvMetod<float> > ec;
    ec.runIn();
}

TEST(EVENT_IN, dtrsv) {
	CHECK_DOUBLE;
    EventClass<TrsvMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_IN, ctrsv) {
    EventClass<TrsvMetod<FloatComplex> > ec;
    ec.runIn();
}

TEST(EVENT_IN, ztrsv) {
	CHECK_DOUBLE;
    EventClass<TrsvMetod<DoubleComplex> > ec;
    ec.runIn();
}
#endif

#ifdef DO_TPSV
TEST(EVENT_OUT, stpsv) {
    EventClass<TpsvMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, dtpsv) {
    CHECK_DOUBLE;
    EventClass<TpsvMetod<double> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, ctpsv) {
    EventClass<TpsvMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, ztpsv) {
    CHECK_DOUBLE;
    EventClass<TpsvMetod<DoubleComplex> > ec;
    ec.runOut();
}

TEST(EVENT_IN, stpsv) {
    EventClass<TpsvMetod<float> > ec;
    ec.runIn();
}

TEST(EVENT_IN, dtpsv) {
    CHECK_DOUBLE;
    EventClass<TpsvMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_IN, ctpsv) {
    EventClass<TpsvMetod<FloatComplex> > ec;
    ec.runIn();
}

TEST(EVENT_IN, ztpsv) {
    CHECK_DOUBLE;
    EventClass<TpsvMetod<DoubleComplex> > ec;
    ec.runIn();
}
#endif


#ifdef DO_SYMM
TEST(EVENT_IN, Ssymm) {
    EventClass<SymmMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Dsymm) {
	CHECK_DOUBLE;
    EventClass<SymmMetod<double> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Csymm) {
    EventClass<SymmMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Zsymm) {
	CHECK_DOUBLE;
    EventClass<SymmMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Ssymm) {
    EventClass<SymmMetod<float> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Dsymm) {
	CHECK_DOUBLE;
    EventClass<SymmMetod<double> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Csymm) {
    EventClass<SymmMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Zsymm) {
	CHECK_DOUBLE;
    EventClass<SymmMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_SYR
TEST(EVENT_IN, Ssyr) {
    EventClass<SyrMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Dsyr) {
    CHECK_DOUBLE;
    EventClass<SyrMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Ssyr) {
    EventClass<SyrMetod<float> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Dsyr) {
    CHECK_DOUBLE;
    EventClass<SyrMetod<double> > ec;
    ec.runOut();
}
#endif

#ifdef DO_SPR
TEST(EVENT_IN, Sspr) {
    EventClass<SprMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Dspr) {
    CHECK_DOUBLE;
    EventClass<SprMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Sspr) {
    EventClass<SprMetod<float> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Dspr) {
    CHECK_DOUBLE;
    EventClass<SprMetod<double> > ec;
    ec.runOut();
}
#endif


#ifdef DO_SYR2
TEST(EVENT_IN, Ssyr2) {
    EventClass<Syr2Metod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Dsyr2) {
    CHECK_DOUBLE;
    EventClass<Syr2Metod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Ssyr2) {
    EventClass<Syr2Metod<float> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Dsyr2) {
    CHECK_DOUBLE;
    EventClass<Syr2Metod<double> > ec;
    ec.runOut();
}
#endif

#ifdef DO_GER
TEST(EVENT_IN, Sger) {
    EventClass<GerMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Dger) {
	CHECK_DOUBLE;
    EventClass<GerMetod<double> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Cgeru) {
    EventClass<GerMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Zgeru) {
	CHECK_DOUBLE;
    EventClass<GerMetod<DoubleComplex> > ec;
    ec.runIn();
}
TEST(EVENT_OUT, Sger) {
    EventClass<GerMetod<float> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Dger) {
    CHECK_DOUBLE;
    EventClass<GerMetod<double> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Cgeru) {
    EventClass<GerMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Zgeru) {
    CHECK_DOUBLE;
    EventClass<GerMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif


#ifdef DO_HER
TEST(EVENT_IN, Cher) {
    EventClass<HerMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Zher) {
    CHECK_DOUBLE;
    EventClass<HerMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Cher) {
    EventClass<HerMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Zher) {
    CHECK_DOUBLE;
    EventClass<HerMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_GERC

TEST(EVENT_IN, Cgerc) {
    EventClass<GercMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Zgerc) {
	CHECK_DOUBLE;
    EventClass<GercMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Cgerc) {
    EventClass<GercMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Zgerc) {
    CHECK_DOUBLE;
    EventClass<GercMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_HER2
TEST(EVENT_IN, Cher2) {
    EventClass<Her2Metod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Zher2) {
    CHECK_DOUBLE;
    EventClass<Her2Metod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Cher2) {
    EventClass<Her2Metod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Zher2) {
    CHECK_DOUBLE;
    EventClass<Her2Metod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_HEMM
TEST(EVENT_IN, Chemm) {
    EventClass<HemmMetod<FloatComplex> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Zhemm) {
    CHECK_DOUBLE;
    EventClass<HemmMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Chemm) {
    EventClass<HemmMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Zhemm) {
    CHECK_DOUBLE;
    EventClass<HemmMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif


#ifdef DO_HEMV
TEST(EVENT_IN, Chemv) {
    EventClass<HemvMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Zhemv) {
    CHECK_DOUBLE;
    EventClass<HemvMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Chemv) {
    EventClass<HemvMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Zhemv) {
    CHECK_DOUBLE;
    EventClass<HemvMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_HERK
TEST(EVENT_IN, Cherk) {
    EventClass<HerkMetod<FloatComplex> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Zherk) {
    CHECK_DOUBLE;
    EventClass<HerkMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Cherk) {
    EventClass<HerkMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Zherk) {
    CHECK_DOUBLE;
    EventClass<HerkMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif


#ifdef DO_HPMV
TEST(EVENT_IN, Chpmv) {
    EventClass<HpmvMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Zhpmv) {
    CHECK_DOUBLE;
    EventClass<HpmvMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Chpmv) {
    EventClass<HpmvMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Zhpmv) {
    CHECK_DOUBLE;
    EventClass<HpmvMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif


#ifdef DO_SPMV
TEST(EVENT_IN, Sspmv) {
    EventClass<SpmvMetod<cl_float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Dspmv) {
    CHECK_DOUBLE;
    EventClass<SpmvMetod<cl_double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Sspmv) {
    EventClass<SpmvMetod<cl_float> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Dspmv) {
    CHECK_DOUBLE;
    EventClass<SpmvMetod<cl_double> > ec;
    ec.runOut();
}
#endif

#ifdef DO_SPR2
TEST(EVENT_IN, Sspr2) {
    EventClass<Spr2Metod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Dspr2) {
    CHECK_DOUBLE;
    EventClass<Spr2Metod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Sspr2) {
    EventClass<Spr2Metod<float> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Dspr2) {
    CHECK_DOUBLE;
    EventClass<Spr2Metod<double> > ec;
    ec.runOut();
}
#endif


#ifdef DO_HPR
TEST(EVENT_IN, Chpr) {
    EventClass<HprMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Zhpr) {
    CHECK_DOUBLE;
    EventClass<HprMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Chpr) {
    EventClass<HprMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Zhpr) {
    CHECK_DOUBLE;
    EventClass<HprMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_HPR2
TEST(EVENT_IN, Chpr2) {
    EventClass<Hpr2Metod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Zhpr2) {
    CHECK_DOUBLE;
    EventClass<Hpr2Metod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Chpr2) {
    EventClass<Hpr2Metod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Zhpr2) {
    CHECK_DOUBLE;
    EventClass<Hpr2Metod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_GBMV
TEST(EVENT_IN, CGBMV) {
    EventClass<GbmvMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, ZGBMV) {
    CHECK_DOUBLE;
    EventClass<GbmvMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, CGBMV) {
    EventClass<GbmvMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, ZGBMV) {
    CHECK_DOUBLE;
    EventClass<GbmvMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_SBMV
TEST(EVENT_IN, Ssbmv) {
    EventClass<SbmvMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Dsbmv) {
    CHECK_DOUBLE;
    EventClass<SbmvMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Ssbmv) {
    EventClass<SbmvMetod<float> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Dsbmv) {
    CHECK_DOUBLE;
    EventClass<SbmvMetod<double> > ec;
    ec.runOut();
}
#endif

//DOT

#ifdef DO_DOT
TEST(EVENT_IN, Sdot) {
    EventClass<DotMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Ddot) {
    CHECK_DOUBLE;
    EventClass<DotMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Sdot) {
    EventClass<DotMetod<float> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Ddot) {
    CHECK_DOUBLE;
    EventClass<DotMetod<double> > ec;
    ec.runOut();
}

TEST(EVENT_IN, Cdotu) {
    EventClass<DotMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Zdotu) {
    CHECK_DOUBLE;
    EventClass<DotMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Cdotu) {
    EventClass<DotMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Zdotu) {
    CHECK_DOUBLE;
    EventClass<DotMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

//ASUM

#ifdef DO_ASUM
TEST(EVENT_IN, Sasum) {
    EventClass<AsumMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Dasum) {
    CHECK_DOUBLE;
    EventClass<AsumMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Sasum) {
    EventClass<AsumMetod<float> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Dasum) {
    CHECK_DOUBLE;
    EventClass<AsumMetod<double> > ec;
    ec.runOut();
}

TEST(EVENT_IN, Scasum) {
    EventClass<AsumMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Dzasum) {
    CHECK_DOUBLE;
    EventClass<AsumMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Scasum) {
    EventClass<AsumMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Dzasum) {
    CHECK_DOUBLE;
    EventClass<AsumMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

//iAMAX

#ifdef DO_iAMAX
TEST(EVENT_IN, iSamax) {
    EventClass<iAmaxMetod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, iDamax) {
    CHECK_DOUBLE;
    EventClass<iAmaxMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, iSamax) {
    EventClass<iAmaxMetod<float> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, iDamax) {
    CHECK_DOUBLE;
    EventClass<iAmaxMetod<double> > ec;
    ec.runOut();
}

TEST(EVENT_IN, iCamax) {
    EventClass<iAmaxMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, iZamax) {
    CHECK_DOUBLE;
    EventClass<iAmaxMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, iCamax) {
    EventClass<iAmaxMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, iZamax) {
    CHECK_DOUBLE;
    EventClass<iAmaxMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

//DOTC
#ifdef DO_DOTC
TEST(EVENT_IN, Cdotc) {
    EventClass<DotcMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Zdotc) {
    CHECK_DOUBLE;
    EventClass<DotcMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Cdotc) {
    EventClass<DotcMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Zdotc) {
    CHECK_DOUBLE;
    EventClass<DotcMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif


#ifdef DO_HBMV
TEST(EVENT_IN, Chbmv) {
    EventClass<HbmvMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Zhbmv) {
    CHECK_DOUBLE;
    EventClass<HbmvMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Chbmv) {
    EventClass<HbmvMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Zhbmv) {
    CHECK_DOUBLE;
    EventClass<HbmvMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif


#ifdef DO_TBMV
TEST(EVENT_IN, CTBMV) {
    EventClass<TbmvMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, ZTBMV) {
    CHECK_DOUBLE;
    EventClass<TbmvMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, CTBMV) {
    EventClass<TbmvMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, ZTBMV) {
    CHECK_DOUBLE;
    EventClass<TbmvMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_TBSV
TEST(EVENT_IN, CTBSV) {
    EventClass<TbsvMetod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, ZTBSV) {
    CHECK_DOUBLE;
    EventClass<TbsvMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, CTBSV) {
    EventClass<TbsvMetod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, ZTBSV) {
    CHECK_DOUBLE;
    EventClass<TbsvMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_HER2K
TEST(EVENT_IN, Cher2k) {
    EventClass<Her2kMetod<FloatComplex> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Zher2k) {
    CHECK_DOUBLE;
    EventClass<Her2kMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Cher2k) {
    EventClass<Her2kMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Zher2k) {
    CHECK_DOUBLE;
    EventClass<Her2kMetod<DoubleComplex> > ec;
    ec.runOut();
}

#endif


#ifdef DO_SCAL
TEST(EVENT_IN, Sscal) {
    EventClass<ScalMetod<float> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Dscal) {
    CHECK_DOUBLE;
    EventClass<ScalMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Sscal) {
    EventClass<ScalMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Dscal) {
    CHECK_DOUBLE;
    EventClass<ScalMetod<double> > ec;
    ec.runOut();
}
TEST(EVENT_IN, Cscal) {
    EventClass<ScalMetod<FloatComplex> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Zscal) {
    CHECK_DOUBLE;
    EventClass<ScalMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Cscal) {
    EventClass<ScalMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Zscal) {
    CHECK_DOUBLE;
    EventClass<ScalMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_SSCAL
TEST(EVENT_IN, Csscal) {
    EventClass<SscalMetod<FloatComplex> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Zdscal) {
    CHECK_DOUBLE;
    EventClass<SscalMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Csscal) {
    EventClass<SscalMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Zdscal) {
    CHECK_DOUBLE;
    EventClass<SscalMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_SWAP
TEST(EVENT_IN, Sswap) {
    EventClass<SwapMetod<float> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Dswap) {
    CHECK_DOUBLE;
    EventClass<SwapMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Sswap) {
    EventClass<SwapMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Dswap) {
    CHECK_DOUBLE;
    EventClass<SwapMetod<double> > ec;
    ec.runOut();
}
TEST(EVENT_IN, Cswap) {
    EventClass<SwapMetod<FloatComplex> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Zswap) {
    CHECK_DOUBLE;
    EventClass<SwapMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Cswap) {
    EventClass<SwapMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Zswap) {
    CHECK_DOUBLE;
    EventClass<SwapMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif


//copy

#ifdef DO_COPY
TEST(EVENT_IN, Scopy) {
    EventClass<CopyMetod<float> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Dcopy) {
    CHECK_DOUBLE;
    EventClass<CopyMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Scopy) {
    EventClass<CopyMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Dcopy) {
    CHECK_DOUBLE;
    EventClass<CopyMetod<double> > ec;
    ec.runOut();
}
TEST(EVENT_IN, Ccopy) {
    EventClass<CopyMetod<FloatComplex> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Zcopy) {
    CHECK_DOUBLE;
    EventClass<CopyMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Ccopy) {
    EventClass<CopyMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Zcopy) {
    CHECK_DOUBLE;
    EventClass<CopyMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_AXPY
TEST(EVENT_IN, Saxpy) {
    EventClass<AxpyMetod<float> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Daxpy) {
    CHECK_DOUBLE;
    EventClass<AxpyMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Saxpy) {
    EventClass<AxpyMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Daxpy) {
    CHECK_DOUBLE;
    EventClass<AxpyMetod<double> > ec;
    ec.runOut();
}

TEST(EVENT_IN, Caxpy) {
    EventClass<AxpyMetod<FloatComplex> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Zaxpy) {
    CHECK_DOUBLE;
    EventClass<AxpyMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Caxpy) {
    EventClass<AxpyMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Zaxpy) {
    CHECK_DOUBLE;
    EventClass<AxpyMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif



#ifdef DO_ROTG
TEST(EVENT_IN, Srotg) {
    EventClass<RotgMetod<float> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Drotg) {
    CHECK_DOUBLE;
    EventClass<RotgMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Srotg) {
    EventClass<RotgMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Drotg) {
    CHECK_DOUBLE;
    EventClass<RotgMetod<double> > ec;
    ec.runOut();
}
TEST(EVENT_IN, Crotg) {
    EventClass<RotgMetod<FloatComplex> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Zrotg) {
    CHECK_DOUBLE;
    EventClass<RotgMetod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Crotg) {
    EventClass<RotgMetod<FloatComplex> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Zrotg) {
    CHECK_DOUBLE;
    EventClass<RotgMetod<DoubleComplex> > ec;
    ec.runOut();
}
#endif

#ifdef DO_ROTM
TEST(EVENT_IN, Srotm) {
    EventClass<RotmMetod<float> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Drotm) {
    CHECK_DOUBLE;
    EventClass<RotmMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Srotm) {
    EventClass<RotmMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Drotm) {
    CHECK_DOUBLE;
    EventClass<RotmMetod<double> > ec;
    ec.runOut();
}
#endif

#ifdef DO_ROT
TEST(EVENT_IN, Srot) {
    EventClass<RotMetod<float> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Drot) {
    CHECK_DOUBLE;
    EventClass<RotMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Csrot) {
    EventClass<RotMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Zdrot) {
    CHECK_DOUBLE;
    EventClass<RotMetod<double> > ec;
    ec.runOut();
}
#endif

#ifdef DO_ROTMG
TEST(EVENT_IN, Srotmg) {
    EventClass<RotmgMetod<float> > ec;
    ec.runIn();
}

TEST(EVENT_IN, Drotmg) {
    CHECK_DOUBLE;
    EventClass<RotmgMetod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Srotmg) {
    EventClass<RotmgMetod<float> > ec;
    ec.runOut();
}

TEST(EVENT_OUT, Drotmg) {
    CHECK_DOUBLE;
    EventClass<RotmgMetod<double> > ec;
    ec.runOut();
}
#endif

#ifdef DO_NRM2
TEST(EVENT_IN, Snrm2) {
    EventClass<Nrm2Metod<float> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Dnrm2) {
    CHECK_DOUBLE;
    EventClass<Nrm2Metod<double> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Snrm2) {
    EventClass<Nrm2Metod<float> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Dnrm2) {
    CHECK_DOUBLE;
    EventClass<Nrm2Metod<double> > ec;
    ec.runOut();
}

TEST(EVENT_IN, Scnrm2) {
    EventClass<Nrm2Metod<FloatComplex> > ec;
    ec.runIn();
}
TEST(EVENT_IN, Dznrm2) {
    CHECK_DOUBLE;
    EventClass<Nrm2Metod<DoubleComplex> > ec;
    ec.runIn();
}

TEST(EVENT_OUT, Scnrm2) {
    EventClass<Nrm2Metod<FloatComplex> > ec;
    ec.runOut();
}
TEST(EVENT_OUT, Dznrm2) {
    CHECK_DOUBLE;
    EventClass<Nrm2Metod<DoubleComplex> > ec;
    ec.runOut();
}
#endif
