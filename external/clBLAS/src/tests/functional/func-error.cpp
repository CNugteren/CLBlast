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


#include <gtest/gtest.h>
#include <clBLAS.h>
#include "blas-wrapper.h"
#include "clBLAS-wrapper.h"
#include "BlasBase.h"
#include "blas-random.h"
#include "timer.h"
#include "func.h"



template <typename M>
class ErrorClass
{
    M metod;
protected:
    bool generateData();
public:
    void error(cl_int err_etalon);
//    nano_time_t runRepeat(int rep, cl_int* err);
};

template <typename T> bool
ErrorClass<T>::generateData()
{
    metod.generateData();
    bool ret = metod.prepareDataToRun();

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
ErrorClass<M>::error(cl_int err_etalon)
{
    metod.initDefault(1024, 1);
    cl_command_queue queues = metod.queues[0];
    if (generateData()) {
        switch (err_etalon) {
        case CL_INVALID_EVENT_WAIT_LIST:
            metod.inEvent = NULL;
            metod.inEventCount = 1;
            break;
        case CL_INVALID_EVENT:
            metod.outEvent = NULL;
            metod.inEventCount = 1;
            break;
            case CL_INVALID_CONTEXT:
            clReleaseContext(metod.context);
            break;
        case CL_INVALID_COMMAND_QUEUE:
            metod.queues[0] = NULL;
            break;
        case clblasInvalidMatA:
        case clblasInvalidVecX:
        case CL_INVALID_MEM_OBJECT:
            metod.bufA = NULL;
            metod.bufAP = NULL;
            metod.bufX = NULL;
            metod.bufY = NULL;
            break;
        case CL_INVALID_DEVICE:
            break;
        case clblasInsufficientMemMatA:
        case clblasInsufficientMemMatB:
        case clblasInsufficientMemVecX:
        case CL_INVALID_VALUE:
			metod.size = 2048;
            //metod.bufA = NULL;
            break;
        default:
            FAIL() << "Unknown Error cod " << err_etalon;
        }

        cl_int err = metod.run();

	    metod.queues[0] = queues;

        ASSERT_EQ(err, err_etalon) << "clFinish()";

    }
    metod.destroy();
}

#ifdef DO_THEIRS
// Instantiate the test
TEST(ERROR, InvalidCommandQueue) {
    ErrorClass<GemmMetod<float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitList) {
    ErrorClass<GemmMetod<float> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObject) {
    ErrorClass<GemmMetod<float> > ec;
    ec.error(clblasInvalidMatA);
}
TEST(ERROR, InvalidValue) {
    ErrorClass<GemmMetod<float> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevice) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
        ErrorClass<GemmMetod<double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}

// Instantiate the test
#endif

#ifdef DO_TRMV
TEST(ERROR, InvalidCommandQueuetrmv) {
    ErrorClass<TrmvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListtrmv) {
    ErrorClass<TrmvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjecttrmv) {
    ErrorClass<TrmvMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}
TEST(ERROR, InvalidValuetrmv) {
    ErrorClass<TrmvMetod<FloatComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevicetrmv) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
	ErrorClass<TrmvMetod<cl_double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_TRSV
TEST(ERROR, InvalidCommandQueue_trsv) {
    ErrorClass<TrsvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitList_trsv) {
    ErrorClass<TrsvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObject_trsv) {
    ErrorClass<TrsvMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}
TEST(ERROR, InvalidValue_trsv) {
    ErrorClass<TrsvMetod<FloatComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevice_trsv) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
        ErrorClass<TrsvMetod<double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_TPSV
TEST(ERROR, InvalidCommandQueue_tpsv) {
    ErrorClass<TpsvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitList_tpsv) {
    ErrorClass<TpsvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObject_tpsv) {
    ErrorClass<TpsvMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}
TEST(ERROR, InvalidValue_tpsv) {
    ErrorClass<TpsvMetod<FloatComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevice_tpsv) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
        ErrorClass<TpsvMetod<double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_TPMV
TEST(ERROR, InvalidCommandQueue_tpmv) {
    ErrorClass<TpmvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitList_tpmv) {
    ErrorClass<TpmvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObject_tpmv) {
    ErrorClass<TpmvMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}
TEST(ERROR, InvalidValue_tpmv) {
    ErrorClass<TpmvMetod<FloatComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevice_tpmv) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
        ErrorClass<TpmvMetod<double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif


#ifdef DO_SYMM
TEST(ERROR, InvalidCommandQueuesymm) {
    ErrorClass<SymmMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListsymm) {
    ErrorClass<SymmMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectsymm) {
    ErrorClass<SymmMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}
TEST(ERROR, InvalidValuesymm) {
    ErrorClass<SymmMetod<FloatComplex> > ec;
    ec.error(clblasInsufficientMemMatB);
}

TEST(ERROR, InvalidDevicesymm) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<SymmMetod<cl_double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_SYR
TEST(ERROR, InvalidCommandQueuesyr) {
    ErrorClass<SyrMetod<cl_float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListsyr) {
    ErrorClass<SyrMetod<cl_float> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectsyr) {
    ErrorClass<SyrMetod<cl_float> > ec;
    ec.error(clblasInvalidMatA);
}
TEST(ERROR, InvalidValuesyr) {
    ErrorClass<SyrMetod<cl_float> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevicesyr) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<SyrMetod<cl_double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_SPR
TEST(ERROR, InvalidCommandQueuespr) {
    ErrorClass<SprMetod<cl_float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListspr) {
    ErrorClass<SprMetod<cl_float> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectspr) {
    ErrorClass<SprMetod<cl_float> > ec;
    ec.error(clblasInvalidMatA);
}
TEST(ERROR, InvalidValuespr) {
    ErrorClass<SprMetod<cl_float> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevicespr) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<SprMetod<cl_double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif


#ifdef DO_SYR2
TEST(ERROR, InvalidCommandQueuesyr2) {
    ErrorClass<Syr2Metod<cl_float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListsyr2) {
    ErrorClass<Syr2Metod<cl_float> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectsyr2) {
    ErrorClass<Syr2Metod<cl_float> > ec;
    ec.error(clblasInvalidMatA);
}
TEST(ERROR, InvalidValuesyr2) {
    ErrorClass<Syr2Metod<cl_float> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevicesyr2) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<Syr2Metod<cl_double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_GER
TEST(ERROR, InvalidCommandQueueger) {
    ErrorClass<GerMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListger) {
    ErrorClass<GerMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectger) {
    ErrorClass<GerMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}
TEST(ERROR, InvalidValueger) {
    ErrorClass<GerMetod<FloatComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDeviceger) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
        ErrorClass<GerMetod<cl_double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_GERC
TEST(ERROR, InvalidCommandQueuegerc) {
    ErrorClass<GercMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListgerc) {
    ErrorClass<GercMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectgerc) {
    ErrorClass<GercMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}
TEST(ERROR, InvalidValuegerc) {
    ErrorClass<GercMetod<FloatComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevicegerc) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
        ErrorClass<GercMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif


#ifdef DO_HER
TEST(ERROR, InvalidCommandQueueher) {
    ErrorClass<HerMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListher) {
    ErrorClass<HerMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjecther) {
    ErrorClass<HerMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValueher) {

    ErrorClass<HerMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDeviceher) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<HerMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_HER2
TEST(ERROR, InvalidCommandQueueher2) {
    ErrorClass<Her2Metod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListher2) {
    ErrorClass<Her2Metod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjecther2) {
    ErrorClass<Her2Metod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValueher2) {

    ErrorClass<Her2Metod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDeviceher2) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<Her2Metod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_HEMM
TEST(ERROR, InvalidCommandQueuehemm) {
    ErrorClass<HemmMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListhemm) {
    ErrorClass<HemmMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjecthemm) {
    ErrorClass<HemmMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValuehemm) {

    ErrorClass<HemmMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemMatB);
}

TEST(ERROR, InvalidDevicehemm) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<HemmMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_HEMV
TEST(ERROR, InvalidCommandQueuehemv) {
    ErrorClass<HemvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListhemv) {
    ErrorClass<HemvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjecthemv) {
    ErrorClass<HemvMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValuehemv) {

    ErrorClass<HemvMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevicehemv) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<HemvMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_HERK
TEST(ERROR, InvalidCommandQueueherk) {
    ErrorClass<HerkMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListherk) {
    ErrorClass<HerkMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectherk) {
    ErrorClass<HerkMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValueherk) {
    ErrorClass<HerkMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDeviceherk) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<HerkMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif


#ifdef DO_HPMV

TEST(ERROR, InvalidCommandQueuehpmv) {
    ErrorClass<HpmvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListhpmv) {
    ErrorClass<HpmvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjecthpmv) {
    ErrorClass<HpmvMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValuehpmv) {

    ErrorClass<HpmvMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevicehpmv) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<HpmvMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif


#ifdef DO_SPMV
TEST(ERROR, InvalidCommandQueuespmv) {
    ErrorClass<SpmvMetod<cl_float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListspmv) {
    ErrorClass<SpmvMetod<cl_float> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectspmv) {
    ErrorClass<SpmvMetod<cl_float> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValuespmv) {

    ErrorClass<SpmvMetod<cl_double> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevicespmv) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<SpmvMetod<cl_double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_SPR2
TEST(ERROR, InvalidCommandQueuespr2) {
    ErrorClass<Spr2Metod<cl_float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListspr2) {
    ErrorClass<Spr2Metod<cl_float> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectspr2) {
    ErrorClass<Spr2Metod<cl_float> > ec;
    ec.error(clblasInvalidMatA);
}
TEST(ERROR, InvalidValuespr2) {
    ErrorClass<Spr2Metod<cl_float> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevicespr2) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<Spr2Metod<cl_double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif


#ifdef DO_HPR
TEST(ERROR, InvalidCommandQueuehpr) {
    ErrorClass<HprMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListhpr) {
    ErrorClass<HprMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjecthpr) {
    ErrorClass<HprMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValuehpr) {

    ErrorClass<HprMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevicehpr) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<HprMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_HPR2
TEST(ERROR, InvalidCommandQueuehpr2) {
    ErrorClass<Hpr2Metod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListhpr2) {
    ErrorClass<Hpr2Metod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjecthpr2) {
    ErrorClass<Hpr2Metod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValuehpr2) {

    ErrorClass<Hpr2Metod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevicehpr2) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<Hpr2Metod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif


#ifdef DO_GBMV
TEST(ERROR, InvalidCommandQueueGBMV) {
    ErrorClass<GbmvMetod<cl_float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListGBMV) {
    ErrorClass<GbmvMetod<cl_float> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectGBMV) {
    ErrorClass<GbmvMetod<cl_float> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValueGBMV) {
    ErrorClass<GbmvMetod<cl_double> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDeviceGBMV) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<GbmvMetod<cl_double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_SBMV
TEST(ERROR, InvalidCommandQueuesbmv) {
    ErrorClass<SbmvMetod<cl_float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListsbmv) {
    ErrorClass<SbmvMetod<cl_float> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectsbmv) {
    ErrorClass<SbmvMetod<cl_float> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValuesbmv) {
    ErrorClass<SbmvMetod<cl_float> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevicesbmv) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<SbmvMetod<cl_double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_HBMV
TEST(ERROR, InvalidCommandQueuehbmv) {
    ErrorClass<HbmvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListhbmv) {
    ErrorClass<HbmvMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjecthbmv) {
    ErrorClass<HbmvMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValuehbmv) {
    ErrorClass<HbmvMetod<FloatComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDevicehbmv) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<HbmvMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif


#ifdef DO_TBMV
TEST(ERROR, InvalidCommandQueueTBMV) {
    ErrorClass<TbmvMetod<cl_float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListTBMV) {
    ErrorClass<TbmvMetod<cl_float> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectTBMV) {
    ErrorClass<TbmvMetod<cl_float> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValueTBMV) {

    ErrorClass<TbmvMetod<cl_double> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDeviceTBMV) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<TbmvMetod<cl_double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_TBSV
TEST(ERROR, InvalidCommandQueueTBSV) {
    ErrorClass<TbsvMetod<cl_float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListTBSV) {
    ErrorClass<TbsvMetod<cl_float> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectTBSV) {
    ErrorClass<TbsvMetod<cl_float> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValueTBSV) {

    ErrorClass<TbsvMetod<cl_double> > ec;
    ec.error(clblasInsufficientMemVecX);
}

TEST(ERROR, InvalidDeviceTBSV) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<TbsvMetod<cl_double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_HER2K
TEST(ERROR, InvalidCommandQueueher2k) {
    ErrorClass<Her2kMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListher2k) {
    ErrorClass<Her2kMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjecther2k) {
    ErrorClass<Her2kMetod<FloatComplex> > ec;
    ec.error(clblasInvalidMatA);
}

TEST(ERROR, InvalidValueher2k) {
    ErrorClass<Her2kMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemMatA);
}

TEST(ERROR, InvalidDeviceher2k) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<Her2kMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_SCAL
TEST(ERROR, InvalidCommandQueuescal) {
    ErrorClass<ScalMetod<float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListscal) {
    ErrorClass<ScalMetod<double> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectscal) {
    ErrorClass<ScalMetod<FloatComplex> > ec;
    ec.error(clblasInvalidVecX);
}

TEST(ERROR, InvalidValuescal) {
    ErrorClass<ScalMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemVecX);
}

TEST(ERROR, InvalidDevicescal) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<ScalMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_SSCAL
TEST(ERROR, InvalidCommandQueuesscal) {
    ErrorClass<SscalMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListsscal) {
    ErrorClass<SscalMetod<DoubleComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectsscal) {
    ErrorClass<SscalMetod<FloatComplex> > ec;
    ec.error(clblasInvalidVecX);
}

TEST(ERROR, InvalidValuesscal) {
    ErrorClass<SscalMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemVecX);
}

TEST(ERROR, InvalidDevicesscal) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<SscalMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_SWAP
TEST(ERROR, InvalidCommandQueueswap) {
    ErrorClass<SwapMetod<float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListswap) {
    ErrorClass<SwapMetod<double> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectswap) {
    ErrorClass<SwapMetod<FloatComplex> > ec;
    ec.error(clblasInvalidVecX);
}

TEST(ERROR, InvalidValueswap) {
    ErrorClass<SwapMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemVecX);
}

TEST(ERROR, InvalidDeviceswap) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<SwapMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_COPY
TEST(ERROR, InvalidCommandQueuecopy) {
    ErrorClass<CopyMetod<float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListcopy) {
    ErrorClass<CopyMetod<double> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectcopy) {
    ErrorClass<CopyMetod<FloatComplex> > ec;
    ec.error(clblasInvalidVecX);
}

TEST(ERROR, InvalidValuecopy) {
    ErrorClass<CopyMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemVecX);
}

TEST(ERROR, InvalidDevicecopy) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<CopyMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif


#ifdef DO_AXPY
TEST(ERROR, InvalidCommandQueueaxpy) {
    ErrorClass<AxpyMetod<float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListaxpy) {
    ErrorClass<AxpyMetod<double> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectaxpy) {
    ErrorClass<AxpyMetod<FloatComplex> > ec;
    ec.error(clblasInvalidVecX);
}

TEST(ERROR, InvalidValueaxpy) {
    ErrorClass<AxpyMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemVecX);
}

TEST(ERROR, InvalidDeviceaxpy) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<AxpyMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

//DOT
#ifdef DO_DOT
TEST(ERROR, InvalidCommandQueuedot) {
    ErrorClass<DotMetod<cl_float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListdot) {
    ErrorClass<DotMetod<cl_double> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectdot) {
    ErrorClass<DotMetod<FloatComplex> > ec;
    ec.error(clblasInvalidVecX);
}
TEST(ERROR, InvalidValuedot) {
    ErrorClass<DotMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemVecX);
}

TEST(ERROR, InvalidDevicedot) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<DotMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_ASUM
TEST(ERROR, InvalidCommandQueueasum) {
    ErrorClass<AsumMetod<cl_float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListasum) {
    ErrorClass<AsumMetod<cl_double> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectasum) {
    ErrorClass<AsumMetod<FloatComplex> > ec;
    ec.error(clblasInvalidVecX);
}
TEST(ERROR, InvalidValueasum) {
    ErrorClass<AsumMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemVecX);
}

TEST(ERROR, InvalidDeviceasum) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<AsumMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_iAMAX
TEST(ERROR, InvalidCommandQueueiamax) {
    ErrorClass<iAmaxMetod<cl_float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListiamax) {
    ErrorClass<iAmaxMetod<cl_double> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectiamax) {
    ErrorClass<iAmaxMetod<FloatComplex> > ec;
    ec.error(clblasInvalidVecX);
}
TEST(ERROR, InvalidValueiamax) {
    ErrorClass<iAmaxMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemVecX);
}

TEST(ERROR, InvalidDeviceiamax) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<iAmaxMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

//DOTC
#ifdef DO_DOTC
TEST(ERROR, InvalidCommandQueuedotc) {
    ErrorClass<DotcMetod<FloatComplex> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListdotc) {
    ErrorClass<DotcMetod<DoubleComplex> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectdotc) {
    ErrorClass<DotcMetod<FloatComplex> > ec;
    ec.error(clblasInvalidVecX);
}
TEST(ERROR, InvalidValuedotc) {
    ErrorClass<DotcMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemVecX);
}

TEST(ERROR, InvalidDevicedotc) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<DotMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif


#ifdef DO_ROTG
TEST(ERROR, InvalidCommandQueuerotg) {
    ErrorClass<RotgMetod<float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListrotg) {
    ErrorClass<RotgMetod<double> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectrotg) {
    ErrorClass<RotgMetod<FloatComplex> > ec;
    ec.error(clblasInvalidVecX);
}

/*  Skipping Invalid value- because rotg doesn't depend on parameter N,
                            So even passing an invalid N doesn't matter
TEST(ERROR, InvalidValuerotg) {
    ErrorClass<RotgMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemVecX);
}
*/

TEST(ERROR, InvalidDevicerotg) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<RotgMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_ROTM
TEST(ERROR, InvalidCommandQueuerotm) {
    ErrorClass<RotmMetod<float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListrotm) {
    ErrorClass<RotmMetod<double> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectrotm) {
    ErrorClass<RotmMetod<float> > ec;
    ec.error(clblasInvalidVecX);
}

TEST(ERROR, InvalidValuerotm) {
    ErrorClass<RotmMetod<double> > ec;
    ec.error(clblasInsufficientMemVecX);
}

TEST(ERROR, InvalidDevicerotm) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<RotmMetod<double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_ROT
TEST(ERROR, InvalidCommandQueuerot) {
    ErrorClass<RotMetod<float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListrot) {
    ErrorClass<RotMetod<double> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectrot) {
    ErrorClass<RotMetod<FloatComplex> > ec;
    ec.error(clblasInvalidVecX);
}

TEST(ERROR, InvalidValuerot) {
    ErrorClass<RotMetod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemVecX);
}

TEST(ERROR, InvalidDevicerot) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<RotMetod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_ROTMG
TEST(ERROR, InvalidCommandQueuerotmg) {
    ErrorClass<RotmgMetod<float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListrotmg) {
    ErrorClass<RotmgMetod<double> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectrotmg) {
    ErrorClass<RotmgMetod<float> > ec;
    ec.error(clblasInvalidVecX);
}

/*  Skipping Invalid value- because rotg doesn't depend on parameter N,
                            So even passing an invalid N doesn't matter
TEST(ERROR, InvalidValuerotmg) {
    ErrorClass<RotmgMetod<double> > ec;
    ec.error(clblasInsufficientMemVecX);
}
*/

TEST(ERROR, InvalidDevicerotmg) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<RotmgMetod<double> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif

#ifdef DO_NRM2
TEST(ERROR, InvalidCommandQueuenrm2) {
    ErrorClass<Nrm2Metod<cl_float> > ec;
    ec.error(CL_INVALID_COMMAND_QUEUE);
}

TEST(ERROR, InvalidEventWaitListnrm2) {
    ErrorClass<Nrm2Metod<cl_double> > ec;
    ec.error(CL_INVALID_EVENT_WAIT_LIST);
}

TEST(ERROR, InvalidMemObjectnrm2) {
    ErrorClass<Nrm2Metod<FloatComplex> > ec;
    ec.error(clblasInvalidVecX);
}
TEST(ERROR, InvalidValuenrm2) {
    ErrorClass<Nrm2Metod<DoubleComplex> > ec;
    ec.error(clblasInsufficientMemVecX);
}

TEST(ERROR, InvalidDevicenrm2) {
    clMath::BlasBase* base = clMath::BlasBase::getInstance();
    if (!base->isDevSupportDoublePrecision()) {
    ErrorClass<Nrm2Metod<DoubleComplex> > ec;
        ec.error(CL_INVALID_DEVICE);
    }
}
#endif


