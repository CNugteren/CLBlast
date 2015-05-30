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

#include <stdio.h>

template <typename M>
class ImagesClass
{
    enum
    {
        I_DEFAULT = -1,
        I_BUFERS,
        I_IMAGES,
        I_CASHES
    };

    M metod;
protected:
    bool generateData();
    void setImplementation(int i);
public:
    void images();
    nano_time_t runRepeat(int rep, cl_int* err);
};
template <typename T> void
ImagesClass<T>::setImplementation(int i)
{
    char str[100];
    clMath::BlasBase *base = clMath::BlasBase::getInstance();


    if (i != I_IMAGES) {
        if (base->useImages()) {
            base->removeScratchImages();
        }
        base->setUseImages(false);
    }

#if WIN32
    if (i == I_DEFAULT) {
        sprintf (str, "%s=", metod.env);
    }
    else {
        sprintf (str, "%s=%i",metod.env, i);
    }
    _putenv(str);
#else
    if (i == I_DEFAULT) {
        str[0] = '\0';
    }
    else {
        sprintf (str, "%i", i);
    }

    setenv(metod.env, str, 1);
#endif

    if (i == I_IMAGES) {
        base->setUseImages(true);
        if (base->useImages()) {
            if (base->addScratchImages()) {
                std::cerr << ">> FATAL ERROR, CANNOT CREATE SCRATCH IMAGES!"
                          << std::endl
                          << ">> Test skipped." << ::std::endl;
                SUCCEED();
            }
        }
   }

}

template <typename T> bool
ImagesClass<T>::generateData()
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

template <typename M> nano_time_t
ImagesClass<M>::runRepeat(int rep, cl_int* err)
{
    nano_time_t time1 = getCurrentTime();
    for (int i= 0; i < rep; i++) {
        nano_time_t time = getCurrentTime();
        *err = metod.run();
        if (*err != CL_SUCCESS) {
            return 0;
        }
        *err = clFinish(metod.queues[0]);
        if (*err != CL_SUCCESS) {
            return 0;
        }
        time = getCurrentTime() - time;
        time1 = (time < time1)?time:time1 ;
    }
    return time1;
}

template <typename M> void
ImagesClass<M>::images()
{
    cl_int err;
    int i= 6;
    int iMax = 30;
    nano_time_t maxTime = 1000;
    nano_time_t minTime = 100;
    bool next = true;

    do {
        nano_time_t time;

        metod.initDefault(256*i, 1);
        bool b = generateData();
        ASSERT_EQ(b, true) << "generateData()";
        setImplementation(I_BUFERS);
        metod.initOutEvent();
        time = runRepeat(2, &err);
        ASSERT_EQ(err, CL_SUCCESS) << "clFinish()";
        //std::cerr << "size = " << 256*i << "/" << i << " time = " << conv2millisec(time) << std::endl;
        if (conv2millisec(time) < minTime) {
            i += (((int)minTime - (int)conv2millisec(time)) /20) + 1;
			metod.destroy();
            continue;
        }
        if (conv2millisec(time) > maxTime) {
            i = iMax;
			metod.destroy();
            continue;
        }
		next = false;

		nano_time_t time1 = runRepeat(5, &err);
        ASSERT_EQ(err, CL_SUCCESS) << "clFinish()";

        setImplementation(I_IMAGES);

        nano_time_t time2 = runRepeat(5, &err);
        ASSERT_EQ(err, CL_SUCCESS) << "clFinish()";

        setImplementation(I_DEFAULT);

        //nano_time_t time3 = runRepeat(5, & err);
        //ASSERT_EQ(err, CL_SUCCESS) << "clFinish()";

        double d = (double)(time1) / time2;
        std::cerr << "size = " << 256*i
                  << "  timeBufer = " << conv2millisec(time1)
                  << "  timeImage = " << conv2millisec(time2)
                  << "  t1/t2 = " << d << std::endl;

        if (d < 1.2) {
            next = true;
            i++;
        }
        metod.destroy();

    } while (i < iMax && next);

    ASSERT_TRUE(!next) ;

}

// Instantiate the test

//******************************************************/
TEST(IMAGES, sgemm) {
    ImagesClass<GemmMetod<float> > ec;
    ec.images();
}

TEST(IMAGES, cgemm) {
    ImagesClass<GemmMetod<FloatComplex> > ec;
    ec.images();
}

TEST(IMAGES, dgemm) {
    CHECK_DOUBLE;
    ImagesClass<GemmMetod<cl_double> > ec;
    ec.images();
}

TEST(IMAGES, zgemm) {
    CHECK_DOUBLE;
    ImagesClass<GemmMetod<DoubleComplex> > ec;
    ec.images();
}//******************************************************/
TEST(IMAGES, strmm) {
    ImagesClass<TrmmMetod<float> > ec;
    ec.images();
}

TEST(IMAGES, ctrmm) {
    ImagesClass<TrmmMetod<FloatComplex> > ec;
    ec.images();
}

TEST(IMAGES, dtrmm) {
    CHECK_DOUBLE;
    ImagesClass<TrmmMetod<cl_double> > ec;
    ec.images();
}

TEST(IMAGES, ztrmm) {
    CHECK_DOUBLE;
    ImagesClass<TrmmMetod<DoubleComplex> > ec;
    ec.images();
}
//******************************************************/
TEST(IMAGES, strsm) {
    ImagesClass<TrsmMetod<float> > ec;
    ec.images();
}

TEST(IMAGES, ctrsm) {
    ImagesClass<TrsmMetod<FloatComplex> > ec;
    ec.images();
}

TEST(IMAGES, dtrsm) {
    CHECK_DOUBLE;
    ImagesClass<TrsmMetod<cl_double> > ec;
    ec.images();
}

TEST(IMAGES, ztrsm) {
    CHECK_DOUBLE;
    ImagesClass<TrsmMetod<DoubleComplex> > ec;
    ec.images();
}
//******************************************************/
