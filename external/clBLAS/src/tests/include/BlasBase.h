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


#ifndef BLASBASE_H_
#define BLASBASE_H_

#include <clBLAS.h>
#include <common.h>
#include <algorithm>

#if _MSC_VER
#pragma warning (disable:4127)
#endif

// check it is double precision error and return
#define CHECK_DP_ERROR_AND_RETURN(err, funcName)                            \
do {                                                                        \
    clMath::BlasBase *base = clMath::BlasBase::getInstance();                     \
                                                                            \
    if (err == CL_INVALID_DEVICE && !base->isDevSupportDoublePrecision()) { \
        ::std::cerr << std::endl << ">> " << funcName <<                    \
                "() reported that this device doesn't support double "      \
                "precision floating point arithmetic. Test is skipped" <<   \
        ::std::endl;                                                        \
        SUCCEED();                                                          \
                                                                            \
        return;                                                             \
    }                                                                       \
} while (0)                                                                 \

#define DEFAULT_SEED 12345
#define MAX_COMMAND_QUEUES 10

namespace clMath {

// This class is a singleton

class BlasBase {
private:
    cl_platform_id platform_;
    // used in all cases
    cl_device_id primaryDevice_;
    /*
     * used only  in cases with MultipleQueues to cover problem distribution
     * among different devices, not only different queues belonging to the same
     * device
     */
    cl_device_id additionalDevice_;
    cl_context context_;
    cl_command_queue commandQueues_[MAX_COMMAND_QUEUES];

    bool useNumCommandQueues_;
    cl_uint numCommandQueues_;

    bool useAlpha_;
    bool useBeta_;
    ComplexLong alpha_;
    ComplexLong beta_;

    bool useSeed_;
    unsigned int seed_;

    bool useM_, useN_, useK_;
    size_t M_, N_, K_;

    bool useIncX_, useIncY_;
    int incX_, incY_;

    bool useImages_;
    cl_device_type devType_;
    const char* devName_;
    cl_ulong imageA_;
    cl_ulong imageB_;

    BlasBase();
    ~BlasBase();
    BlasBase(const BlasBase &);             // intentionally undefined
    BlasBase & operator=(const BlasBase &); // intentionally undefined

    void SetUp();
    void TearDown();
    bool initialized();

    cl_int getPlatforms(cl_platform_id** platforms, cl_int *error);
    cl_device_id getDevice(cl_device_type type, const char* name,
        cl_int *error);
    void printDevInfoStr(cl_device_info param, const char *paramName,
                         int primAdd);

public:
    static BlasBase* getInstance();

    cl_context context()
    {
        return context_;
    };

    cl_command_queue* commandQueues() const
    {
        return const_cast<cl_command_queue*>(commandQueues_);
    };

    bool useNumCommandQueues() const    { return useNumCommandQueues_; };
    cl_uint numCommandQueues() const    { return numCommandQueues_; };
    void setNumCommandQueues(cl_uint numCommandQueues)
    {
        if (numCommandQueues <= MAX_COMMAND_QUEUES) {
            numCommandQueues_ = numCommandQueues;
            useNumCommandQueues_ = true;
        }
    }

    bool useAlpha() const        { return useAlpha_; }
    ComplexLong alpha() const   { return alpha_; }
    void setAlpha(ComplexLong alpha)
    {
        alpha_ = alpha;
        useAlpha_ = true;
    }

    bool useBeta() const         { return useBeta_; }
    ComplexLong beta() const    { return beta_; }
    void setBeta(ComplexLong beta)
    {
        beta_ = beta;
        useBeta_ = true;
    }

    bool useSeed() const        { return useSeed_; };
    unsigned int seed() const   { return seed_; };
    void setSeed(unsigned int seed)
    {
        seed_ = seed;
        useSeed_ = true;
    }

    bool useM() const           { return useM_; };
    size_t M() const            { return M_; }
    void setM(size_t M)
    {
        M_ = M;
        useM_ = true;
    }

    bool useN() const           { return useN_; };
    size_t N() const            { return N_; }
    void setN(size_t N)
    {
        N_ = N;
        useN_ = true;
    }

    bool useK() const           { return useK_; };
    size_t K() const            { return K_; }
    void setK(size_t K)
    {
        K_ = K;
        useK_ = true;
    }

    bool useIncX() const        { return useIncX_; };
    int incX() const            { return incX_; }
    void setIncX(int incX)
    {
        incX_ = incX;
        useIncX_ = true;
    }

    bool useIncY() const        { return useIncY_; };
    int incY() const            { return incY_; }
    void setIncY(int incY)
    {
        incY_ = incY;
        useIncY_ = true;
    }

    bool useImages() const      { return useImages_; };
    void setUseImages(bool value)
    {
        useImages_ = value;
    }
    void setUseImages(int value)
    {
        useImages_ = (value != 0);
    }

    bool setDeviceType(cl_device_type* devType, const char* devName);
    cl_mem createEnqueueBuffer(const void *data, size_t matrSize, size_t off,
                               cl_mem_flags mode);
    cl_mem readBuffer(void *ptr, size_t off, size_t size);

    clblasStatus addScratchImages(void);
    void removeScratchImages(void);
    size_t scratchImageWidth(void);
    size_t scratchImageHeight(void);

    cl_ulong maxMemAllocSize(void);
    cl_ulong availGlobalMemSize(int primAdd);

    bool isDevSupportDoublePrecision(void);
    // print information on environment the test run in
    void printEnvInfo(void);

    void release(void)
    {
        TearDown();
    }
};

}   // namespace

#endif  // BLASBASE_H_
