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


#include <string.h>
#include <iostream>
#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <BlasBase.h>

namespace clMath {

BlasBase* BlasBase::getInstance()
{
    static BlasBase instance;

    if (!instance.initialized()) {
      return NULL;
    }
    return &instance;
}

BlasBase::BlasBase()
    : platform_(0), primaryDevice_(0), additionalDevice_(0), context_(NULL),
    useNumCommandQueues_(false), numCommandQueues_(1),
    useAlpha_(false), useBeta_(false), useSeed_(false),
    useM_(false), useN_(false), useK_(false),
    M_(0), N_(0), K_(0),
    useIncX_(false), useIncY_(false),
    incX_(0), incY_(0),
    useImages_(false), devType_(CL_DEVICE_TYPE_GPU), imageA_(0), imageB_(0)
{
    memset(&alpha_, 0, sizeof(alpha_));
    memset(&beta_, 0, sizeof(beta_));
    memset(commandQueues_, 0, sizeof(commandQueues_));

    SetUp();
}

BlasBase::~BlasBase()
{
    /*
     * Teardown() is disabled due to troubles with test interrupting
     * with CTRL-C in windows. This occurs since after pressing of these keys
     * the OpenCL runtime is destroyed before calling global object destructors.
     */
#if 0
    TearDown();
#endif
}

cl_int
BlasBase::getPlatforms(cl_platform_id **platforms, cl_int *error)
{
    cl_int err;
    cl_uint nrPlatforms;

    //platforms = NULL;

    if (error != NULL) {
        *error = CL_SUCCESS;
    }

    err = clGetPlatformIDs(0, NULL, &nrPlatforms);
    if (err != CL_SUCCESS) {
        if (error != NULL) {
            *error = err;
        }
        return 0;
    }
    if (nrPlatforms == 0) {
        return 0;
    }

    *platforms = new cl_platform_id[nrPlatforms];
    err = clGetPlatformIDs(nrPlatforms, *platforms, NULL);
    if (err != CL_SUCCESS) {
        if (error != NULL) {
            *error = err;
        }
        delete[] platforms;
        return 0;
    }
    return nrPlatforms;
}

cl_device_id
BlasBase::getDevice(cl_device_type type, const char* name,
                       cl_int *error)
{
    cl_int err;
    cl_uint nrDevices, i, p;
    cl_device_id *devices, result = NULL;
    size_t sz;
    char *str;
    cl_platform_id *platforms, selPlatform = NULL;
    cl_uint nrPlatforms;
    cl_device_info devInfo;

    nrPlatforms = getPlatforms(&platforms, &err);

    if (error != NULL) {
        *error = CL_SUCCESS;
    }

    /*
     * If device name is not specified, then any AMD device is preferable.
     * It there are not AMD devices of such a type presented in the system,
     * then get a device of another vendor. If this is the additional device
     * which is being tried to get, it must be supported by the same platform
     * as the primary device does.
     */

    if (name == NULL) {
        name = "Advanced Micro Devices, Inc.";
        devInfo = CL_DEVICE_VENDOR;
    }
    else {
        devInfo = CL_DEVICE_NAME;
        type = CL_DEVICE_TYPE_ALL;
    }

    for (p = 0; p < nrPlatforms; p++) {
        cl_platform_id platform = platforms[p];
        err = clGetDeviceIDs(platform, type, 0, NULL, &nrDevices);
        if (err == CL_DEVICE_NOT_FOUND) {
            continue;
        }
        if (err != CL_SUCCESS) {
            if (error != NULL) {
                *error = err;
            }
            return NULL;
        }
        if (nrDevices == 0) {
            return NULL;
        }

        devices = new cl_device_id[nrDevices];
        err = clGetDeviceIDs(platform, type, nrDevices, devices, NULL);
        if (err != CL_SUCCESS) {
            if (error != NULL) {
                *error = err;
            }
            delete[] devices;
            return NULL;
        }

        for (i = 0; i < nrDevices; i++) {
            err = clGetDeviceInfo(devices[i], devInfo, 0, NULL, &sz);
            if (err != CL_SUCCESS) {
                continue;
            }
            str = new char[sz + 1];
            memset(str, 0, sz + 1);
            err = clGetDeviceInfo(devices[i], devInfo, sz, str, NULL);
            if (err != CL_SUCCESS) {
                delete[] str;
                continue;
            }
            if ((devInfo == CL_DEVICE_VENDOR) && (result == NULL) &&
                ((platform_ == NULL) || (platform == platform_))) {

                result = devices[i];
                selPlatform = platform;
            }
                printf("---- %s\n", str);
            if (strcmp(str, name) == 0) {
                //printf("---- %s\n", str);
                platform_ = platform;
                result = devices[i];
                delete[] str;
                break;
            }
            delete[] str;
        }
        delete[] devices;
        devices = NULL;
    }

    if (platform_ == NULL) {
        platform_ = selPlatform;
    }

    delete[] platforms;
    return result;
}

void
BlasBase::SetUp()
{
    cl_int err = CL_SUCCESS;
    cl_context_properties props[] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_uint i = 1;
    cl_uint addDevQueueIdx = MAX_COMMAND_QUEUES;
    cl_device_id devices[2] = {NULL, NULL};

    primaryDevice_ = getDevice(devType_, devName_, &err);
    if ((err != CL_SUCCESS) || (primaryDevice_ == NULL)) {
        ASSERT_EQ(CL_SUCCESS, clGetPlatformIDs(1, &platform_, NULL));
        ASSERT_EQ(CL_SUCCESS,
            clGetDeviceIDs(platform_, devType_, 1, &primaryDevice_, NULL));
    }

    devices[0] = primaryDevice_;

#if !defined(TEST_WITH_SINGLE_DEVICE)
    cl_device_type addDevType;

    if (MAX_COMMAND_QUEUES > 1) {
        addDevType = (devType_ == CL_DEVICE_TYPE_GPU) ? CL_DEVICE_TYPE_CPU :
                                                    CL_DEVICE_TYPE_GPU;
        additionalDevice_ = getDevice(addDevType, NULL, NULL);
        if (additionalDevice_ != NULL) {
            addDevQueueIdx = (MAX_COMMAND_QUEUES <= 3) ?
                (MAX_COMMAND_QUEUES - 1) : 2;
            devices[1] = additionalDevice_;
            i = 2;
        }
    }
#endif  /* !TEST_WITH_SINGLE_DEVICE */

    props[1] = (cl_context_properties)platform_;

    context_ = clCreateContext(props, i, devices, NULL, NULL, &err);
    ASSERT_EQ(CL_SUCCESS, err) << "clCreateContext() failed";
	#ifdef DEBUG_CONTEXT
	printf("SetUp: Created context %p\n", context_);
	#endif
	printf("SetUp: about to create command queues\n");
    for (i = 0; i < MAX_COMMAND_QUEUES; i++) {
        cl_device_id dev;

        dev = (i == addDevQueueIdx) ? additionalDevice_ : primaryDevice_;
        commandQueues_[i] = clCreateCommandQueue(context_, dev,
            0 /*CL_QUEUE_PROFILING_ENABLE*/, &err);
        ASSERT_EQ(CL_SUCCESS, err) << "clCreateCommandQueue() failed";
    }

    ASSERT_EQ(CL_SUCCESS, clblasSetup());
}

void
BlasBase::TearDown()
{
    cl_uint i;

    for (i = 0; i < MAX_COMMAND_QUEUES; i++) {
        clReleaseCommandQueue(commandQueues_[i]);
    }
    numCommandQueues_ = 1;

    if (context_ != NULL) {
        clReleaseContext(context_);
        context_ = NULL;
    }

    primaryDevice_ = additionalDevice_ = NULL;

    clblasTeardown();
}

bool
BlasBase::initialized()
{
    return (context_ != NULL);
}

bool
BlasBase::setDeviceType(cl_device_type* devType, const char* devName)
{
    if (devType_ == *devType && devName_ == devName) {
        return true;
    }

    devType_ = *devType;
    devName_ = devName;
    if (!initialized()) {
        return true;
    }
    TearDown();
    SetUp();
    *devType = devType_;
    return initialized();
}

cl_mem
BlasBase::createEnqueueBuffer(
    const void *data,
    size_t matrSize,
    size_t off,
    cl_mem_flags mode)
{
    cl_int err;
    cl_mem buf;
    cl_uint i;

	#ifdef DEBUG_CONTEXT
	cl_uint refcnt;
	printf("BLASBASE: createEnqBuff - Querying context %p\n", context_);
	if (clGetContextInfo(context_, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &refcnt, NULL) != CL_SUCCESS)
	{
		printf("BLASBASE: clGetContextInfo FAILED\n");
	} else {
		printf("BLASBASE: REFCNT = %u\n", refcnt);
	}
	#endif
    buf = clCreateBuffer(context_, mode, matrSize + off, NULL, &err);

	if ( data != NULL ) {
    if (err == CL_SUCCESS ) {
        for (i = 0; i < numCommandQueues_; i++) {
            err = clEnqueueWriteBuffer(commandQueues_[i], buf, CL_TRUE,
                                       off, matrSize, data, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(buf);
                return NULL;
            }
        }
    }
	}

    return buf;
}

bool
BlasBase::isDevSupportDoublePrecision(void)
{
    cl_int err;
    cl_uint v;
    size_t len;
    char *extensions, *s;

    /* Check for cl_khr_fp64 extension */

    err = clGetDeviceInfo(primaryDevice_, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
            sizeof(cl_uint), &v, NULL);
    if (err != CL_SUCCESS) {
        return false;
    }

    if (v != 0) {
        return true;
    }

    /* Check extensions */

    err = clGetDeviceInfo(primaryDevice_, CL_DEVICE_EXTENSIONS, 0, NULL, &len);
    if (err != CL_SUCCESS) {
        return false;
    }

    extensions = new char[len];
    err = clGetDeviceInfo(primaryDevice_, CL_DEVICE_EXTENSIONS, len, extensions, NULL);
    if (err != CL_SUCCESS) {
        delete[] extensions;
        return false;
    }

    /* Check for cl_amd_fp64 extension */
    s = strstr(extensions, "cl_amd_fp64");      /* strlen("cl_amd_fp64") = 11 */
    if (s != NULL) {
        if ((s[11] == ' ') || (s[11] == '\0')) {
            delete[] extensions;
            return true;
        }
    }

    delete[] extensions;

    return false;
}

void
BlasBase::removeScratchImages(void)
{
    //if (imageB_) {
    //    clblasRemoveScratchImage(imageB_);
    //}
    //if (imageA_) {
    //    clblasRemoveScratchImage(imageA_);
    //}
}

size_t
BlasBase::scratchImageWidth(void)
{
    size_t width;

    clGetImageInfo(reinterpret_cast<cl_mem>(imageA_), CL_IMAGE_WIDTH,
                   sizeof(width), &width, NULL);
    return width;
}

size_t
BlasBase::scratchImageHeight(void)
{
    size_t height;

    clGetImageInfo(reinterpret_cast<cl_mem>(imageA_), CL_IMAGE_HEIGHT,
                   sizeof(height), &height, NULL);

    return height;
}

cl_ulong
BlasBase::maxMemAllocSize(void)
{
    cl_int err;
    cl_ulong rc = 0;

    err = clGetDeviceInfo(primaryDevice_, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                          sizeof(rc), &rc, NULL);
    if ((err == CL_SUCCESS) && (additionalDevice_ != NULL)) {
        cl_ulong u;

        err = clGetDeviceInfo(additionalDevice_, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                              sizeof(u), &u, NULL);
        if (err == CL_SUCCESS) {
            rc = std::min(rc, u);
        }
    }

    return rc;
}

cl_ulong
BlasBase::availGlobalMemSize(int primAdd)
{
    cl_ulong gmemSize;
    cl_device_id dev;

    dev = (primAdd) ? additionalDevice_ : primaryDevice_;
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmemSize),
                    &gmemSize, NULL);

    return gmemSize;
}

void
BlasBase::printDevInfoStr(cl_device_info param, const char *paramName,
                          int primAdd)
{
    char buf[4096];
    cl_device_id dev;

    dev = (primAdd) ? additionalDevice_ : primaryDevice_;
    if (clGetDeviceInfo(dev, param, sizeof(buf), buf, NULL) == CL_SUCCESS) {
        std::cout << paramName << ": " << buf << std::endl;
    }
}

void
BlasBase::printEnvInfo(void)
{
    cl_ulong memSize;
    int i;

    if (primaryDevice_ == NULL) {
        return;
    }

    cl_uint libMajor, libMinor, libPatch;
    clblasGetVersion( &libMajor, &libMinor, &libPatch );

    std::cout << std::endl << "Test environment:" << std::endl << std::endl;

    for (i = 0; i < 2; i++) {
        if (additionalDevice_ != NULL) {
            if (!i) {
                std::cout << "PRIMARY DEVICE (used in all cases):" << std::endl;
            }
            else {
                std::cout << "ADDITIONAL DEVICE (used only in cases with "
                             "multiple command queues to cover cases with "
                             "problem distribution among command queues "
                             "belonging to different devices):" << std::endl;
            }
        }
        else if (i) {
            break;
        }

        printDevInfoStr(CL_DEVICE_NAME, "Device name", i);
        printDevInfoStr(CL_DEVICE_VENDOR, "Device vendor", i);
        std::cout << "Platform (bit): ";
#if defined( _WIN32 )
        std::cout << "Windows ";
    #if defined( _WIN64 )
            std::cout << "(x64)" << std::endl;
    #else
            std::cout << "(x32)" << std::endl;
    #endif
#elif defined( __APPLE__ )
        std::cout << "Apple OS X" << std::endl;
#else
        std::cout << "Linux" << std::endl;
#endif
        std::cout << "clblas version: " << libMajor << "." << libMinor << "."
            << libPatch << std::endl;
        printDevInfoStr(CL_DRIVER_VERSION, "Driver version", i);
        printDevInfoStr(CL_DEVICE_VERSION, "Device version", i);
        memSize = availGlobalMemSize(i);
        std::cout << "Global mem size: " << memSize / (1024 * 1024) <<
                     " MB" << std::endl;

        std::cout << "---------------------------------------------------------"
                  << std::endl << std::endl;
    }
}

}   // namespace
