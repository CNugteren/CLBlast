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

#ifndef CLBLAS_BENCHMARK_COMMON_HXX__
#define CLBLAS_BENCHMARK_COMMON_HXX__

#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>

#include "blas-math.h"
#include "test-limits.h"
#include "dis_warning.h"

#include "clBLAS.h"
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl_ext.h>
#endif

template<typename T>
static T
makeScalar(double val)
{
    return static_cast<T>(val);
}

template<>
__template_static FloatComplex
makeScalar(double val)
{
    FloatComplex c;

    c.s[0] = static_cast<float>(val);
    c.s[1] = 0;

    return c;
}

template<>
__template_static DoubleComplex
makeScalar(double val)
{
    DoubleComplex c;

    c.s[0] = val;
    c.s[1] = 0;

    return c;
}

template<typename T>
static T
randomScale()
{
    T t = random<T>(UPPER_BOUND<T>());
    if (module(t) == 0) {
        t = t + ONE<T>();
    }

    return t;
}

std::string
prettyPrintClStatus( const cl_int& status )
{
    switch( status )
    {
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_MIP_LEVEL:
        return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_GL_OBJECT:
        return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_OPERATION:
        return "CL_INVALID_OPERATION";
    case CL_INVALID_EVENT:
        return "CL_INVALID_EVENT";
    case CL_INVALID_EVENT_WAIT_LIST:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_GLOBAL_OFFSET:
        return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_DIMENSION:
        return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_KERNEL:
        return "CL_INVALID_KERNEL";
    case CL_INVALID_KERNEL_DEFINITION:
        return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_PROGRAM:
        return "CL_INVALID_PROGRAM";
    case CL_INVALID_BUILD_OPTIONS:
        return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_BINARY:
        return "CL_INVALID_BINARY";
    case CL_INVALID_SAMPLER:
        return "CL_INVALID_SAMPLER";
    case CL_INVALID_IMAGE_SIZE:
        return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_HOST_PTR:
        return "CL_INVALID_HOST_PTR";
    case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_QUEUE_PROPERTIES:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_CONTEXT:
        return "CL_INVALID_CONTEXT";
    case CL_INVALID_DEVICE:
        return "CL_INVALID_DEVICE";
    case CL_INVALID_PLATFORM:
        return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE_TYPE:
        return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
    case CL_MAP_FAILURE:
        return "CL_MAP_FAILURE";
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_IMAGE_FORMAT_MISMATCH:
        return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_MEM_COPY_OVERLAP:
        return "CL_MEM_COPY_OVERLAP";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_COMPILER_NOT_AVAILABLE:
        return "CL_COMPILER_NOT_AVAILABLE";
    case CL_DEVICE_NOT_AVAILABLE:
        return "CL_DEVICE_NOT_AVAILABLE";
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
    case CL_SUCCESS:
        return "CL_SUCCESS";
    default:
        return "Error code not defined";
        break;
    }
}

// This is used to either wrap an OpenCL function call, or to
// explicitly check a variable for an OpenCL error condition.
// If an error occurs, we throw.
// Note: std::runtime_error does not take unicode strings as input, so
// only strings supported
inline cl_int
OpenCL_V_Throw( cl_int res, const std::string& msg, size_t lineno )
{
    switch( res )
    {
    case CL_SUCCESS: /**< No error */
        break;
    default:
        {
            std::stringstream tmp;

            tmp << "OPENCL_V_THROWERROR< ";
            tmp << prettyPrintClStatus(res) ;
            tmp << " > (";
            tmp << lineno;
            tmp << "): ";
            tmp << msg;
            std::string errorm(tmp.str());
            std::cout << errorm<< std::endl;
            throw std::runtime_error( errorm );
        }
    }

    return res;
}

#define OPENCL_V_THROW(_status,_message) OpenCL_V_Throw(_status, _message, \
                                                        __LINE__)

inline cl_ulong
queryMemAllocSize( cl_device_id device_ )
{
    cl_int err;
    cl_ulong rc = 0;

    err = clGetDeviceInfo(device_, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                          sizeof(rc), &rc, NULL);

    return rc;
}

class clblasFunc
{
public:
    clblasFunc(StatisticalTimer& _timer, cl_device_type devType)
          : timer(_timer)
    {
        cl_int err;

        /* Setup OpenCL environment. */
        OPENCL_V_THROW(clGetPlatformIDs(1, &platform_, NULL),
                       "getting platform IDs");
        OPENCL_V_THROW(clGetDeviceIDs(platform_, devType, 1,
                                      &device_, NULL), "getting device IDs");
        props_[0] = CL_CONTEXT_PLATFORM;
        props_[1] = (cl_context_properties)platform_;
        props_[2] = 0;
        ctx_ = clCreateContext(props_, 1, &device_, NULL, NULL, &err);
        OPENCL_V_THROW(err, "creating context");
        queue_ = clCreateCommandQueue(ctx_, device_, 0, &err);


        timer_id = timer.getUniqueID( "clfunc", 0 );


        maxMemAllocSize = queryMemAllocSize( device_ );

    /* Setup clblas. */
        err = clblasSetup();
        if (err != CL_SUCCESS) {
            std::cerr << "clblasSetup() failed with %d\n";
            clReleaseCommandQueue(queue_);
            clReleaseContext(ctx_);
        }
    }

    virtual ~clblasFunc()
    {
        clblasTeardown();
        OPENCL_V_THROW( clReleaseCommandQueue(queue_),
                        "releasing command queue" );
        OPENCL_V_THROW( clReleaseContext(ctx_), "releasing context" );
    }

    void wait_and_check()
    {
		cl_int err;
        cl_int wait_status = clWaitForEvents(1, &event_);

        if( wait_status != CL_SUCCESS )
        {
    	    if( wait_status == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST )
    	    {
    	    	clGetEventInfo( event_, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                sizeof(cl_int), &err, NULL );
    	    	std::cout << "blas function execution status error: " << err << std::endl;
                exit(1);
    	    }
            else
            {
    	    	std::cout << "blas function wait status error: " << wait_status << std::endl;
                exit(1);
            }
        }
    }

    double time_in_ns()
    {
	    StatisticalTimer& timer = StatisticalTimer::getInstance( );
        return timer.getAverageTime( timer_id ) * 1e9;
    }

    virtual void call_func() = 0;
    virtual double gflops() = 0;
    virtual std::string gflops_formula() = 0;
    virtual void setup_buffer(int order_option, int side_option,
                              int uplo_option, int diag_option, int
                              transA_option, int transB_option,
                              size_t M, size_t N, size_t K, size_t lda,
                              size_t ldb, size_t ldc, size_t offA, size_t offBX,
                              size_t offCY, double alpha, double beta) = 0;
    virtual void initialize_cpu_buffer() = 0;
    virtual void initialize_gpu_buffer() = 0;
    virtual void reset_gpu_write_buffer() = 0;
	virtual void read_gpu_buffer() = 0;
	virtual void roundtrip_func() = 0;
	virtual void roundtrip_func_rect() {}
	virtual void allochostptr_roundtrip_func() {}
	virtual void usehostptr_roundtrip_func() {}
	virtual void copyhostptr_roundtrip_func() {}
	virtual void usepersismem_roundtrip_func() {}
	virtual void roundtrip_setup_buffer(int order_option, int side_option,
                              int uplo_option, int diag_option, int
                              transA_option, int transB_option,
                              size_t M, size_t N, size_t K, size_t lda,
                              size_t ldb, size_t ldc, size_t offA, size_t offBX,
                              size_t offCY, double alpha, double beta) = 0;
	virtual void releaseGPUBuffer_deleteCPUBuffer()=0;
    StatisticalTimer& timer;
    StatisticalTimer::sTimerID timer_id;

protected:
    virtual void initialize_scalars(double alpha, double beta) = 0;

protected:
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context_properties props_[3];
    cl_context ctx_;
    cl_command_queue queue_;
    clblasOrder order_;
    cl_event event_;
    size_t maxMemAllocSize;
}; // class clblasFunc

#endif // ifndef CLBLAS_BENCHMARK_COMMON_HXX__

