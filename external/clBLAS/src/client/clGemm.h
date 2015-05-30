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
#include <clBLAS.h>
#include <string>
#include <map>

cl_int gemm_err;

std::string prettyPrintClStatus( const cl_int& status )
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

//	This is used to either wrap an OpenCL function call, or to explicitly check a variable for an OpenCL error condition.
//	If an error occurs, we throw.
//	Note: std::runtime_error does not take unicode strings as input, so only strings supported
inline cl_int OpenCL_V_Throw( cl_int res, const std::string& msg, size_t lineno )
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
			throw	std::runtime_error( errorm );
		}
	}

	return	res;
}
#define OPENCL_V_THROW(_status,_message) OpenCL_V_Throw(_status, _message, __LINE__)

enum complexity_t { not_complex = 1, yes_complex = 2 };

//can be cl_float, cl_double
//TODO should be cl_float2 and cl_double2 instead of using float/double * complexity?
template< class T >
class buffers
{
public:
    size_t M, N, K;
    size_t lda, ldb, ldc;
    complexity_t complexity;
    T* A;
    T* B;
    T* C;
    cl_mem bufA, bufB, bufC;
    cl_command_queue queue;
    std::map<std::string, T*> buffer_map;
    std::map<std::string, size_t> rows_map;
    std::map<std::string, size_t> ldx_map;

    buffers( cl_context ctx, cl_command_queue _queue,
             size_t _M, size_t _N, size_t _K,
             size_t _lda, size_t _ldb, size_t _ldc,
             complexity_t _complexity )
    : M(_M)
    , N(_N)
    , K(_K)
    , lda(_lda)
    , ldb(_ldb)
    , ldc(_ldc)
    , complexity(_complexity)
    , A(new T[M*lda*sizeof(T)*complexity])
    , B(new T[K*ldb*sizeof(T)*complexity])
    , C(new T[M*ldc*sizeof(T)*complexity])
    , queue(_queue)
    {
        // request and initialize openCL memory
        bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * lda * sizeof(*A) * complexity,
                              NULL, &gemm_err);
        OPENCL_V_THROW( gemm_err, "creating buffer A" );
        bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * ldb * sizeof(*B) * complexity,
                              NULL, &gemm_err);
        OPENCL_V_THROW( gemm_err, "creating buffer B" );
        bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * ldc * sizeof(*C) * complexity,
                              NULL, &gemm_err);
        OPENCL_V_THROW( gemm_err, "creating buffer C" );

        buffer_map.insert(std::pair<std::string,T*>("A",A));
        buffer_map.insert(std::pair<std::string,T*>("B",B));
        buffer_map.insert(std::pair<std::string,T*>("C",C));
        rows_map.insert(std::pair<std::string,size_t>("A",M));
        rows_map.insert(std::pair<std::string,size_t>("B",K));
        rows_map.insert(std::pair<std::string,size_t>("C",M));
        ldx_map.insert(std::pair<std::string,size_t>("A",lda));
        ldx_map.insert(std::pair<std::string,size_t>("B",ldb));
        ldx_map.insert(std::pair<std::string,size_t>("C",ldc));

        initialize_data();
    }

    ~buffers()
    {
        OPENCL_V_THROW( clReleaseMemObject(bufC), "releasing buffer A");
        OPENCL_V_THROW( clReleaseMemObject(bufB), "releasing buffer B");
        OPENCL_V_THROW( clReleaseMemObject(bufA), "releasing buffer C");
        delete[] A;
        delete[] B;
        delete[] C;
    }

    void initialize_data()
    {
        initializeLocalMatrix("A");
        initializeLocalMatrix("B");
        initializeLocalMatrix("C");

        gemm_err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
            M * K * sizeof(*A) * complexity, A, 0, NULL, NULL);
        OPENCL_V_THROW( gemm_err, "writing to buffer A" );
        gemm_err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
            K * N * sizeof(*B) * complexity, A, 0, NULL, NULL);
        OPENCL_V_THROW( gemm_err, "writing to buffer B" );
        gemm_err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0,
            M * N * sizeof(*C) * complexity, C, 0, NULL, NULL);
        OPENCL_V_THROW( gemm_err, "writing to buffer C" );
    }

    void read_back_result()
    {
        OPENCL_V_THROW( clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M * N * sizeof(*C) * complexity, C, 0, NULL, NULL),
                        "reading from buffer C" );
    }

    void initializeLocalMatrix(std::string matrix)
    {
        for (size_t i = 0; i < rows_map[matrix]*complexity; i++) {
            for (size_t j = 0; j < ldx_map[matrix]; j++) {
                buffer_map[matrix][i * ldx_map[matrix] + j] = (i+1)*10 + (j+1);
            }
        }
    }

    void printLocalMatrix(std::string matrix)
    {
        for (size_t i = 0; i < rows_map[matrix]*complexity; i++) {
            for (size_t j = 0; j < ldx_map[matrix]; j++) {
                std::cout << (int)buffer_map[matrix][i * ldx_map[matrix] + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};

class clGemm
{
public:
    size_t M;
    size_t N;
    size_t K;
    size_t lda;
    size_t ldb;
    size_t ldc;
    clblasOrder order;
    clblasTranspose transA;
    clblasTranspose transB;
    cl_context_properties props[3];
    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_device_type deviceType;
    cl_command_queue queue;
    cl_event event;
    cl_uint commandQueueFlags;
    bool useimages;
    cl_ulong imgA;
    cl_ulong imgB;
    StatisticalTimer& timer;
	StatisticalTimer::sTimerID gemm_timer_id;

    clGemm( size_t _M, size_t _N, size_t _K,
            size_t _lda, size_t _ldb, size_t _ldc,
            bool _useimages,
            clblasOrder _order,
            clblasTranspose _transA, clblasTranspose _transB,
            cl_device_type _deviceType, cl_uint _commandQueueFlags,
            StatisticalTimer& _timer )
    : M(_M)
    , N(_N)
    , K(_K)
    , lda(_lda)
    , ldb(_ldb)
    , ldc(_ldc)
    , order(_order)
    , transA(_transA)
    , transB(_transB)
    , deviceType(_deviceType)
    , event(NULL)
    , commandQueueFlags(_commandQueueFlags)
    , useimages(_useimages)
    , imgA(0)
    , imgB(0)
    , timer(_timer)
    {
        props[0] = CL_CONTEXT_PLATFORM;
        props[1] = 0;
        props[2] = 0;
        OPENCL_V_THROW( clGetPlatformIDs(1, &platform, NULL), "getting platform IDs" );
        OPENCL_V_THROW( clGetDeviceIDs(platform, deviceType, 1, &device, NULL), "getting device IDs" );
        props[1] = (cl_context_properties)platform;
        ctx = clCreateContext(props, 1, &device, NULL, NULL, &gemm_err);
        OPENCL_V_THROW( gemm_err, "creating context" );
        queue = clCreateCommandQueue(ctx, device, commandQueueFlags, &gemm_err);
        OPENCL_V_THROW( gemm_err, "creating command queue" );

        gemm_err = clblasSetup();
        if (gemm_err != CL_SUCCESS) {
            std::cout << "clblasSetup() failed with " << gemm_err << std::endl;
            clReleaseCommandQueue(queue);
            clReleaseContext(ctx);
            exit(1);
        }

        if (useimages) {
            imgA = clblasAddScratchImage(ctx, 16, 64, NULL);
            imgB = clblasAddScratchImage(ctx, 16, 64, NULL);
        }

	    gemm_timer_id = timer.getUniqueID( "clGemm", 0 );
    }

    ~clGemm()
    {
        if (useimages) {
            clblasRemoveScratchImage(imgA);
            clblasRemoveScratchImage(imgB);
        }

        clblasTeardown();
        OPENCL_V_THROW( clReleaseCommandQueue(queue), "releasing command queue" );
        OPENCL_V_THROW( clReleaseContext(ctx), "releasing context" );
    }

    void wait_and_check()
    {
        cl_int wait_status = clWaitForEvents(1, &event);

        if( wait_status != CL_SUCCESS )
        {
    	    if( wait_status == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST )
    	    {
    	    	clGetEventInfo( event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &gemm_err, NULL );
    	    	std::cout << "blas function execution status error: " << gemm_err << std::endl;
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
        return timer.getAverageTime( gemm_timer_id ) * 1e9;
    }

    virtual void call_gemm() = 0;
    virtual void clear_buffers() = 0;
    virtual double gflops() = 0;
    virtual std::string gflops_formula() = 0;
};

class clSgemm : public clGemm
{
public:
    cl_float alpha;
    cl_float beta;
    buffers<cl_float> mybuffers;

    clSgemm( size_t _M, size_t _N, size_t _K,
            size_t _lda, size_t _ldb, size_t _ldc,
            bool _useimages,
            clblasOrder _order,
            clblasTranspose _transA, clblasTranspose _transB,
            cl_float _alpha, cl_float _beta,
            cl_device_type _deviceType, cl_uint _commandQueueFlags,
            StatisticalTimer& _timer)
    : clGemm( _M, _N, _K,
              _lda, _ldb, _ldc,
              _useimages, _order, _transA, _transB,
              _deviceType, _commandQueueFlags, _timer )
    , alpha(_alpha)
    , beta(_beta)
    , mybuffers( ctx, queue, M, N, K, lda, ldb, ldc, not_complex )
    {}

    void call_gemm()
    {
	    timer.Start(gemm_timer_id);
        OPENCL_V_THROW( clblasSgemm(order, transA, transB,
                                       M, N, K,
                                       alpha,
                                       mybuffers.bufA, lda,
                                       mybuffers.bufB, ldb,
                                       beta,
                                       mybuffers.bufC, ldc,
                                       1, &queue, 0, NULL, &event),
                        "clblasSgemm" );
        wait_and_check();
	    timer.Stop(gemm_timer_id);
        //mybuffers.read_back_result();
        //mybuffers.printLocalMatrix("C");
    }

    void clear_buffers()
    {
        mybuffers.initialize_data();
    }

    double gflops()
    {
        return (2*M*N*K)/time_in_ns();
    }

    std::string gflops_formula()
    {
        return "(2*M*N*K)/time_in_ns";
    }
};

class clDgemm : public clGemm
{
public:
    cl_double alpha;
    cl_double beta;
    buffers<cl_double> mybuffers;

    clDgemm( size_t _M, size_t _N, size_t _K,
            size_t _lda, size_t _ldb, size_t _ldc,
            bool _useimages,
            clblasOrder _order,
            clblasTranspose _transA, clblasTranspose _transB,
            cl_double _alpha, cl_double _beta,
            cl_device_type _deviceType, cl_uint _commandQueueFlags,
            StatisticalTimer& _timer)
    : clGemm( _M, _N, _K,
              _lda, _ldb, _ldc,
              _useimages, _order, _transA, _transB,
              _deviceType, _commandQueueFlags, _timer )
    , alpha(_alpha)
    , beta(_beta)
    , mybuffers( ctx, queue, M, N, K, lda, ldb, ldc, not_complex )
    {}

    void call_gemm()
    {
	    timer.Start(gemm_timer_id);
        OPENCL_V_THROW( clblasDgemm(order, transA, transB,
                                       M, N, K,
                                       alpha,
                                       mybuffers.bufA, lda,
                                       mybuffers.bufB, ldb,
                                       beta,
                                       mybuffers.bufC, ldc,
                                       1, &queue, 0, NULL, &event),
                        "clblasDgemm" );
        wait_and_check();
	    timer.Stop(gemm_timer_id);
        //mybuffers.read_back_result();
        //mybuffers.printLocalMatrix("C");
    }

    void clear_buffers()
    {
        mybuffers.initialize_data();
    }

    double gflops()
    {
        return (2*M*N*K)/time_in_ns();
    }

    std::string gflops_formula()
    {
        return "(2*M*N*K)/time_in_ns";
    }
};

class clCgemm : public clGemm
{
public:
    cl_float2 alpha;
    cl_float2 beta;
    buffers<cl_float> mybuffers;

    clCgemm( size_t _M, size_t _N, size_t _K,
            size_t _lda, size_t _ldb, size_t _ldc,
            bool _useimages,
            clblasOrder _order,
            clblasTranspose _transA, clblasTranspose _transB,
            cl_float _alpha, cl_float _beta,
            cl_device_type _deviceType, cl_uint _commandQueueFlags,
            StatisticalTimer& _timer)
    : clGemm( _M, _N, _K,
              _lda, _ldb, _ldc,
              _useimages, _order, _transA, _transB,
              _deviceType, _commandQueueFlags, _timer )
    , mybuffers( ctx, queue, M, N, K, lda, ldb, ldc, yes_complex )
    {
        alpha.s[0] = _alpha;
        alpha.s[1] = _alpha;
        beta.s[0] = _beta;
        beta.s[1] = _beta;
    }

    void call_gemm()
    {
	    timer.Start(gemm_timer_id);
        OPENCL_V_THROW( clblasCgemm(order, transA, transB,
                                       M, N, K,
                                       alpha,
                                       mybuffers.bufA, lda,
                                       mybuffers.bufB, ldb,
                                       beta,
                                       mybuffers.bufC, ldc,
                                       1, &queue, 0, NULL, &event),
                        "clblasCgemm" );
        wait_and_check();
	    timer.Stop(gemm_timer_id);
        //mybuffers.read_back_result();
        //mybuffers.printLocalMatrix("C");
    }

    void clear_buffers()
    {
        mybuffers.initialize_data();
    }

    double gflops()
    {
        return (8*M*N*K)/time_in_ns();
    }

    std::string gflops_formula()
    {
        return "(8*M*N*K)/time_in_ns";
    }
};

class clZgemm : public clGemm
{
public:
    cl_double2 alpha;
    cl_double2 beta;
    buffers<cl_double> mybuffers;

    clZgemm( size_t _M, size_t _N, size_t _K,
            size_t _lda, size_t _ldb, size_t _ldc,
            bool _useimages,
            clblasOrder _order,
            clblasTranspose _transA, clblasTranspose _transB,
            cl_double _alpha, cl_double _beta,
            cl_device_type _deviceType, cl_uint _commandQueueFlags,
            StatisticalTimer& _timer)
    : clGemm( _M, _N, _K,
              _lda, _ldb, _ldc,
              _useimages, _order, _transA, _transB,
              _deviceType, _commandQueueFlags, _timer )
    , mybuffers( ctx, queue, M, N, K, lda, ldb, ldc, yes_complex )
    {
        alpha.s[0] = _alpha;
        alpha.s[1] = _alpha;
        beta.s[0] = _beta;
        beta.s[1] = _beta;
    }

    void call_gemm()
    {
	    timer.Start(gemm_timer_id);
        OPENCL_V_THROW( clblasZgemm(order, transA, transB,
                                       M, N, K,
                                       alpha,
                                       mybuffers.bufA, lda,
                                       mybuffers.bufB, ldb,
                                       beta,
                                       mybuffers.bufC, ldc,
                                       1, &queue, 0, NULL, &event),
                        "clblasZgemm" );
        wait_and_check();
	    timer.Stop(gemm_timer_id);
        //mybuffers.read_back_result();
        //mybuffers.printLocalMatrix("C");
    }

    void clear_buffers()
    {
        mybuffers.initialize_data();
    }

    double gflops()
    {
        return (8*M*N*K)/time_in_ns();
    }

    std::string gflops_formula()
    {
        return "(8*M*N*K)/time_in_ns";
    }
};
