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


// $Id

#ifndef CLBLAS_BENCHMARK_XTRSM_HXX__
#define CLBLAS_BENCHMARK_XTRSM_HXX__

#include "clfunc_common.hpp"


template <typename T>
struct xTrsmBuffer
{
    clblasOrder order_;
    size_t m_;
    size_t n_;
    size_t lda_;
    size_t ldb_;
    size_t offA_;
    size_t offB_;
    size_t a_num_vectors_;
    size_t b_num_vectors_;
    clblasTranspose trans_a_;
    clblasSide side_;
    clblasUplo uplo_;
    clblasDiag diag_;
    T* a_;
    T* b_;
    cl_mem buf_a_;
    cl_mem buf_b_;
    T alpha_;
}; // struct buffer

template <typename T>
class xTrsm : public clblasFunc
{
public:
    xTrsm(StatisticalTimer& timer, cl_device_type devType) :
        clblasFunc(timer, devType)
    {
        timer.getUniqueID("clTrsm", 0);
    }

    ~xTrsm()
    {
    }

    void call_func()
    {
    timer.Start(timer_id);
	xTrsm_Function(true);
    timer.Stop(timer_id);
    }

    double gflops()
    {
        if (buffer_.side_ == clblasLeft)
        {
            return buffer_.m_*(buffer_.m_+1)*buffer_.n_/time_in_ns();
        }
        else
        {
            return 20*buffer_.m_*(buffer_.n_+1)*buffer_.n_/time_in_ns();
        }
    }

    std::string gflops_formula()
    {
        if (buffer_.side_ == clblasLeft)
        {
            return "M*(M+1)*N/time";
        }
        else
        {
            return "M*(N+1)*N/time";
        }
    }

    void setup_buffer(int order_option, int side_option, int uplo_option,
                      int diag_option, int transA_option, int transB_option,
                      size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                      size_t ldc, size_t offA, size_t offBX, size_t offCY,
                      double alpha, double beta)
    {
        DUMMY_ARGS_USAGE_3(transB_option, K, beta);
        DUMMY_ARGS_USAGE_2(ldc, offCY);

        initialize_scalars(alpha, beta);

        buffer_.m_ = M;
        buffer_.n_ = N;
        buffer_.offA_ = offA;
        buffer_.offB_ = offBX;

        if (transA_option == 0)
        {
            buffer_.trans_a_ = clblasNoTrans;
        }
        else if (transA_option == 1)
        {
            buffer_.trans_a_ = clblasTrans;
        }
        else if (transA_option == 2)
        {
            buffer_.trans_a_ = clblasConjTrans;
        }

        if (side_option == 0)
        {
            buffer_.side_ = clblasLeft;
            buffer_.a_num_vectors_ = M;
        }
        else
        {
            buffer_.side_ = clblasRight;
            buffer_.a_num_vectors_ = N;
        }

        if (uplo_option == 0)
        {
            buffer_.uplo_ = clblasUpper;
        }
        else
        {
            buffer_.uplo_ = clblasLower;
        }

        if (diag_option == 0)
        {
            buffer_.diag_ = clblasUnit;
        }
        else
        {
            buffer_.diag_ = clblasNonUnit;
        }

        if (order_option == 0)
        {
            order_ = clblasRowMajor;
            buffer_.b_num_vectors_ = M;
            if (ldb == 0)
            {
                buffer_.ldb_ = N;
            }
            else
            {
                if (ldb < N)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.ldb_ = ldb;
                }
            }
        }
        else
        {
            buffer_.order_ = clblasColumnMajor;
            buffer_.b_num_vectors_ = N;
            if (ldb == 0)
            {
                buffer_.ldb_ = M;
            }
            else
            {
                if (ldb < M)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.ldb_ = ldb;
                }
            }
        }

        if (lda == 0)
        {
            if (side_option == 0)
            {
                buffer_.lda_ = M;
            }
            else
            {
                buffer_.lda_ = N;
            }
        }
        else
        {
            if( side_option == 0 && lda < M )
            {
                std::cerr << "ERROR: when side is 0, lda must be set to 0 "
                             "or a value >= M" << std::endl;
            }
            else if(side_option == 0 && lda >= M )
            {
                buffer_.lda_ = lda;
            }
            else if(side_option != 0 && lda < N)
            {
                std::cerr << "ERROR: when side is 1, lda must be set to 0 "
                             "or a value >= N" << std::endl;
            }
            else if (side_option != 0 && lda >= N)
            {
                buffer_.lda_ = lda;
            }
        }

        buffer_.a_ = new T[buffer_.lda_*buffer_.a_num_vectors_];
        buffer_.b_ = new T[buffer_.ldb_*buffer_.b_num_vectors_];

        cl_int err;
        buffer_.buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offA_) * sizeof(T),
                                        NULL, &err);

        buffer_.buf_b_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offB_) * sizeof(T),
                                         NULL, &err);
    }

    void initialize_cpu_buffer()
    {
        srand(10);

        for (size_t i = 0; i < buffer_.b_num_vectors_; ++i)
        {
            for (size_t j = 0; j < buffer_.ldb_; ++j)
            {
                buffer_.b_[i*buffer_.ldb_+j] = random<T>(UPPER_BOUND<T>()) /
                    randomScale<T>();
            }
        }

        for (size_t i = 0; i < buffer_.a_num_vectors_; ++i)
        {
            for (size_t j = 0; j < buffer_.lda_; ++j)
            {
                if (i == j)
                {
                    if (buffer_.diag_ == clblasUnit)
                    {
                        buffer_.a_[i*buffer_.lda_+j] = ONE<T>();
                    }
                    else
                    {
                        buffer_.a_[i*buffer_.lda_+j] =
                            random<T>(UPPER_BOUND<T>()) /
                            randomScale<T>();
                    }
                }
                else
                {
                    buffer_.a_[i*buffer_.lda_+j] = ZERO<T>();
                }
            }
        }
    }

    void initialize_gpu_buffer()
    {
        cl_int err;

        err = clEnqueueWriteBuffer(queue_, buffer_.buf_a_, CL_TRUE,
                                   buffer_.offA_ * sizeof(T),
                                   buffer_.lda_ * buffer_.a_num_vectors_ *
                                       sizeof(T),
                                   buffer_.a_, 0, NULL, NULL);

        err = clEnqueueWriteBuffer(queue_, buffer_.buf_b_, CL_TRUE,
                                   buffer_.offB_ * sizeof(T),
                                   buffer_.ldb_ * buffer_.b_num_vectors_ *
                                       sizeof(T),
                                   buffer_.b_, 0, NULL, NULL);
    }

    void reset_gpu_write_buffer()
    {
        cl_int err;
        err = clEnqueueWriteBuffer(queue_, buffer_.buf_b_, CL_TRUE,
                                   buffer_.offB_ * sizeof(T),
                                   buffer_.ldb_ * buffer_.b_num_vectors_ *
                                       sizeof(T),
                                   buffer_.b_, 0, NULL, NULL);
    }
	void read_gpu_buffer()
	{
		cl_int err;
		err = clEnqueueReadBuffer(queue_, buffer_.buf_b_, CL_TRUE,
			                      buffer_.offB_ * sizeof(T), buffer_.ldb_ * buffer_.b_num_vectors_ *
                                       sizeof(T),
								  buffer_.b_, 0, NULL, NULL);
	}
	void roundtrip_func()
	{
	timer.Start(timer_id);
	    //set up buffer
        cl_int err;
        buffer_.buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offA_) * sizeof(T),
                                        NULL, &err);

        buffer_.buf_b_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offB_) * sizeof(T),
                                         NULL, &err);
		//initialize gpu buffer
		err = clEnqueueWriteBuffer(queue_, buffer_.buf_a_, CL_TRUE,
                                   buffer_.offA_ * sizeof(T),
                                   buffer_.lda_ * buffer_.a_num_vectors_ *
                                       sizeof(T),
                                   buffer_.a_, 0, NULL, NULL);

        err = clEnqueueWriteBuffer(queue_, buffer_.buf_b_, CL_TRUE,
                                   buffer_.offB_ * sizeof(T),
                                   buffer_.ldb_ * buffer_.b_num_vectors_ *
                                       sizeof(T),
                                   buffer_.b_, 0, NULL, NULL);
		//call func
		xTrsm_Function(false);
		//read gpu buffer
		err = clEnqueueReadBuffer(queue_, buffer_.buf_b_, CL_TRUE,
			                      buffer_.offB_ * sizeof(T), buffer_.ldb_ * buffer_.b_num_vectors_ *
                                       sizeof(T),
								  buffer_.b_, 0, NULL, &event_);
		clWaitForEvents(1, &event_);
	timer.Stop(timer_id);
	}
	void allochostptr_roundtrip_func()
	{
	timer.Start(timer_id);
	    //set up buffer
        cl_int err;
        buffer_.buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offA_) * sizeof(T),
                                        NULL, &err);

        buffer_.buf_b_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                        (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offB_) * sizeof(T),
                                         NULL, &err);
		// Map the buffers to pointers at host device
		T *map_a,*map_b;
		map_a = (T*)clEnqueueMapBuffer(queue_, buffer_.buf_a_, CL_TRUE, CL_MAP_WRITE, 0,
                                          (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offB_) * sizeof(T),
											0, NULL, NULL, &err);
		map_b = (T*)clEnqueueMapBuffer(queue_, buffer_.buf_b_, CL_TRUE, CL_MAP_WRITE, 0,
                                          (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offB_) * sizeof(T),
											0, NULL, NULL, &err);
		// memcpy the input A, B to the mapped regions
		memcpy( map_a, buffer_.a_, ( buffer_.lda_*buffer_.a_num_vectors_ + buffer_.offA_) * sizeof( T ) );
		memcpy( map_b, buffer_.b_, ( buffer_.ldb_*buffer_.b_num_vectors_ + buffer_.offB_) * sizeof( T ) );
		// unmap the buffers
		clEnqueueUnmapMemObject(queue_, buffer_.buf_a_, map_a, 0, NULL, NULL);
		clEnqueueUnmapMemObject(queue_, buffer_.buf_b_, map_b, 0, NULL, NULL);
		//call func
		xTrsm_Function(false);
		// map the B buffer again to read the output
		map_b = (T*)clEnqueueMapBuffer(queue_, buffer_.buf_b_, CL_TRUE, CL_MAP_READ, 0,
                                          (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offB_) * sizeof(T),
											0, NULL, NULL, &err);
		memcpy( map_b, buffer_.b_, ( buffer_.ldb_*buffer_.b_num_vectors_ + buffer_.offB_) * sizeof( T ) );
		clEnqueueUnmapMemObject(queue_, buffer_.buf_b_, map_b, 0, NULL, NULL);
		clWaitForEvents(1, &event_);
	timer.Stop(timer_id);
	}
	void usehostptr_roundtrip_func()
	{
	timer.Start(timer_id);
	    //set up buffer
        cl_int err;
        buffer_.buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offA_) * sizeof(T),
                                        buffer_.a_, &err);

        buffer_.buf_b_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                        (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offB_) * sizeof(T),
                                         buffer_.b_, &err);
		//call func
		xTrsm_Function(false);
		//read gpu buffer
		err = clEnqueueReadBuffer(queue_, buffer_.buf_b_, CL_TRUE,
			                      buffer_.offB_ * sizeof(T), buffer_.ldb_ * buffer_.b_num_vectors_ *
                                       sizeof(T),
								  buffer_.b_, 0, NULL, &event_);
		clWaitForEvents(1, &event_);
	timer.Stop(timer_id);
	}
	void copyhostptr_roundtrip_func()
	{
	timer.Start(timer_id);
	    //set up buffer
        cl_int err;
        buffer_.buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offA_) * sizeof(T),
                                        buffer_.a_, &err);

        buffer_.buf_b_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offB_) * sizeof(T),
                                         buffer_.b_, &err);
		//call func
		xTrsm_Function(false);
		//read gpu buffer
		err = clEnqueueReadBuffer(queue_, buffer_.buf_b_, CL_TRUE,
			                      buffer_.offB_ * sizeof(T), buffer_.ldb_ * buffer_.b_num_vectors_ *
                                       sizeof(T),
								  buffer_.b_, 0, NULL, &event_);
	clWaitForEvents(1, &event_);
	timer.Stop(timer_id);
	}
	void usepersismem_roundtrip_func()
	{
#if defined(CL_MEM_USE_PERSISTENT_MEM_AMD)
	timer.Start(timer_id);
	    //set up buffer
        cl_int err;
        buffer_.buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_USE_PERSISTENT_MEM_AMD,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offA_) * sizeof(T),
                                        NULL, &err);

        buffer_.buf_b_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
                                        (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offB_) * sizeof(T),
                                         NULL, &err);
		// Map the buffers to pointers at host device
		T *map_a,*map_b;
		map_a = (T*)clEnqueueMapBuffer(queue_, buffer_.buf_a_, CL_TRUE, CL_MAP_WRITE, 0,
                                          (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offB_) * sizeof(T),
											0, NULL, NULL, &err);
		map_b = (T*)clEnqueueMapBuffer(queue_, buffer_.buf_b_, CL_TRUE, CL_MAP_WRITE, 0,
                                          (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offB_) * sizeof(T),
											0, NULL, NULL, &err);
		// memcpy the input A, B to the mapped regions
		memcpy( map_a, buffer_.a_, ( buffer_.lda_*buffer_.a_num_vectors_ + buffer_.offA_) * sizeof( T ) );
		memcpy( map_b, buffer_.b_, ( buffer_.ldb_*buffer_.b_num_vectors_ + buffer_.offB_) * sizeof( T ) );
		// unmap the buffers
		clEnqueueUnmapMemObject(queue_, buffer_.buf_a_, map_a, 0, NULL, NULL);
		clEnqueueUnmapMemObject(queue_, buffer_.buf_b_, map_b, 0, NULL, NULL);
		//call func
		xTrsm_Function(false);
		// map the B buffer again to read the output
		map_b = (T*)clEnqueueMapBuffer(queue_, buffer_.buf_b_, CL_TRUE, CL_MAP_READ, 0,
                                          (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offB_) * sizeof(T),
											0, NULL, NULL, &err);
		memcpy( map_b, buffer_.b_, ( buffer_.ldb_*buffer_.b_num_vectors_ + buffer_.offB_) * sizeof( T ) );
		clEnqueueUnmapMemObject(queue_, buffer_.buf_b_, map_b, 0, NULL, NULL);
	clWaitForEvents(1, &event_);
	timer.Stop(timer_id);
#else
		std::cout<<"CL_MEM_USE_PERSISTENT_MEM_AMD is only supported on AMD hardware"<<std::endl;
#endif
	}
	void zerocopy_roundtrip_func()
	{
		std::cout << "xTrmm::zerocopy_roundtrip_func\n";
	}
	void roundtrip_setup_buffer(int order_option, int side_option, int uplo_option,
                      int diag_option, int transA_option, int  transB_option,
                      size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                      size_t ldc, size_t offA, size_t offBX, size_t offCY,
                      double alpha, double beta)
	{
        DUMMY_ARGS_USAGE_3(transB_option, K, beta);
        DUMMY_ARGS_USAGE_2(ldc, offCY);

        initialize_scalars(alpha, beta);

        buffer_.m_ = M;
        buffer_.n_ = N;
        buffer_.offA_ = offA;
        buffer_.offB_ = offBX;

        if (transA_option == 0)
        {
            buffer_.trans_a_ = clblasNoTrans;
        }
        else if (transA_option == 1)
        {
            buffer_.trans_a_ = clblasTrans;
        }
        else if (transA_option == 2)
        {
            buffer_.trans_a_ = clblasConjTrans;
        }

        if (side_option == 0)
        {
            buffer_.side_ = clblasLeft;
            buffer_.a_num_vectors_ = M;
        }
        else
        {
            buffer_.side_ = clblasRight;
            buffer_.a_num_vectors_ = N;
        }

        if (uplo_option == 0)
        {
            buffer_.uplo_ = clblasUpper;
        }
        else
        {
            buffer_.uplo_ = clblasLower;
        }

        if (diag_option == 0)
        {
            buffer_.diag_ = clblasUnit;
        }
        else
        {
            buffer_.diag_ = clblasNonUnit;
        }

        if (order_option == 0)
        {
            order_ = clblasRowMajor;
            buffer_.b_num_vectors_ = M;
            if (ldb == 0)
            {
                buffer_.ldb_ = N;
            }
            else
            {
                if (ldb < N)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.ldb_ = ldb;
                }
            }
        }
        else
        {
            buffer_.order_ = clblasColumnMajor;
            buffer_.b_num_vectors_ = N;
            if (ldb == 0)
            {
                buffer_.ldb_ = M;
            }
            else
            {
                if (ldb < M)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.ldb_ = ldb;
                }
            }
        }

        if (lda == 0)
        {
            if (side_option == 0)
            {
                buffer_.lda_ = M;
            }
            else
            {
                buffer_.lda_ = N;
            }
        }
        else
        {
            if( side_option == 0 && lda < M )
            {
                std::cerr << "ERROR: when side is 0, lda must be set to 0 "
                             "or a value >= M" << std::endl;
            }
            else if(side_option == 0 && lda >= M )
            {
                buffer_.lda_ = lda;
            }
            else if(side_option != 0 && lda < N)
            {
                std::cerr << "ERROR: when side is 1, lda must be set to 0 "
                             "or a value >= N" << std::endl;
            }
            else if (side_option != 0 && lda >= N)
            {
                buffer_.lda_ = lda;
            }
        }

        buffer_.a_ = new T[buffer_.lda_*buffer_.a_num_vectors_];
        buffer_.b_ = new T[buffer_.ldb_*buffer_.b_num_vectors_];
	}
	void releaseGPUBuffer_deleteCPUBuffer()
	{
		//this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
		//need to do this before we eventually hit the destructor
        delete buffer_.a_;
        delete buffer_.b_;
        OPENCL_V_THROW(clReleaseMemObject(buffer_.buf_a_),
                       "releasing buffer A");
        OPENCL_V_THROW(clReleaseMemObject(buffer_.buf_b_),
                       "releasing buffer B");
	}
protected:
    void initialize_scalars(double alpha, double beta)
    {
        DUMMY_ARG_USAGE(beta);
        buffer_.alpha_ = makeScalar<T>(alpha);
    }

private:
    xTrsmBuffer<T> buffer_;
	void xTrsm_Function(bool flush);

}; // class xtrsm

template<>
void
xTrsm<cl_float>::
xTrsm_Function(bool flush)
{
    clblasStrsm(order_, buffer_.side_, buffer_.uplo_,
                     buffer_.trans_a_, buffer_.diag_,
                     buffer_.m_, buffer_.n_, buffer_.alpha_,
                     buffer_.buf_a_, buffer_.offA_, buffer_.lda_,
                     buffer_.buf_b_, buffer_.offB_, buffer_.ldb_,
                     1, &queue_, 0, NULL, &event_);
	if(flush==true)
	{
		clWaitForEvents(1, &event_);
	}
}

template<>
void
xTrsm<cl_double>::
xTrsm_Function(bool flush)
{
    clblasDtrsm(order_, buffer_.side_, buffer_.uplo_,
                     buffer_.trans_a_, buffer_.diag_,
                     buffer_.m_, buffer_.n_, buffer_.alpha_,
                     buffer_.buf_a_, buffer_.offA_, buffer_.lda_,
                     buffer_.buf_b_, buffer_.offB_, buffer_.ldb_,
                     1, &queue_, 0, NULL, &event_);
	if(flush==true)
	{
		clWaitForEvents(1, &event_);
	}
}

template<>
void
xTrsm<cl_float2>::
xTrsm_Function(bool flush)
{
    clblasCtrsm(order_, buffer_.side_, buffer_.uplo_,
                     buffer_.trans_a_, buffer_.diag_,
                     buffer_.m_, buffer_.n_, buffer_.alpha_,
                     buffer_.buf_a_, buffer_.offA_, buffer_.lda_,
                     buffer_.buf_b_, buffer_.offB_, buffer_.ldb_,
                     1, &queue_, 0, NULL, &event_);
	if(flush==true)
	{
		clWaitForEvents(1, &event_);
	}
}

template<>
void
xTrsm<cl_double2>::
xTrsm_Function(bool flush)
{
    clblasZtrsm(order_, buffer_.side_, buffer_.uplo_,
                     buffer_.trans_a_, buffer_.diag_,
                     buffer_.m_, buffer_.n_, buffer_.alpha_,
                     buffer_.buf_a_, buffer_.offA_, buffer_.lda_,
                     buffer_.buf_b_, buffer_.offB_, buffer_.ldb_,
                     1, &queue_, 0, NULL, &event_);
	if(flush==true)
	{
		clWaitForEvents(1, &event_);
	}
}


template<>
double
xTrsm<cl_float2>::
gflops()
{
    if (buffer_.side_ == clblasLeft)
    {
        return 4.0*buffer_.m_*(buffer_.m_+1)*buffer_.n_/time_in_ns();
    }
    else
    {
        return 4.0*buffer_.m_*(buffer_.n_+1)*buffer_.n_/time_in_ns();
    }
}


template<>
double
xTrsm<cl_double2>::
gflops()
{
    if (buffer_.side_ == clblasLeft)
    {
        return 4.0*buffer_.m_*(buffer_.m_+1)*buffer_.n_/time_in_ns();
    }
    else
    {
        return 4.0*buffer_.m_*(buffer_.n_+1)*buffer_.n_/time_in_ns();
    }
}

template<>
std::string
xTrsm<cl_float2>::
gflops_formula()
{
    if (buffer_.side_ == clblasLeft)
    {
        return "4.0*M*(M+1)*N/time";
    }
    else
    {
        return "4.0*M*(N+1)*N/time";
    }
}

template<>
std::string
xTrsm<cl_double2>::
gflops_formula()
{
    if (buffer_.side_ == clblasLeft)
    {
        return "4.0*M*(M+1)*N/time";
    }
    else
    {
        return "4.0*M*(N+1)*N/time";
    }
}


#endif // ifndef CLBLAS_BENCHMARK_XTRSM_HXX__
