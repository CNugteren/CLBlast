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

#ifndef CLBLAS_BENCHMARK_XSYRK_HXX__
#define CLBLAS_BENCHMARK_XSYRK_HXX__

#include "clfunc_common.hpp"

template <typename T>
struct xSyrkBuffer
{
    clblasOrder order_;
    size_t n_;
    size_t k_;
    size_t lda_;
    size_t ldc_;
    size_t offA_;
    size_t offC_;
    size_t a_num_vectors_;
    size_t c_num_vectors_;
    clblasTranspose trans_a_;
    clblasUplo uplo_;
    T* a_;
    T* c_;
    cl_mem buf_a_;
    cl_mem buf_c_;
    T alpha_;
    T beta_;
}; // struct buffer

template <typename T>
class xSyrk : public clblasFunc
{
public:
    xSyrk(StatisticalTimer& _timer, cl_device_type devType) :
        clblasFunc(_timer, devType)
    {
        timer.getUniqueID("clSyrk", 0);
    }

    ~xSyrk()
    {
    }

    void call_func()
    {
    }

    double gflops()
    {
        return buffer_.n_*(buffer_.n_+1)*buffer_.n_/time_in_ns();
    }

    std::string gflops_formula()
    {
        return "(N*(N+1)*K)/time";
    }

    void setup_buffer(int order_option, int side_option, int uplo_option,
                      int diag_option, int transA_option, int transB_option,
                      size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                      size_t ldc, size_t offA, size_t offBX, size_t offCY,
                      double alpha, double beta)
    {
        DUMMY_ARGS_USAGE_4(side_option, diag_option, transB_option, M);
        DUMMY_ARGS_USAGE_2(ldb, offBX);

        initialize_scalars(alpha, beta);

        buffer_.n_ = N;
        buffer_.k_ = K;
        buffer_.offA_ = offA;
        buffer_.offC_ = offCY;

        if (uplo_option == 0)
        {
            buffer_.uplo_ = clblasUpper;
        }
        else
        {
            buffer_.uplo_ = clblasLower;
        }


        if (ldc == 0)
        {
            buffer_.ldc_ = N;
        }
        else if (ldc < N)
        {
            std::cerr << "ldc:wrong size\n";
        }
        else
        {
            buffer_.ldc_ = ldc;
        }
        buffer_.c_num_vectors_ = N;

        if (order_option == 0)
        {
            order_ = clblasRowMajor;
            if (transA_option == 0)
            {
                buffer_.trans_a_ = clblasNoTrans;
                buffer_.a_num_vectors_ = N;
                if (lda == 0)
                {
                    buffer_.lda_ = K;
                }
                else if (lda < K)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.lda_ = lda;
                }
            }
            else
            {
                buffer_.a_num_vectors_ = K;
                if (transA_option == 1)
                {
                    buffer_.trans_a_ = clblasTrans;
                }
                else if (transA_option == 2)
                {
                    buffer_.trans_a_ = clblasConjTrans;
                }
                if (lda == 0)
                {
                    buffer_.lda_ = N;
                }
                else if (lda < N)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.lda_ = lda;
                }
            }
        }
        else
        {
            order_ = clblasColumnMajor;
            if (transA_option == 0)
            {
                buffer_.a_num_vectors_ = K;
                buffer_.trans_a_ = clblasNoTrans;
                if (lda == 0)
                {
                    buffer_.lda_ = N;
                }
                else if (lda < N)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.lda_ = lda;
                }
            }
            else
            {
                buffer_.a_num_vectors_ = N;
                if (transA_option == 1)
                {
                    buffer_.trans_a_ = clblasTrans;
                }
                else if (transA_option == 2)
                {
                    buffer_.trans_a_ = clblasConjTrans;
                }

                if (lda == 0)
                {
                    buffer_.lda_ = K;
                }
                else if (lda < K)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.lda_ = lda;
                }
            }
        }

        buffer_.a_ = new T[buffer_.lda_*buffer_.a_num_vectors_];
        buffer_.c_ = new T[buffer_.ldc_*buffer_.c_num_vectors_];

        cl_int err;
        buffer_.buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offA_) * sizeof(T),
                                        NULL, &err);

        buffer_.buf_c_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (buffer_.ldc_ * buffer_.c_num_vectors_ +
                                            buffer_.offC_) * sizeof(T),
                                        NULL, &err);
    }

    void initialize_cpu_buffer()
    {
        srand(10);
        for (size_t i = 0; i < buffer_.a_num_vectors_; ++i)
        {
            for (size_t j = 0; j < buffer_.lda_; ++j)
            {
                buffer_.a_[i*buffer_.lda_+j] = random<T>(UPPER_BOUND<T>()) /
                                               randomScale<T>();
            }
        }

        for (size_t i = 0; i < buffer_.c_num_vectors_; ++i)
        {
            for (size_t j = 0; j < buffer_.ldc_; ++j)
            {
                buffer_.c_[i*buffer_.ldc_+j] = random<T>(UPPER_BOUND<T>()) /
                                               randomScale<T>();
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

        err = clEnqueueWriteBuffer(queue_, buffer_.buf_c_, CL_TRUE,
                                   buffer_.offA_ * sizeof(T),
                                   buffer_.ldc_ * buffer_.c_num_vectors_ *
                                       sizeof(T),
                                   buffer_.c_, 0, NULL, NULL);
    }

    void reset_gpu_write_buffer()
    {
        cl_int err;

        err = clEnqueueWriteBuffer(queue_, buffer_.buf_c_, CL_TRUE,
                                   buffer_.offC_ * sizeof(T),
                                   buffer_.ldc_ * buffer_.c_num_vectors_ *
                                       sizeof(T),
                                   buffer_.c_, 0, NULL, NULL);
    }
 	void read_gpu_buffer()
	{
		cl_int err;
		err = clEnqueueReadBuffer(queue_, buffer_.buf_c_, CL_TRUE,
								  buffer_.offC_*sizeof(T), buffer_.ldc_*buffer_.c_num_vectors_*sizeof(T),
								  buffer_.c_, 0, NULL, NULL);
	}
	void roundtrip_func()
	{
	}
	void zerocopy_roundtrip_func()
	{
		std::cout << "xSyrk::zerocopy_roundtrip_func\n";
	}
	void roundtrip_setup_buffer(int order_option, int side_option, int uplo_option,
                      int diag_option, int transA_option, int  transB_option,
                      size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                      size_t ldc, size_t offA, size_t offBX, size_t offCY,
                      double alpha, double beta)
	{
		DUMMY_ARGS_USAGE_4(side_option, diag_option, transB_option, M);
        DUMMY_ARGS_USAGE_2(ldb, offBX);

        initialize_scalars(alpha, beta);

        buffer_.n_ = N;
        buffer_.k_ = K;
        buffer_.offA_ = offA;
        buffer_.offC_ = offCY;

        if (uplo_option == 0)
        {
            buffer_.uplo_ = clblasUpper;
        }
        else
        {
            buffer_.uplo_ = clblasLower;
        }


        if (ldc == 0)
        {
            buffer_.ldc_ = N;
        }
        else if (ldc < N)
        {
            std::cerr << "ldc:wrong size\n";
        }
        else
        {
            buffer_.ldc_ = ldc;
        }
        buffer_.c_num_vectors_ = N;

        if (order_option == 0)
        {
            order_ = clblasRowMajor;
            if (transA_option == 0)
            {
                buffer_.trans_a_ = clblasNoTrans;
                buffer_.a_num_vectors_ = N;
                if (lda == 0)
                {
                    buffer_.lda_ = K;
                }
                else if (lda < K)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.lda_ = lda;
                }
            }
            else
            {
                buffer_.a_num_vectors_ = K;
                if (transA_option == 1)
                {
                    buffer_.trans_a_ = clblasTrans;
                }
                else if (transA_option == 2)
                {
                    buffer_.trans_a_ = clblasConjTrans;
                }
                if (lda == 0)
                {
                    buffer_.lda_ = N;
                }
                else if (lda < N)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.lda_ = lda;
                }
            }
        }
        else
        {
            order_ = clblasColumnMajor;
            if (transA_option == 0)
            {
                buffer_.a_num_vectors_ = K;
                buffer_.trans_a_ = clblasNoTrans;
                if (lda == 0)
                {
                    buffer_.lda_ = N;
                }
                else if (lda < N)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.lda_ = lda;
                }
            }
            else
            {
                buffer_.a_num_vectors_ = N;
                if (transA_option == 1)
                {
                    buffer_.trans_a_ = clblasTrans;
                }
                else if (transA_option == 2)
                {
                    buffer_.trans_a_ = clblasConjTrans;
                }

                if (lda == 0)
                {
                    buffer_.lda_ = K;
                }
                else if (lda < K)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.lda_ = lda;
                }
            }
        }

        buffer_.a_ = new T[buffer_.lda_*buffer_.a_num_vectors_];
        buffer_.c_ = new T[buffer_.ldc_*buffer_.c_num_vectors_];
	}
	void releaseGPUBuffer_deleteCPUBuffer()
	{
		//this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
		//need to do this before we eventually hit the destructor
		delete buffer_.a_;
        delete buffer_.c_;
        OPENCL_V_THROW( clReleaseMemObject(buffer_.buf_a_),
                        "releasing buffer A");
        OPENCL_V_THROW( clReleaseMemObject(buffer_.buf_c_),
                        "releasing buffer C");
	}
protected:
    void initialize_scalars(double alpha, double beta)
    {
        buffer_.alpha_ = makeScalar<T>(alpha);
        buffer_.beta_ = makeScalar<T>(beta);
    }

private:
    xSyrkBuffer<T> buffer_;

}; // class xsyrk

template<>
void
xSyrk<float>::
call_func()
{
    timer.Start(timer_id);

    clblasSsyrk(order_, buffer_.uplo_, buffer_.trans_a_, buffer_.n_,
                     buffer_.k_, buffer_.alpha_, buffer_.buf_a_, buffer_.offA_,
                     buffer_.lda_, buffer_.beta_, buffer_.buf_c_, buffer_.offC_,
                     buffer_.ldc_, 1, &queue_, 0, NULL, &event_);

    clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

template<>
void
xSyrk<float>::roundtrip_func()
{
	timer.Start(timer_id);

	cl_int err;
    buffer_.buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offA_) * sizeof(float),
                                        NULL, &err);

    buffer_.buf_c_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (buffer_.ldc_ * buffer_.c_num_vectors_ +
                                            buffer_.offC_) * sizeof(float),
                                        NULL, &err);
	this->initialize_gpu_buffer();
	clblasSsyrk(order_, buffer_.uplo_, buffer_.trans_a_, buffer_.n_,
                     buffer_.k_, buffer_.alpha_, buffer_.buf_a_, buffer_.offA_,
                     buffer_.lda_, buffer_.beta_, buffer_.buf_c_, buffer_.offC_,
                     buffer_.ldc_, 1, &queue_, 0, NULL, NULL);
	err = clEnqueueReadBuffer(queue_, buffer_.buf_c_, CL_TRUE,
								  buffer_.offC_*sizeof(float), buffer_.ldc_*buffer_.c_num_vectors_*sizeof(float),
								  buffer_.c_, 0, NULL, &event_);

    clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

template<>
void
xSyrk<double>::
call_func()
{
    timer.Start(timer_id);

    clblasDsyrk(order_, buffer_.uplo_, buffer_.trans_a_, buffer_.n_,
                     buffer_.k_, buffer_.alpha_, buffer_.buf_a_, buffer_.offA_,
                     buffer_.lda_, buffer_.beta_, buffer_.buf_c_, buffer_.offC_,
                     buffer_.ldc_, 1, &queue_, 0, NULL, &event_);

    clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

template<>
void
xSyrk<double>::roundtrip_func()
{
	timer.Start(timer_id);

	cl_int err;
    buffer_.buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offA_) * sizeof(double),
                                        NULL, &err);

    buffer_.buf_c_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (buffer_.ldc_ * buffer_.c_num_vectors_ +
                                            buffer_.offC_) * sizeof(double),
                                        NULL, &err);
	this->initialize_gpu_buffer();
	clblasDsyrk(order_, buffer_.uplo_, buffer_.trans_a_, buffer_.n_,
                     buffer_.k_, buffer_.alpha_, buffer_.buf_a_, buffer_.offA_,
                     buffer_.lda_, buffer_.beta_, buffer_.buf_c_, buffer_.offC_,
                     buffer_.ldc_, 1, &queue_, 0, NULL, NULL);
	err = clEnqueueReadBuffer(queue_, buffer_.buf_c_, CL_TRUE,
								  buffer_.offC_*sizeof(double), buffer_.ldc_*buffer_.c_num_vectors_*sizeof(double),
								  buffer_.c_, 0, NULL, &event_);

    clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

template<>
void
xSyrk<cl_float2>::
call_func()
{
    timer.Start(timer_id);

    clblasCsyrk(order_, buffer_.uplo_, buffer_.trans_a_, buffer_.n_,
                     buffer_.k_, buffer_.alpha_, buffer_.buf_a_, buffer_.offA_,
                     buffer_.lda_, buffer_.beta_, buffer_.buf_c_, buffer_.offC_,
                     buffer_.ldc_, 1, &queue_, 0, NULL, &event_);

    clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

template<>
void
xSyrk<cl_float2>::roundtrip_func()
{
	timer.Start(timer_id);

	cl_int err;
    buffer_.buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offA_) * sizeof(cl_float2),
                                        NULL, &err);

    buffer_.buf_c_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (buffer_.ldc_ * buffer_.c_num_vectors_ +
                                            buffer_.offC_) * sizeof(cl_float2),
                                        NULL, &err);
	this->initialize_gpu_buffer();
	clblasCsyrk(order_, buffer_.uplo_, buffer_.trans_a_, buffer_.n_,
                     buffer_.k_, buffer_.alpha_, buffer_.buf_a_, buffer_.offA_,
                     buffer_.lda_, buffer_.beta_, buffer_.buf_c_, buffer_.offC_,
                     buffer_.ldc_, 1, &queue_, 0, NULL, NULL);
	err = clEnqueueReadBuffer(queue_, buffer_.buf_c_, CL_TRUE,
								  buffer_.offC_*sizeof(cl_float2), buffer_.ldc_*buffer_.c_num_vectors_*sizeof(cl_float2),
								  buffer_.c_, 0, NULL, &event_);

    clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

template<>
double
xSyrk<cl_float2>::gflops()
{
        return 4*buffer_.n_*(buffer_.n_+1)*buffer_.n_/time_in_ns();
}

template<>
std::string 
xSyrk<cl_float2>::gflops_formula()
{
        return "(4*N*(N+1)*K)/time";
}
template<>
void
xSyrk<cl_double2>::
call_func()
{
    timer.Start(timer_id);

    clblasZsyrk(order_, buffer_.uplo_, buffer_.trans_a_, buffer_.n_,
                     buffer_.k_, buffer_.alpha_, buffer_.buf_a_, buffer_.offA_,
                     buffer_.lda_, buffer_.beta_, buffer_.buf_c_, buffer_.offC_,
                     buffer_.ldc_, 1, &queue_, 0, NULL, &event_);

    clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

template<>
void
xSyrk<cl_double2>::roundtrip_func()
{
	timer.Start(timer_id);

	cl_int err;
    buffer_.buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offA_) * sizeof(cl_double2),
                                        NULL, &err);

    buffer_.buf_c_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (buffer_.ldc_ * buffer_.c_num_vectors_ +
                                            buffer_.offC_) * sizeof(cl_double2),
                                        NULL, &err);
	this->initialize_gpu_buffer();
	clblasZsyrk(order_, buffer_.uplo_, buffer_.trans_a_, buffer_.n_,
                     buffer_.k_, buffer_.alpha_, buffer_.buf_a_, buffer_.offA_,
                     buffer_.lda_, buffer_.beta_, buffer_.buf_c_, buffer_.offC_,
                     buffer_.ldc_, 1, &queue_, 0, NULL, NULL);
	err = clEnqueueReadBuffer(queue_, buffer_.buf_c_, CL_TRUE,
								  buffer_.offC_*sizeof(cl_double2), buffer_.ldc_*buffer_.c_num_vectors_*sizeof(cl_double2),
								  buffer_.c_, 0, NULL, &event_);

    clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

template<>
double
xSyrk<cl_double2>::gflops()
{
        return 4*buffer_.n_*(buffer_.n_+1)*buffer_.n_/time_in_ns();
}

template<>
std::string 
xSyrk<cl_double2>::gflops_formula()
{
        return "(4*N*(N+1)*K)/time";
}

#endif // ifndef CLBLAS_BENCHMARK_XSYRK_HXX__
