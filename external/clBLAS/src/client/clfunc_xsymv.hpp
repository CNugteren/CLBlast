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

#ifndef CLBLAS_BENCHMARK_XSYMV_HXX__
#define CLBLAS_BENCHMARK_XSYMV_HXX__

#include "clfunc_common.hpp"

template <typename T>
struct xSymvBuffer
{
    clblasOrder order_;
    size_t n_;
    size_t lda_;
    size_t offA_;
    size_t a_num_vectors_;
    clblasUplo uplo_;
    T* a_;
    T* x_;
    T* y_;
    cl_mem buf_a_;
    cl_mem buf_x_;
    cl_mem buf_y_;
    T alpha_;
    T beta_;
}; // struct buffer

template <typename T>
class xSymv : public clblasFunc
{
public:
    xSymv(StatisticalTimer& _timer, cl_device_type devType) :
        clblasFunc(_timer, devType)
    {
        timer.getUniqueID("clSymv", 0);
    }

    ~xSymv()
    {
        delete buffer_.a_;
        delete buffer_.x_;
        delete buffer_.y_;
        OPENCL_V_THROW( clReleaseMemObject(buffer_.buf_a_),
                        "releasing buffer A");
        OPENCL_V_THROW( clReleaseMemObject(buffer_.buf_x_),
                        "releasing buffer X");
        OPENCL_V_THROW( clReleaseMemObject(buffer_.buf_y_),
                        "releasing buffer Y");
    }

    void call_func()
    {
    }

    double gflops()
    {
        return (2.0*buffer_.n_*buffer_.n_)/time_in_ns();
    }

    std::string gflops_formula()
    {
        return "2.0*N*N/time";
    }

    void setup_buffer(int order_option, int side_option, int uplo_option,
                      int diag_option, int transA_option, int transB_option,
                      size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                      size_t ldc, size_t offA, size_t offBX, size_t offCY,
                      double alpha, double beta)
    {
        DUMMY_ARGS_USAGE_3(side_option, diag_option, transB_option);
        DUMMY_ARGS_USAGE_4(M, K, ldb, ldc);
        DUMMY_ARGS_USAGE_3(transA_option, offBX, offCY);

        initialize_scalars(alpha, beta);

        buffer_.n_ = N;
        buffer_.a_num_vectors_ = N;
        buffer_.offA_ = offA;

        if (uplo_option == 0)
        {
            buffer_.uplo_ = clblasUpper;
        }
        else
        {
            buffer_.uplo_ = clblasLower;
        }

        buffer_.x_ = new T[buffer_.n_];
        buffer_.y_ = new T[buffer_.n_];

        if (order_option == 0)
        {
            order_ = clblasRowMajor;
        }
        else
        {
            order_ = clblasColumnMajor;
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

        buffer_.a_ = new T[buffer_.lda_*buffer_.a_num_vectors_];

        cl_int err;
        buffer_.buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offA_) * sizeof(T),
                                        NULL, &err);

        buffer_.buf_x_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        buffer_.n_*sizeof(T),
                                        NULL, &err);

        buffer_.buf_y_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        buffer_.n_*sizeof(T),
                                        NULL, &err);
    }

    void initialize_cpu_buffer()
    {
        srand(10);
        for (size_t i = 0; i < buffer_.a_num_vectors_; ++i)
        {
            for (size_t j = 0; j < buffer_.lda_; ++j)
            {
                buffer_.a_[i*buffer_.lda_+j] = static_cast<T>(rand()) /
                                               static_cast<T>(RAND_MAX);
            }
        }

        for (size_t i = 0; i < buffer_.n_; ++i)
        {
            buffer_.x_[i] = static_cast<T>(rand()) /
                            static_cast<T>(RAND_MAX);
            buffer_.y_[i] = static_cast<T>(rand()) /
                            static_cast<T>(RAND_MAX);
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

        err = clEnqueueWriteBuffer(queue_, buffer_.buf_x_, CL_TRUE, 0,
                                   buffer_.n_*sizeof(T),
                                   buffer_.x_, 0, NULL, NULL);

        err = clEnqueueWriteBuffer(queue_, buffer_.buf_y_, CL_TRUE, 0,
                                   buffer_.n_*sizeof(T),
                                   buffer_.y_, 0, NULL, NULL);
    }

    void reset_gpu_write_buffer()
    {
        cl_int err;

        err = clEnqueueWriteBuffer(queue_, buffer_.buf_y_, CL_TRUE, 0,
                                   buffer_.n_*sizeof(T),
                                   buffer_.y_, 0, NULL, NULL);
    }
 	void read_gpu_buffer()
	{
		//cl_int err;
		//to-do need to fill up
	}
	void roundtrip_func()
	{//to-do need to fill up
	}
	void roundtrip_setup_buffer(int order_option, int side_option, int uplo_option,
                      int diag_option, int transA_option, int  transB_option,
                      size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                      size_t ldc, size_t offA, size_t offBX, size_t offCY,
                      double alpha, double beta)
		{}
	void releaseGPUBuffer_deleteCPUBuffer()
	{
		//this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
		//need to do this before we eventually hit the destructor
		//to-do
	}
protected:
    void initialize_scalars(double alpha, double beta)
    {
        buffer_.alpha_ = static_cast<T>(alpha);
        buffer_.beta_ = static_cast<T>(beta);
    }

private:
    xSymvBuffer<T> buffer_;

}; // class xsymv

template<>
void
xSymv<float>::
call_func()
{
    timer.Start(timer_id);

    clblasSsymv(order_, buffer_.uplo_, buffer_.n_, buffer_.alpha_,
                     buffer_.buf_a_, buffer_.offA_, buffer_.lda_,
                     buffer_.buf_x_, 0, 1, buffer_.beta_, buffer_.buf_y_,
                     0, 1, 1, &queue_, 0, NULL, &event_);

	clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

template<>
void
xSymv<double>::
call_func()
{
    timer.Start(timer_id);

	clblasDsymv(order_, buffer_.uplo_, buffer_.n_, buffer_.alpha_,
                     buffer_.buf_a_, buffer_.offA_, buffer_.lda_,
                     buffer_.buf_x_, 0, 1, buffer_.beta_, buffer_.buf_y_,
                     0, 1, 1, &queue_, 0, NULL, &event_);

    clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

#endif // ifndef CLBLAS_BENCHMARK_XSYMV_HXX__
