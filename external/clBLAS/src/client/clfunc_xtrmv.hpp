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

#ifndef CLBLAS_BENCHMARK_XTRMV_HXX__
#define CLBLAS_BENCHMARK_XTRMV_HXX__

#include "clfunc_common.hpp"

template <typename T>
struct xTrmvBuffer
{
  size_t m_;
  size_t lda_;
  size_t a_num_vectors_;
  clblasTranspose trans_a_;
  clblasUplo uplo_;
  clblasDiag diag_;
  T* a_;
  T* x_;
  cl_mem buf_a_;
  cl_mem buf_x_;
  cl_mem scratch_;
}; // struct buffer

template <typename T>
class xTrmv : public clblasFunc
{
public:
  xTrmv(StatisticalTimer& timer,  cl_device_type devType) : clblasFunc(timer,  devType)
  {
    timer.getUniqueID("clTrmv", 0);
  }

  ~xTrmv()
  {
    delete buffer_.a_;
    delete buffer_.x_;
    OPENCL_V_THROW( clReleaseMemObject(buffer_.buf_a_), "releasing buffer A");
    OPENCL_V_THROW( clReleaseMemObject(buffer_.buf_x_), "releasing buffer X");
    OPENCL_V_THROW( clReleaseMemObject(buffer_.scratch_), "releasing buffer X");
  }

  void call_func() {}

  double gflops()
  {
    return static_cast<double>(buffer_.m_ * buffer_.m_ )/time_in_ns();
  }

  std::string gflops_formula()
  {
    return "M*M/time";
  }

  void setup_buffer(int order_option, int side_option, int
                    uplo_option, int diag_option, int transA_option, int
                    transB_option, size_t M, size_t N, size_t K,
                    size_t lda, size_t ldb, size_t ldc,size_t offA,
					size_t offB, size_t offC, double alpha,
                    double beta)
  {
    initialize_scalars(alpha, beta);

    buffer_.m_ = M;

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
    }
    else
    {
      order_ = clblasColumnMajor;
    }


    if (lda == 0)
    {
      buffer_.lda_ = M;
    }
    else
    {
      if( lda < M )
      {
        std::cerr << "ERROR: lda must be set to 0 or a value >= M" << std::endl;
      }
      else if (lda >= M)
      {
        buffer_.lda_ = lda;
      }
    }


    buffer_.a_num_vectors_ = buffer_.m_;

    buffer_.a_ = new T[buffer_.lda_*buffer_.a_num_vectors_];
    buffer_.x_ = new T[buffer_.m_];


    cl_int err;
    buffer_.buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                    buffer_.lda_*buffer_.a_num_vectors_*sizeof(T),
                                    NULL, &err);

    buffer_.buf_x_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                    buffer_.m_*sizeof(T),
                                    NULL, &err);

    buffer_.scratch_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                    buffer_.m_*sizeof(T),
                                    NULL, &err);

  }

  void initialize_cpu_buffer()
  {
    srand(10);

    for (size_t i = 0; i < buffer_.m_; ++i)
    {
      buffer_.x_[i] = static_cast<T>(rand())/static_cast<T>(RAND_MAX);
    }

    for (size_t i = 0; i < buffer_.a_num_vectors_; ++i)
    {
      for (size_t j = 0; j < buffer_.lda_; ++j)
      {
        if (i == j)
        {
          if (buffer_.diag_ == clblasUnit)
          {
            buffer_.a_[i*buffer_.lda_+j] = static_cast<T>(1.0);
          }
          else
          {
            buffer_.a_[i*buffer_.lda_+j] =
              static_cast<T>(rand())/static_cast<T>(RAND_MAX);
          }
        }
        else
        {
          buffer_.a_[i*buffer_.lda_+j] = static_cast<T>(0.0);
        }
      }
    }
  }

  void initialize_gpu_buffer()
  {
    cl_int err;

    err = clEnqueueWriteBuffer(queue_, buffer_.buf_a_, CL_TRUE, 0,
                               buffer_.lda_*buffer_.a_num_vectors_*sizeof(T),
                               buffer_.a_, 0, NULL, NULL);

    err = clEnqueueWriteBuffer(queue_, buffer_.buf_x_, CL_TRUE, 0,
                               buffer_.m_*sizeof(T),
                               buffer_.x_, 0, NULL, NULL);
  }

  void reset_gpu_write_buffer()
  {
    cl_int err;
    err = clEnqueueWriteBuffer(queue_, buffer_.buf_x_, CL_TRUE, 0,
                               buffer_.m_,
                               buffer_.x_, 0, NULL, NULL);
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
  }

private:
  xTrmvBuffer<T> buffer_;

}; // class xtrmv

template<>
void
xTrmv<cl_float2>::
initialize_scalars(double alpha, double beta)
{
}

template<>
void
xTrmv<cl_double2>::
initialize_scalars(double alpha, double beta)
{
}

template<>
void
xTrmv<cl_float>::
call_func()
{
  timer.Start(timer_id);
  clblasStrmv(order_, buffer_.uplo_, buffer_.trans_a_,
                 buffer_.diag_, buffer_.m_, buffer_.buf_a_, 0,
                 buffer_.lda_, buffer_.buf_x_, 0, 1, buffer_.scratch_,
                 1, &queue_, 0, NULL, &event_);
  clWaitForEvents(1, &event_);
  timer.Stop(timer_id);
}

template<>
void
xTrmv<cl_double>::
call_func()
{
  timer.Start(timer_id);
  clblasDtrmv(order_, buffer_.uplo_, buffer_.trans_a_,
                 buffer_.diag_, buffer_.m_, buffer_.buf_a_, 0,
                 buffer_.lda_, buffer_.buf_x_, 0, 1, buffer_.scratch_,
                 1, &queue_, 0, NULL, &event_);
  clWaitForEvents(1, &event_);
  timer.Stop(timer_id);
}

template<>
void
xTrmv<cl_float2>::
call_func()
{
  timer.Start(timer_id);
  clblasCtrmv(order_, buffer_.uplo_, buffer_.trans_a_,
                 buffer_.diag_, buffer_.m_, buffer_.buf_a_, 0,
                 buffer_.lda_, buffer_.buf_x_, 0, 1, buffer_.scratch_,
                 1, &queue_, 0, NULL, &event_);
  clWaitForEvents(1, &event_);
  timer.Stop(timer_id);
}

template<>
void
xTrmv<cl_double2>::
call_func()
{
  timer.Start(timer_id);
  clblasZtrmv(order_, buffer_.uplo_, buffer_.trans_a_,
                 buffer_.diag_, buffer_.m_, buffer_.buf_a_, 0,
                 buffer_.lda_, buffer_.buf_x_, 0, 1, buffer_.scratch_,
                 1, &queue_, 0, NULL, &event_);
  clWaitForEvents(1, &event_);
  timer.Stop(timer_id);
}

template<>
void
xTrmv<cl_float2>::
initialize_cpu_buffer()
{
  srand(10);
  for (size_t i = 0; i < buffer_.m_; ++i)
  {
    buffer_.x_[i].s[0] =
      static_cast<cl_float>(rand())/static_cast<cl_float>(RAND_MAX);
    buffer_.x_[i].s[1] =
      static_cast<cl_float>(rand())/static_cast<cl_float>(RAND_MAX);
  }

  for (size_t i = 0; i < buffer_.a_num_vectors_; ++i)
  {
    for (size_t j = 0; j < buffer_.lda_; ++j)
    {
      if (i == j)
      {
        if (buffer_.diag_ == clblasUnit)
        {
          buffer_.a_[i*buffer_.lda_+j].s[0] = 1.0f;
          buffer_.a_[i*buffer_.lda_+j].s[1] = 0.0f;
        }
        else
        {
          buffer_.a_[i*buffer_.lda_+j].s[0] =
            static_cast<cl_float>(rand())/static_cast<cl_float>(RAND_MAX);
          buffer_.a_[i*buffer_.lda_+j].s[1] =
            static_cast<cl_float>(rand())/static_cast<cl_float>(RAND_MAX);
        }
      }
      else
      {
        buffer_.a_[i*buffer_.lda_+j].s[0] = 0.0f;
        buffer_.a_[i*buffer_.lda_+j].s[1] = 0.0f;
      }
    }
  }


}

template<>
void
xTrmv<cl_double2>::
initialize_cpu_buffer()
{
  srand(10);
  for (size_t i = 0; i < buffer_.m_; ++i)
  {
    buffer_.x_[i].s[0] =
      static_cast<cl_double>(rand())/static_cast<cl_double>(RAND_MAX);
    buffer_.x_[i].s[1] =
      static_cast<cl_double>(rand())/static_cast<cl_double>(RAND_MAX);
  }

  for (size_t i = 0; i < buffer_.a_num_vectors_; ++i)
  {
    for (size_t j = 0; j < buffer_.lda_; ++j)
    {
      if (i == j)
      {
        if (buffer_.diag_ == clblasUnit)
        {
          buffer_.a_[i*buffer_.lda_+j].s[0] = 1.0;
          buffer_.a_[i*buffer_.lda_+j].s[1] = 0.0;
        }
        else
        {
          buffer_.a_[i*buffer_.lda_+j].s[0] =
            static_cast<cl_double>(rand())/static_cast<cl_double>(RAND_MAX);
          buffer_.a_[i*buffer_.lda_+j].s[1] =
            static_cast<cl_double>(rand())/static_cast<cl_double>(RAND_MAX);
        }
      }
      else
      {
        buffer_.a_[i*buffer_.lda_+j].s[0] = 0.0;
        buffer_.a_[i*buffer_.lda_+j].s[1] = 0.0;
      }
    }
  }
}

template<>
double
xTrmv<cl_float2>::
gflops()
{
  return static_cast<double>(4 * buffer_.m_ * buffer_.m_ )/time_in_ns();
}

template<>
double
xTrmv<cl_double2>::
gflops()
{
  return static_cast<double>(4 * buffer_.m_ * buffer_.m_ )/time_in_ns();
}

template<>
std::string
xTrmv<cl_float2>::
gflops_formula()
{
  return "4*M*M/time";
}

template<>
std::string
xTrmv<cl_double2>::
gflops_formula()
{
  return "4*M*M/time";
}


#endif // ifndef CLBLAS_BENCHMARK_XTRMV_HXX__
