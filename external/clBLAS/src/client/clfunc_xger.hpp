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

#ifndef CLBLAS_BENCHMARK_XGER_HXX__
#define CLBLAS_BENCHMARK_XGER_HXX__

#include "clfunc_common.hpp"

template <typename T>
struct xGerBuffer
{
  clblasOrder order_;
  size_t m_;
  size_t n_;
  T alpha;
  T* X;
  cl_mem x_;
  size_t offX;
  int incx_;
  T* Y;
  cl_mem y_;
  size_t offY;
  int incy_;
  T* A;
  cl_mem a_;
  size_t a_num_vectors_;
  size_t offA;
  size_t lda_;
}; // struct buffer

template <typename T>
class xGer : public clblasFunc
{
public:
  xGer(StatisticalTimer& timer, cl_device_type devType) : clblasFunc(timer,  devType)
  {
    timer.getUniqueID("clGer", 0);
  }

  ~xGer()
  {
    delete buffer_.X;
    delete buffer_.Y;
    delete buffer_.A;
    OPENCL_V_THROW( clReleaseMemObject(buffer_.x_), "releasing buffer X");
    OPENCL_V_THROW( clReleaseMemObject(buffer_.y_), "releasing buffer Y");
    OPENCL_V_THROW( clReleaseMemObject(buffer_.a_), "releasing buffer A");
  }

  //void call_func() {}

  double gflops()
  {
    return (buffer_.m_*(buffer_.m_+1))/time_in_ns();
  }

  std::string gflops_formula()
  {
    return "M*(M+1)/time";
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
    buffer_.n_ = N;
    buffer_.incx_ = 1;
    buffer_.incy_ = 1;

    if (order_option == 0)
    {
      buffer_.order_ = clblasRowMajor;
    }
    else
    {
      buffer_.order_ = clblasColumnMajor;
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
    buffer_.offA = offA;
    buffer_.offX = offB;
    buffer_.offY = offC;


    buffer_.a_num_vectors_ = buffer_.n_;
    size_t sizeA = buffer_.lda_*buffer_.a_num_vectors_;
    size_t sizeX = buffer_.m_;
    size_t sizeY = buffer_.n_;
    buffer_.A = new T[sizeA];
    buffer_.X = new T[sizeX];
    buffer_.Y = new T[sizeY];


    cl_int err;
    buffer_.a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                    sizeA*sizeof(T),
                                    NULL, &err);

    buffer_.x_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                    sizeX*sizeof(T),
                                    NULL, &err);
    buffer_.y_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                    sizeY*sizeof(T),
                                    NULL, &err);
  }

  void initialize_cpu_buffer()
  {
    srand(10);

    for (size_t i = 0; i < buffer_.m_; ++i)
    {
      buffer_.X[i] = static_cast<T>(rand())/static_cast<T>(RAND_MAX);
    }
    for (size_t i = 0; i < buffer_.n_; ++i)
    {
      buffer_.Y[i] = static_cast<T>(rand())/static_cast<T>(RAND_MAX);
    }

    for (size_t i = 0; i < buffer_.a_num_vectors_; ++i)
    {
      for (size_t j = 0; j < buffer_.lda_; ++j)
      {
        if (i == j)
        {
          /*if (buffer_.diag_ == clblasUnit)
          {
            buffer_.a_[i*buffer_.lda_+j] = static_cast<T>(1.0);
          }
          else
          {*/
            buffer_.A[i*buffer_.lda_+j] =
              static_cast<T>(rand())/static_cast<T>(RAND_MAX);
          //}
        }
        else
        {
          buffer_.A[i*buffer_.lda_+j] = static_cast<T>(0.0);
        }
      }
    }
  }

  void initialize_gpu_buffer()
  {
    cl_int err;

    err = clEnqueueWriteBuffer(queue_, buffer_.a_, CL_TRUE, 0,
                               buffer_.lda_*buffer_.a_num_vectors_*sizeof(T),
                               buffer_.A, 0, NULL, NULL);

    err = clEnqueueWriteBuffer(queue_, buffer_.x_, CL_TRUE, 0,
                               buffer_.m_*sizeof(T),
                               buffer_.X, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue_, buffer_.y_, CL_TRUE, 0,
                               buffer_.n_*sizeof(T),
                               buffer_.Y, 0, NULL, NULL);
  }

  void reset_gpu_write_buffer()
  {
    cl_int err;
    err = clEnqueueWriteBuffer(queue_, buffer_.x_, CL_TRUE, 0,
                               buffer_.m_,
                               buffer_.x_, 0, NULL, NULL);
  }
  void call_func();

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
    buffer_.alpha = alpha;
  }

private:
  xGerBuffer<T> buffer_;

}; // class xger

//template<>
//void
//xGer<cl_float2>::
//initialize_scalars(double alpha, double beta)
//{
//  buffer_.alpha = alpha;
//}

//template<>
//void
//xGer<cl_double2>::
//initialize_scalars(double alpha, double beta)
//{
//}

template<>
void
xGer<cl_float>::
call_func()
{
    timer.Start(timer_id);
    clblasSger(buffer_.order_, buffer_.m_, buffer_.n_, buffer_.alpha, buffer_.x_, buffer_.offX, 1, buffer_.y_, buffer_.offY,
      1, buffer_.a_, buffer_.offA, buffer_.lda_, 1, &queue_, 0, NULL,
                   &event_);
    clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

template<>
void
xGer<cl_double>::
call_func()
{
    timer.Start(timer_id);
    clblasDger(buffer_.order_, buffer_.m_, buffer_.n_, buffer_.alpha, buffer_.x_, buffer_.offX, 1, buffer_.y_, buffer_.offY,
      1, buffer_.a_, buffer_.offA, buffer_.lda_, 1, &queue_, 0, NULL,
                   &event_);
    clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

//template<>
//void
//xGer<cl_float2>::
//call_func()
//{
//  timer.Start(timer_id);
//  clblasCger(order_, buffer_.m_, buffer_.n, buffer_a_, 0,
//                 buffer_.lda_, buffer_x_, 0, 1, 1, &queue_, 0, NULL,
//                 &event_);
//  clWaitForEvents(1, &event_);
//  timer.Stop(timer_id);
//}
//
//template<>
//void
//xGer<cl_double2>::
//call_func()
//{
//  timer.Start(timer_id);
//  clblasZger(order_, buffer_.uplo_, buffer_.trans_a_,
//                 buffer_.diag_, buffer_.m_, buffer_a_, 0,
//                 buffer_.lda_, buffer_x_, 0, 1, 1, &queue_, 0, NULL,
//                 &event_);
//  clWaitForEvents(1, &event_);
//  timer.Stop(timer_id);
//}

//template<>
//void
//xGer<cl_float2>::
//initialize_cpu_buffer()
//{
//  srand(10);
//  for (size_t i = 0; i < buffer_.m_; ++i)
//  {
//    buffer_x_[i].s[0] =
//      static_cast<cl_float>(rand())/static_cast<cl_float>(RAND_MAX);
//    buffer_.x_[i].s[1] =
//      static_cast<cl_float>(rand())/static_cast<cl_float>(RAND_MAX);
//  }
//
//  for (size_t i = 0; i < buffer_.a_num_vectors_; ++i)
//  {
//    for (size_t j = 0; j < buffer_.lda_; ++j)
//    {
//      if (i == j)
//      {
//        if (buffer_.diag_ == clblasUnit)
//        {
//          buffer_.a_[i*buffer_.lda_+j].s[0] = 1.0f;
//          buffer_.a_[i*buffer_.lda_+j].s[1] = 0.0f;
//        }
//        else
//        {
//          buffer_.a_[i*buffer_.lda_+j].s[0] =
//            static_cast<cl_float>(rand())/static_cast<cl_float>(RAND_MAX);
//          buffer_.a_[i*buffer_.lda_+j].s[1] =
//            static_cast<cl_float>(rand())/static_cast<cl_float>(RAND_MAX);
//        }
//      }
//      else
//      {
//        buffer_.a_[i*buffer_.lda_+j].s[0] = 0.0f;
//        buffer_.a_[i*buffer_.lda_+j].s[1] = 0.0f;
//      }
//    }
//  }
//
//
//}

//template<>
//void
//xGer<cl_double2>::
//initialize_cpu_buffer()
//{
//  srand(10);
//  for (size_t i = 0; i < buffer_.m_; ++i)
//  {
//    buffer_.x_[i].s[0] =
//      static_cast<cl_double>(rand())/static_cast<cl_double>(RAND_MAX);
//    buffer_.x_[i].s[1] =
//      static_cast<cl_double>(rand())/static_cast<cl_double>(RAND_MAX);
//  }
//
//  for (size_t i = 0; i < buffer_.a_num_vectors_; ++i)
//  {
//    for (size_t j = 0; j < buffer_.lda_; ++j)
//    {
//      if (i == j)
//      {
//        if (buffer_.diag_ == clblasUnit)
//        {
//          buffer_.a_[i*buffer_.lda_+j].s[0] = 1.0;
//          buffer_.a_[i*buffer_.lda_+j].s[1] = 0.0;
//        }
//        else
//        {
//          buffer_.a_[i*buffer_.lda_+j].s[0] =
//            static_cast<cl_double>(rand())/static_cast<cl_double>(RAND_MAX);
//          buffer_.a_[i*buffer_.lda_+j].s[1] =
//            static_cast<cl_double>(rand())/static_cast<cl_double>(RAND_MAX);
//        }
//      }
//      else
//      {
//        buffer_.a_[i*buffer_.lda_+j].s[0] = 0.0;
//        buffer_.a_[i*buffer_.lda_+j].s[1] = 0.0;
//      }
//    }
//  }
//}

//template<>
//double
//xGer<cl_float2>::
//gflops()
//{
//  return 2.0*buffer_.m_*(buffer_.m_+1)/time_in_ns();
//}
//
//template<>
//double
//xGer<cl_double2>::
//gflops()
//{
//  return 2.0*buffer_.m_*(buffer_.m_+1)/time_in_ns();
//}
//
//template<>
//std::string
//xGer<cl_float2>::
//gflops_formula()
//{
//  return "2.0*M*(M+1)/time";
//}
//
//template<>
//std::string
//xGer<cl_double2>::
//gflops_formula()
//{
//  return "2.0*M*(M+1)/time";
//}


#endif // ifndef CLBLAS_BENCHMARK_XGER_HXX__
