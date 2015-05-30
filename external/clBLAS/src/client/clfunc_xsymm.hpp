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

#ifndef CLBLAS_BENCHMARK_XSYMM_HXX__
#define CLBLAS_BENCHMARK_XSYMM_HXX__

#include "clfunc_common.hpp"

template <typename T>
struct xSymmBuffer
{
  clblasOrder order;
  clblasSide side;
  clblasUplo uplo;
  size_t M;
  size_t N;
  T alpha;
  T* cpuA;
  size_t a_num_vectors;
  cl_mem A;
  size_t offa;
  size_t lda;
  T* cpuB;
  cl_mem B;
  size_t offb;
  size_t ldb;
  T beta;
  T* cpuC;
  cl_mem C;
  size_t offc;
  size_t ldc;
}; // struct buffer

template <typename T>
class xSymm : public clblasFunc
{
public:
  xSymm(StatisticalTimer& timer, cl_device_type devType) : clblasFunc(timer,  devType)
  {
    timer.getUniqueID("clSymm", 0);
  }

  ~xSymm()
  {
  }

  double gflops()
  {
    if (buffer.side == clblasLeft)
      return static_cast<double>((2 * buffer.M * buffer.M * buffer.N)/time_in_ns());
    else
      return static_cast<double>((2 * buffer.N * buffer.N * buffer.M)/time_in_ns());
  }

  std::string gflops_formula()
  {
    if (buffer.side == clblasLeft)
      return "2*M*M*N/time";
    else
      return "2*N*N*M/time";
  }

  void setup_buffer(int order_option, int side_option, int
                    uplo_option, int diag_option, int transA_option, int
                    transB_option, size_t M, size_t N, size_t K,
                    size_t lda, size_t ldb, size_t ldc,size_t offA,
					          size_t offB, size_t offC, double alpha,
                    double beta);
  void initialize_cpu_buffer();
  void initialize_gpu_buffer();
  void reset_gpu_write_buffer();
  void call_func();
  void read_gpu_buffer()
	{
		cl_int err;
		err = clEnqueueReadBuffer(queue_, buffer.C, CL_TRUE,
			                      buffer.offc * sizeof(T), buffer.ldc * buffer.N *
                                       sizeof(T),
								  buffer.cpuC, 0, NULL, NULL);
	}
  void roundtrip_func()
	{
				std::cout << "xSymm::roundtrip_func\n";
	}
	void zerocopy_roundtrip_func()
	{
		std::cout << "xSymm::zerocopy_roundtrip_func\n";
	}
  void roundtrip_setup_buffer(int order_option, int side_option, int uplo_option,
                      int diag_option, int transA_option, int  transB_option,
                      size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                      size_t ldc, size_t offA, size_t offB, size_t offC,
                      double alpha, double beta)
  {
  initialize_scalars(alpha, beta);
  buffer.offa = offA;
  buffer.offb = offB;
  buffer.offc = offC;
  buffer.M = M;
  buffer.N = N;
  if (order_option == 0)
  {
  buffer.order = clblasRowMajor;
  }
  else
  {
  buffer.order = clblasColumnMajor;
  }
  if (uplo_option == 0)
  {
    buffer.uplo = clblasUpper;
  }
  else
  {
    buffer.uplo = clblasLower;
  }
  if (side_option == 0)
  {
      buffer.side = clblasLeft;
      buffer.a_num_vectors = M;
      if (lda == 0)
      {
        buffer.lda = buffer.M;
      }
      else if (lda < buffer.M)
      {
        std::cerr << "lda:wrong size\n";
        exit(1);
      }
      else
      {
        buffer.lda = lda;
      }
  }
  else
  {
      buffer.side = clblasRight;
      buffer.a_num_vectors = N;
      if (lda == 0)
      {
        buffer.lda = buffer.N;
      }
      else if (lda < buffer.N)
      {
        std::cerr << "lda:wrong size\n";
        exit(1);
      }
      else
      {
        buffer.lda = lda;
      }
  }
  /*}
  if (lda == 0)
  {
    buffer.lda = buffer.M;
  }
  else if (lda < buffer.M)
  {
    std::cerr << "lda:wrong size\n";
    exit(1);
  }
  else
  {
    buffer.lda = lda;
  }*/
  if (ldb == 0)
  {
    buffer.ldb = buffer.M;
  }
  else if (ldb < buffer.M)
  {
    std::cerr << "ldb:wrong size\n";
    exit(1);
  }
  else
  {
    buffer.ldb = ldb;
  }
  if (ldc == 0)
  {
    buffer.ldc = buffer.M;
  }
  else if (ldc < buffer.M)
  {
    std::cerr << "ldc:wrong size\n";
    exit(1);
  }
  else
  {
    buffer.ldc = ldc;
  }
  buffer.cpuB = new T[buffer.N * buffer.ldb];
  buffer.cpuC = new T[buffer.N * buffer.ldc];
  buffer.cpuA = new T[buffer.a_num_vectors * buffer.lda];
  }
  	void releaseGPUBuffer_deleteCPUBuffer()
	{
		//this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
		//need to do this before we eventually hit the destructor
		delete buffer.cpuA;
		delete buffer.cpuB;
		delete buffer.cpuC;
		OPENCL_V_THROW( clReleaseMemObject(buffer.A), "releasing buffer A");
		OPENCL_V_THROW( clReleaseMemObject(buffer.B), "releasing buffer B");
		OPENCL_V_THROW( clReleaseMemObject(buffer.C), "releasing buffer C");
	}
protected:
  void initialize_scalars(double alpha, double beta)
  {
      buffer.alpha = makeScalar<T>(alpha);
      buffer.beta = makeScalar<T>(beta);
  }

private:
  xSymmBuffer<T> buffer;
};

template <typename T>
void xSymm<T>::setup_buffer(int order_option, int side_option, int
                    uplo_option, int diag_option, int transA_option, int
                    transB_option, size_t M, size_t N, size_t K,
                    size_t lda, size_t ldb, size_t ldc,size_t offA,
					          size_t offB, size_t offC, double alpha,
                    double beta)
{
  initialize_scalars(alpha, beta);
  buffer.offa = offA;
  buffer.offb = offB;
  buffer.offc = offC;
  buffer.M = M;
  buffer.N = N;
  if (order_option == 0)
  {
  buffer.order = clblasRowMajor;
  }
  else
  {
  buffer.order = clblasColumnMajor;
  }
  if (uplo_option == 0)
  {
    buffer.uplo = clblasUpper;
  }
  else
  {
    buffer.uplo = clblasLower;
  }
  if (side_option == 0)
  {
      buffer.side = clblasLeft;
      buffer.a_num_vectors = M;
      if (lda == 0)
      {
        buffer.lda = buffer.M;
      }
      else if (lda < buffer.M)
      {
        std::cerr << "lda:wrong size\n";
        exit(1);
      }
      else
      {
        buffer.lda = lda;
      }
  }
  else
  {
      buffer.side = clblasRight;
      buffer.a_num_vectors = N;
      if (lda == 0)
      {
        buffer.lda = buffer.N;
      }
      else if (lda < buffer.N)
      {
        std::cerr << "lda:wrong size\n";
        exit(1);
      }
      else
      {
        buffer.lda = lda;
      }
  }
  /*}
  if (lda == 0)
  {
    buffer.lda = buffer.M;
  }
  else if (lda < buffer.M)
  {
    std::cerr << "lda:wrong size\n";
    exit(1);
  }
  else
  {
    buffer.lda = lda;
  }*/
  if (ldb == 0)
  {
    buffer.ldb = buffer.M;
  }
  else if (ldb < buffer.M)
  {
    std::cerr << "ldb:wrong size\n";
    exit(1);
  }
  else
  {
    buffer.ldb = ldb;
  }
  if (ldc == 0)
  {
    buffer.ldc = buffer.M;
  }
  else if (ldc < buffer.M)
  {
    std::cerr << "ldc:wrong size\n";
    exit(1);
  }
  else
  {
    buffer.ldc = ldc;
  }
  buffer.cpuB = new T[buffer.N * buffer.ldb];
  buffer.cpuC = new T[buffer.N * buffer.ldc];
  buffer.cpuA = new T[buffer.a_num_vectors * buffer.lda];
  cl_int err;
  buffer.A = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                buffer.a_num_vectors * buffer.lda*sizeof(T),
                                NULL, &err);

  buffer.B = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                    buffer.N*buffer.ldb*sizeof(T),
                                    NULL, &err);
  buffer.C = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                    buffer.N*buffer.ldc*sizeof(T),
                                    NULL, &err);
}

template <typename T>
void xSymm<T>::initialize_cpu_buffer()
{
  srand(10);
  for (size_t i = 0; i < buffer.a_num_vectors; ++i)
  {
    for (size_t j = 0; j < buffer.lda; ++j)
    {
        buffer.cpuA[i*buffer.lda+j] = random<T>(UPPER_BOUND<T>()) /
                                        randomScale<T>();
    }
  }
  for (size_t i = 0; i < buffer.N; ++i)
  {
    for (size_t j = 0; j < buffer.ldb; ++j)
    {
        buffer.cpuB[i*buffer.ldb+j] = random<T>(UPPER_BOUND<T>()) /
                                      randomScale<T>();
    }
  }
  for (size_t i = 0; i < buffer.N; ++i)
  {
    for (size_t j = 0; j < buffer.ldc; ++j)
    {
        buffer.cpuC[i*buffer.ldc+j] = random<T>(UPPER_BOUND<T>()) /
                                      randomScale<T>();
    }
  }
}

template <typename T>
void xSymm<T>::initialize_gpu_buffer()
{
  cl_int err;

  err = clEnqueueWriteBuffer(queue_, buffer.A, CL_TRUE,
                              buffer.offa * sizeof(T),
                              buffer.a_num_vectors * buffer.lda*sizeof(T),
                              buffer.cpuA, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue_, buffer.B, CL_TRUE, 0,
                              buffer.ldb*buffer.N*sizeof(T),
                              buffer.cpuB, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue_, buffer.C, CL_TRUE, 0,
                              buffer.ldc*buffer.N*sizeof(T),
                              buffer.cpuC, 0, NULL, NULL);
}

template <typename T>
void xSymm<T>::reset_gpu_write_buffer()
{
  cl_int err;
  err = clEnqueueWriteBuffer(queue_, buffer.C, CL_TRUE, 0,
                              buffer.ldc*buffer.N*sizeof(T),
                              buffer.cpuC, 0, NULL, NULL);
}

template <>
void xSymm<cl_float>::call_func()
{
  timer.Start(timer_id);
  clblasSsymm(buffer.order, buffer.side, buffer.uplo, buffer.M, buffer.N,
      buffer.alpha, buffer.A, buffer.offa, buffer.lda, buffer.B, buffer.offb,
      buffer.ldb, buffer.beta, buffer.C, buffer.offc, buffer.ldc, 1, &queue_,
      0, NULL,&event_);
  clWaitForEvents(1, &event_);
  timer.Stop(timer_id);
}

template <>
void xSymm<cl_float>::roundtrip_func()
{
  timer.Start(timer_id);
  //set up buffer
    cl_int err;
  buffer.A = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                buffer.a_num_vectors * buffer.lda*sizeof(cl_float),
                                NULL, &err);

  buffer.B = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                    buffer.N*buffer.ldb*sizeof(cl_float),
                                    NULL, &err);
  buffer.C = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                    buffer.N*buffer.ldc*sizeof(cl_float),
                                    NULL, &err);
  //initialize gpu buffer
  err = clEnqueueWriteBuffer(queue_, buffer.A, CL_TRUE,
                              buffer.offa * sizeof(cl_float),
                              buffer.a_num_vectors * buffer.lda*sizeof(cl_float),
                              buffer.cpuA, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue_, buffer.B, CL_TRUE, 0,
                              buffer.ldb*buffer.N*sizeof(cl_float),
                              buffer.cpuB, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue_, buffer.C, CL_TRUE, 0,
                              buffer.ldc*buffer.N*sizeof(cl_float),
                              buffer.cpuC, 0, NULL, NULL);
  //call func
  clblasSsymm(buffer.order, buffer.side, buffer.uplo, buffer.M, buffer.N,
      buffer.alpha, buffer.A, buffer.offa, buffer.lda, buffer.B, buffer.offb,
      buffer.ldb, buffer.beta, buffer.C, buffer.offc, buffer.ldc, 1, &queue_,
      0, NULL,NULL);
  //read gpu buffer
  err = clEnqueueReadBuffer(queue_, buffer.C, CL_TRUE,
			                      buffer.offc * sizeof(cl_float), buffer.ldc * buffer.N *
                                       sizeof(cl_float),
								  buffer.cpuC, 0, NULL, &event_);
  clWaitForEvents(1, &event_);
  timer.Stop(timer_id);
}

template <>
void xSymm<cl_double>::call_func()
{
  timer.Start(timer_id);
  clblasDsymm(buffer.order, buffer.side, buffer.uplo, buffer.M, buffer.N,
      buffer.alpha, buffer.A, buffer.offa, buffer.lda, buffer.B, buffer.offb,
      buffer.ldb, buffer.beta, buffer.C, buffer.offc, buffer.ldc, 1, &queue_,
      0, NULL,&event_);
  clWaitForEvents(1, &event_);
  timer.Stop(timer_id);
}

template <>
void xSymm<cl_double>::roundtrip_func()
{
  timer.Start(timer_id);
  //set up buffer
    cl_int err;
  buffer.A = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                buffer.a_num_vectors * buffer.lda*sizeof(cl_double),
                                NULL, &err);

  buffer.B = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                    buffer.N*buffer.ldb*sizeof(cl_double),
                                    NULL, &err);
  buffer.C = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                    buffer.N*buffer.ldc*sizeof(cl_double),
                                    NULL, &err);
  //initialize gpu buffer
  err = clEnqueueWriteBuffer(queue_, buffer.A, CL_TRUE,
                              buffer.offa * sizeof(cl_double),
                              buffer.a_num_vectors * buffer.lda*sizeof(cl_double),
                              buffer.cpuA, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue_, buffer.B, CL_TRUE, 0,
                              buffer.ldb*buffer.N*sizeof(cl_double),
                              buffer.cpuB, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue_, buffer.C, CL_TRUE, 0,
                              buffer.ldc*buffer.N*sizeof(cl_double),
                              buffer.cpuC, 0, NULL, NULL);
  //call func
  clblasDsymm(buffer.order, buffer.side, buffer.uplo, buffer.M, buffer.N,
      buffer.alpha, buffer.A, buffer.offa, buffer.lda, buffer.B, buffer.offb,
      buffer.ldb, buffer.beta, buffer.C, buffer.offc, buffer.ldc, 1, &queue_,
      0, NULL,NULL);
  //read gpu buffer
  err = clEnqueueReadBuffer(queue_, buffer.C, CL_TRUE,
			                      buffer.offc * sizeof(cl_double), buffer.ldc * buffer.N *
                                       sizeof(cl_double),
								  buffer.cpuC, 0, NULL, &event_);
  clWaitForEvents(1, &event_);
  timer.Stop(timer_id);
}

template <>
void xSymm<cl_float2>::call_func()
{
  timer.Start(timer_id);
  clblasCsymm(buffer.order, buffer.side, buffer.uplo, buffer.M, buffer.N,
      buffer.alpha, buffer.A, buffer.offa, buffer.lda, buffer.B, buffer.offb,
      buffer.ldb, buffer.beta, buffer.C, buffer.offc, buffer.ldc, 1, &queue_,
      0, NULL,&event_);
  clWaitForEvents(1, &event_);
  timer.Stop(timer_id);
}

template <>
void xSymm<cl_float2>::roundtrip_func()
{
  timer.Start(timer_id);
  //set up buffer
    cl_int err;
  buffer.A = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                buffer.a_num_vectors * buffer.lda*sizeof(cl_float2),
                                NULL, &err);

  buffer.B = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                    buffer.N*buffer.ldb*sizeof(cl_float2),
                                    NULL, &err);
  buffer.C = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                    buffer.N*buffer.ldc*sizeof(cl_float2),
                                    NULL, &err);
  //initialize gpu buffer
  err = clEnqueueWriteBuffer(queue_, buffer.A, CL_TRUE,
                              buffer.offa * sizeof(cl_float2),
                              buffer.a_num_vectors * buffer.lda*sizeof(cl_float2),
                              buffer.cpuA, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue_, buffer.B, CL_TRUE, 0,
                              buffer.ldb*buffer.N*sizeof(cl_float2),
                              buffer.cpuB, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue_, buffer.C, CL_TRUE, 0,
                              buffer.ldc*buffer.N*sizeof(cl_float2),
                              buffer.cpuC, 0, NULL, NULL);
  //call func
  clblasCsymm(buffer.order, buffer.side, buffer.uplo, buffer.M, buffer.N,
      buffer.alpha, buffer.A, buffer.offa, buffer.lda, buffer.B, buffer.offb,
      buffer.ldb, buffer.beta, buffer.C, buffer.offc, buffer.ldc, 1, &queue_,
      0, NULL,NULL);
  //read gpu buffer
  err = clEnqueueReadBuffer(queue_, buffer.C, CL_TRUE,
			                      buffer.offc * sizeof(cl_float2), buffer.ldc * buffer.N *
                                       sizeof(cl_float2),
								  buffer.cpuC, 0, NULL, &event_);
  clWaitForEvents(1, &event_);
  timer.Stop(timer_id);
}

template <>
void xSymm<cl_double2>::call_func()
{
  timer.Start(timer_id);
  clblasZsymm(buffer.order, buffer.side, buffer.uplo, buffer.M, buffer.N,
      buffer.alpha, buffer.A, buffer.offa, buffer.lda, buffer.B, buffer.offb,
      buffer.ldb, buffer.beta, buffer.C, buffer.offc, buffer.ldc, 1, &queue_,
      0, NULL,&event_);
  clWaitForEvents(1, &event_);
  timer.Stop(timer_id);
}

template <>
void xSymm<cl_double2>::roundtrip_func()
{
  timer.Start(timer_id);
  //set up buffer
  cl_int err;
  buffer.A = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                buffer.a_num_vectors * buffer.lda*sizeof(cl_double2),
                                NULL, &err);

  buffer.B = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                    buffer.N*buffer.ldb*sizeof(cl_double2),
                                    NULL, &err);
  buffer.C = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                    buffer.N*buffer.ldc*sizeof(cl_double2),
                                    NULL, &err);
  //initialize gpu buffer
  err = clEnqueueWriteBuffer(queue_, buffer.A, CL_TRUE,
                              buffer.offa * sizeof(cl_double2),
                              buffer.a_num_vectors * buffer.lda*sizeof(cl_double2),
                              buffer.cpuA, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue_, buffer.B, CL_TRUE, 0,
                              buffer.ldb*buffer.N*sizeof(cl_double2),
                              buffer.cpuB, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue_, buffer.C, CL_TRUE, 0,
                              buffer.ldc*buffer.N*sizeof(cl_double2),
                              buffer.cpuC, 0, NULL, NULL);
  //call func
  clblasZsymm(buffer.order, buffer.side, buffer.uplo, buffer.M, buffer.N,
      buffer.alpha, buffer.A, buffer.offa, buffer.lda, buffer.B, buffer.offb,
      buffer.ldb, buffer.beta, buffer.C, buffer.offc, buffer.ldc, 1, &queue_,
      0, NULL,NULL);
  //read gpu buffer
  err = clEnqueueReadBuffer(queue_, buffer.C, CL_TRUE,
			                      buffer.offc * sizeof(cl_double2), buffer.ldc * buffer.N *
                                       sizeof(cl_double2),
								  buffer.cpuC, 0, NULL, &event_);
  clWaitForEvents(1, &event_);
  timer.Stop(timer_id);
}

template<>
double
xSymm<cl_float2>::
gflops()
{
  if (buffer.side == clblasLeft)
    return static_cast<double>((8 * buffer.M * buffer.M * buffer.N)/time_in_ns());
  else
    return static_cast<double>((8 * buffer.N * buffer.N * buffer.M)/time_in_ns());
}

template<>
double
xSymm<cl_double2>::
gflops()
{
  if (buffer.side == clblasLeft)
      return static_cast<double>((8 * buffer.M * buffer.M * buffer.N)/time_in_ns());
  else
      return static_cast<double>((8 * buffer.N * buffer.N * buffer.M)/time_in_ns());
}

template<>
std::string
xSymm<cl_float2>::
gflops_formula()
{
  if (buffer.side == clblasLeft)
      return "8*M*M*N/time";
  else
      return "8*N*N*M/time";
}

template<>
std::string
xSymm<cl_double2>::
gflops_formula()
{
  if (buffer.side == clblasLeft)
      return "8*M*M*N/time";
  else
      return "8*N*N*M/time";
}

#endif // ifndef CLBLAS_BENCHMARK_XSYR_HXX__