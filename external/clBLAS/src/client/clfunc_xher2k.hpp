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

#ifndef CLBLAS_BENCHMARK_XHER2K_HXX__
#define CLBLAS_BENCHMARK_XHER2K_HXX__

#include "clfunc_common.hpp"

template <typename T>
struct xHer2kBuffer
{
    clblasOrder order_;
    clblasUplo uplo_;
    clblasTranspose transA_;
    size_t N_;
    size_t K_;
    T alpha_;
	cl_mem A_;
    size_t offa_;
    size_t lda_;
	cl_mem B_;
	size_t offb_;
	size_t ldb_;
    T beta_;
    cl_mem C_;
    size_t offc_;
    size_t ldc_;
	size_t a_num_vectors_;
	size_t b_num_vectors_;
    size_t c_num_vectors_;
	T* cpuA_;
	T* cpuB_;
	T* cpuC_;
}; // struct buffer

template <typename T>
class xHer2k : public clblasFunc
{
public:
  xHer2k(StatisticalTimer& timer, cl_device_type devType) : clblasFunc(timer,  devType)
  {
    timer.getUniqueID("clHer2k", 0);
  }

  ~xHer2k()
  {
  }

  double gflops()
  {
    return static_cast<double>(8*(buffer_.K_ * buffer_.N_ * buffer_.N_)/time_in_ns()+2*buffer_.N_/time_in_ns());
  }

  std::string gflops_formula()
  {
    return "(8*K*N*N+2*N)/time";
  }

  void setup_buffer(int order_option, int side_option, int
                    uplo_option, int diag_option, int transA_option, int
                    transB_option, size_t M, size_t N, size_t K,
                    size_t lda, size_t ldb, size_t ldc,size_t offA,
					          size_t offB, size_t offC, double alpha,
                    double beta)
  {
        DUMMY_ARGS_USAGE_4(side_option, diag_option, transB_option, M);

		initialize_scalars(alpha,beta);

		buffer_.N_ = N;
		buffer_.K_ = K;
		buffer_.offa_ = offA;
		buffer_.offb_ = offB;
		buffer_.offc_ = offC;

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
                buffer_.transA_ = clblasNoTrans;
                buffer_.a_num_vectors_ = N;
				buffer_.b_num_vectors_ = N;
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

                if (ldb == 0)
                {
                    buffer_.ldb_ = K;
                }
                else if (ldb < K)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.ldb_ = ldb;
                }
            }
            else
            {
                buffer_.a_num_vectors_ = K;
				buffer_.b_num_vectors_ = K;
                if (transA_option == 1)
                {
                    buffer_.transA_ = clblasTrans;
                }
                else if (transA_option == 2)
                {
                    buffer_.transA_ = clblasConjTrans;
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

                if (ldb == 0)
                {
                    buffer_.ldb_ = N;
                }
                else if (ldb < N)
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
            order_ = clblasColumnMajor;
            if (transA_option == 0)
            {
                buffer_.a_num_vectors_ = K;
                buffer_.b_num_vectors_ = K;
                buffer_.transA_ = clblasNoTrans;
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

                if (ldb == 0)
                {
                    buffer_.ldb_ = N;
                }
                else if (ldb < N)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.ldb_ = ldb;
                }
            }
            else
            {
                buffer_.a_num_vectors_ = N;
                buffer_.b_num_vectors_ = N;
                if (transA_option == 1)
                {
                    buffer_.transA_ = clblasTrans;
                }
                else if (transA_option == 2)
                {
                    buffer_.transA_ = clblasConjTrans;
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

                if (ldb == 0)
                {
                    buffer_.ldb_ = K;
                }
                else if (ldb < K)
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

        buffer_.cpuA_ = new T[buffer_.lda_*buffer_.a_num_vectors_];
		buffer_.cpuB_ = new T[buffer_.ldb_*buffer_.b_num_vectors_];
        buffer_.cpuC_ = new T[buffer_.ldc_*buffer_.c_num_vectors_];

        cl_int err;
        buffer_.A_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offa_) * sizeof(T),
                                        NULL, &err);

	    buffer_.B_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offb_) * sizeof(T),
                                        NULL, &err);

        buffer_.C_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (buffer_.ldc_ * buffer_.c_num_vectors_ +
                                            buffer_.offc_) * sizeof(T),
                                        NULL, &err);
  }
  void initialize_cpu_buffer()
  {
	  srand(10);
	  for (size_t i = 0; i < buffer_.a_num_vectors_; ++i)
	  {
		  for (size_t j = 0; j < buffer_.lda_; ++j)
		  {
                buffer_.cpuA_[i*buffer_.lda_+j] = random<T>(UPPER_BOUND<T>()) /
                                               randomScale<T>();
		  }
	  }
	  for (size_t i = 0; i < buffer_.N_; ++i)
	  {
		  for (size_t j = 0; j < buffer_.ldc_; ++j)
		  {
                buffer_.cpuC_[i*buffer_.ldc_+j] = random<T>(UPPER_BOUND<T>()) /
                                               randomScale<T>();
		  }
	  }
  }
  void initialize_gpu_buffer()
  {
	    cl_int err;

        err = clEnqueueWriteBuffer(queue_, buffer_.A_, CL_TRUE,
                                   buffer_.offa_ * sizeof(T),
                                   buffer_.lda_ * buffer_.a_num_vectors_ *
                                       sizeof(T),
                                   buffer_.cpuA_, 0, NULL, NULL);

        err = clEnqueueWriteBuffer(queue_, buffer_.C_, CL_TRUE,
                                   buffer_.offa_ * sizeof(T),
                                   buffer_.ldc_ * buffer_.c_num_vectors_ *
                                       sizeof(T),
                                   buffer_.cpuC_, 0, NULL, NULL);
  }
  void reset_gpu_write_buffer()
  {
	    cl_int err;

        err = clEnqueueWriteBuffer(queue_, buffer_.C_, CL_TRUE,
                                   buffer_.offc_ * sizeof(T),
                                   buffer_.ldc_ * buffer_.c_num_vectors_ *
                                       sizeof(T),
                                   buffer_.cpuC_, 0, NULL, NULL);
  }
  void call_func();
  void read_gpu_buffer()
	{
		cl_int err;
		err = clEnqueueReadBuffer(queue_, buffer_.C_, CL_TRUE,
								  buffer_.offc_*sizeof(T), buffer_.ldc_*buffer_.c_num_vectors_*sizeof(T),
								  buffer_.cpuC_, 0, NULL, NULL);
	}
	void roundtrip_func();
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
        DUMMY_ARGS_USAGE_4(side_option, diag_option, transB_option, M);

		initialize_scalars(alpha,beta);

		buffer_.N_ = N;
		buffer_.K_ = K;
		buffer_.offa_ = offA;
		buffer_.offb_ = offBX;
		buffer_.offc_ = offCY;

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
                buffer_.transA_ = clblasNoTrans;
                buffer_.a_num_vectors_ = N;
				buffer_.b_num_vectors_ = N;
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

                if (ldb == 0)
                {
                    buffer_.ldb_ = K;
                }
                else if (ldb < K)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.ldb_ = ldb;
                }
            }
            else
            {
                buffer_.a_num_vectors_ = K;
				buffer_.b_num_vectors_ = K;
                if (transA_option == 1)
                {
                    buffer_.transA_ = clblasTrans;
                }
                else if (transA_option == 2)
                {
                    buffer_.transA_ = clblasConjTrans;
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

                if (ldb == 0)
                {
                    buffer_.ldb_ = N;
                }
                else if (ldb < N)
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
            order_ = clblasColumnMajor;
            if (transA_option == 0)
            {
                buffer_.a_num_vectors_ = K;
                buffer_.b_num_vectors_ = K;
                buffer_.transA_ = clblasNoTrans;
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

                if (ldb == 0)
                {
                    buffer_.ldb_ = N;
                }
                else if (ldb < N)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer_.ldb_ = ldb;
                }
            }
            else
            {
                buffer_.a_num_vectors_ = N;
                buffer_.b_num_vectors_ = N;
                if (transA_option == 1)
                {
                    buffer_.transA_ = clblasTrans;
                }
                else if (transA_option == 2)
                {
                    buffer_.transA_ = clblasConjTrans;
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

                if (ldb == 0)
                {
                    buffer_.ldb_ = K;
                }
                else if (ldb < K)
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

        buffer_.cpuA_ = new T[buffer_.lda_*buffer_.a_num_vectors_];
		buffer_.cpuB_ = new T[buffer_.ldb_*buffer_.b_num_vectors_];
        buffer_.cpuC_ = new T[buffer_.ldc_*buffer_.c_num_vectors_];
	}
	void releaseGPUBuffer_deleteCPUBuffer()
	{
		//this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
		//need to do this before we eventually hit the destructor
		delete buffer_.cpuA_;
		delete buffer_.cpuB_;
		delete buffer_.cpuC_;
		OPENCL_V_THROW( clReleaseMemObject(buffer_.A_), "releasing buffer A");
		OPENCL_V_THROW( clReleaseMemObject(buffer_.B_), "releasing buffer B");
		OPENCL_V_THROW( clReleaseMemObject(buffer_.C_), "releasing buffer C");
	}
protected:
protected:
  void initialize_scalars(double alpha, double beta)
  {
      buffer_.alpha_ = makeScalar<T>(alpha);
      buffer_.beta_ = makeScalar<T>(beta);
  }

private:
  xHer2kBuffer<T> buffer_;
};

template<>
void 
xHer2k<cl_float2>::call_func()
{
	timer.Start(timer_id);
	clblasCher2k(order_, buffer_.uplo_, buffer_.transA_,
				buffer_.N_, buffer_.K_, buffer_.alpha_,
				buffer_.A_, buffer_.offa_, buffer_.lda_, 
				buffer_.B_, buffer_.offb_, buffer_.ldb_,
				buffer_.beta_.s[0], buffer_.C_, buffer_.offc_,
				buffer_.ldc_, 1, &queue_, 0, NULL, &event_);
    clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

template<>
void 
xHer2k<cl_float2>::roundtrip_func()
{
		timer.Start(timer_id);
        cl_int err;
        buffer_.A_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offa_) * sizeof(cl_float2),
                                        NULL, &err);
	    buffer_.B_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offb_) * sizeof(cl_float2),
                                        NULL, &err);
        buffer_.C_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (buffer_.ldc_ * buffer_.c_num_vectors_ +
                                            buffer_.offc_) * sizeof(cl_float2),
                                        NULL, &err);
		this->initialize_gpu_buffer();
		clblasCher2k(order_, buffer_.uplo_, buffer_.transA_,
				buffer_.N_, buffer_.K_, buffer_.alpha_,
				buffer_.A_, buffer_.offa_, buffer_.lda_, 
				buffer_.B_, buffer_.offb_, buffer_.ldb_,
				buffer_.beta_.s[0], buffer_.C_, buffer_.offc_,
				buffer_.ldc_, 1, &queue_, 0, NULL, NULL);

		err = clEnqueueWriteBuffer(queue_, buffer_.C_, CL_TRUE,
                                   buffer_.offc_ * sizeof(cl_float2),
                                   buffer_.ldc_ * buffer_.c_num_vectors_ *
                                       sizeof(cl_float2),
                                   buffer_.cpuC_, 0, NULL, &event_);
		clWaitForEvents(1, &event_);
		timer.Stop(timer_id);
}

template<>
void 
xHer2k<cl_double2>::call_func()
{
	timer.Start(timer_id);

	clblasZher2k(order_, buffer_.uplo_, buffer_.transA_,
				buffer_.N_, buffer_.K_, buffer_.alpha_,
				buffer_.A_, buffer_.offa_, buffer_.lda_, 
				buffer_.B_, buffer_.offb_, buffer_.ldb_,
				buffer_.beta_.s[0], buffer_.C_, buffer_.offc_,
				buffer_.ldc_, 1, &queue_, 0, NULL, &event_);

    clWaitForEvents(1, &event_);
    timer.Stop(timer_id);
}

template<>
void
xHer2k<cl_double2>::roundtrip_func()
{
		timer.Start(timer_id);
        cl_int err;
        buffer_.A_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.lda_ * buffer_.a_num_vectors_ +
                                            buffer_.offa_) * sizeof(cl_double2),
                                        NULL, &err);
	    buffer_.B_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer_.ldb_ * buffer_.b_num_vectors_ +
                                            buffer_.offb_) * sizeof(cl_double2),
                                        NULL, &err);
        buffer_.C_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (buffer_.ldc_ * buffer_.c_num_vectors_ +
                                            buffer_.offc_) * sizeof(cl_double2),
                                        NULL, &err);
		this->initialize_gpu_buffer();

	   clblasZher2k(order_, buffer_.uplo_, buffer_.transA_,
				buffer_.N_, buffer_.K_, buffer_.alpha_,
				buffer_.A_, buffer_.offa_, buffer_.lda_, 
				buffer_.B_, buffer_.offb_, buffer_.ldb_,
				buffer_.beta_.s[0], buffer_.C_, buffer_.offc_,
				buffer_.ldc_, 1, &queue_, 0, NULL, NULL);

		err = clEnqueueWriteBuffer(queue_, buffer_.C_, CL_TRUE,
                                   buffer_.offc_ * sizeof(cl_double2),
                                   buffer_.ldc_ * buffer_.c_num_vectors_ *
                                       sizeof(cl_double2),
                                   buffer_.cpuC_, 0, NULL, &event_);
		clWaitForEvents(1, &event_);
		timer.Stop(timer_id);
}
#endif // ifndef CLBLAS_BENCHMARK_XSYR_HXX__