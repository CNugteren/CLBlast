
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains all the CUDA related code; used only in case of testing against cuBLAS
//
// =================================================================================================

#ifndef CLBLAST_TEST_WRAPPER_CUDA_H_
#define CLBLAST_TEST_WRAPPER_CUDA_H_

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

#include "utilities/utilities.hpp"

#ifdef CLBLAST_REF_CUBLAS
  #define CUDA_NO_HALF
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
#endif

namespace clblast {
// =================================================================================================

#ifdef CLBLAST_REF_CUBLAS
  template <typename T>
  void cublasSetup(Arguments<T> &args) {
    cudaSetDevice(static_cast<int>(args.device_id));
    auto status = cublasCreate(reinterpret_cast<cublasHandle_t*>(&args.cublas_handle));
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("CUDA cublasCreate error");
    }
  }
#endif

#ifdef CLBLAST_REF_CUBLAS
  template <typename T>
  void cublasTeardown(Arguments<T> &args) {
    auto status = cublasDestroy(reinterpret_cast<cublasHandle_t>(args.cublas_handle));
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("CUDA cublasDestroy error");
    }
  }
#endif

// =================================================================================================

// Copies data from the CUDA device to the host and frees-up the CUDA memory afterwards
#ifdef CLBLAST_REF_CUBLAS
  template <typename T>
  void CUDAToHost(T** buffer_cuda, std::vector<T> &buffer_host, const size_t size) {
    auto status1 = cudaMemcpy(
      reinterpret_cast<void*>(buffer_host.data()),
      reinterpret_cast<void*>(*buffer_cuda),
      size*sizeof(T),
      cudaMemcpyDeviceToHost
    );
    if (status1 != cudaSuccess) {
      throw std::runtime_error("CUDA cudaMemcpy error with status: "+ToString(static_cast<int>(status1)));
    }
    auto status2 = cudaFree(*buffer_cuda);
    if (status2 != cudaSuccess) {
      throw std::runtime_error("CUDA cudaFree error with status: "+ToString(static_cast<int>(status2)));
    }
    *buffer_cuda = nullptr;
}
#else
  template <typename T> void CUDAToHost(T**, const std::vector<T>&, const size_t) { }
#endif

// Allocates space on the CUDA device and copies in data from the host
#ifdef CLBLAST_REF_CUBLAS
  template <typename T>
  void HostToCUDA(T** buffer_cuda, std::vector<T> &buffer_host, const size_t size) {
    if (*buffer_cuda == nullptr) {
      auto status1 = cudaMalloc(reinterpret_cast<void**>(buffer_cuda), size*sizeof(T));
      if (status1 != cudaSuccess) {
        throw std::runtime_error("CUDA cudaMalloc error with status: "+ToString(static_cast<int>(status1)));
      }
    }
    auto status2 = cudaMemcpy(
      reinterpret_cast<void*>(*buffer_cuda),
      reinterpret_cast<void*>(buffer_host.data()),
      size*sizeof(T),
      cudaMemcpyHostToDevice
    );
    if (status2 != cudaSuccess) {
      throw std::runtime_error("CUDA cudaMemcpy error with status: "+ToString(static_cast<int>(status2)));
    }
  }
#else
  template <typename T> void HostToCUDA(T**, const std::vector<T>&, const size_t) { }
#endif

// =================================================================================================

template <typename T>
struct BuffersCUDA {
  T* x_vec = nullptr;
  T* y_vec = nullptr;
  T* a_mat = nullptr;
  T* b_mat = nullptr;
  T* c_mat = nullptr;
  T* ap_mat = nullptr;
  T* scalar = nullptr;
};

template <typename T, typename U>
void CUDAToHost(const Arguments<U> &args, BuffersCUDA<T> &buffers, BuffersHost<T> &buffers_host,
                const std::vector<std::string> &names) {
  for (auto &name: names) {
    if (name == kBufVecX) { buffers_host.x_vec = std::vector<T>(args.x_size, static_cast<T>(0)); CUDAToHost(&buffers.x_vec, buffers_host.x_vec, args.x_size); }
    else if (name == kBufVecY) { buffers_host.y_vec = std::vector<T>(args.y_size, static_cast<T>(0)); CUDAToHost(&buffers.y_vec, buffers_host.y_vec, args.y_size); }
    else if (name == kBufMatA) { buffers_host.a_mat = std::vector<T>(args.a_size, static_cast<T>(0)); CUDAToHost(&buffers.a_mat, buffers_host.a_mat, args.a_size); }
    else if (name == kBufMatB) { buffers_host.b_mat = std::vector<T>(args.b_size, static_cast<T>(0)); CUDAToHost(&buffers.b_mat, buffers_host.b_mat, args.b_size); }
    else if (name == kBufMatC) { buffers_host.c_mat = std::vector<T>(args.c_size, static_cast<T>(0)); CUDAToHost(&buffers.c_mat, buffers_host.c_mat, args.c_size); }
    else if (name == kBufMatAP) { buffers_host.ap_mat = std::vector<T>(args.ap_size, static_cast<T>(0)); CUDAToHost(&buffers.ap_mat, buffers_host.ap_mat, args.ap_size); }
    else if (name == kBufScalar) { buffers_host.scalar = std::vector<T>(args.scalar_size, static_cast<T>(0)); CUDAToHost(&buffers.scalar, buffers_host.scalar, args.scalar_size); }
    else { throw std::runtime_error("Invalid buffer name"); }
  }
}

template <typename T, typename U>
void HostToCUDA(const Arguments<U> &args, BuffersCUDA<T> &buffers, BuffersHost<T> &buffers_host,
                const std::vector<std::string> &names) {
  for (auto &name: names) {
    if (name == kBufVecX) { HostToCUDA(&buffers.x_vec, buffers_host.x_vec, args.x_size); }
    else if (name == kBufVecY) { HostToCUDA(&buffers.y_vec, buffers_host.y_vec, args.y_size); }
    else if (name == kBufMatA) { HostToCUDA(&buffers.a_mat, buffers_host.a_mat, args.a_size); }
    else if (name == kBufMatB) { HostToCUDA(&buffers.b_mat, buffers_host.b_mat, args.b_size); }
    else if (name == kBufMatC) { HostToCUDA(&buffers.c_mat, buffers_host.c_mat, args.c_size); }
    else if (name == kBufMatAP) { HostToCUDA(&buffers.ap_mat, buffers_host.ap_mat, args.ap_size); }
    else if (name == kBufScalar) { HostToCUDA(&buffers.scalar, buffers_host.scalar, args.scalar_size); }
    else { throw std::runtime_error("Invalid buffer name"); }
  }
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_WRAPPER_CUDA_H_
#endif
