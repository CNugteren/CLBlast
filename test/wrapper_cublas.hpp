
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a wrapper around the cuBLAS library, such that its routines can be called
// in a similar way as the CLBlast routines: using alpha and beta to determine the precision.
//
// =================================================================================================

#ifndef CLBLAST_TEST_WRAPPER_CUBLAS_H_
#define CLBLAST_TEST_WRAPPER_CUBLAS_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utilities/utilities.hpp"

namespace clblast {

// Conversions from CLBlast types
cublasOperation_t convertToCUBLAS(const Transpose v) { return (v == Transpose::kNo) ? CUBLAS_OP_N : (v == Transpose::kYes) ? CUBLAS_OP_T : CUBLAS_OP_C; }
cublasFillMode_t convertToCUBLAS(const Triangle v) { return (v == Triangle::kUpper) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER; }
cublasDiagType_t convertToCUBLAS(const Diagonal v) { return (v == Diagonal::kUnit) ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT; }
cublasSideMode_t convertToCUBLAS(const Side v) { return (v == Side::kLeft) ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT; }

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// Forwards the cuBLAS calls for SROTG/DROTG
template <typename T>
cublasStatus_t cublasXrotg(cublasHandle_t handle, T* sa_buffer, const size_t sa_offset,
                           T* sb_buffer, const size_t sb_offset,
                           T* sc_buffer, const size_t sc_offset,
                           T* ss_buffer, const size_t ss_offset);
template <>
cublasStatus_t cublasXrotg<float>(cublasHandle_t handle, float* sa_buffer, const size_t sa_offset,
                                  float* sb_buffer, const size_t sb_offset,
                                  float* sc_buffer, const size_t sc_offset,
                                  float* ss_buffer, const size_t ss_offset) {
  auto status = cublasSrotg(handle, &sa_buffer[sa_offset],
                            &sb_buffer[sb_offset],
                            &sc_buffer[sc_offset],
                            &ss_buffer[ss_offset]);
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXrotg<double>(cublasHandle_t handle, double* sa_buffer, const size_t sa_offset,
                                   double* sb_buffer, const size_t sb_offset,
                                   double* sc_buffer, const size_t sc_offset,
                                   double* ss_buffer, const size_t ss_offset) {
  auto status = cublasDrotg(handle, &sa_buffer[sa_offset],
                            &sb_buffer[sb_offset],
                            &sc_buffer[sc_offset],
                            &ss_buffer[ss_offset]);
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for SROTMG/DROTMG
template <typename T>
cublasStatus_t cublasXrotmg(cublasHandle_t handle, T* sd1_buffer, const size_t sd1_offset,
                            T* sd2_buffer, const size_t sd2_offset,
                            T* sx1_buffer, const size_t sx1_offset,
                            const T* sy1_buffer, const size_t sy1_offset,
                            T* sparam_buffer, const size_t sparam_offset);
template <>
cublasStatus_t cublasXrotmg<float>(cublasHandle_t handle, float* sd1_buffer, const size_t sd1_offset,
                                   float* sd2_buffer, const size_t sd2_offset,
                                   float* sx1_buffer, const size_t sx1_offset,
                                   const float* sy1_buffer, const size_t sy1_offset,
                                   float* sparam_buffer, const size_t sparam_offset) {
  auto status = cublasSrotmg(handle, &sd1_buffer[sd1_offset],
                             &sd2_buffer[sd2_offset],
                             &sx1_buffer[sx1_offset],
                             &sy1_buffer[sy1_offset],
                             &sparam_buffer[sparam_offset]);
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXrotmg<double>(cublasHandle_t handle, double* sd1_buffer, const size_t sd1_offset,
                                    double* sd2_buffer, const size_t sd2_offset,
                                    double* sx1_buffer, const size_t sx1_offset,
                                    const double* sy1_buffer, const size_t sy1_offset,
                                    double* sparam_buffer, const size_t sparam_offset) {
  auto status = cublasDrotmg(handle, &sd1_buffer[sd1_offset],
                             &sd2_buffer[sd2_offset],
                             &sx1_buffer[sx1_offset],
                             &sy1_buffer[sy1_offset],
                             &sparam_buffer[sparam_offset]);
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for SROT/DROT
cublasStatus_t cublasXrot(cublasHandle_t handle, const size_t n,
                          float* x_buffer, const size_t x_offset, const size_t x_inc,
                          float* y_buffer, const size_t y_offset, const size_t y_inc,
                          const float cos,
                          const float sin) {
  auto status = cublasSrot(handle, static_cast<int>(n),
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &y_buffer[y_offset], static_cast<int>(y_inc),
                           &cos,
                           &sin);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXrot(cublasHandle_t handle, const size_t n,
                          double* x_buffer, const size_t x_offset, const size_t x_inc,
                          double* y_buffer, const size_t y_offset, const size_t y_inc,
                          const double cos,
                          const double sin) {
  auto status = cublasDrot(handle, static_cast<int>(n),
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &y_buffer[y_offset], static_cast<int>(y_inc),
                           &cos,
                           &sin);
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for SROTM/DROTM
template <typename T>
cublasStatus_t cublasXrotm(cublasHandle_t handle, const size_t n,
                           T* x_buffer, const size_t x_offset, const size_t x_inc,
                           T* y_buffer, const size_t y_offset, const size_t y_inc,
                           T* sparam_buffer, const size_t sparam_offset);
template <>
cublasStatus_t cublasXrotm<float>(cublasHandle_t handle, const size_t n,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc,
                                  float* y_buffer, const size_t y_offset, const size_t y_inc,
                                  float* sparam_buffer, const size_t sparam_offset) {
  auto status = cublasSrotm(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc),
                            &sparam_buffer[sparam_offset]);
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXrotm<double>(cublasHandle_t handle, const size_t n,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc,
                                   double* y_buffer, const size_t y_offset, const size_t y_inc,
                                   double* sparam_buffer, const size_t sparam_offset) {
  auto status = cublasDrotm(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc),
                            &sparam_buffer[sparam_offset]);
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for SSWAP/DSWAP/CSWAP/ZSWAP
template <typename T>
cublasStatus_t cublasXswap(cublasHandle_t handle, const size_t n,
                           T* x_buffer, const size_t x_offset, const size_t x_inc,
                           T* y_buffer, const size_t y_offset, const size_t y_inc);
template <>
cublasStatus_t cublasXswap<float>(cublasHandle_t handle, const size_t n,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc,
                                  float* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasSswap(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXswap<double>(cublasHandle_t handle, const size_t n,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc,
                                   double* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasDswap(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXswap<float2>(cublasHandle_t handle, const size_t n,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc,
                                   float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasCswap(handle, static_cast<int>(n),
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXswap<double2>(cublasHandle_t handle, const size_t n,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc,
                                    double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasZswap(handle, static_cast<int>(n),
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXswap<half>(cublasHandle_t handle, const size_t n,
                                 half* x_buffer, const size_t x_offset, const size_t x_inc,
                                 half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSCAL/DSCAL/CSCAL/ZSCAL
cublasStatus_t cublasXscal(cublasHandle_t handle, const size_t n,
                           const float alpha,
                           float* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasSscal(handle, static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXscal(cublasHandle_t handle, const size_t n,
                           const double alpha,
                           double* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasDscal(handle, static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXscal(cublasHandle_t handle, const size_t n,
                           const float2 alpha,
                           float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasCscal(handle, static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXscal(cublasHandle_t handle, const size_t n,
                           const double2 alpha,
                           double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasZscal(handle, static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXscal(cublasHandle_t handle, const size_t n,
                           const half alpha,
                           half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SCOPY/DCOPY/CCOPY/ZCOPY
template <typename T>
cublasStatus_t cublasXcopy(cublasHandle_t handle, const size_t n,
                           const T* x_buffer, const size_t x_offset, const size_t x_inc,
                           T* y_buffer, const size_t y_offset, const size_t y_inc);
template <>
cublasStatus_t cublasXcopy<float>(cublasHandle_t handle, const size_t n,
                                  const float* x_buffer, const size_t x_offset, const size_t x_inc,
                                  float* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasScopy(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXcopy<double>(cublasHandle_t handle, const size_t n,
                                   const double* x_buffer, const size_t x_offset, const size_t x_inc,
                                   double* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasDcopy(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXcopy<float2>(cublasHandle_t handle, const size_t n,
                                   const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                                   float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasCcopy(handle, static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXcopy<double2>(cublasHandle_t handle, const size_t n,
                                    const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                                    double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasZcopy(handle, static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXcopy<half>(cublasHandle_t handle, const size_t n,
                                 const half* x_buffer, const size_t x_offset, const size_t x_inc,
                                 half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SAXPY/DAXPY/CAXPY/ZAXPY
cublasStatus_t cublasXaxpy(cublasHandle_t handle, const size_t n,
                           const float alpha,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           float* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasSaxpy(handle, static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXaxpy(cublasHandle_t handle, const size_t n,
                           const double alpha,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           double* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasDaxpy(handle, static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXaxpy(cublasHandle_t handle, const size_t n,
                           const float2 alpha,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasCaxpy(handle, static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXaxpy(cublasHandle_t handle, const size_t n,
                           const double2 alpha,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasZaxpy(handle, static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXaxpy(cublasHandle_t handle, const size_t n,
                           const half alpha,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SDOT/DDOT
template <typename T>
cublasStatus_t cublasXdot(cublasHandle_t handle, const size_t n,
                          T* dot_buffer, const size_t dot_offset,
                          const T* x_buffer, const size_t x_offset, const size_t x_inc,
                          const T* y_buffer, const size_t y_offset, const size_t y_inc);
template <>
cublasStatus_t cublasXdot<float>(cublasHandle_t handle, const size_t n,
                                 float* dot_buffer, const size_t dot_offset,
                                 const float* x_buffer, const size_t x_offset, const size_t x_inc,
                                 const float* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasSdot(handle, static_cast<int>(n),
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &y_buffer[y_offset], static_cast<int>(y_inc),
                           &dot_buffer[dot_offset]);
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXdot<double>(cublasHandle_t handle, const size_t n,
                                  double* dot_buffer, const size_t dot_offset,
                                  const double* x_buffer, const size_t x_offset, const size_t x_inc,
                                  const double* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasDdot(handle, static_cast<int>(n),
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &y_buffer[y_offset], static_cast<int>(y_inc),
                           &dot_buffer[dot_offset]);
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXdot<half>(cublasHandle_t handle, const size_t n,
                                half* dot_buffer, const size_t dot_offset,
                                const half* x_buffer, const size_t x_offset, const size_t x_inc,
                                const half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for CDOTU/ZDOTU
template <typename T>
cublasStatus_t cublasXdotu(cublasHandle_t handle, const size_t n,
                           T* dot_buffer, const size_t dot_offset,
                           const T* x_buffer, const size_t x_offset, const size_t x_inc,
                           const T* y_buffer, const size_t y_offset, const size_t y_inc);
template <>
cublasStatus_t cublasXdotu<float2>(cublasHandle_t handle, const size_t n,
                                   float2* dot_buffer, const size_t dot_offset,
                                   const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                                   const float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasCdotu(handle, static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuComplex*>(&dot_buffer[dot_offset]));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXdotu<double2>(cublasHandle_t handle, const size_t n,
                                    double2* dot_buffer, const size_t dot_offset,
                                    const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                                    const double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasZdotu(handle, static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuDoubleComplex*>(&dot_buffer[dot_offset]));
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for CDOTC/ZDOTC
template <typename T>
cublasStatus_t cublasXdotc(cublasHandle_t handle, const size_t n,
                           T* dot_buffer, const size_t dot_offset,
                           const T* x_buffer, const size_t x_offset, const size_t x_inc,
                           const T* y_buffer, const size_t y_offset, const size_t y_inc);
template <>
cublasStatus_t cublasXdotc<float2>(cublasHandle_t handle, const size_t n,
                                   float2* dot_buffer, const size_t dot_offset,
                                   const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                                   const float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasCdotc(handle, static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuComplex*>(&dot_buffer[dot_offset]));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXdotc<double2>(cublasHandle_t handle, const size_t n,
                                    double2* dot_buffer, const size_t dot_offset,
                                    const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                                    const double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  auto status = cublasZdotc(handle, static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuDoubleComplex*>(&dot_buffer[dot_offset]));
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for SNRM2/DNRM2/ScNRM2/DzNRM2
template <typename T>
cublasStatus_t cublasXnrm2(cublasHandle_t handle, const size_t n,
                           T* nrm2_buffer, const size_t nrm2_offset,
                           const T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXnrm2<float>(cublasHandle_t handle, const size_t n,
                                  float* nrm2_buffer, const size_t nrm2_offset,
                                  const float* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasSnrm2(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &nrm2_buffer[nrm2_offset]);
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXnrm2<double>(cublasHandle_t handle, const size_t n,
                                   double* nrm2_buffer, const size_t nrm2_offset,
                                   const double* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasDnrm2(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &nrm2_buffer[nrm2_offset]);
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXnrm2<float2>(cublasHandle_t handle, const size_t n,
                                   float2* nrm2_buffer, const size_t nrm2_offset,
                                   const float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasScnrm2(handle, static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<float*>(&nrm2_buffer[nrm2_offset]));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXnrm2<double2>(cublasHandle_t handle, const size_t n,
                                    double2* nrm2_buffer, const size_t nrm2_offset,
                                    const double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasDznrm2(handle, static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<double*>(&nrm2_buffer[nrm2_offset]));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXnrm2<half>(cublasHandle_t handle, const size_t n,
                                 half* nrm2_buffer, const size_t nrm2_offset,
                                 const half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SASUM/DASUM/ScASUM/DzASUM
template <typename T>
cublasStatus_t cublasXasum(cublasHandle_t handle, const size_t n,
                           T* asum_buffer, const size_t asum_offset,
                           const T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXasum<float>(cublasHandle_t handle, const size_t n,
                                  float* asum_buffer, const size_t asum_offset,
                                  const float* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasSasum(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &asum_buffer[asum_offset]);
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXasum<double>(cublasHandle_t handle, const size_t n,
                                   double* asum_buffer, const size_t asum_offset,
                                   const double* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasDasum(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &asum_buffer[asum_offset]);
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXasum<float2>(cublasHandle_t handle, const size_t n,
                                   float2* asum_buffer, const size_t asum_offset,
                                   const float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasScasum(handle, static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<float*>(&asum_buffer[asum_offset]));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXasum<double2>(cublasHandle_t handle, const size_t n,
                                    double2* asum_buffer, const size_t asum_offset,
                                    const double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasDzasum(handle, static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<double*>(&asum_buffer[asum_offset]));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXasum<half>(cublasHandle_t handle, const size_t n,
                                 half* asum_buffer, const size_t asum_offset,
                                 const half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
template <typename T>
cublasStatus_t cublasXamax(cublasHandle_t handle, const size_t n,
                           T* imax_buffer, const size_t imax_offset,
                           const T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXamax<float>(cublasHandle_t handle, const size_t n,
                                  float* imax_buffer, const size_t imax_offset,
                                  const float* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasIsamax(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            reinterpret_cast<int*>(&imax_buffer[imax_offset]));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXamax<double>(cublasHandle_t handle, const size_t n,
                                   double* imax_buffer, const size_t imax_offset,
                                   const double* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasIdamax(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            reinterpret_cast<int*>(&imax_buffer[imax_offset]));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXamax<float2>(cublasHandle_t handle, const size_t n,
                                   float2* imax_buffer, const size_t imax_offset,
                                   const float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasIcamax(handle, static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<int*>(&imax_buffer[imax_offset]));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXamax<double2>(cublasHandle_t handle, const size_t n,
                                    double2* imax_buffer, const size_t imax_offset,
                                    const double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  auto status = cublasIzamax(handle, static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<int*>(&imax_buffer[imax_offset]));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXamax<half>(cublasHandle_t handle, const size_t n,
                                 half* imax_buffer, const size_t imax_offset,
                                 const half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// Forwards the cuBLAS calls for SGEMV/DGEMV/CGEMV/ZGEMV
cublasStatus_t cublasXgemv(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float beta,
                           float* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSgemv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgemv(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double beta,
                           double* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDgemv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgemv(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n,
                           const float2 alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float2 beta,
                           float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasCgemv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgemv(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n,
                           const double2 alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double2 beta,
                           double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuDoubleComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasZgemv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgemv(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           const half beta,
                           half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SGBMV/DGBMV/CGBMV/ZGBMV
cublasStatus_t cublasXgbmv(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n, const size_t kl, const size_t ku,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float beta,
                           float* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSgbmv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(kl), static_cast<int>(ku),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgbmv(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n, const size_t kl, const size_t ku,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double beta,
                           double* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDgbmv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(kl), static_cast<int>(ku),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgbmv(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n, const size_t kl, const size_t ku,
                           const float2 alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float2 beta,
                           float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasCgbmv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(kl), static_cast<int>(ku),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgbmv(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n, const size_t kl, const size_t ku,
                           const double2 alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double2 beta,
                           double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuDoubleComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasZgbmv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(kl), static_cast<int>(ku),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgbmv(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n, const size_t kl, const size_t ku,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           const half beta,
                           half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for CHEMV/ZHEMV
cublasStatus_t cublasXhemv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float2 alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float2 beta,
                           float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasChemv(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXhemv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double2 alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double2 beta,
                           double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuDoubleComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasZhemv(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for CHBMV/ZHBMV
cublasStatus_t cublasXhbmv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n, const size_t k,
                           const float2 alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float2 beta,
                           float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasChbmv(handle, triangle,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXhbmv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n, const size_t k,
                           const double2 alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double2 beta,
                           double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuDoubleComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasZhbmv(handle, triangle,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for CHPMV/ZHPMV
cublasStatus_t cublasXhpmv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float2 alpha,
                           const float2* ap_buffer, const size_t ap_offset,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float2 beta,
                           float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasChpmv(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&ap_buffer[ap_offset]),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXhpmv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double2 alpha,
                           const double2* ap_buffer, const size_t ap_offset,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double2 beta,
                           double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuDoubleComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasZhpmv(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&ap_buffer[ap_offset]),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for SSYMV/DSYMV
cublasStatus_t cublasXsymv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float beta,
                           float* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSsymv(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsymv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double beta,
                           double* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDsymv(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsymv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           const half beta,
                           half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSBMV/DSBMV
cublasStatus_t cublasXsbmv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n, const size_t k,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float beta,
                           float* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSsbmv(handle, triangle,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsbmv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n, const size_t k,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double beta,
                           double* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDsbmv(handle, triangle,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsbmv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n, const size_t k,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           const half beta,
                           half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSPMV/DSPMV
cublasStatus_t cublasXspmv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float alpha,
                           const float* ap_buffer, const size_t ap_offset,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float beta,
                           float* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSspmv(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &ap_buffer[ap_offset],
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXspmv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double alpha,
                           const double* ap_buffer, const size_t ap_offset,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double beta,
                           double* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDspmv(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &ap_buffer[ap_offset],
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXspmv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const half alpha,
                           const half* ap_buffer, const size_t ap_offset,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           const half beta,
                           half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for STRMV/DTRMV/CTRMV/ZTRMV
template <typename T>
cublasStatus_t cublasXtrmv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t n,
                           const T* a_buffer, const size_t a_offset, const size_t a_ld,
                           T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXtrmv<float>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                  const size_t n,
                                  const float* a_buffer, const size_t a_offset, const size_t a_ld,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasStrmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtrmv<double>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const double* a_buffer, const size_t a_offset, const size_t a_ld,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDtrmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtrmv<float2>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasCtrmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtrmv<double2>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                    const size_t n,
                                    const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasZtrmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtrmv<half>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                 const size_t n,
                                 const half* a_buffer, const size_t a_offset, const size_t a_ld,
                                 half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for STBMV/DTBMV/CTBMV/ZTBMV
template <typename T>
cublasStatus_t cublasXtbmv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t n, const size_t k,
                           const T* a_buffer, const size_t a_offset, const size_t a_ld,
                           T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXtbmv<float>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                  const size_t n, const size_t k,
                                  const float* a_buffer, const size_t a_offset, const size_t a_ld,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasStbmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtbmv<double>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n, const size_t k,
                                   const double* a_buffer, const size_t a_offset, const size_t a_ld,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDtbmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtbmv<float2>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n, const size_t k,
                                   const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasCtbmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtbmv<double2>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                    const size_t n, const size_t k,
                                    const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasZtbmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtbmv<half>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                 const size_t n, const size_t k,
                                 const half* a_buffer, const size_t a_offset, const size_t a_ld,
                                 half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for STPMV/DTPMV/CTPMV/ZTPMV
template <typename T>
cublasStatus_t cublasXtpmv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t n,
                           const T* ap_buffer, const size_t ap_offset,
                           T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXtpmv<float>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                  const size_t n,
                                  const float* ap_buffer, const size_t ap_offset,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasStpmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &ap_buffer[ap_offset],
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtpmv<double>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const double* ap_buffer, const size_t ap_offset,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDtpmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &ap_buffer[ap_offset],
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtpmv<float2>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const float2* ap_buffer, const size_t ap_offset,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasCtpmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&ap_buffer[ap_offset]),
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtpmv<double2>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                    const size_t n,
                                    const double2* ap_buffer, const size_t ap_offset,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasZtpmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&ap_buffer[ap_offset]),
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtpmv<half>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                 const size_t n,
                                 const half* ap_buffer, const size_t ap_offset,
                                 half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for STRSV/DTRSV/CTRSV/ZTRSV
template <typename T>
cublasStatus_t cublasXtrsv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t n,
                           const T* a_buffer, const size_t a_offset, const size_t a_ld,
                           T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXtrsv<float>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                  const size_t n,
                                  const float* a_buffer, const size_t a_offset, const size_t a_ld,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasStrsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtrsv<double>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const double* a_buffer, const size_t a_offset, const size_t a_ld,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDtrsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtrsv<float2>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasCtrsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtrsv<double2>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                    const size_t n,
                                    const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasZtrsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for STBSV/DTBSV/CTBSV/ZTBSV
template <typename T>
cublasStatus_t cublasXtbsv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t n, const size_t k,
                           const T* a_buffer, const size_t a_offset, const size_t a_ld,
                           T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXtbsv<float>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                  const size_t n, const size_t k,
                                  const float* a_buffer, const size_t a_offset, const size_t a_ld,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasStbsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtbsv<double>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n, const size_t k,
                                   const double* a_buffer, const size_t a_offset, const size_t a_ld,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDtbsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtbsv<float2>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n, const size_t k,
                                   const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasCtbsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtbsv<double2>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                    const size_t n, const size_t k,
                                    const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasZtbsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for STPSV/DTPSV/CTPSV/ZTPSV
template <typename T>
cublasStatus_t cublasXtpsv(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t n,
                           const T* ap_buffer, const size_t ap_offset,
                           T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXtpsv<float>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                  const size_t n,
                                  const float* ap_buffer, const size_t ap_offset,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasStpsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &ap_buffer[ap_offset],
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtpsv<double>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const double* ap_buffer, const size_t ap_offset,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDtpsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &ap_buffer[ap_offset],
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtpsv<float2>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const float2* ap_buffer, const size_t ap_offset,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasCtpsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&ap_buffer[ap_offset]),
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}
template <>
cublasStatus_t cublasXtpsv<double2>(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                    const size_t n,
                                    const double2* ap_buffer, const size_t ap_offset,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasZtpsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&ap_buffer[ap_offset]),
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for SGER/DGER
cublasStatus_t cublasXger(cublasHandle_t handle, const Layout layout,
                          const size_t m, const size_t n,
                          const float alpha,
                          const float* x_buffer, const size_t x_offset, const size_t x_inc,
                          const float* y_buffer, const size_t y_offset, const size_t y_inc,
                          float* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSger(handle, static_cast<int>(m), static_cast<int>(n),
                           &alpha,
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &y_buffer[y_offset], static_cast<int>(y_inc),
                           &a_buffer[a_offset], a_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXger(cublasHandle_t handle, const Layout layout,
                          const size_t m, const size_t n,
                          const double alpha,
                          const double* x_buffer, const size_t x_offset, const size_t x_inc,
                          const double* y_buffer, const size_t y_offset, const size_t y_inc,
                          double* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDger(handle, static_cast<int>(m), static_cast<int>(n),
                           &alpha,
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &y_buffer[y_offset], static_cast<int>(y_inc),
                           &a_buffer[a_offset], a_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXger(cublasHandle_t handle, const Layout layout,
                          const size_t m, const size_t n,
                          const half alpha,
                          const half* x_buffer, const size_t x_offset, const size_t x_inc,
                          const half* y_buffer, const size_t y_offset, const size_t y_inc,
                          half* a_buffer, const size_t a_offset, const size_t a_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for CGERU/ZGERU
cublasStatus_t cublasXgeru(cublasHandle_t handle, const Layout layout,
                           const size_t m, const size_t n,
                           const float2 alpha,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float2* y_buffer, const size_t y_offset, const size_t y_inc,
                           float2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasCgeru(handle, static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuComplex*>(&a_buffer[a_offset]), a_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgeru(cublasHandle_t handle, const Layout layout,
                           const size_t m, const size_t n,
                           const double2 alpha,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double2* y_buffer, const size_t y_offset, const size_t y_inc,
                           double2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasZgeru(handle, static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuDoubleComplex*>(&a_buffer[a_offset]), a_ld);
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for CGERC/ZGERC
cublasStatus_t cublasXgerc(cublasHandle_t handle, const Layout layout,
                           const size_t m, const size_t n,
                           const float2 alpha,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float2* y_buffer, const size_t y_offset, const size_t y_inc,
                           float2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasCgerc(handle, static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuComplex*>(&a_buffer[a_offset]), a_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgerc(cublasHandle_t handle, const Layout layout,
                           const size_t m, const size_t n,
                           const double2 alpha,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double2* y_buffer, const size_t y_offset, const size_t y_inc,
                           double2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasZgerc(handle, static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuDoubleComplex*>(&a_buffer[a_offset]), a_ld);
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for CHER/ZHER
cublasStatus_t cublasXher(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const float alpha,
                          const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                          float2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasCher(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                           reinterpret_cast<cuComplex*>(&a_buffer[a_offset]), a_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXher(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const double alpha,
                          const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                          double2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasZher(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                           reinterpret_cast<cuDoubleComplex*>(&a_buffer[a_offset]), a_ld);
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for CHPR/ZHPR
cublasStatus_t cublasXhpr(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const float alpha,
                          const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                          float2* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasChpr(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                           reinterpret_cast<cuComplex*>(&ap_buffer[ap_offset]));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXhpr(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const double alpha,
                          const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                          double2* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasZhpr(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                           reinterpret_cast<cuDoubleComplex*>(&ap_buffer[ap_offset]));
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for CHER2/ZHER2
cublasStatus_t cublasXher2(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float2 alpha,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float2* y_buffer, const size_t y_offset, const size_t y_inc,
                           float2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasCher2(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuComplex*>(&a_buffer[a_offset]), a_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXher2(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double2 alpha,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double2* y_buffer, const size_t y_offset, const size_t y_inc,
                           double2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasZher2(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuDoubleComplex*>(&a_buffer[a_offset]), a_ld);
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for CHPR2/ZHPR2
cublasStatus_t cublasXhpr2(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float2 alpha,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float2* y_buffer, const size_t y_offset, const size_t y_inc,
                           float2* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasChpr2(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuComplex*>(&ap_buffer[ap_offset]));
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXhpr2(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double2 alpha,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double2* y_buffer, const size_t y_offset, const size_t y_inc,
                           double2* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasZhpr2(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuDoubleComplex*>(&ap_buffer[ap_offset]));
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for SSYR/DSYR
cublasStatus_t cublasXsyr(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const float alpha,
                          const float* x_buffer, const size_t x_offset, const size_t x_inc,
                          float* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSsyr(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &a_buffer[a_offset], a_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsyr(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const double alpha,
                          const double* x_buffer, const size_t x_offset, const size_t x_inc,
                          double* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDsyr(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &a_buffer[a_offset], a_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsyr(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const half alpha,
                          const half* x_buffer, const size_t x_offset, const size_t x_inc,
                          half* a_buffer, const size_t a_offset, const size_t a_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSPR/DSPR
cublasStatus_t cublasXspr(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const float alpha,
                          const float* x_buffer, const size_t x_offset, const size_t x_inc,
                          float* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSspr(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &ap_buffer[ap_offset]);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXspr(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const double alpha,
                          const double* x_buffer, const size_t x_offset, const size_t x_inc,
                          double* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDspr(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &ap_buffer[ap_offset]);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXspr(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const half alpha,
                          const half* x_buffer, const size_t x_offset, const size_t x_inc,
                          half* ap_buffer, const size_t ap_offset) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSYR2/DSYR2
cublasStatus_t cublasXsyr2(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float alpha,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float* y_buffer, const size_t y_offset, const size_t y_inc,
                           float* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSsyr2(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc),
                            &a_buffer[a_offset], a_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsyr2(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double alpha,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double* y_buffer, const size_t y_offset, const size_t y_inc,
                           double* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDsyr2(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc),
                            &a_buffer[a_offset], a_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsyr2(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const half alpha,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           const half* y_buffer, const size_t y_offset, const size_t y_inc,
                           half* a_buffer, const size_t a_offset, const size_t a_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSPR2/DSPR2
cublasStatus_t cublasXspr2(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float alpha,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float* y_buffer, const size_t y_offset, const size_t y_inc,
                           float* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSspr2(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc),
                            &ap_buffer[ap_offset]);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXspr2(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double alpha,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double* y_buffer, const size_t y_offset, const size_t y_inc,
                           double* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDspr2(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc),
                            &ap_buffer[ap_offset]);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXspr2(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const half alpha,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           const half* y_buffer, const size_t y_offset, const size_t y_inc,
                           half* ap_buffer, const size_t ap_offset) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// Forwards the cuBLAS calls for SGEMM/DGEMM/CGEMM/ZGEMM
cublasStatus_t cublasXgemm(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose, const cublasOperation_t b_transpose,
                           const size_t m, const size_t n, const size_t k,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float* b_buffer, const size_t b_offset, const size_t b_ld,
                           const float beta,
                           float* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSgemm(handle, a_transpose, b_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld,
                            &beta,
                            &c_buffer[c_offset], c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgemm(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose, const cublasOperation_t b_transpose,
                           const size_t m, const size_t n, const size_t k,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double* b_buffer, const size_t b_offset, const size_t b_ld,
                           const double beta,
                           double* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDgemm(handle, a_transpose, b_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld,
                            &beta,
                            &c_buffer[c_offset], c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgemm(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose, const cublasOperation_t b_transpose,
                           const size_t m, const size_t n, const size_t k,
                           const float2 alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float2* b_buffer, const size_t b_offset, const size_t b_ld,
                           const float2 beta,
                           float2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasCgemm(handle, a_transpose, b_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&b_buffer[b_offset]), b_ld,
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgemm(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose, const cublasOperation_t b_transpose,
                           const size_t m, const size_t n, const size_t k,
                           const double2 alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double2* b_buffer, const size_t b_offset, const size_t b_ld,
                           const double2 beta,
                           double2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuDoubleComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasZgemm(handle, a_transpose, b_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&b_buffer[b_offset]), b_ld,
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXgemm(cublasHandle_t handle, const Layout layout, const cublasOperation_t a_transpose, const cublasOperation_t b_transpose,
                           const size_t m, const size_t n, const size_t k,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half* b_buffer, const size_t b_offset, const size_t b_ld,
                           const half beta,
                           half* c_buffer, const size_t c_offset, const size_t c_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSYMM/DSYMM/CSYMM/ZSYMM
cublasStatus_t cublasXsymm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
                           const size_t m, const size_t n,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float* b_buffer, const size_t b_offset, const size_t b_ld,
                           const float beta,
                           float* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSsymm(handle, side, triangle,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld,
                            &beta,
                            &c_buffer[c_offset], c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsymm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
                           const size_t m, const size_t n,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double* b_buffer, const size_t b_offset, const size_t b_ld,
                           const double beta,
                           double* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDsymm(handle, side, triangle,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld,
                            &beta,
                            &c_buffer[c_offset], c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsymm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
                           const size_t m, const size_t n,
                           const float2 alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float2* b_buffer, const size_t b_offset, const size_t b_ld,
                           const float2 beta,
                           float2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasCsymm(handle, side, triangle,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&b_buffer[b_offset]), b_ld,
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsymm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
                           const size_t m, const size_t n,
                           const double2 alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double2* b_buffer, const size_t b_offset, const size_t b_ld,
                           const double2 beta,
                           double2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuDoubleComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasZsymm(handle, side, triangle,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&b_buffer[b_offset]), b_ld,
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsymm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
                           const size_t m, const size_t n,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half* b_buffer, const size_t b_offset, const size_t b_ld,
                           const half beta,
                           half* c_buffer, const size_t c_offset, const size_t c_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for CHEMM/ZHEMM
cublasStatus_t cublasXhemm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
                           const size_t m, const size_t n,
                           const float2 alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float2* b_buffer, const size_t b_offset, const size_t b_ld,
                           const float2 beta,
                           float2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasChemm(handle, side, triangle,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&b_buffer[b_offset]), b_ld,
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXhemm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
                           const size_t m, const size_t n,
                           const double2 alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double2* b_buffer, const size_t b_offset, const size_t b_ld,
                           const double2 beta,
                           double2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuDoubleComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasZhemm(handle, side, triangle,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&b_buffer[b_offset]), b_ld,
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for SSYRK/DSYRK/CSYRK/ZSYRK
cublasStatus_t cublasXsyrk(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
                           const size_t n, const size_t k,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float beta,
                           float* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSsyrk(handle, triangle, a_transpose,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &beta,
                            &c_buffer[c_offset], c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsyrk(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
                           const size_t n, const size_t k,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double beta,
                           double* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDsyrk(handle, triangle, a_transpose,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &beta,
                            &c_buffer[c_offset], c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsyrk(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
                           const size_t n, const size_t k,
                           const float2 alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float2 beta,
                           float2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasCsyrk(handle, triangle, a_transpose,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsyrk(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
                           const size_t n, const size_t k,
                           const double2 alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double2 beta,
                           double2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuDoubleComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasZsyrk(handle, triangle, a_transpose,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsyrk(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
                           const size_t n, const size_t k,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half beta,
                           half* c_buffer, const size_t c_offset, const size_t c_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for CHERK/ZHERK
cublasStatus_t cublasXherk(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
                           const size_t n, const size_t k,
                           const float alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float beta,
                           float2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasCherk(handle, triangle, a_transpose,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            &beta,
                            reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXherk(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
                           const size_t n, const size_t k,
                           const double alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double beta,
                           double2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasZherk(handle, triangle, a_transpose,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            &beta,
                            reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for SSYR2K/DSYR2K/CSYR2K/ZSYR2K
cublasStatus_t cublasXsyr2k(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
                            const size_t n, const size_t k,
                            const float alpha,
                            const float* a_buffer, const size_t a_offset, const size_t a_ld,
                            const float* b_buffer, const size_t b_offset, const size_t b_ld,
                            const float beta,
                            float* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasSsyr2k(handle, triangle, ab_transpose,
                             static_cast<int>(n), static_cast<int>(k),
                             &alpha,
                             &a_buffer[a_offset], a_ld,
                             &b_buffer[b_offset], b_ld,
                             &beta,
                             &c_buffer[c_offset], c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsyr2k(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
                            const size_t n, const size_t k,
                            const double alpha,
                            const double* a_buffer, const size_t a_offset, const size_t a_ld,
                            const double* b_buffer, const size_t b_offset, const size_t b_ld,
                            const double beta,
                            double* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDsyr2k(handle, triangle, ab_transpose,
                             static_cast<int>(n), static_cast<int>(k),
                             &alpha,
                             &a_buffer[a_offset], a_ld,
                             &b_buffer[b_offset], b_ld,
                             &beta,
                             &c_buffer[c_offset], c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsyr2k(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
                            const size_t n, const size_t k,
                            const float2 alpha,
                            const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                            const float2* b_buffer, const size_t b_offset, const size_t b_ld,
                            const float2 beta,
                            float2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasCsyr2k(handle, triangle, ab_transpose,
                             static_cast<int>(n), static_cast<int>(k),
                             &alpha_cuda,
                             reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                             reinterpret_cast<const cuComplex*>(&b_buffer[b_offset]), b_ld,
                             &beta_cuda,
                             reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsyr2k(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
                            const size_t n, const size_t k,
                            const double2 alpha,
                            const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                            const double2* b_buffer, const size_t b_offset, const size_t b_ld,
                            const double2 beta,
                            double2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cuDoubleComplex beta_cuda;
  beta_cuda.x = beta.real();
  beta_cuda.y = beta.imag();
  auto status = cublasZsyr2k(handle, triangle, ab_transpose,
                             static_cast<int>(n), static_cast<int>(k),
                             &alpha_cuda,
                             reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                             reinterpret_cast<const cuDoubleComplex*>(&b_buffer[b_offset]), b_ld,
                             &beta_cuda,
                             reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXsyr2k(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
                            const size_t n, const size_t k,
                            const half alpha,
                            const half* a_buffer, const size_t a_offset, const size_t a_ld,
                            const half* b_buffer, const size_t b_offset, const size_t b_ld,
                            const half beta,
                            half* c_buffer, const size_t c_offset, const size_t c_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for CHER2K/ZHER2K
cublasStatus_t cublasXher2k(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
                            const size_t n, const size_t k,
                            const float2 alpha,
                            const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                            const float2* b_buffer, const size_t b_offset, const size_t b_ld,
                            const float beta,
                            float2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasCher2k(handle, triangle, ab_transpose,
                             static_cast<int>(n), static_cast<int>(k),
                             &alpha_cuda,
                             reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                             reinterpret_cast<const cuComplex*>(&b_buffer[b_offset]), b_ld,
                             &beta,
                             reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXher2k(cublasHandle_t handle, const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
                            const size_t n, const size_t k,
                            const double2 alpha,
                            const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                            const double2* b_buffer, const size_t b_offset, const size_t b_ld,
                            const double beta,
                            double2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasZher2k(handle, triangle, ab_transpose,
                             static_cast<int>(n), static_cast<int>(k),
                             &alpha_cuda,
                             reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                             reinterpret_cast<const cuDoubleComplex*>(&b_buffer[b_offset]), b_ld,
                             &beta,
                             reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cudaDeviceSynchronize();
  return status;
}

// Forwards the cuBLAS calls for STRMM/DTRMM/CTRMM/ZTRMM
cublasStatus_t cublasXtrmm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           float* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasStrmm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXtrmm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           double* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDtrmm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXtrmm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const float2 alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           float2* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasCtrmm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuComplex*>(&b_buffer[b_offset]), b_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXtrmm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const double2 alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           double2* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasZtrmm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuDoubleComplex*>(&b_buffer[b_offset]), b_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXtrmm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           half* b_buffer, const size_t b_offset, const size_t b_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for STRSM/DTRSM/CTRSM/ZTRSM
cublasStatus_t cublasXtrsm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           float* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasStrsm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXtrsm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           double* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  auto status = cublasDtrsm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXtrsm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const float2 alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           float2* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasCtrsm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuComplex*>(&b_buffer[b_offset]), b_ld);
  cudaDeviceSynchronize();
  return status;
}
cublasStatus_t cublasXtrsm(cublasHandle_t handle, const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const double2 alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           double2* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  auto status = cublasZtrsm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuDoubleComplex*>(&b_buffer[b_offset]), b_ld);
  cudaDeviceSynchronize();
  return status;
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_WRAPPER_CUBLAS_H_
#endif
