
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
cublasStatus_t cublasXrotg(T* sa_buffer, const size_t sa_offset,
                           T* sb_buffer, const size_t sb_offset,
                           T* sc_buffer, const size_t sc_offset,
                           T* ss_buffer, const size_t ss_offset);
template <>
cublasStatus_t cublasXrotg<float>(float* sa_buffer, const size_t sa_offset,
                                  float* sb_buffer, const size_t sb_offset,
                                  float* sc_buffer, const size_t sc_offset,
                                  float* ss_buffer, const size_t ss_offset) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSrotg(handle, &sa_buffer[sa_offset],
                            &sb_buffer[sb_offset],
                            &sc_buffer[sc_offset],
                            &ss_buffer[ss_offset]);
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXrotg<double>(double* sa_buffer, const size_t sa_offset,
                                   double* sb_buffer, const size_t sb_offset,
                                   double* sc_buffer, const size_t sc_offset,
                                   double* ss_buffer, const size_t ss_offset) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDrotg(handle, &sa_buffer[sa_offset],
                            &sb_buffer[sb_offset],
                            &sc_buffer[sc_offset],
                            &ss_buffer[ss_offset]);
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for SROTMG/DROTMG
template <typename T>
cublasStatus_t cublasXrotmg(T* sd1_buffer, const size_t sd1_offset,
                            T* sd2_buffer, const size_t sd2_offset,
                            T* sx1_buffer, const size_t sx1_offset,
                            const T* sy1_buffer, const size_t sy1_offset,
                            T* sparam_buffer, const size_t sparam_offset);
template <>
cublasStatus_t cublasXrotmg<float>(float* sd1_buffer, const size_t sd1_offset,
                                   float* sd2_buffer, const size_t sd2_offset,
                                   float* sx1_buffer, const size_t sx1_offset,
                                   const float* sy1_buffer, const size_t sy1_offset,
                                   float* sparam_buffer, const size_t sparam_offset) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSrotmg(handle, &sd1_buffer[sd1_offset],
                             &sd2_buffer[sd2_offset],
                             &sx1_buffer[sx1_offset],
                             &sy1_buffer[sy1_offset],
                             &sparam_buffer[sparam_offset]);
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXrotmg<double>(double* sd1_buffer, const size_t sd1_offset,
                                    double* sd2_buffer, const size_t sd2_offset,
                                    double* sx1_buffer, const size_t sx1_offset,
                                    const double* sy1_buffer, const size_t sy1_offset,
                                    double* sparam_buffer, const size_t sparam_offset) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDrotmg(handle, &sd1_buffer[sd1_offset],
                             &sd2_buffer[sd2_offset],
                             &sx1_buffer[sx1_offset],
                             &sy1_buffer[sy1_offset],
                             &sparam_buffer[sparam_offset]);
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for SROT/DROT
cublasStatus_t cublasXrot(const size_t n,
                          float* x_buffer, const size_t x_offset, const size_t x_inc,
                          float* y_buffer, const size_t y_offset, const size_t y_inc,
                          const float cos,
                          const float sin) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSrot(handle, static_cast<int>(n),
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &y_buffer[y_offset], static_cast<int>(y_inc),
                           &cos,
                           &sin);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXrot(const size_t n,
                          double* x_buffer, const size_t x_offset, const size_t x_inc,
                          double* y_buffer, const size_t y_offset, const size_t y_inc,
                          const double cos,
                          const double sin) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDrot(handle, static_cast<int>(n),
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &y_buffer[y_offset], static_cast<int>(y_inc),
                           &cos,
                           &sin);
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for SROTM/DROTM
template <typename T>
cublasStatus_t cublasXrotm(const size_t n,
                           T* x_buffer, const size_t x_offset, const size_t x_inc,
                           T* y_buffer, const size_t y_offset, const size_t y_inc,
                           T* sparam_buffer, const size_t sparam_offset);
template <>
cublasStatus_t cublasXrotm<float>(const size_t n,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc,
                                  float* y_buffer, const size_t y_offset, const size_t y_inc,
                                  float* sparam_buffer, const size_t sparam_offset) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSrotm(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc),
                            &sparam_buffer[sparam_offset]);
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXrotm<double>(const size_t n,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc,
                                   double* y_buffer, const size_t y_offset, const size_t y_inc,
                                   double* sparam_buffer, const size_t sparam_offset) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDrotm(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc),
                            &sparam_buffer[sparam_offset]);
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for SSWAP/DSWAP/CSWAP/ZSWAP
template <typename T>
cublasStatus_t cublasXswap(const size_t n,
                           T* x_buffer, const size_t x_offset, const size_t x_inc,
                           T* y_buffer, const size_t y_offset, const size_t y_inc);
template <>
cublasStatus_t cublasXswap<float>(const size_t n,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc,
                                  float* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSswap(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXswap<double>(const size_t n,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc,
                                   double* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDswap(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXswap<float2>(const size_t n,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc,
                                   float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCswap(handle, static_cast<int>(n),
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXswap<double2>(const size_t n,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc,
                                    double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZswap(handle, static_cast<int>(n),
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXswap<half>(const size_t n,
                                 half* x_buffer, const size_t x_offset, const size_t x_inc,
                                 half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSCAL/DSCAL/CSCAL/ZSCAL
cublasStatus_t cublasXscal(const size_t n,
                           const float alpha,
                           float* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSscal(handle, static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXscal(const size_t n,
                           const double alpha,
                           double* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDscal(handle, static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXscal(const size_t n,
                           const float2 alpha,
                           float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCscal(handle, static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXscal(const size_t n,
                           const double2 alpha,
                           double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZscal(handle, static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXscal(const size_t n,
                           const half alpha,
                           half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SCOPY/DCOPY/CCOPY/ZCOPY
template <typename T>
cublasStatus_t cublasXcopy(const size_t n,
                           const T* x_buffer, const size_t x_offset, const size_t x_inc,
                           T* y_buffer, const size_t y_offset, const size_t y_inc);
template <>
cublasStatus_t cublasXcopy<float>(const size_t n,
                                  const float* x_buffer, const size_t x_offset, const size_t x_inc,
                                  float* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasScopy(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXcopy<double>(const size_t n,
                                   const double* x_buffer, const size_t x_offset, const size_t x_inc,
                                   double* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDcopy(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXcopy<float2>(const size_t n,
                                   const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                                   float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCcopy(handle, static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXcopy<double2>(const size_t n,
                                    const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                                    double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZcopy(handle, static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXcopy<half>(const size_t n,
                                 const half* x_buffer, const size_t x_offset, const size_t x_inc,
                                 half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SAXPY/DAXPY/CAXPY/ZAXPY
cublasStatus_t cublasXaxpy(const size_t n,
                           const float alpha,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           float* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSaxpy(handle, static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXaxpy(const size_t n,
                           const double alpha,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           double* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDaxpy(handle, static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXaxpy(const size_t n,
                           const float2 alpha,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCaxpy(handle, static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXaxpy(const size_t n,
                           const double2 alpha,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZaxpy(handle, static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXaxpy(const size_t n,
                           const half alpha,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SDOT/DDOT
template <typename T>
cublasStatus_t cublasXdot(const size_t n,
                          T* dot_buffer, const size_t dot_offset,
                          const T* x_buffer, const size_t x_offset, const size_t x_inc,
                          const T* y_buffer, const size_t y_offset, const size_t y_inc);
template <>
cublasStatus_t cublasXdot<float>(const size_t n,
                                 float* dot_buffer, const size_t dot_offset,
                                 const float* x_buffer, const size_t x_offset, const size_t x_inc,
                                 const float* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSdot(handle, static_cast<int>(n),
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &y_buffer[y_offset], static_cast<int>(y_inc),
                           &dot_buffer[dot_offset]);
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXdot<double>(const size_t n,
                                  double* dot_buffer, const size_t dot_offset,
                                  const double* x_buffer, const size_t x_offset, const size_t x_inc,
                                  const double* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDdot(handle, static_cast<int>(n),
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &y_buffer[y_offset], static_cast<int>(y_inc),
                           &dot_buffer[dot_offset]);
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXdot<half>(const size_t n,
                                half* dot_buffer, const size_t dot_offset,
                                const half* x_buffer, const size_t x_offset, const size_t x_inc,
                                const half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for CDOTU/ZDOTU
template <typename T>
cublasStatus_t cublasXdotu(const size_t n,
                           T* dot_buffer, const size_t dot_offset,
                           const T* x_buffer, const size_t x_offset, const size_t x_inc,
                           const T* y_buffer, const size_t y_offset, const size_t y_inc);
template <>
cublasStatus_t cublasXdotu<float2>(const size_t n,
                                   float2* dot_buffer, const size_t dot_offset,
                                   const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                                   const float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCdotu(handle, static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuComplex*>(&dot_buffer[dot_offset]));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXdotu<double2>(const size_t n,
                                    double2* dot_buffer, const size_t dot_offset,
                                    const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                                    const double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZdotu(handle, static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuDoubleComplex*>(&dot_buffer[dot_offset]));
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for CDOTC/ZDOTC
template <typename T>
cublasStatus_t cublasXdotc(const size_t n,
                           T* dot_buffer, const size_t dot_offset,
                           const T* x_buffer, const size_t x_offset, const size_t x_inc,
                           const T* y_buffer, const size_t y_offset, const size_t y_inc);
template <>
cublasStatus_t cublasXdotc<float2>(const size_t n,
                                   float2* dot_buffer, const size_t dot_offset,
                                   const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                                   const float2* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCdotc(handle, static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuComplex*>(&dot_buffer[dot_offset]));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXdotc<double2>(const size_t n,
                                    double2* dot_buffer, const size_t dot_offset,
                                    const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                                    const double2* y_buffer, const size_t y_offset, const size_t y_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZdotc(handle, static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuDoubleComplex*>(&dot_buffer[dot_offset]));
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for SNRM2/DNRM2/ScNRM2/DzNRM2
template <typename T>
cublasStatus_t cublasXnrm2(const size_t n,
                           T* nrm2_buffer, const size_t nrm2_offset,
                           const T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXnrm2<float>(const size_t n,
                                  float* nrm2_buffer, const size_t nrm2_offset,
                                  const float* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSnrm2(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &nrm2_buffer[nrm2_offset]);
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXnrm2<double>(const size_t n,
                                   double* nrm2_buffer, const size_t nrm2_offset,
                                   const double* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDnrm2(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &nrm2_buffer[nrm2_offset]);
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXnrm2<float2>(const size_t n,
                                   float2* nrm2_buffer, const size_t nrm2_offset,
                                   const float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasScnrm2(handle, static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<float*>(&nrm2_buffer[nrm2_offset]));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXnrm2<double2>(const size_t n,
                                    double2* nrm2_buffer, const size_t nrm2_offset,
                                    const double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDznrm2(handle, static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<double*>(&nrm2_buffer[nrm2_offset]));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXnrm2<half>(const size_t n,
                                 half* nrm2_buffer, const size_t nrm2_offset,
                                 const half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SASUM/DASUM/ScASUM/DzASUM
template <typename T>
cublasStatus_t cublasXasum(const size_t n,
                           T* asum_buffer, const size_t asum_offset,
                           const T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXasum<float>(const size_t n,
                                  float* asum_buffer, const size_t asum_offset,
                                  const float* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSasum(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &asum_buffer[asum_offset]);
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXasum<double>(const size_t n,
                                   double* asum_buffer, const size_t asum_offset,
                                   const double* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDasum(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &asum_buffer[asum_offset]);
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXasum<float2>(const size_t n,
                                   float2* asum_buffer, const size_t asum_offset,
                                   const float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasScasum(handle, static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<float*>(&asum_buffer[asum_offset]));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXasum<double2>(const size_t n,
                                    double2* asum_buffer, const size_t asum_offset,
                                    const double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDzasum(handle, static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<double*>(&asum_buffer[asum_offset]));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXasum<half>(const size_t n,
                                 half* asum_buffer, const size_t asum_offset,
                                 const half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
template <typename T>
cublasStatus_t cublasXamax(const size_t n,
                           T* imax_buffer, const size_t imax_offset,
                           const T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXamax<float>(const size_t n,
                                  float* imax_buffer, const size_t imax_offset,
                                  const float* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasIsamax(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            reinterpret_cast<int*>(&imax_buffer[imax_offset]));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXamax<double>(const size_t n,
                                   double* imax_buffer, const size_t imax_offset,
                                   const double* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasIdamax(handle, static_cast<int>(n),
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            reinterpret_cast<int*>(&imax_buffer[imax_offset]));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXamax<float2>(const size_t n,
                                   float2* imax_buffer, const size_t imax_offset,
                                   const float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasIcamax(handle, static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<int*>(&imax_buffer[imax_offset]));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXamax<double2>(const size_t n,
                                    double2* imax_buffer, const size_t imax_offset,
                                    const double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasIzamax(handle, static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<int*>(&imax_buffer[imax_offset]));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXamax<half>(const size_t n,
                                 half* imax_buffer, const size_t imax_offset,
                                 const half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// Forwards the cuBLAS calls for SGEMV/DGEMV/CGEMV/ZGEMV
cublasStatus_t cublasXgemv(const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float beta,
                           float* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSgemv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgemv(const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double beta,
                           double* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDgemv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgemv(const Layout layout, const cublasOperation_t a_transpose,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCgemv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgemv(const Layout layout, const cublasOperation_t a_transpose,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZgemv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgemv(const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           const half beta,
                           half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SGBMV/DGBMV/CGBMV/ZGBMV
cublasStatus_t cublasXgbmv(const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n, const size_t kl, const size_t ku,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float beta,
                           float* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSgbmv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(kl), static_cast<int>(ku),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgbmv(const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n, const size_t kl, const size_t ku,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double beta,
                           double* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDgbmv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(kl), static_cast<int>(ku),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgbmv(const Layout layout, const cublasOperation_t a_transpose,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCgbmv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(kl), static_cast<int>(ku),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgbmv(const Layout layout, const cublasOperation_t a_transpose,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZgbmv(handle, a_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(kl), static_cast<int>(ku),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgbmv(const Layout layout, const cublasOperation_t a_transpose,
                           const size_t m, const size_t n, const size_t kl, const size_t ku,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           const half beta,
                           half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for CHEMV/ZHEMV
cublasStatus_t cublasXhemv(const Layout layout, const cublasFillMode_t triangle,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasChemv(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXhemv(const Layout layout, const cublasFillMode_t triangle,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZhemv(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for CHBMV/ZHBMV
cublasStatus_t cublasXhbmv(const Layout layout, const cublasFillMode_t triangle,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasChbmv(handle, triangle,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXhbmv(const Layout layout, const cublasFillMode_t triangle,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZhbmv(handle, triangle,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for CHPMV/ZHPMV
cublasStatus_t cublasXhpmv(const Layout layout, const cublasFillMode_t triangle,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasChpmv(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&ap_buffer[ap_offset]),
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXhpmv(const Layout layout, const cublasFillMode_t triangle,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZhpmv(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&ap_buffer[ap_offset]),
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for SSYMV/DSYMV
cublasStatus_t cublasXsymv(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float beta,
                           float* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSsymv(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsymv(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double beta,
                           double* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDsymv(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsymv(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           const half beta,
                           half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSBMV/DSBMV
cublasStatus_t cublasXsbmv(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n, const size_t k,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float beta,
                           float* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSsbmv(handle, triangle,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsbmv(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n, const size_t k,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double beta,
                           double* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDsbmv(handle, triangle,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsbmv(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n, const size_t k,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           const half beta,
                           half* y_buffer, const size_t y_offset, const size_t y_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSPMV/DSPMV
cublasStatus_t cublasXspmv(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float alpha,
                           const float* ap_buffer, const size_t ap_offset,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float beta,
                           float* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSspmv(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &ap_buffer[ap_offset],
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXspmv(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double alpha,
                           const double* ap_buffer, const size_t ap_offset,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double beta,
                           double* y_buffer, const size_t y_offset, const size_t y_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDspmv(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &ap_buffer[ap_offset],
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &beta,
                            &y_buffer[y_offset], static_cast<int>(y_inc));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXspmv(const Layout layout, const cublasFillMode_t triangle,
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
cublasStatus_t cublasXtrmv(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t n,
                           const T* a_buffer, const size_t a_offset, const size_t a_ld,
                           T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXtrmv<float>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                  const size_t n,
                                  const float* a_buffer, const size_t a_offset, const size_t a_ld,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasStrmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtrmv<double>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const double* a_buffer, const size_t a_offset, const size_t a_ld,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDtrmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtrmv<float2>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCtrmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtrmv<double2>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                    const size_t n,
                                    const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZtrmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtrmv<half>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                 const size_t n,
                                 const half* a_buffer, const size_t a_offset, const size_t a_ld,
                                 half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for STBMV/DTBMV/CTBMV/ZTBMV
template <typename T>
cublasStatus_t cublasXtbmv(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t n, const size_t k,
                           const T* a_buffer, const size_t a_offset, const size_t a_ld,
                           T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXtbmv<float>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                  const size_t n, const size_t k,
                                  const float* a_buffer, const size_t a_offset, const size_t a_ld,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasStbmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtbmv<double>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n, const size_t k,
                                   const double* a_buffer, const size_t a_offset, const size_t a_ld,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDtbmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtbmv<float2>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n, const size_t k,
                                   const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCtbmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtbmv<double2>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                    const size_t n, const size_t k,
                                    const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZtbmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtbmv<half>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                 const size_t n, const size_t k,
                                 const half* a_buffer, const size_t a_offset, const size_t a_ld,
                                 half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for STPMV/DTPMV/CTPMV/ZTPMV
template <typename T>
cublasStatus_t cublasXtpmv(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t n,
                           const T* ap_buffer, const size_t ap_offset,
                           T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXtpmv<float>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                  const size_t n,
                                  const float* ap_buffer, const size_t ap_offset,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasStpmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &ap_buffer[ap_offset],
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtpmv<double>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const double* ap_buffer, const size_t ap_offset,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDtpmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &ap_buffer[ap_offset],
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtpmv<float2>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const float2* ap_buffer, const size_t ap_offset,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCtpmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&ap_buffer[ap_offset]),
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtpmv<double2>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                    const size_t n,
                                    const double2* ap_buffer, const size_t ap_offset,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZtpmv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&ap_buffer[ap_offset]),
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtpmv<half>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                 const size_t n,
                                 const half* ap_buffer, const size_t ap_offset,
                                 half* x_buffer, const size_t x_offset, const size_t x_inc) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for STRSV/DTRSV/CTRSV/ZTRSV
template <typename T>
cublasStatus_t cublasXtrsv(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t n,
                           const T* a_buffer, const size_t a_offset, const size_t a_ld,
                           T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXtrsv<float>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                  const size_t n,
                                  const float* a_buffer, const size_t a_offset, const size_t a_ld,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasStrsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtrsv<double>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const double* a_buffer, const size_t a_offset, const size_t a_ld,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDtrsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtrsv<float2>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCtrsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtrsv<double2>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                    const size_t n,
                                    const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZtrsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for STBSV/DTBSV/CTBSV/ZTBSV
template <typename T>
cublasStatus_t cublasXtbsv(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t n, const size_t k,
                           const T* a_buffer, const size_t a_offset, const size_t a_ld,
                           T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXtbsv<float>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                  const size_t n, const size_t k,
                                  const float* a_buffer, const size_t a_offset, const size_t a_ld,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasStbsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtbsv<double>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n, const size_t k,
                                   const double* a_buffer, const size_t a_offset, const size_t a_ld,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDtbsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            &a_buffer[a_offset], a_ld,
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtbsv<float2>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n, const size_t k,
                                   const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCtbsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtbsv<double2>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                    const size_t n, const size_t k,
                                    const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZtbsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n), static_cast<int>(k),
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for STPSV/DTPSV/CTPSV/ZTPSV
template <typename T>
cublasStatus_t cublasXtpsv(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t n,
                           const T* ap_buffer, const size_t ap_offset,
                           T* x_buffer, const size_t x_offset, const size_t x_inc);
template <>
cublasStatus_t cublasXtpsv<float>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                  const size_t n,
                                  const float* ap_buffer, const size_t ap_offset,
                                  float* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasStpsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &ap_buffer[ap_offset],
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtpsv<double>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const double* ap_buffer, const size_t ap_offset,
                                   double* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDtpsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            &ap_buffer[ap_offset],
                            &x_buffer[x_offset], static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtpsv<float2>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                   const size_t n,
                                   const float2* ap_buffer, const size_t ap_offset,
                                   float2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCtpsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuComplex*>(&ap_buffer[ap_offset]),
                            reinterpret_cast<cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}
template <>
cublasStatus_t cublasXtpsv<double2>(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                                    const size_t n,
                                    const double2* ap_buffer, const size_t ap_offset,
                                    double2* x_buffer, const size_t x_offset, const size_t x_inc) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZtpsv(handle, triangle, a_transpose, diagonal,
                            static_cast<int>(n),
                            reinterpret_cast<const cuDoubleComplex*>(&ap_buffer[ap_offset]),
                            reinterpret_cast<cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc));
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for SGER/DGER
cublasStatus_t cublasXger(const Layout layout,
                          const size_t m, const size_t n,
                          const float alpha,
                          const float* x_buffer, const size_t x_offset, const size_t x_inc,
                          const float* y_buffer, const size_t y_offset, const size_t y_inc,
                          float* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSger(handle, static_cast<int>(m), static_cast<int>(n),
                           &alpha,
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &y_buffer[y_offset], static_cast<int>(y_inc),
                           &a_buffer[a_offset], a_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXger(const Layout layout,
                          const size_t m, const size_t n,
                          const double alpha,
                          const double* x_buffer, const size_t x_offset, const size_t x_inc,
                          const double* y_buffer, const size_t y_offset, const size_t y_inc,
                          double* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDger(handle, static_cast<int>(m), static_cast<int>(n),
                           &alpha,
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &y_buffer[y_offset], static_cast<int>(y_inc),
                           &a_buffer[a_offset], a_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXger(const Layout layout,
                          const size_t m, const size_t n,
                          const half alpha,
                          const half* x_buffer, const size_t x_offset, const size_t x_inc,
                          const half* y_buffer, const size_t y_offset, const size_t y_inc,
                          half* a_buffer, const size_t a_offset, const size_t a_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for CGERU/ZGERU
cublasStatus_t cublasXgeru(const Layout layout,
                           const size_t m, const size_t n,
                           const float2 alpha,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float2* y_buffer, const size_t y_offset, const size_t y_inc,
                           float2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCgeru(handle, static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuComplex*>(&a_buffer[a_offset]), a_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgeru(const Layout layout,
                           const size_t m, const size_t n,
                           const double2 alpha,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double2* y_buffer, const size_t y_offset, const size_t y_inc,
                           double2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZgeru(handle, static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuDoubleComplex*>(&a_buffer[a_offset]), a_ld);
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for CGERC/ZGERC
cublasStatus_t cublasXgerc(const Layout layout,
                           const size_t m, const size_t n,
                           const float2 alpha,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float2* y_buffer, const size_t y_offset, const size_t y_inc,
                           float2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCgerc(handle, static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuComplex*>(&a_buffer[a_offset]), a_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgerc(const Layout layout,
                           const size_t m, const size_t n,
                           const double2 alpha,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double2* y_buffer, const size_t y_offset, const size_t y_inc,
                           double2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZgerc(handle, static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuDoubleComplex*>(&a_buffer[a_offset]), a_ld);
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for CHER/ZHER
cublasStatus_t cublasXher(const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const float alpha,
                          const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                          float2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCher(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                           reinterpret_cast<cuComplex*>(&a_buffer[a_offset]), a_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXher(const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const double alpha,
                          const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                          double2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZher(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                           reinterpret_cast<cuDoubleComplex*>(&a_buffer[a_offset]), a_ld);
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for CHPR/ZHPR
cublasStatus_t cublasXhpr(const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const float alpha,
                          const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                          float2* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasChpr(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                           reinterpret_cast<cuComplex*>(&ap_buffer[ap_offset]));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXhpr(const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const double alpha,
                          const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                          double2* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZhpr(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                           reinterpret_cast<cuDoubleComplex*>(&ap_buffer[ap_offset]));
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for CHER2/ZHER2
cublasStatus_t cublasXher2(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float2 alpha,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float2* y_buffer, const size_t y_offset, const size_t y_inc,
                           float2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCher2(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuComplex*>(&a_buffer[a_offset]), a_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXher2(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double2 alpha,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double2* y_buffer, const size_t y_offset, const size_t y_inc,
                           double2* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZher2(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuDoubleComplex*>(&a_buffer[a_offset]), a_ld);
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for CHPR2/ZHPR2
cublasStatus_t cublasXhpr2(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float2 alpha,
                           const float2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float2* y_buffer, const size_t y_offset, const size_t y_inc,
                           float2* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasChpr2(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuComplex*>(&ap_buffer[ap_offset]));
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXhpr2(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double2 alpha,
                           const double2* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double2* y_buffer, const size_t y_offset, const size_t y_inc,
                           double2* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZhpr2(handle, triangle,
                            static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&x_buffer[x_offset]), static_cast<int>(x_inc),
                            reinterpret_cast<const cuDoubleComplex*>(&y_buffer[y_offset]), static_cast<int>(y_inc),
                            reinterpret_cast<cuDoubleComplex*>(&ap_buffer[ap_offset]));
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for SSYR/DSYR
cublasStatus_t cublasXsyr(const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const float alpha,
                          const float* x_buffer, const size_t x_offset, const size_t x_inc,
                          float* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSsyr(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &a_buffer[a_offset], a_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsyr(const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const double alpha,
                          const double* x_buffer, const size_t x_offset, const size_t x_inc,
                          double* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDsyr(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &a_buffer[a_offset], a_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsyr(const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const half alpha,
                          const half* x_buffer, const size_t x_offset, const size_t x_inc,
                          half* a_buffer, const size_t a_offset, const size_t a_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSPR/DSPR
cublasStatus_t cublasXspr(const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const float alpha,
                          const float* x_buffer, const size_t x_offset, const size_t x_inc,
                          float* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSspr(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &ap_buffer[ap_offset]);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXspr(const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const double alpha,
                          const double* x_buffer, const size_t x_offset, const size_t x_inc,
                          double* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDspr(handle, triangle,
                           static_cast<int>(n),
                           &alpha,
                           &x_buffer[x_offset], static_cast<int>(x_inc),
                           &ap_buffer[ap_offset]);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXspr(const Layout layout, const cublasFillMode_t triangle,
                          const size_t n,
                          const half alpha,
                          const half* x_buffer, const size_t x_offset, const size_t x_inc,
                          half* ap_buffer, const size_t ap_offset) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSYR2/DSYR2
cublasStatus_t cublasXsyr2(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float alpha,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float* y_buffer, const size_t y_offset, const size_t y_inc,
                           float* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSsyr2(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc),
                            &a_buffer[a_offset], a_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsyr2(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double alpha,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double* y_buffer, const size_t y_offset, const size_t y_inc,
                           double* a_buffer, const size_t a_offset, const size_t a_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDsyr2(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc),
                            &a_buffer[a_offset], a_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsyr2(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const half alpha,
                           const half* x_buffer, const size_t x_offset, const size_t x_inc,
                           const half* y_buffer, const size_t y_offset, const size_t y_inc,
                           half* a_buffer, const size_t a_offset, const size_t a_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSPR2/DSPR2
cublasStatus_t cublasXspr2(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const float alpha,
                           const float* x_buffer, const size_t x_offset, const size_t x_inc,
                           const float* y_buffer, const size_t y_offset, const size_t y_inc,
                           float* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSspr2(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc),
                            &ap_buffer[ap_offset]);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXspr2(const Layout layout, const cublasFillMode_t triangle,
                           const size_t n,
                           const double alpha,
                           const double* x_buffer, const size_t x_offset, const size_t x_inc,
                           const double* y_buffer, const size_t y_offset, const size_t y_inc,
                           double* ap_buffer, const size_t ap_offset) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDspr2(handle, triangle,
                            static_cast<int>(n),
                            &alpha,
                            &x_buffer[x_offset], static_cast<int>(x_inc),
                            &y_buffer[y_offset], static_cast<int>(y_inc),
                            &ap_buffer[ap_offset]);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXspr2(const Layout layout, const cublasFillMode_t triangle,
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
cublasStatus_t cublasXgemm(const Layout layout, const cublasOperation_t a_transpose, const cublasOperation_t b_transpose,
                           const size_t m, const size_t n, const size_t k,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float* b_buffer, const size_t b_offset, const size_t b_ld,
                           const float beta,
                           float* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSgemm(handle, a_transpose, b_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld,
                            &beta,
                            &c_buffer[c_offset], c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgemm(const Layout layout, const cublasOperation_t a_transpose, const cublasOperation_t b_transpose,
                           const size_t m, const size_t n, const size_t k,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double* b_buffer, const size_t b_offset, const size_t b_ld,
                           const double beta,
                           double* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDgemm(handle, a_transpose, b_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld,
                            &beta,
                            &c_buffer[c_offset], c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgemm(const Layout layout, const cublasOperation_t a_transpose, const cublasOperation_t b_transpose,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCgemm(handle, a_transpose, b_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&b_buffer[b_offset]), b_ld,
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgemm(const Layout layout, const cublasOperation_t a_transpose, const cublasOperation_t b_transpose,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZgemm(handle, a_transpose, b_transpose,
                            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&b_buffer[b_offset]), b_ld,
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXgemm(const Layout layout, const cublasOperation_t a_transpose, const cublasOperation_t b_transpose,
                           const size_t m, const size_t n, const size_t k,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half* b_buffer, const size_t b_offset, const size_t b_ld,
                           const half beta,
                           half* c_buffer, const size_t c_offset, const size_t c_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for SSYMM/DSYMM/CSYMM/ZSYMM
cublasStatus_t cublasXsymm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
                           const size_t m, const size_t n,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float* b_buffer, const size_t b_offset, const size_t b_ld,
                           const float beta,
                           float* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSsymm(handle, side, triangle,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld,
                            &beta,
                            &c_buffer[c_offset], c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsymm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
                           const size_t m, const size_t n,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double* b_buffer, const size_t b_offset, const size_t b_ld,
                           const double beta,
                           double* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDsymm(handle, side, triangle,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld,
                            &beta,
                            &c_buffer[c_offset], c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsymm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCsymm(handle, side, triangle,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&b_buffer[b_offset]), b_ld,
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsymm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZsymm(handle, side, triangle,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&b_buffer[b_offset]), b_ld,
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsymm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
                           const size_t m, const size_t n,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half* b_buffer, const size_t b_offset, const size_t b_ld,
                           const half beta,
                           half* c_buffer, const size_t c_offset, const size_t c_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for CHEMM/ZHEMM
cublasStatus_t cublasXhemm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasChemm(handle, side, triangle,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&b_buffer[b_offset]), b_ld,
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXhemm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZhemm(handle, side, triangle,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&b_buffer[b_offset]), b_ld,
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for SSYRK/DSYRK/CSYRK/ZSYRK
cublasStatus_t cublasXsyrk(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
                           const size_t n, const size_t k,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float beta,
                           float* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSsyrk(handle, triangle, a_transpose,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &beta,
                            &c_buffer[c_offset], c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsyrk(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
                           const size_t n, const size_t k,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double beta,
                           double* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDsyrk(handle, triangle, a_transpose,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &beta,
                            &c_buffer[c_offset], c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsyrk(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCsyrk(handle, triangle, a_transpose,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            &beta_cuda,
                            reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsyrk(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZsyrk(handle, triangle, a_transpose,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            &beta_cuda,
                            reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsyrk(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
                           const size_t n, const size_t k,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           const half beta,
                           half* c_buffer, const size_t c_offset, const size_t c_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for CHERK/ZHERK
cublasStatus_t cublasXherk(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
                           const size_t n, const size_t k,
                           const float alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const float beta,
                           float2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCherk(handle, triangle, a_transpose,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            &beta,
                            reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXherk(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t a_transpose,
                           const size_t n, const size_t k,
                           const double alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           const double beta,
                           double2* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZherk(handle, triangle, a_transpose,
                            static_cast<int>(n), static_cast<int>(k),
                            &alpha,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            &beta,
                            reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for SSYR2K/DSYR2K/CSYR2K/ZSYR2K
cublasStatus_t cublasXsyr2k(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
                            const size_t n, const size_t k,
                            const float alpha,
                            const float* a_buffer, const size_t a_offset, const size_t a_ld,
                            const float* b_buffer, const size_t b_offset, const size_t b_ld,
                            const float beta,
                            float* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasSsyr2k(handle, triangle, ab_transpose,
                             static_cast<int>(n), static_cast<int>(k),
                             &alpha,
                             &a_buffer[a_offset], a_ld,
                             &b_buffer[b_offset], b_ld,
                             &beta,
                             &c_buffer[c_offset], c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsyr2k(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
                            const size_t n, const size_t k,
                            const double alpha,
                            const double* a_buffer, const size_t a_offset, const size_t a_ld,
                            const double* b_buffer, const size_t b_offset, const size_t b_ld,
                            const double beta,
                            double* c_buffer, const size_t c_offset, const size_t c_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDsyr2k(handle, triangle, ab_transpose,
                             static_cast<int>(n), static_cast<int>(k),
                             &alpha,
                             &a_buffer[a_offset], a_ld,
                             &b_buffer[b_offset], b_ld,
                             &beta,
                             &c_buffer[c_offset], c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsyr2k(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCsyr2k(handle, triangle, ab_transpose,
                             static_cast<int>(n), static_cast<int>(k),
                             &alpha_cuda,
                             reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                             reinterpret_cast<const cuComplex*>(&b_buffer[b_offset]), b_ld,
                             &beta_cuda,
                             reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsyr2k(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZsyr2k(handle, triangle, ab_transpose,
                             static_cast<int>(n), static_cast<int>(k),
                             &alpha_cuda,
                             reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                             reinterpret_cast<const cuDoubleComplex*>(&b_buffer[b_offset]), b_ld,
                             &beta_cuda,
                             reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXsyr2k(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
                            const size_t n, const size_t k,
                            const half alpha,
                            const half* a_buffer, const size_t a_offset, const size_t a_ld,
                            const half* b_buffer, const size_t b_offset, const size_t b_ld,
                            const half beta,
                            half* c_buffer, const size_t c_offset, const size_t c_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for CHER2K/ZHER2K
cublasStatus_t cublasXher2k(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCher2k(handle, triangle, ab_transpose,
                             static_cast<int>(n), static_cast<int>(k),
                             &alpha_cuda,
                             reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                             reinterpret_cast<const cuComplex*>(&b_buffer[b_offset]), b_ld,
                             &beta,
                             reinterpret_cast<cuComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXher2k(const Layout layout, const cublasFillMode_t triangle, const cublasOperation_t ab_transpose,
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
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZher2k(handle, triangle, ab_transpose,
                             static_cast<int>(n), static_cast<int>(k),
                             &alpha_cuda,
                             reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                             reinterpret_cast<const cuDoubleComplex*>(&b_buffer[b_offset]), b_ld,
                             &beta,
                             reinterpret_cast<cuDoubleComplex*>(&c_buffer[c_offset]), c_ld);
  cublasDestroy(handle);
  return status;
}

// Forwards the cuBLAS calls for STRMM/DTRMM/CTRMM/ZTRMM
cublasStatus_t cublasXtrmm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           float* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasStrmm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXtrmm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           double* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDtrmm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXtrmm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const float2 alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           float2* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCtrmm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuComplex*>(&b_buffer[b_offset]), b_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXtrmm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const double2 alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           double2* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZtrmm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuDoubleComplex*>(&b_buffer[b_offset]), b_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXtrmm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const half alpha,
                           const half* a_buffer, const size_t a_offset, const size_t a_ld,
                           half* b_buffer, const size_t b_offset, const size_t b_ld) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// Forwards the cuBLAS calls for STRSM/DTRSM/CTRSM/ZTRSM
cublasStatus_t cublasXtrsm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const float alpha,
                           const float* a_buffer, const size_t a_offset, const size_t a_ld,
                           float* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasStrsm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXtrsm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const double alpha,
                           const double* a_buffer, const size_t a_offset, const size_t a_ld,
                           double* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasDtrsm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha,
                            &a_buffer[a_offset], a_ld,
                            &b_buffer[b_offset], b_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXtrsm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const float2 alpha,
                           const float2* a_buffer, const size_t a_offset, const size_t a_ld,
                           float2* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasCtrsm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuComplex*>(&b_buffer[b_offset]), b_ld);
  cublasDestroy(handle);
  return status;
}
cublasStatus_t cublasXtrsm(const Layout layout, const cublasSideMode_t side, const cublasFillMode_t triangle, const cublasOperation_t a_transpose, const cublasDiagType_t diagonal,
                           const size_t m, const size_t n,
                           const double2 alpha,
                           const double2* a_buffer, const size_t a_offset, const size_t a_ld,
                           double2* b_buffer, const size_t b_offset, const size_t b_ld) {
  if (layout == Layout::kRowMajor) { return CUBLAS_STATUS_NOT_SUPPORTED; }
  cuDoubleComplex alpha_cuda;
  alpha_cuda.x = alpha.real();
  alpha_cuda.y = alpha.imag();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { return CUBLAS_STATUS_NOT_INITIALIZED; }
  auto status = cublasZtrsm(handle, side, triangle, a_transpose, diagonal,
                            static_cast<int>(m), static_cast<int>(n),
                            &alpha_cuda,
                            reinterpret_cast<const cuDoubleComplex*>(&a_buffer[a_offset]), a_ld,
                            reinterpret_cast<cuDoubleComplex*>(&b_buffer[b_offset]), b_ld);
  cublasDestroy(handle);
  return status;
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_WRAPPER_CUBLAS_H_
#endif
