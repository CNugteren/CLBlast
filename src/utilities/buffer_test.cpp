
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the OpenCL buffers (matrices and vectors). These tests are
// templated and thus header-only.
//
// =================================================================================================

#ifndef CLBLAST_BUFFER_TEST_H_
#define CLBLAST_BUFFER_TEST_H_

#include <algorithm>
#include <vector>

#include "clblast_half.h"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Tests matrix 'A' for validity
template <typename T>
void TestMatrixA(const size_t one, const size_t two, const Buffer<T>& buffer, const size_t offset, const size_t ld,
                 const bool test_lead_dim = true) {
  if (test_lead_dim && ld < one) {
    throw BLASError(StatusCode::kInvalidLeadDimA);
  }
  try {
    const auto required_size = (ld * (two - 1) + one + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) {
      throw BLASError(StatusCode::kInsufficientMemoryA);
    }
  } catch (const Error<std::runtime_error>& e) {
    throw BLASError(StatusCode::kInvalidMatrixA, e.what());
  }
}

template void TestMatrixA(const size_t, const size_t, const Buffer<half>&, const size_t, const size_t, const bool);
template void TestMatrixA(const size_t, const size_t, const Buffer<float>&, const size_t, const size_t, const bool);
template void TestMatrixA(const size_t, const size_t, const Buffer<double>&, const size_t, const size_t, const bool);
template void TestMatrixA(const size_t, const size_t, const Buffer<float2>&, const size_t, const size_t, const bool);
template void TestMatrixA(const size_t, const size_t, const Buffer<double2>&, const size_t, const size_t, const bool);

// Tests matrix 'B' for validity
template <typename T>
void TestMatrixB(const size_t one, const size_t two, const Buffer<T>& buffer, const size_t offset, const size_t ld,
                 const bool test_lead_dim = true) {
  if (test_lead_dim && ld < one) {
    throw BLASError(StatusCode::kInvalidLeadDimB);
  }
  try {
    const auto required_size = (ld * (two - 1) + one + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) {
      throw BLASError(StatusCode::kInsufficientMemoryB);
    }
  } catch (const Error<std::runtime_error>& e) {
    throw BLASError(StatusCode::kInvalidMatrixB, e.what());
  }
}

template void TestMatrixB(const size_t, const size_t, const Buffer<unsigned short>&, const size_t, const size_t,
                          const bool);
template void TestMatrixB(const size_t, const size_t, const Buffer<float>&, const size_t, const size_t, const bool);
template void TestMatrixB(const size_t, const size_t, const Buffer<double>&, const size_t, const size_t, const bool);
template void TestMatrixB(const size_t, const size_t, const Buffer<float2>&, const size_t, const size_t, const bool);
template void TestMatrixB(const size_t, const size_t, const Buffer<double2>&, const size_t, const size_t, const bool);

// Tests matrix 'C' for validity
template <typename T>
void TestMatrixC(const size_t one, const size_t two, const Buffer<T>& buffer, const size_t offset, const size_t ld) {
  if (ld < one) {
    throw BLASError(StatusCode::kInvalidLeadDimC);
  }
  try {
    const auto required_size = (ld * (two - 1) + one + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) {
      throw BLASError(StatusCode::kInsufficientMemoryC);
    }
  } catch (const Error<std::runtime_error>& e) {
    throw BLASError(StatusCode::kInvalidMatrixC, e.what());
  }
}

template void TestMatrixC(const size_t, const size_t, const Buffer<half>&, const size_t, const size_t);
template void TestMatrixC(const size_t, const size_t, const Buffer<float>&, const size_t, const size_t);
template void TestMatrixC(const size_t, const size_t, const Buffer<double>&, const size_t, const size_t);
template void TestMatrixC(const size_t, const size_t, const Buffer<float2>&, const size_t, const size_t);
template void TestMatrixC(const size_t, const size_t, const Buffer<double2>&, const size_t, const size_t);

// Tests matrix 'AP' for validity
template <typename T>
void TestMatrixAP(const size_t n, const Buffer<T>& buffer, const size_t offset) {
  try {
    const auto required_size = (((n * (n + 1)) / 2) + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) {
      throw BLASError(StatusCode::kInsufficientMemoryA);
    }
  } catch (const Error<std::runtime_error>& e) {
    throw BLASError(StatusCode::kInvalidMatrixA, e.what());
  }
}

template void TestMatrixAP(const size_t, const Buffer<half>&, const size_t);
template void TestMatrixAP(const size_t, const Buffer<float>&, const size_t);
template void TestMatrixAP(const size_t, const Buffer<double>&, const size_t);
template void TestMatrixAP(const size_t, const Buffer<float2>&, const size_t);
template void TestMatrixAP(const size_t, const Buffer<double2>&, const size_t);

// =================================================================================================

// Tests vector 'X' for validity
template <typename T>
void TestVectorX(const size_t n, const Buffer<T>& buffer, const size_t offset, const size_t inc) {
  if (inc == 0) {
    throw BLASError(StatusCode::kInvalidIncrementX);
  }
  try {
    const auto required_size = ((n - 1) * inc + 1 + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) {
      throw BLASError(StatusCode::kInsufficientMemoryX);
    }
  } catch (const Error<std::runtime_error>& e) {
    throw BLASError(StatusCode::kInvalidVectorX, e.what());
  }
}

template void TestVectorX(const size_t n, const Buffer<half>& buffer, const size_t offset, const size_t inc);
template void TestVectorX(const size_t n, const Buffer<float>& buffer, const size_t offset, const size_t inc);
template void TestVectorX(const size_t n, const Buffer<double>& buffer, const size_t offset, const size_t inc);
template void TestVectorX(const size_t n, const Buffer<float2>& buffer, const size_t offset, const size_t inc);
template void TestVectorX(const size_t n, const Buffer<double2>& buffer, const size_t offset, const size_t inc);

// Tests vector 'Y' for validity
template <typename T>
void TestVectorY(const size_t n, const Buffer<T>& buffer, const size_t offset, const size_t inc) {
  if (inc == 0) {
    throw BLASError(StatusCode::kInvalidIncrementY);
  }
  try {
    const auto required_size = ((n - 1) * inc + 1 + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) {
      throw BLASError(StatusCode::kInsufficientMemoryY);
    }
  } catch (const Error<std::runtime_error>& e) {
    throw BLASError(StatusCode::kInvalidVectorY, e.what());
  }
}

template void TestVectorY(const size_t n, const Buffer<half>& buffer, const size_t offset, const size_t inc);
template void TestVectorY(const size_t n, const Buffer<float>& buffer, const size_t offset, const size_t inc);
template void TestVectorY(const size_t n, const Buffer<double>& buffer, const size_t offset, const size_t inc);
template void TestVectorY(const size_t n, const Buffer<float2>& buffer, const size_t offset, const size_t inc);
template void TestVectorY(const size_t n, const Buffer<double2>& buffer, const size_t offset, const size_t inc);

// =================================================================================================

// Tests vector 'scalar' for validity
template <typename T>
void TestVectorScalar(const size_t n, const Buffer<T>& buffer, const size_t offset) {
  try {
    const auto required_size = (n + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) {
      throw BLASError(StatusCode::kInsufficientMemoryScalar);
    }
  } catch (const Error<std::runtime_error>& e) {
    throw BLASError(StatusCode::kInvalidVectorScalar, e.what());
  }
}

template void TestVectorScalar(const size_t, const Buffer<half>&, const size_t);
template void TestVectorScalar(const size_t, const Buffer<float>&, const size_t);
template void TestVectorScalar(const size_t, const Buffer<double>&, const size_t);
template void TestVectorScalar(const size_t, const Buffer<float2>&, const size_t);
template void TestVectorScalar(const size_t, const Buffer<double2>&, const size_t);

// Tests vector 'index' for validity
template <typename T>
void TestVectorIndex(const size_t n, const Buffer<T>& buffer, const size_t offset) {
  try {
    const auto required_size = (n + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) {
      throw BLASError(StatusCode::kInsufficientMemoryScalar);
    }
  } catch (const Error<std::runtime_error>& e) {
    throw BLASError(StatusCode::kInvalidVectorScalar, e.what());
  }
}

template void TestVectorIndex(const size_t, const Buffer<unsigned int>&, const size_t);

// =================================================================================================

// Tests matrix 'A' for validity in a batched setting
template <typename T>
void TestBatchedMatrixA(const size_t one, const size_t two, const Buffer<T>& buffer, const std::vector<size_t>& offsets,
                        const size_t ld, const bool test_lead_dim = true) {
  const auto max_offset = *std::max_element(offsets.begin(), offsets.end());
  TestMatrixA(one, two, buffer, max_offset, ld, test_lead_dim);
}

template void TestBatchedMatrixA(const size_t, const size_t, const Buffer<half>&, const std::vector<size_t>&,
                                 const size_t, const bool);
template void TestBatchedMatrixA(const size_t, const size_t, const Buffer<float>&, const std::vector<size_t>&,
                                 const size_t, const bool);
template void TestBatchedMatrixA(const size_t, const size_t, const Buffer<double>&, const std::vector<size_t>&,
                                 const size_t, const bool);
template void TestBatchedMatrixA(const size_t, const size_t, const Buffer<float2>&, const std::vector<size_t>&,
                                 const size_t, const bool);
template void TestBatchedMatrixA(const size_t, const size_t, const Buffer<double2>&, const std::vector<size_t>&,
                                 const size_t, const bool);

// Tests matrix 'B' for validity in a batched setting
template <typename T>
void TestBatchedMatrixB(const size_t one, const size_t two, const Buffer<T>& buffer, const std::vector<size_t>& offsets,
                        const size_t ld, const bool test_lead_dim = true) {
  const auto max_offset = *std::max_element(offsets.begin(), offsets.end());
  TestMatrixB(one, two, buffer, max_offset, ld, test_lead_dim);
}

template void TestBatchedMatrixB(const size_t, const size_t, const Buffer<half>&, const std::vector<size_t>&,
                                 const size_t, const bool);
template void TestBatchedMatrixB(const size_t, const size_t, const Buffer<float>&, const std::vector<size_t>&,
                                 const size_t, const bool);
template void TestBatchedMatrixB(const size_t, const size_t, const Buffer<double>&, const std::vector<size_t>&,
                                 const size_t, const bool);
template void TestBatchedMatrixB(const size_t, const size_t, const Buffer<float2>&, const std::vector<size_t>&,
                                 const size_t, const bool);
template void TestBatchedMatrixB(const size_t, const size_t, const Buffer<double2>&, const std::vector<size_t>&,
                                 const size_t, const bool);

// Tests matrix 'C' for validity in a batched setting
template <typename T>
void TestBatchedMatrixC(const size_t one, const size_t two, const Buffer<T>& buffer, const std::vector<size_t>& offsets,
                        const size_t ld) {
  const auto max_offset = *std::max_element(offsets.begin(), offsets.end());
  TestMatrixC(one, two, buffer, max_offset, ld);
}

template void TestBatchedMatrixC(const size_t, const size_t, const Buffer<half>&, const std::vector<size_t>&,
                                 const size_t);
template void TestBatchedMatrixC(const size_t, const size_t, const Buffer<float>&, const std::vector<size_t>&,
                                 const size_t);
template void TestBatchedMatrixC(const size_t, const size_t, const Buffer<double>&, const std::vector<size_t>&,
                                 const size_t);
template void TestBatchedMatrixC(const size_t, const size_t, const Buffer<float2>&, const std::vector<size_t>&,
                                 const size_t);
template void TestBatchedMatrixC(const size_t, const size_t, const Buffer<double2>&, const std::vector<size_t>&,
                                 const size_t);

// =================================================================================================

// Tests matrix 'A' for validity in a strided batched setting
template <typename T>
void TestStridedBatchedMatrixA(const size_t one, const size_t two, const Buffer<T>& buffer, const size_t offset,
                               const size_t stride, const size_t batch_count, const size_t ld,
                               const bool test_lead_dim = true) {
  const auto last_batch_offset = (batch_count - 1) * stride;
  TestMatrixA(one, two, buffer, offset + last_batch_offset, ld, test_lead_dim);
}

template void TestStridedBatchedMatrixA(const size_t, const size_t, const Buffer<half>&, const size_t, const size_t,
                                        const size_t, const size_t, const bool);
template void TestStridedBatchedMatrixA(const size_t, const size_t, const Buffer<float>&, const size_t, const size_t,
                                        const size_t, const size_t, const bool);
template void TestStridedBatchedMatrixA(const size_t, const size_t, const Buffer<double>&, const size_t, const size_t,
                                        const size_t, const size_t, const bool);
template void TestStridedBatchedMatrixA(const size_t, const size_t, const Buffer<float2>&, const size_t, const size_t,
                                        const size_t, const size_t, const bool);
template void TestStridedBatchedMatrixA(const size_t, const size_t, const Buffer<double2>&, const size_t, const size_t,
                                        const size_t, const size_t, const bool);

// Tests matrix 'B' for validity in a strided batched setting
template <typename T>
void TestStridedBatchedMatrixB(const size_t one, const size_t two, const Buffer<T>& buffer, const size_t offset,
                               const size_t stride, const size_t batch_count, const size_t ld,
                               const bool test_lead_dim = true) {
  const auto last_batch_offset = (batch_count - 1) * stride;
  TestMatrixB(one, two, buffer, offset + last_batch_offset, ld, test_lead_dim);
}

template void TestStridedBatchedMatrixB(const size_t, const size_t, const Buffer<half>&, const size_t, const size_t,
                                        const size_t, const size_t, const bool);
template void TestStridedBatchedMatrixB(const size_t, const size_t, const Buffer<float>&, const size_t, const size_t,
                                        const size_t, const size_t, const bool);
template void TestStridedBatchedMatrixB(const size_t, const size_t, const Buffer<double>&, const size_t, const size_t,
                                        const size_t, const size_t, const bool);
template void TestStridedBatchedMatrixB(const size_t, const size_t, const Buffer<float2>&, const size_t, const size_t,
                                        const size_t, const size_t, const bool);
template void TestStridedBatchedMatrixB(const size_t, const size_t, const Buffer<double2>&, const size_t, const size_t,
                                        const size_t, const size_t, const bool);

// Tests matrix 'C' for validity in a strided batched setting
template <typename T>
void TestStridedBatchedMatrixC(const size_t one, const size_t two, const Buffer<T>& buffer, const size_t offset,
                               const size_t stride, const size_t batch_count, const size_t ld) {
  const auto last_batch_offset = (batch_count - 1) * stride;
  TestMatrixC(one, two, buffer, offset + last_batch_offset, ld);
}

template void TestStridedBatchedMatrixC(const size_t, const size_t, const Buffer<half>&, const size_t, const size_t,
                                        const size_t, const size_t);
template void TestStridedBatchedMatrixC(const size_t, const size_t, const Buffer<float>&, const size_t, const size_t,
                                        const size_t, const size_t);
template void TestStridedBatchedMatrixC(const size_t, const size_t, const Buffer<double>&, const size_t, const size_t,
                                        const size_t, const size_t);
template void TestStridedBatchedMatrixC(const size_t, const size_t, const Buffer<float2>&, const size_t, const size_t,
                                        const size_t, const size_t);
template void TestStridedBatchedMatrixC(const size_t, const size_t, const Buffer<double2>&, const size_t, const size_t,
                                        const size_t, const size_t);

// =================================================================================================
}  // namespace clblast

// CLBLAST_BUFFER_TEST_H_
#endif
