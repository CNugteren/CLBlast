
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the OpenCL buffers (matrices and vectors). These tests are
// templated and thus header-only.
//
// =================================================================================================

#ifndef CLBLAST_BUFFER_TEST_H_
#define CLBLAST_BUFFER_TEST_H_

#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Tests matrix 'A' for validity
template <typename T>
void TestMatrixA(const size_t one, const size_t two, const Buffer<T> &buffer,
                 const size_t offset, const size_t ld, const bool test_lead_dim = true) {
  if (test_lead_dim && ld < one) { throw BLASError(StatusCode::kInvalidLeadDimA); }
  try {
    const auto required_size = (ld * (two - 1) + one + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { throw BLASError(StatusCode::kInsufficientMemoryA); }
  } catch (const Error<std::runtime_error> &e) { throw BLASError(StatusCode::kInvalidMatrixA, e.what()); }
}

// Tests matrix 'B' for validity
template <typename T>
void TestMatrixB(const size_t one, const size_t two, const Buffer<T> &buffer,
                 const size_t offset, const size_t ld, const bool test_lead_dim = true) {
  if (test_lead_dim && ld < one) { throw BLASError(StatusCode::kInvalidLeadDimB); }
  try {
    const auto required_size = (ld * (two - 1) + one + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { throw BLASError(StatusCode::kInsufficientMemoryB); }
  } catch (const Error<std::runtime_error> &e) { throw BLASError(StatusCode::kInvalidMatrixB, e.what()); }
}

// Tests matrix 'C' for validity
template <typename T>
void TestMatrixC(const size_t one, const size_t two, const Buffer<T> &buffer,
                 const size_t offset, const size_t ld) {
  if (ld < one) { throw BLASError(StatusCode::kInvalidLeadDimC); }
  try {
    const auto required_size = (ld * (two - 1) + one + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { throw BLASError(StatusCode::kInsufficientMemoryC); }
  } catch (const Error<std::runtime_error> &e) { throw BLASError(StatusCode::kInvalidMatrixC, e.what()); }
}

// Tests matrix 'AP' for validity
template <typename T>
void TestMatrixAP(const size_t n, const Buffer<T> &buffer, const size_t offset) {
  try {
    const auto required_size = (((n * (n + 1)) / 2) + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { throw BLASError(StatusCode::kInsufficientMemoryA); }
  } catch (const Error<std::runtime_error> &e) { throw BLASError(StatusCode::kInvalidMatrixA, e.what()); }
}

// =================================================================================================

// Tests vector 'X' for validity
template <typename T>
void TestVectorX(const size_t n, const Buffer<T> &buffer, const size_t offset, const size_t inc) {
  if (inc == 0) { throw BLASError(StatusCode::kInvalidIncrementX); }
  try {
    const auto required_size = ((n - 1) * inc + 1 + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { throw BLASError(StatusCode::kInsufficientMemoryX); }
  } catch (const Error<std::runtime_error> &e) { throw BLASError(StatusCode::kInvalidVectorX, e.what()); }
}

// Tests vector 'Y' for validity
template <typename T>
void TestVectorY(const size_t n, const Buffer<T> &buffer, const size_t offset, const size_t inc) {
  if (inc == 0) { throw BLASError(StatusCode::kInvalidIncrementY); }
  try {
    const auto required_size = ((n - 1) * inc + 1 + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { throw BLASError(StatusCode::kInsufficientMemoryY); }
  } catch (const Error<std::runtime_error> &e) { throw BLASError(StatusCode::kInvalidVectorY, e.what()); }
}

// =================================================================================================

// Tests vector 'scalar' for validity
template <typename T>
void TestVectorScalar(const size_t n, const Buffer<T> &buffer, const size_t offset) {
  try {
    const auto required_size = (n + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { throw BLASError(StatusCode::kInsufficientMemoryScalar); }
  } catch (const Error<std::runtime_error> &e) { throw BLASError(StatusCode::kInvalidVectorScalar, e.what()); }
}

// Tests vector 'index' for validity
template <typename T>
void TestVectorIndex(const size_t n, const Buffer<T> &buffer, const size_t offset) {
  try {
    const auto required_size = (n + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { throw BLASError(StatusCode::kInsufficientMemoryScalar); }
  } catch (const Error<std::runtime_error> &e) { throw BLASError(StatusCode::kInvalidVectorScalar, e.what()); }
}

// =================================================================================================
} // namespace clblast

// CLBLAST_BUFFER_TEST_H_
#endif
