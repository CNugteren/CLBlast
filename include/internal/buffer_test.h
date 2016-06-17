
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

#include "clblast.h"

namespace clblast {
// =================================================================================================

// Tests matrix 'A' for validity
template <typename T>
StatusCode TestMatrixA(const size_t one, const size_t two, const Buffer<T> &buffer,
                       const size_t offset, const size_t ld) {
  if (ld < one) { return StatusCode::kInvalidLeadDimA; }
  try {
    const auto required_size = (ld * (two - 1) + one + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { return StatusCode::kInsufficientMemoryA; }
  } catch (...) { return StatusCode::kInvalidMatrixA; }
  return StatusCode::kSuccess;
}

// Tests matrix 'B' for validity
template <typename T>
StatusCode TestMatrixB(const size_t one, const size_t two, const Buffer<T> &buffer,
                       const size_t offset, const size_t ld) {
  if (ld < one) { return StatusCode::kInvalidLeadDimB; }
  try {
    const auto required_size = (ld * (two - 1) + one + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { return StatusCode::kInsufficientMemoryB; }
  } catch (...) { return StatusCode::kInvalidMatrixB; }
  return StatusCode::kSuccess;
}

// Tests matrix 'C' for validity
template <typename T>
StatusCode TestMatrixC(const size_t one, const size_t two, const Buffer<T> &buffer,
                       const size_t offset, const size_t ld) {
  if (ld < one) { return StatusCode::kInvalidLeadDimC; }
  try {
    const auto required_size = (ld * (two - 1) + one + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { return StatusCode::kInsufficientMemoryC; }
  } catch (...) { return StatusCode::kInvalidMatrixC; }
  return StatusCode::kSuccess;
}

// Tests matrix 'AP' for validity
template <typename T>
StatusCode TestMatrixAP(const size_t n, const Buffer<T> &buffer, const size_t offset) {
  try {
    const auto required_size = (((n * (n + 1)) / 2) + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { return StatusCode::kInsufficientMemoryA; }
  } catch (...) { return StatusCode::kInvalidMatrixA; }
  return StatusCode::kSuccess;
}

// =================================================================================================

// Tests vector 'X' for validity
template <typename T>
StatusCode TestVectorX(const size_t n, const Buffer<T> &buffer, const size_t offset,
                       const size_t inc) {
  if (inc == 0) { return StatusCode::kInvalidIncrementX; }
  try {
    const auto required_size = ((n - 1) * inc + 1 + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { return StatusCode::kInsufficientMemoryX; }
  } catch (...) { return StatusCode::kInvalidVectorX; }
  return StatusCode::kSuccess;
}

// Tests vector 'Y' for validity
template <typename T>
StatusCode TestVectorY(const size_t n, const Buffer<T> &buffer, const size_t offset,
                       const size_t inc) {
  if (inc == 0) { return StatusCode::kInvalidIncrementY; }
  try {
    const auto required_size = ((n - 1) * inc + 1 + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { return StatusCode::kInsufficientMemoryY; }
  } catch (...) { return StatusCode::kInvalidVectorY; }
  return StatusCode::kSuccess;
}

// =================================================================================================

// Tests vector 'scalar' for validity
template <typename T>
StatusCode TestVectorScalar(const size_t n, const Buffer<T> &buffer, const size_t offset) {
  try {
    const auto required_size = (n + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { return StatusCode::kInsufficientMemoryScalar; }
  } catch (...) { return StatusCode::kInvalidVectorScalar; }
  return StatusCode::kSuccess;
}

// Tests vector 'index' for validity
template <typename T>
StatusCode TestVectorIndex(const size_t n, const Buffer<T> &buffer, const size_t offset) {
  try {
    const auto required_size = (n + offset) * sizeof(T);
    if (buffer.GetSize() < required_size) { return StatusCode::kInsufficientMemoryScalar; }
  } catch (...) { return StatusCode::kInvalidVectorScalar; }
  return StatusCode::kSuccess;
}

// =================================================================================================
} // namespace clblast

// CLBLAST_BUFFER_TEST_H_
#endif
