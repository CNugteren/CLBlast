#include "utilities/buffer_test.hpp"

#include "clblast_half.h"

namespace clblast {
template <typename T>
void TestMatrixA(const size_t one, const size_t two, const Buffer<T>& buffer, const size_t offset, const size_t ld,
                 const bool test_lead_dim) {
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

template void TestMatrixA<half>(const size_t, const size_t, const Buffer<half>&, const size_t, const size_t,
                                const bool);
template void TestMatrixA<float>(const size_t, const size_t, const Buffer<float>&, const size_t, const size_t,
                                 const bool);
template void TestMatrixA<double>(const size_t, const size_t, const Buffer<double>&, const size_t, const size_t,
                                  const bool);
template void TestMatrixA<float2>(const size_t, const size_t, const Buffer<float2>&, const size_t, const size_t,
                                  const bool);
template void TestMatrixA<double2>(const size_t, const size_t, const Buffer<double2>&, const size_t, const size_t,
                                   const bool);

template <typename T>
void TestMatrixB(const size_t one, const size_t two, const Buffer<T>& buffer, const size_t offset, const size_t ld,
                 const bool test_lead_dim) {
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

template void TestMatrixB<half>(const size_t, const size_t, const Buffer<half>&, const size_t, const size_t,
                                const bool);
template void TestMatrixB<float>(const size_t, const size_t, const Buffer<float>&, const size_t, const size_t,
                                 const bool);
template void TestMatrixB<double>(const size_t, const size_t, const Buffer<double>&, const size_t, const size_t,
                                  const bool);
template void TestMatrixB<float2>(const size_t, const size_t, const Buffer<float2>&, const size_t, const size_t,
                                  const bool);
template void TestMatrixB<double2>(const size_t, const size_t, const Buffer<double2>&, const size_t, const size_t,
                                   const bool);

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

template void TestMatrixC<half>(const size_t, const size_t, const Buffer<half>&, const size_t, const size_t);
template void TestMatrixC<float>(const size_t, const size_t, const Buffer<float>&, const size_t, const size_t);
template void TestMatrixC<double>(const size_t, const size_t, const Buffer<double>&, const size_t, const size_t);
template void TestMatrixC<float2>(const size_t, const size_t, const Buffer<float2>&, const size_t, const size_t);
template void TestMatrixC<double2>(const size_t, const size_t, const Buffer<double2>&, const size_t, const size_t);

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

template void TestMatrixAP<half>(const size_t, const Buffer<half>&, const size_t);
template void TestMatrixAP<float>(const size_t, const Buffer<float>&, const size_t);
template void TestMatrixAP<double>(const size_t, const Buffer<double>&, const size_t);
template void TestMatrixAP<float2>(const size_t, const Buffer<float2>&, const size_t);
template void TestMatrixAP<double2>(const size_t, const Buffer<double2>&, const size_t);

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

template void TestVectorX<half>(const size_t, const Buffer<half>&, const size_t, const size_t);
template void TestVectorX<float>(const size_t, const Buffer<float>&, const size_t, const size_t);
template void TestVectorX<double>(const size_t, const Buffer<double>&, const size_t, const size_t);
template void TestVectorX<float2>(const size_t, const Buffer<float2>&, const size_t, const size_t);
template void TestVectorX<double2>(const size_t, const Buffer<double2>&, const size_t, const size_t);

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

template void TestVectorY<half>(const size_t, const Buffer<half>&, const size_t, const size_t);
template void TestVectorY<float>(const size_t, const Buffer<float>&, const size_t, const size_t);
template void TestVectorY<double>(const size_t, const Buffer<double>&, const size_t, const size_t);
template void TestVectorY<float2>(const size_t, const Buffer<float2>&, const size_t, const size_t);
template void TestVectorY<double2>(const size_t, const Buffer<double2>&, const size_t, const size_t);

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

template void TestVectorScalar<half>(const size_t, const Buffer<half>&, const size_t);
template void TestVectorScalar<float>(const size_t, const Buffer<float>&, const size_t);
template void TestVectorScalar<double>(const size_t, const Buffer<double>&, const size_t);
template void TestVectorScalar<float2>(const size_t, const Buffer<float2>&, const size_t);
template void TestVectorScalar<double2>(const size_t, const Buffer<double2>&, const size_t);

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

template void TestVectorIndex<half>(const size_t, const Buffer<half>&, const size_t);
template void TestVectorIndex<float>(const size_t, const Buffer<float>&, const size_t);
template void TestVectorIndex<double>(const size_t, const Buffer<double>&, const size_t);
template void TestVectorIndex<float2>(const size_t, const Buffer<float2>&, const size_t);
template void TestVectorIndex<double2>(const size_t, const Buffer<double2>&, const size_t);
}  // namespace clblast