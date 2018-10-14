
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the TestBlas class (see the header for information about the class).
//
// =================================================================================================

#include <algorithm>
#include <iostream>
#include <random>

#include "utilities/utilities.hpp"
#include "test/correctness/testblas.hpp"

namespace clblast {
// =================================================================================================

// The transpose configurations to test with: template parameter dependent
template <> const std::vector<Transpose> TestBlas<half,half>::kTransposes = {Transpose::kNo, Transpose::kYes};
template <> const std::vector<Transpose> TestBlas<float,float>::kTransposes = {Transpose::kNo, Transpose::kYes};
template <> const std::vector<Transpose> TestBlas<double,double>::kTransposes = {Transpose::kNo, Transpose::kYes};
template <> const std::vector<Transpose> TestBlas<float2,float2>::kTransposes = {Transpose::kNo, Transpose::kYes, Transpose::kConjugate};
template <> const std::vector<Transpose> TestBlas<double2,double2>::kTransposes = {Transpose::kNo, Transpose::kYes, Transpose::kConjugate};
template <> const std::vector<Transpose> TestBlas<float2,float>::kTransposes = {Transpose::kNo, Transpose::kConjugate};
template <> const std::vector<Transpose> TestBlas<double2,double>::kTransposes = {Transpose::kNo, Transpose::kConjugate};

// Constructor, initializes the base class tester and input data
template <typename T, typename U>
TestBlas<T,U>::TestBlas(const std::vector<std::string> &arguments, const bool silent,
                        const std::string &name, const std::vector<std::string> &options,
                        const DataPrepare prepare_data,
                        const Routine run_routine,
                        const Routine run_reference1, const Routine run_reference2,
                        const ResultGet get_result, const ResultIndex get_index,
                        const ResultIterator get_id1, const ResultIterator get_id2):
    Tester<T,U>(arguments, silent, name, options),
    kOffsets(GetOffsets()),
    kAlphaValues(GetExampleScalars<U>(full_test_)),
    kBetaValues(GetExampleScalars<U>(full_test_)),
    prepare_data_(prepare_data),
    run_routine_(run_routine),
    run_reference1_(run_reference1),
    run_reference2_(run_reference2),
    get_result_(get_result),
    get_index_(get_index),
    get_id1_(get_id1),
    get_id2_(get_id2) {

  // Sanity check
  if (!compare_clblas_ && !compare_cblas_) {
    throw std::runtime_error("Invalid configuration: no reference to test against");
  }

  // Computes the maximum sizes. This allows for a single set of input/output buffers.
  const auto max_vec = *std::max_element(kVectorDims.begin(), kVectorDims.end());
  const auto max_inc = *std::max_element(kIncrements.begin(), kIncrements.end());
  const auto max_mat = *std::max_element(kMatrixDims.begin(), kMatrixDims.end());
  const auto max_ld = *std::max_element(kMatrixDims.begin(), kMatrixDims.end());
  const auto max_matvec = *std::max_element(kMatrixVectorDims.begin(), kMatrixVectorDims.end());
  const auto max_offset = *std::max_element(kOffsets.begin(), kOffsets.end());
  const auto max_batch_count = *std::max_element(kBatchCounts.begin(), kBatchCounts.end());

  // Creates test input data. Adds a 'canary' region to detect buffer overflows
  x_source_.resize(max_batch_count * std::max(max_vec, max_matvec)*max_inc + max_offset + kCanarySize);
  y_source_.resize(max_batch_count * std::max(max_vec, max_matvec)*max_inc + max_offset + kCanarySize);
  a_source_.resize(max_batch_count * std::max(max_mat, max_matvec)*std::max(max_ld, max_matvec) + max_offset + kCanarySize);
  b_source_.resize(max_batch_count * std::max(max_mat, max_matvec)*std::max(max_ld, max_matvec) + max_offset + kCanarySize);
  c_source_.resize(max_batch_count * std::max(max_mat, max_matvec)*std::max(max_ld, max_matvec) + max_offset + kCanarySize);
  ap_source_.resize(max_batch_count * std::max(max_mat, max_matvec)*std::max(max_mat, max_matvec) + max_offset + kCanarySize);
  scalar_source_.resize(max_batch_count * std::max(max_mat, max_matvec) + max_offset + kCanarySize);
  std::mt19937 mt(kSeed);
  std::uniform_real_distribution<double> dist(kTestDataLowerLimit, kTestDataUpperLimit);
  PopulateVector(x_source_, mt, dist);
  PopulateVector(y_source_, mt, dist);
  PopulateVector(a_source_, mt, dist);
  PopulateVector(b_source_, mt, dist);
  PopulateVector(c_source_, mt, dist);
  PopulateVector(ap_source_, mt, dist);
  PopulateVector(scalar_source_, mt, dist);
}

// ===============================================================================================

// Tests the routine for a wide variety of parameters
template <typename T, typename U>
void TestBlas<T,U>::TestRegular(std::vector<Arguments<U>> &test_vector, const std::string &name) {
  if (!PrecisionSupported<T>(device_)) { return; }
  TestStart("regular behaviour", name);

  // Iterates over all the to-be-tested combinations of arguments
  for (auto &args: test_vector) {

    // Adds a 'canary' region to detect buffer overflows
    args.x_size += kCanarySize;
    args.y_size += kCanarySize;
    args.a_size += kCanarySize;
    args.b_size += kCanarySize;
    args.c_size += kCanarySize;
    args.ap_size += kCanarySize;
    args.scalar_size += kCanarySize;

    // Prints the current test configuration
    if (verbose_) {
      fprintf(stdout, "   Testing: %s", GetOptionsString(args).c_str());
      std::cout << std::flush;
    }

    // Optionally prepares the input data
    prepare_data_(args, queue_, kSeed,
                  x_source_, y_source_, a_source_, b_source_, c_source_,
                  ap_source_, scalar_source_);

    // Set-up for the CLBlast run
    auto x_vec2 = Buffer<T>(context_, args.x_size);
    auto y_vec2 = Buffer<T>(context_, args.y_size);
    auto a_mat2 = Buffer<T>(context_, args.a_size);
    auto b_mat2 = Buffer<T>(context_, args.b_size);
    auto c_mat2 = Buffer<T>(context_, args.c_size);
    auto ap_mat2 = Buffer<T>(context_, args.ap_size);
    auto scalar2 = Buffer<T>(context_, args.scalar_size);
    x_vec2.Write(queue_, args.x_size, x_source_);
    y_vec2.Write(queue_, args.y_size, y_source_);
    a_mat2.Write(queue_, args.a_size, a_source_);
    b_mat2.Write(queue_, args.b_size, b_source_);
    c_mat2.Write(queue_, args.c_size, c_source_);
    ap_mat2.Write(queue_, args.ap_size, ap_source_);
    scalar2.Write(queue_, args.scalar_size, scalar_source_);
    auto buffers2 = Buffers<T>{x_vec2, y_vec2, a_mat2, b_mat2, c_mat2, ap_mat2, scalar2};

    // Runs CLBlast
    if (verbose_) {
      fprintf(stdout, "[CLBlast]");
      std::cout << std::flush;
    }
    const auto status2 = run_routine_(args, buffers2, queue_);

    // Don't continue with CBLAS if there are incorrect parameters
    if (compare_cblas_ && status2 != StatusCode::kSuccess) {
      if (verbose_) {
        fprintf(stdout, " -> %d -> ", static_cast<int>(status2));
        std::cout << std::flush;
      }
      TestErrorCodes(status2, status2, args);
      continue;
    }

    // Set-up for the reference run
    auto x_vec1 = Buffer<T>(context_, args.x_size);
    auto y_vec1 = Buffer<T>(context_, args.y_size);
    auto a_mat1 = Buffer<T>(context_, args.a_size);
    auto b_mat1 = Buffer<T>(context_, args.b_size);
    auto c_mat1 = Buffer<T>(context_, args.c_size);
    auto ap_mat1 = Buffer<T>(context_, args.ap_size);
    auto scalar1 = Buffer<T>(context_, args.scalar_size);
    x_vec1.Write(queue_, args.x_size, x_source_);
    y_vec1.Write(queue_, args.y_size, y_source_);
    a_mat1.Write(queue_, args.a_size, a_source_);
    b_mat1.Write(queue_, args.b_size, b_source_);
    c_mat1.Write(queue_, args.c_size, c_source_);
    ap_mat1.Write(queue_, args.ap_size, ap_source_);
    scalar1.Write(queue_, args.scalar_size, scalar_source_);
    auto buffers1 = Buffers<T>{x_vec1, y_vec1, a_mat1, b_mat1, c_mat1, ap_mat1, scalar1};

    // Runs the reference code
    if (verbose_) {
      if (compare_clblas_) { fprintf(stdout, " [clBLAS]"); }
      else if (compare_cblas_) { fprintf(stdout, " [CPU BLAS]"); }
      std::cout << std::flush;
    }
    auto status1 = StatusCode::kSuccess;
    if (compare_clblas_) { status1 = run_reference1_(args, buffers1, queue_); }
    else if (compare_cblas_) { status1 = run_reference2_(args, buffers1, queue_); }

    // Tests for equality of the two status codes
    if (verbose_) { fprintf(stdout, " -> "); std::cout << std::flush; }
    if (status1 != StatusCode::kSuccess || status2 != StatusCode::kSuccess) {
      TestErrorCodes(status1, status2, args);
      continue;
    }

    // Downloads the results
    auto result1 = get_result_(args, buffers1, queue_);
    auto result2 = get_result_(args, buffers2, queue_);

    // Computes the L2 error
    auto l2error = 0.0;
    const auto kErrorMarginL2 = getL2ErrorMargin<T>();
    for (auto id1=size_t{0}; id1<get_id1_(args); ++id1) {
      for (auto id2=size_t{0}; id2<get_id2_(args); ++id2) {
        auto index = get_index_(args, id1, id2);
        l2error += SquaredDifference(result1[index], result2[index]);
      }
    }
    l2error /= static_cast<double>(get_id1_(args) * get_id2_(args));

    // Checks for differences in the output
    auto errors = size_t{0};
    for (auto id1=size_t{0}; id1<get_id1_(args); ++id1) {
      for (auto id2=size_t{0}; id2<get_id2_(args); ++id2) {
        auto index = get_index_(args, id1, id2);
        if (!TestSimilarity(result1[index], result2[index])) {
          if (l2error >= kErrorMarginL2) { errors++; }
          if (verbose_) {
            if (get_id2_(args) == 1) { std::cout << std::endl << "   Error at index " << id1 << ": "; }
            else { std::cout << std::endl << "   Error at " << id1 << "," << id2 << ": "; }
            std::cout << " " << ToString(result1[index]) << " (reference) versus ";
            std::cout << " " << ToString(result2[index]) << " (CLBlast)";
            if (l2error < kErrorMarginL2) {
              std::cout << " - error suppressed by a low total L2 error" << std::endl;
            }
          }
        }
      }
    }
    // Checks for differences in the 'canary' region to detect buffer overflows
    for (auto canary_id=size_t{0}; canary_id<kCanarySize; ++canary_id) {
      auto index = get_index_(args, get_id1_(args) - 1, get_id2_(args) - 1) + canary_id;
      if (!TestSimilarity(result1[index], result2[index])) {
        errors++;
        if (verbose_) {
          if (get_id2_(args) == 1) { std::cout << std::endl << "   Buffer overflow index " << index << ": "; }
          else { std::cout << std::endl << "   Buffer overflow " << index << ": "; }
          std::cout << " " << ToString(result1[index]) << " (reference) versus ";
          std::cout << " " << ToString(result2[index]) << " (CLBlast)";
        }
      }
    }


    // Report the results
    if (verbose_ && errors > 0) {
      fprintf(stdout, "\n   Combined average L2 error: %.2e\n   ", l2error);
    }

    // Tests the error count (should be zero)
    TestErrorCount(errors, get_id1_(args)*get_id2_(args) + kCanarySize, args);
  }
  TestEnd();
}

// =================================================================================================

// Tests the routine for cases with invalid OpenCL memory buffer sizes. Tests only on return-types,
// does not test for results (if any).
template <typename T, typename U>
void TestBlas<T,U>::TestInvalid(std::vector<Arguments<U>> &test_vector, const std::string &name) {
  if (!PrecisionSupported<T>(device_)) { return; }
  if (!compare_clblas_) { return; } // not supported for CPU BLAS routines
  if (std::is_same<T, half>::value) { return; } // not supported for half-precision
  TestStart("invalid buffer sizes", name);

  // Iterates over all the to-be-tested combinations of arguments
  for (const auto &args: test_vector) {

    // Prints the current test configuration
    if (verbose_) {
      fprintf(stdout, "   Testing: %s", GetSizesString(args).c_str());
      std::cout << std::flush;
    }

    // Creates the buffers. Note: we are not using the cxpp11.h C++ version since we explicitly
    // want to be able to create invalid buffers (no error checking here).
    auto x_vec1 = CreateInvalidBuffer<T>(context_, args.x_size);
    auto y_vec1 = CreateInvalidBuffer<T>(context_, args.y_size);
    auto a_mat1 = CreateInvalidBuffer<T>(context_, args.a_size);
    auto b_mat1 = CreateInvalidBuffer<T>(context_, args.b_size);
    auto c_mat1 = CreateInvalidBuffer<T>(context_, args.c_size);
    auto ap_mat1 = CreateInvalidBuffer<T>(context_, args.ap_size);
    auto scalar1 = CreateInvalidBuffer<T>(context_, args.scalar_size);
    auto x_vec2 = CreateInvalidBuffer<T>(context_, args.x_size);
    auto y_vec2 = CreateInvalidBuffer<T>(context_, args.y_size);
    auto a_mat2 = CreateInvalidBuffer<T>(context_, args.a_size);
    auto b_mat2 = CreateInvalidBuffer<T>(context_, args.b_size);
    auto c_mat2 = CreateInvalidBuffer<T>(context_, args.c_size);
    auto ap_mat2 = CreateInvalidBuffer<T>(context_, args.ap_size);
    auto scalar2 = CreateInvalidBuffer<T>(context_, args.scalar_size);
    auto buffers1 = Buffers<T>{x_vec1, y_vec1, a_mat1, b_mat1, c_mat1, ap_mat1, scalar1};
    auto buffers2 = Buffers<T>{x_vec2, y_vec2, a_mat2, b_mat2, c_mat2, ap_mat2, scalar2};

    // Runs CLBlast
    if (verbose_) {
      fprintf(stdout, "[CLBlast]");
      std::cout << std::flush;
    }
    const auto status2 = run_routine_(args, buffers2, queue_);

    // Runs the reference code
    if (verbose_) {
      if (compare_clblas_) { fprintf(stdout, " [clBLAS]"); }
      else if (compare_cblas_) { fprintf(stdout, " [CPU BLAS]"); }
      std::cout << std::flush;
    }
    auto status1 = StatusCode::kSuccess;
    if (compare_clblas_) { status1 = run_reference1_(args, buffers1, queue_); }
    else if (compare_cblas_) { status1 = run_reference2_(args, buffers1, queue_); }

    // Tests for equality of the two status codes
    if (verbose_) { fprintf(stdout, " -> "); std::cout << std::flush; }
    TestErrorCodes(status1, status2, args);
  }
  TestEnd();
}

// =================================================================================================

// Compiles the templated class
template class TestBlas<half, half>;
template class TestBlas<float, float>;
template class TestBlas<double, double>;
template class TestBlas<float2, float2>;
template class TestBlas<double2, double2>;
template class TestBlas<float2, float>;
template class TestBlas<double2, double>;

// =================================================================================================
} // namespace clblast
