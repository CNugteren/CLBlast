
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

#include "test/correctness/testblas.hpp"

namespace clblast {
// =================================================================================================

// Test settings for the regular test. Append to these lists in case more tests are required.
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kVectorDims = { 7, 93, 4096 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kIncrements = { 1, 2, 7 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kMatrixDims = { 7, 64 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kMatrixVectorDims = { 61, 256 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kBandSizes = { 4, 19 };

// Test settings for the invalid tests
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kInvalidIncrements = { 0, 1 };
template <typename T, typename U> const size_t TestBlas<T,U>::kBufferSize = 64;
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kMatSizes = {0, kBufferSize*kBufferSize-1, kBufferSize*kBufferSize};
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kVecSizes = {0, kBufferSize - 1, kBufferSize};

// The layout/transpose/triangle options to test with
template <typename T, typename U> const std::vector<Layout> TestBlas<T,U>::kLayouts = {Layout::kRowMajor, Layout::kColMajor};
template <typename T, typename U> const std::vector<Triangle> TestBlas<T,U>::kTriangles = {Triangle::kUpper, Triangle::kLower};
template <typename T, typename U> const std::vector<Side> TestBlas<T,U>::kSides = {Side::kLeft, Side::kRight};
template <typename T, typename U> const std::vector<Diagonal> TestBlas<T,U>::kDiagonals = {Diagonal::kUnit, Diagonal::kNonUnit};
template <> const std::vector<Transpose> TestBlas<half,half>::kTransposes = {Transpose::kNo, Transpose::kYes};
template <> const std::vector<Transpose> TestBlas<float,float>::kTransposes = {Transpose::kNo, Transpose::kYes};
template <> const std::vector<Transpose> TestBlas<double,double>::kTransposes = {Transpose::kNo, Transpose::kYes};
template <> const std::vector<Transpose> TestBlas<float2,float2>::kTransposes = {Transpose::kNo, Transpose::kYes, Transpose::kConjugate};
template <> const std::vector<Transpose> TestBlas<double2,double2>::kTransposes = {Transpose::kNo, Transpose::kYes, Transpose::kConjugate};
template <> const std::vector<Transpose> TestBlas<float2,float>::kTransposes = {Transpose::kNo, Transpose::kConjugate};
template <> const std::vector<Transpose> TestBlas<double2,double>::kTransposes = {Transpose::kNo, Transpose::kConjugate};

// =================================================================================================

// Constructor, initializes the base class tester and input data
template <typename T, typename U>
TestBlas<T,U>::TestBlas(const std::vector<std::string> &arguments, const bool silent,
                        const std::string &name, const std::vector<std::string> &options,
                        const Routine run_routine,
                        const Routine run_reference1, const Routine run_reference2,
                        const ResultGet get_result, const ResultIndex get_index,
                        const ResultIterator get_id1, const ResultIterator get_id2):
    Tester<T,U>(arguments, silent, name, options),
    kOffsets(GetOffsets()),
    kAlphaValues(GetExampleScalars<U>(full_test_)),
    kBetaValues(GetExampleScalars<U>(full_test_)),
    run_routine_(run_routine),
    get_result_(get_result),
    get_index_(get_index),
    get_id1_(get_id1),
    get_id2_(get_id2) {

  // Sets the reference to test against
  if (compare_clblas_) { run_reference_ = run_reference1; }
  else if (compare_cblas_) { run_reference_ = run_reference2; }
  else { throw std::runtime_error("Invalid configuration: no reference to test against"); }

  // Computes the maximum sizes. This allows for a single set of input/output buffers.
  const auto max_vec = *std::max_element(kVectorDims.begin(), kVectorDims.end());
  const auto max_inc = *std::max_element(kIncrements.begin(), kIncrements.end());
  const auto max_mat = *std::max_element(kMatrixDims.begin(), kMatrixDims.end());
  const auto max_ld = *std::max_element(kMatrixDims.begin(), kMatrixDims.end());
  const auto max_matvec = *std::max_element(kMatrixVectorDims.begin(), kMatrixVectorDims.end());
  const auto max_offset = *std::max_element(kOffsets.begin(), kOffsets.end());

  // Creates test input data
  x_source_.resize(std::max(max_vec, max_matvec)*max_inc + max_offset);
  y_source_.resize(std::max(max_vec, max_matvec)*max_inc + max_offset);
  a_source_.resize(std::max(max_mat, max_matvec)*std::max(max_ld, max_matvec) + max_offset);
  b_source_.resize(std::max(max_mat, max_matvec)*std::max(max_ld, max_matvec) + max_offset);
  c_source_.resize(std::max(max_mat, max_matvec)*std::max(max_ld, max_matvec) + max_offset);
  ap_source_.resize(std::max(max_mat, max_matvec)*std::max(max_mat, max_matvec) + max_offset);
  scalar_source_.resize(std::max(max_mat, max_matvec) + max_offset);
  PopulateVector(x_source_, kSeed);
  PopulateVector(y_source_, kSeed);
  PopulateVector(a_source_, kSeed);
  PopulateVector(b_source_, kSeed);
  PopulateVector(c_source_, kSeed);
  PopulateVector(ap_source_, kSeed);
  PopulateVector(scalar_source_, kSeed);
}

// ===============================================================================================

// Tests the routine for a wide variety of parameters
template <typename T, typename U>
void TestBlas<T,U>::TestRegular(std::vector<Arguments<U>> &test_vector, const std::string &name) {
  if (!PrecisionSupported<T>(device_)) { return; }
  TestStart("regular behaviour", name);

  // Iterates over all the to-be-tested combinations of arguments
  for (const auto &args: test_vector) {

    // Prints the current test configuration
    if (verbose_) {
      fprintf(stdout, "   Testing: %s", GetOptionsString(args).c_str());
      std::cout << std::flush;
    }

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
    const auto status1 = run_reference_(args, buffers1, queue_);

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
    const auto kErrorMarginL2 = getL2ErrorMargin<T>();
    auto l2error = 0.0;
    for (auto id1=size_t{0}; id1<get_id1_(args); ++id1) {
      for (auto id2=size_t{0}; id2<get_id2_(args); ++id2) {
        auto index = get_index_(args, id1, id2);
        l2error += SquaredDifference(result1[index], result2[index]);
      }
    }
    l2error /= (get_id1_(args) * get_id2_(args));

    // Checks for differences in the output
    auto errors = size_t{0};
    for (auto id1=size_t{0}; id1<get_id1_(args); ++id1) {
      for (auto id2=size_t{0}; id2<get_id2_(args); ++id2) {
        auto index = get_index_(args, id1, id2);
        if (!TestSimilarity(result1[index], result2[index])) {
          if (l2error >= kErrorMarginL2) { errors++; }
          if (verbose_) {
            if (get_id2_(args) == 1) { fprintf(stdout, "\n   Error at index %zu: ", id1); }
            else { fprintf(stdout, "\n   Error at %zu,%zu: ", id1, id2); }
            fprintf(stdout, " %s (reference) versus ", ToString(result1[index]).c_str());
            fprintf(stdout, " %s (CLBlast)", ToString(result2[index]).c_str());
            if (l2error < kErrorMarginL2) {
              fprintf(stdout, " - error suppressed by a low total L2 error\n");
            }
          }
        }
      }
    }
    if (verbose_ && errors > 0) {
      fprintf(stdout, "\n   Combined L2 error: %.2e\n   ", l2error);
    }

    // Tests the error count (should be zero)
    TestErrorCount(errors, get_id1_(args)*get_id2_(args), args);
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

    // Creates the OpenCL buffers. Note: we are not using the C++ version since we explicitly
    // want to be able to create invalid buffers (no error checking here).
    auto x1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.x_size*sizeof(T), nullptr,nullptr);
    auto y1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.y_size*sizeof(T), nullptr,nullptr);
    auto a1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.a_size*sizeof(T), nullptr,nullptr);
    auto b1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.b_size*sizeof(T), nullptr,nullptr);
    auto c1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.c_size*sizeof(T), nullptr,nullptr);
    auto ap1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.ap_size*sizeof(T), nullptr,nullptr);
    auto d1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.scalar_size*sizeof(T), nullptr,nullptr);
    auto x_vec1 = Buffer<T>(x1);
    auto y_vec1 = Buffer<T>(y1);
    auto a_mat1 = Buffer<T>(a1);
    auto b_mat1 = Buffer<T>(b1);
    auto c_mat1 = Buffer<T>(c1);
    auto ap_mat1 = Buffer<T>(ap1);
    auto scalar1 = Buffer<T>(d1);
    auto x2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.x_size*sizeof(T), nullptr,nullptr);
    auto y2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.y_size*sizeof(T), nullptr,nullptr);
    auto a2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.a_size*sizeof(T), nullptr,nullptr);
    auto b2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.b_size*sizeof(T), nullptr,nullptr);
    auto c2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.c_size*sizeof(T), nullptr,nullptr);
    auto ap2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.ap_size*sizeof(T), nullptr,nullptr);
    auto d2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.scalar_size*sizeof(T), nullptr,nullptr);
    auto x_vec2 = Buffer<T>(x2);
    auto y_vec2 = Buffer<T>(y2);
    auto a_mat2 = Buffer<T>(a2);
    auto b_mat2 = Buffer<T>(b2);
    auto c_mat2 = Buffer<T>(c2);
    auto ap_mat2 = Buffer<T>(ap2);
    auto scalar2 = Buffer<T>(d2);
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
    const auto status1 = run_reference_(args, buffers1, queue_);

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
