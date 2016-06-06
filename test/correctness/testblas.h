
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file tests any CLBlast routine. It contains two types of tests: one testing all sorts of
// input combinations, and one deliberatly testing with invalid values.
// Typename T: the data-type of the routine's memory buffers (==precision)
// Typename U: the data-type of the alpha and beta arguments
//
// =================================================================================================

#ifndef CLBLAST_TEST_CORRECTNESS_TESTBLAS_H_
#define CLBLAST_TEST_CORRECTNESS_TESTBLAS_H_

#include <vector>
#include <string>
#include <algorithm>

#include "correctness/tester.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class TestBlas: public Tester<T,U> {
 public:

  // Uses several variables from the Tester class
  using Tester<T,U>::context_;
  using Tester<T,U>::queue_;
  using Tester<T,U>::full_test_;
  using Tester<T,U>::verbose_;
  using Tester<T,U>::device_;
  using Tester<T,U>::compare_clblas_;
  using Tester<T,U>::compare_cblas_;

  // Uses several helper functions from the Tester class
  using Tester<T,U>::TestStart;
  using Tester<T,U>::TestEnd;
  using Tester<T,U>::TestErrorCount;
  using Tester<T,U>::TestErrorCodes;
  using Tester<T,U>::GetOffsets;
  using Tester<T,U>::GetOptionsString;
  using Tester<T,U>::GetSizesString;

  // Test settings for the regular test. Append to these lists in case more tests are required.
  const std::vector<size_t> kVectorDims = { 7, 93, 4096 };
  const std::vector<size_t> kIncrements = { 1, 2, 7 };
  const std::vector<size_t> kMatrixDims = { 7, 64 };
  const std::vector<size_t> kMatrixVectorDims = { 61, 512 };
  const std::vector<size_t> kBandSizes = { 4, 19 };
  const std::vector<size_t> kOffsets = GetOffsets();
  const std::vector<U> kAlphaValues = GetExampleScalars<U>(full_test_);
  const std::vector<U> kBetaValues = GetExampleScalars<U>(full_test_);

  // Test settings for the invalid tests
  const std::vector<size_t> kInvalidIncrements = { 0, 1 };
  const size_t kBufferSize = 64;
  const std::vector<size_t> kMatSizes = {0, kBufferSize*kBufferSize-1, kBufferSize*kBufferSize};
  const std::vector<size_t> kVecSizes = {0, kBufferSize - 1, kBufferSize};

  // The layout/transpose/triangle options to test with
  const std::vector<Layout> kLayouts = {Layout::kRowMajor, Layout::kColMajor};
  const std::vector<Triangle> kTriangles = {Triangle::kUpper, Triangle::kLower};
  const std::vector<Side> kSides = {Side::kLeft, Side::kRight};
  const std::vector<Diagonal> kDiagonals = {Diagonal::kUnit, Diagonal::kNonUnit};
  static const std::vector<Transpose> kTransposes; // Data-type dependent, see .cc-file

  // Shorthand for the routine-specific functions passed to the tester
  using Routine = std::function<StatusCode(const Arguments<U>&, Buffers<T>&, Queue&)>;
  using ResultGet = std::function<std::vector<T>(const Arguments<U>&, Buffers<T>&, Queue&)>;
  using ResultIndex = std::function<size_t(const Arguments<U>&, const size_t, const size_t)>;
  using ResultIterator = std::function<size_t(const Arguments<U>&)>;

  // Constructor, initializes the base class tester and input data
  TestBlas(int argc, char *argv[], const bool silent,
           const std::string &name, const std::vector<std::string> &options,
           const Routine run_routine,
           const Routine run_reference1, const Routine run_reference2,
           const ResultGet get_result, const ResultIndex get_index,
           const ResultIterator get_id1, const ResultIterator get_id2);

  // The test functions, taking no inputs
  void TestRegular(std::vector<Arguments<U>> &test_vector, const std::string &name);
  void TestInvalid(std::vector<Arguments<U>> &test_vector, const std::string &name);

 private:

  // Source data to test with
  std::vector<T> x_source_;
  std::vector<T> y_source_;
  std::vector<T> a_source_;
  std::vector<T> b_source_;
  std::vector<T> c_source_;
  std::vector<T> ap_source_;
  std::vector<T> scalar_source_;
  
  // The routine-specific functions passed to the tester
  Routine run_routine_;
  Routine run_reference_;
  ResultGet get_result_;
  ResultIndex get_index_;
  ResultIterator get_id1_;
  ResultIterator get_id2_;
};

// =================================================================================================

// The interface to the correctness tester. This is a separate function in the header such that it
// is automatically compiled for each routine, templated by the parameter "C".
template <typename C, typename T, typename U>
size_t RunTests(int argc, char *argv[], const bool silent, const std::string &name) {

  // Sets the reference to test against
  #if defined(CLBLAST_REF_CLBLAS) && defined(CLBLAST_REF_CBLAS)
    const auto reference_routine1 = C::RunReference1; // clBLAS
    const auto reference_routine2 = C::RunReference2; // CBLAS
  #elif CLBLAST_REF_CLBLAS
    const auto reference_routine1 = C::RunReference1; // clBLAS
    const auto reference_routine2 = C::RunReference1; // not used, dummy
  #elif CLBLAST_REF_CBLAS
    const auto reference_routine1 = C::RunReference2; // not used, dummy
    const auto reference_routine2 = C::RunReference2; // CBLAS
  #endif

  // Creates a tester
  auto options = C::GetOptions();
  TestBlas<T,U> tester{argc, argv, silent, name, options,
                       C::RunRoutine, reference_routine1, reference_routine2,
                       C::DownloadResult, C::GetResultIndex, C::ResultID1, C::ResultID2};

  // This variable holds the arguments relevant for this routine
  auto args = Arguments<U>{};

  // Initializes the vectors with a single element. If this particular option is relevant for this
  // routine, this vector is overridden. Otherwise, it is unused - the value here does not matter.
  auto ms = std::vector<size_t>{args.m};
  auto ns = std::vector<size_t>{args.n};
  auto ks = std::vector<size_t>{args.k};
  auto kus = std::vector<size_t>{args.ku};
  auto kls = std::vector<size_t>{args.kl};
  auto layouts = std::vector<Layout>{args.layout};
  auto a_transposes = std::vector<Transpose>{args.a_transpose};
  auto b_transposes = std::vector<Transpose>{args.b_transpose};
  auto sides = std::vector<Side>{args.side};
  auto triangles = std::vector<Triangle>{args.triangle};
  auto diagonals = std::vector<Diagonal>{args.diagonal};
  auto x_incs = std::vector<size_t>{args.x_inc};
  auto y_incs = std::vector<size_t>{args.y_inc};
  auto x_offsets = std::vector<size_t>{args.x_offset};
  auto y_offsets = std::vector<size_t>{args.y_offset};
  auto a_lds = std::vector<size_t>{args.a_ld};
  auto b_lds = std::vector<size_t>{args.b_ld};
  auto c_lds = std::vector<size_t>{args.c_ld};
  auto a_offsets = std::vector<size_t>{args.a_offset};
  auto b_offsets = std::vector<size_t>{args.b_offset};
  auto c_offsets = std::vector<size_t>{args.c_offset};
  auto ap_offsets = std::vector<size_t>{args.ap_offset};
  auto dot_offsets = std::vector<size_t>{args.dot_offset};
  auto nrm2_offsets = std::vector<size_t>{args.nrm2_offset};
  auto asum_offsets = std::vector<size_t>{args.asum_offset};
  auto imax_offsets = std::vector<size_t>{args.imax_offset};
  auto alphas = std::vector<U>{args.alpha};
  auto betas = std::vector<U>{args.beta};
  auto x_sizes = std::vector<size_t>{args.x_size};
  auto y_sizes = std::vector<size_t>{args.y_size};
  auto a_sizes = std::vector<size_t>{args.a_size};
  auto b_sizes = std::vector<size_t>{args.b_size};
  auto c_sizes = std::vector<size_t>{args.c_size};
  auto ap_sizes = std::vector<size_t>{args.ap_size};

  // Sets the dimensions of the matrices or vectors depending on the BLAS level
  auto dimensions = (C::BLASLevel() == 3) ? tester.kMatrixDims :
                    (C::BLASLevel() == 2) ? tester.kMatrixVectorDims :
                    tester.kVectorDims; // else: level 1

  // For the options relevant to this routine, sets the vectors to proper values
  for (auto &option: options) {
    if (option == kArgM) { ms = dimensions; }
    if (option == kArgN) { ns = dimensions; }
    if (option == kArgK) { ks = dimensions; }
    if (option == kArgKU) { kus = tester.kBandSizes; }
    if (option == kArgKL) { kls = tester.kBandSizes; }
    if (option == kArgLayout) { layouts = tester.kLayouts; }
    if (option == kArgATransp) { a_transposes = C::GetATransposes(tester.kTransposes); }
    if (option == kArgBTransp) { b_transposes = C::GetBTransposes(tester.kTransposes); }
    if (option == kArgSide) { sides = tester.kSides; }
    if (option == kArgTriangle) { triangles = tester.kTriangles; }
    if (option == kArgDiagonal) { diagonals = tester.kDiagonals; }
    if (option == kArgXInc) { x_incs = tester.kIncrements; }
    if (option == kArgYInc) { y_incs = tester.kIncrements; }
    if (option == kArgXOffset) { x_offsets = tester.kOffsets; }
    if (option == kArgYOffset) { y_offsets = tester.kOffsets; }
    if (option == kArgALeadDim) { a_lds = dimensions; }
    if (option == kArgBLeadDim) { b_lds = dimensions; }
    if (option == kArgCLeadDim) { c_lds = dimensions; }
    if (option == kArgAOffset) { a_offsets = tester.kOffsets; }
    if (option == kArgBOffset) { b_offsets = tester.kOffsets; }
    if (option == kArgCOffset) { c_offsets = tester.kOffsets; }
    if (option == kArgAPOffset) { ap_offsets = tester.kOffsets; }
    if (option == kArgDotOffset) { dot_offsets = tester.kOffsets; }
    if (option == kArgNrm2Offset) { nrm2_offsets = tester.kOffsets; }
    if (option == kArgAsumOffset) { asum_offsets = tester.kOffsets; }
    if (option == kArgImaxOffset) { imax_offsets = tester.kOffsets; }
    if (option == kArgAlpha) { alphas = tester.kAlphaValues; }
    if (option == kArgBeta) { betas = tester.kBetaValues; }

    if (option == kArgXOffset) { x_sizes = tester.kVecSizes; }
    if (option == kArgYOffset) { y_sizes = tester.kVecSizes; }
    if (option == kArgAOffset) { a_sizes = tester.kMatSizes; }
    if (option == kArgBOffset) { b_sizes = tester.kMatSizes; }
    if (option == kArgCOffset) { c_sizes = tester.kMatSizes; }
    if (option == kArgAPOffset) { ap_sizes = tester.kMatSizes; }
  }

  // Loops over the test-cases from a data-layout point of view
  for (auto &layout: layouts) { args.layout = layout;
    for (auto &a_transpose: a_transposes) { args.a_transpose = a_transpose;
      for (auto &b_transpose: b_transposes) { args.b_transpose = b_transpose;
        for (auto &side: sides) { args.side = side;
          for (auto &triangle: triangles) { args.triangle = triangle;
            for (auto &diagonal: diagonals) { args.diagonal = diagonal;

              // Creates the arguments vector for the regular tests
              auto regular_test_vector = std::vector<Arguments<U>>{};
              auto r_args = args;
              for (auto &m: ms) { r_args.m = m;
                for (auto &n: ns) { r_args.n = n;
                  for (auto &k: ks) { r_args.k = k;
                    for (auto &ku: kus) { r_args.ku = ku;
                      for (auto &kl: kls) { r_args.kl = kl;
                        for (auto &x_inc: x_incs) { r_args.x_inc = x_inc;
                          for (auto &x_offset: x_offsets) { r_args.x_offset = x_offset;
                            for (auto &y_inc: y_incs) { r_args.y_inc = y_inc;
                              for (auto &y_offset: y_offsets) { r_args.y_offset = y_offset;
                                for (auto &a_ld: a_lds) { r_args.a_ld = a_ld;
                                  for (auto &a_offset: a_offsets) { r_args.a_offset = a_offset;
                                    for (auto &b_ld: b_lds) { r_args.b_ld = b_ld;
                                      for (auto &b_offset: b_offsets) { r_args.b_offset = b_offset;
                                        for (auto &c_ld: c_lds) { r_args.c_ld = c_ld;
                                          for (auto &c_offset: c_offsets) { r_args.c_offset = c_offset;
                                            for (auto &ap_offset: ap_offsets) { r_args.ap_offset = ap_offset;
                                              for (auto &dot_offset: dot_offsets) { r_args.dot_offset = dot_offset;
                                                for (auto &nrm2_offset: nrm2_offsets) { r_args.nrm2_offset = nrm2_offset;
                                                  for (auto &asum_offset: asum_offsets) { r_args.asum_offset = asum_offset;
                                                    for (auto &imax_offset: imax_offsets) { r_args.imax_offset = imax_offset;
                                                      for (auto &alpha: alphas) { r_args.alpha = alpha;
                                                        for (auto &beta: betas) { r_args.beta = beta;
                                                          C::SetSizes(r_args);
                                                          regular_test_vector.push_back(r_args);
                                                        }
                                                      }
                                                    }
                                                  }
                                                }
                                              }
                                            }
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }

              // Creates the arguments vector for the invalid-buffer tests
              #ifdef CLBLAST_REF_CLBLAS
                auto invalid_test_vector = std::vector<Arguments<U>>{};
                auto i_args = args;
                i_args.m = i_args.n = i_args.k = i_args.kl = i_args.ku = tester.kBufferSize;
                i_args.a_ld = i_args.b_ld = i_args.c_ld = tester.kBufferSize;
                for (auto &x_size: x_sizes) { i_args.x_size = x_size;
                  for (auto &y_size: y_sizes) { i_args.y_size = y_size;
                    for (auto &a_size: a_sizes) { i_args.a_size = a_size;
                      for (auto &b_size: b_sizes) { i_args.b_size = b_size;
                        for (auto &c_size: c_sizes) { i_args.c_size = c_size;
                          for (auto &ap_size: ap_sizes) { i_args.ap_size = ap_size;
                            invalid_test_vector.push_back(i_args);
                          }
                        }
                      }
                    }
                  }
                }
              #endif

              // Sets the name of this test-case
              auto names = std::vector<std::string>{};
              for (auto &option: options) {
                if (option == kArgLayout) { names.push_back(ToString(layout)); }
                if (option == kArgATransp) { names.push_back(ToString(a_transpose)); }
                if (option == kArgBTransp) { names.push_back(ToString(b_transpose)); }
                if (option == kArgSide) { names.push_back(ToString(side)); }
                if (option == kArgTriangle) { names.push_back(ToString(triangle)); }
                if (option == kArgDiagonal) { names.push_back(ToString(diagonal)); }
              }
              if (names.size() == 0) { names.push_back("default"); }
              auto case_name = std::string{};
              for (auto i=size_t{0}; i<names.size(); ++i) {
                case_name += names[i];
                if (i != names.size()-1) { case_name += " "; }
              }

              // Runs the tests
              tester.TestRegular(regular_test_vector, case_name);
              #ifdef CLBLAST_REF_CLBLAS
                tester.TestInvalid(invalid_test_vector, case_name);
              #endif
            }
          }
        }
      }
    }
  }
  return tester.NumFailedTests();
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_CORRECTNESS_TESTBLAS_H_
#endif
