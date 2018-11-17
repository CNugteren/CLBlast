
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

#include "test/correctness/tester.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class TestBlas: public Tester<T,U> {
 public:
  static const int kSeed;

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
  static const std::vector<size_t> kVectorDims;
  static const std::vector<size_t> kIncrements;
  static const std::vector<size_t> kMatrixDims;
  static const std::vector<size_t> kMatrixVectorDims;
  static const std::vector<size_t> kBandSizes;
  static const std::vector<size_t> kPadSizes;
  static const std::vector<size_t> kDilationSizes;
  static const std::vector<size_t> kKernelSizes;
  static const std::vector<size_t> kBatchCounts;
  static const std::vector<size_t> kNumKernels;
  static const std::vector<size_t> kStrideValues;
  static const std::vector<size_t> kChannelValues;
  static const std::vector<KernelMode> kKernelModes;
  const std::vector<size_t> kOffsets;
  const std::vector<U> kAlphaValues;
  const std::vector<U> kBetaValues;

  // Test settings for the invalid tests
  static const std::vector<size_t> kInvalidIncrements;
  static const size_t kBufferSize;
  static const std::vector<size_t> kMatSizes;
  static const std::vector<size_t> kVecSizes;

  // The layout/transpose/triangle options to test with
  static const std::vector<Layout> kLayouts;
  static const std::vector<Triangle> kTriangles;
  static const std::vector<Side> kSides;
  static const std::vector<Diagonal> kDiagonals;
  static const std::vector<Transpose> kTransposes; // Data-type dependent, see .cpp-file

  // Shorthand for the routine-specific functions passed to the tester
  using DataPrepare = std::function<void(const Arguments<U>&, Queue&, const int,
                                         std::vector<T>&, std::vector<T>&,
                                         std::vector<T>&, std::vector<T>&, std::vector<T>&,
                                         std::vector<T>&, std::vector<T>&)>;
  using Routine = std::function<StatusCode(const Arguments<U>&, Buffers<T>&, Queue&)>;
  using ResultGet = std::function<std::vector<T>(const Arguments<U>&, Buffers<T>&, Queue&)>;
  using ResultIndex = std::function<size_t(const Arguments<U>&, const size_t, const size_t)>;
  using ResultIterator = std::function<size_t(const Arguments<U>&)>;

  // Constructor, initializes the base class tester and input data
  TestBlas(const std::vector<std::string> &arguments, const bool silent,
           const std::string &name, const std::vector<std::string> &options,
           const DataPrepare prepare_data,
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
  const DataPrepare prepare_data_;
  const Routine run_routine_;
  const Routine run_reference1_;
  const Routine run_reference2_;
  const ResultGet get_result_;
  const ResultIndex get_index_;
  const ResultIterator get_id1_;
  const ResultIterator get_id2_;
};

// =================================================================================================

template <typename T, typename U> const int TestBlas<T,U>::kSeed = 42; // fixed seed for reproducibility

// Test settings for the regular test. Append to these lists in case more tests are required.
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kVectorDims = { 7, 93, 144, 4096 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kIncrements = { 1, 2, 7 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kMatrixDims = { 7, 64 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kMatrixVectorDims = { 61, 256 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kBandSizes = { 4, 19 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kBatchCounts = { 1, 3 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kPadSizes = { 0, 1 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kDilationSizes = { 1, 2 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kKernelSizes = { 1, 3 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kNumKernels = { 1, 6 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kStrideValues = { 1, 3 };
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kChannelValues = { 1, 2 };
template <typename T, typename U> const std::vector<KernelMode> TestBlas<T,U>::kKernelModes = { KernelMode::kCrossCorrelation, KernelMode::kConvolution };

// Test settings for the invalid tests
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kInvalidIncrements = { 0, 1 };
template <typename T, typename U> const size_t TestBlas<T,U>::kBufferSize = 64;
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kMatSizes = {0, kBufferSize*kBufferSize-1, kBufferSize*kBufferSize};
template <typename T, typename U> const std::vector<size_t> TestBlas<T,U>::kVecSizes = {0, kBufferSize - 1, kBufferSize};

// The layout/triangle options to test with
template <typename T, typename U> const std::vector<Layout> TestBlas<T,U>::kLayouts = {Layout::kRowMajor, Layout::kColMajor};
template <typename T, typename U> const std::vector<Triangle> TestBlas<T,U>::kTriangles = {Triangle::kUpper, Triangle::kLower};
template <typename T, typename U> const std::vector<Side> TestBlas<T,U>::kSides = {Side::kLeft, Side::kRight};
template <typename T, typename U> const std::vector<Diagonal> TestBlas<T,U>::kDiagonals = {Diagonal::kUnit, Diagonal::kNonUnit};

// =================================================================================================

// Bogus reference function, in case a comparison library is not available
template <typename T, typename U, typename BufferType>
static StatusCode ReferenceNotAvailable(const Arguments<U> &, BufferType &, Queue &) {
  return StatusCode::kNotImplemented;
}

// Helper for the below function: MSVC's C1061 error requires part of the for-loops to be in a
// separate file. This part handles the im2col/xconv arguments.
template <typename C, typename T, typename U>
void handle_remaining_of_options(std::vector<Arguments<U>> &regular_test_vector, Arguments<U> &r_args,
                                 TestBlas<T,U> &tester,
                                 const std::vector<KernelMode> &kernel_modes,
                                 const std::vector<size_t> &channelss,
                                 const std::vector<size_t> &heights,
                                 const std::vector<size_t> &widths,
                                 const std::vector<size_t> &kernel_hs,
                                 const std::vector<size_t> &kernel_ws,
                                 const std::vector<size_t> &pad_hs,
                                 const std::vector<size_t> &pad_ws,
                                 const std::vector<size_t> &stride_hs,
                                 const std::vector<size_t> &stride_ws,
                                 const std::vector<size_t> &dilation_hs,
                                 const std::vector<size_t> &dilation_ws,
                                 const std::vector<size_t> &batch_counts,
                                 const std::vector<size_t> &num_kernelss) {
  for (auto &kernel_mode: kernel_modes) { r_args.kernel_mode = kernel_mode;
    for (auto &channels: channelss) { r_args.channels = channels;
      for (auto &height: heights) { r_args.height = height;
        for (auto &width: widths) { r_args.width = width;
          for (auto &kernel_h: kernel_hs) { r_args.kernel_h = kernel_h;
            for (auto &kernel_w: kernel_ws) { r_args.kernel_w = kernel_w;
              for (auto &pad_h: pad_hs) { r_args.pad_h = pad_h;
                for (auto &pad_w: pad_ws) { r_args.pad_w = pad_w;
                  for (auto &stride_h: stride_hs) { r_args.stride_h = stride_h;
                    for (auto &stride_w: stride_ws) { r_args.stride_w = stride_w;
                      for (auto &dilation_h: dilation_hs) { r_args.dilation_h = dilation_h;
                        for (auto &dilation_w: dilation_ws) { r_args.dilation_w = dilation_w;
                          for (auto &batch_count: batch_counts) { r_args.batch_count = batch_count;
                            for (auto &num_kernels: num_kernelss) { r_args.num_kernels = num_kernels;
                              C::SetSizes(r_args, tester.queue_);
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

// The interface to the correctness tester. This is a separate function in the header such that it
// is automatically compiled for each routine, templated by the parameter "C".
template <typename C, typename T, typename U>
size_t RunTests(int argc, char *argv[], const bool silent, const std::string &name) {
  auto command_line_args = RetrieveCommandLineArguments(argc, argv);

  // Sets the clBLAS reference to test against
  #ifdef CLBLAST_REF_CLBLAS
    auto reference_routine1 = C::RunReference1; // clBLAS when available
  #else
    auto reference_routine1 = ReferenceNotAvailable<T,U,Buffers<T>>;
  #endif

  // Sets the CBLAS reference to test against
  #ifdef CLBLAST_REF_CBLAS
    auto reference_routine2 = [](const Arguments<U> &args, Buffers<T> &buffers, Queue &queue) -> StatusCode {
      auto buffers_host = BuffersHost<T>();
      DeviceToHost(args, buffers, buffers_host, queue, C::BuffersIn());
      C::RunReference2(args, buffers_host, queue);
      HostToDevice(args, buffers, buffers_host, queue, C::BuffersOut());
      return StatusCode::kSuccess;
    };
  #else
    auto reference_routine2 = ReferenceNotAvailable<T,U,Buffers<T>>;
  #endif

  // Non-BLAS routines cannot be fully tested
  if (!silent && C::BLASLevel() == 4) {
    fprintf(stdout, "\n* NOTE: This non-BLAS routine is tested against a custom implementation,\n");
    fprintf(stdout, "  not against clBLAS or a CPU BLAS library. Thus, the arguments '-clblas'\n");
    fprintf(stdout, "  and '-cblas' have no effect.\n");
  }

  // Creates a tester
  auto options = C::GetOptions();
  TestBlas<T,U> tester{command_line_args, silent, name, options,
                       C::PrepareData, C::RunRoutine, reference_routine1, reference_routine2,
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
  auto kernel_modes = std::vector<KernelMode>{args.kernel_mode};
  auto channelss = std::vector<size_t>{args.channels};
  auto heights = std::vector<size_t>{args.height};
  auto widths = std::vector<size_t>{args.width};
  auto kernel_hs = std::vector<size_t>{args.kernel_h};
  auto kernel_ws = std::vector<size_t>{args.kernel_w};
  auto pad_hs = std::vector<size_t>{args.pad_h};
  auto pad_ws = std::vector<size_t>{args.pad_w};
  auto stride_hs = std::vector<size_t>{args.stride_h};
  auto stride_ws = std::vector<size_t>{args.stride_w};
  auto dilation_hs = std::vector<size_t>{args.dilation_h};
  auto dilation_ws = std::vector<size_t>{args.dilation_w};
  auto batch_counts = std::vector<size_t>{args.batch_count};
  auto num_kernelss = std::vector<size_t>{args.num_kernels};
  auto x_sizes = std::vector<size_t>{args.x_size};
  auto y_sizes = std::vector<size_t>{args.y_size};
  auto a_sizes = std::vector<size_t>{args.a_size};
  auto b_sizes = std::vector<size_t>{args.b_size};
  auto c_sizes = std::vector<size_t>{args.c_size};
  auto ap_sizes = std::vector<size_t>{args.ap_size};

  // Sets the dimensions of the matrices or vectors depending on the BLAS level
  auto dimensions = (C::BLASLevel() == 4) ? tester.kMatrixDims : // non-BLAS extra routines
                    (C::BLASLevel() == 3) ? tester.kMatrixDims : // level 3
                    (C::BLASLevel() == 2) ? tester.kMatrixVectorDims : // level 2
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
    if (option == kArgKernelMode) { kernel_modes = tester.kKernelModes; }
    if (option == kArgChannels) { channelss = tester.kChannelValues; }
    if (option == kArgHeight) { heights = tester.kMatrixDims; }
    if (option == kArgWidth) { widths = tester.kMatrixDims; }
    if (option == kArgKernelH) { kernel_hs = tester.kKernelSizes; }
    if (option == kArgKernelW) { kernel_ws = tester.kKernelSizes; }
    if (option == kArgPadH) { pad_hs = tester.kPadSizes; }
    if (option == kArgPadW) { pad_ws = tester.kPadSizes; }
    if (option == kArgStrideH) { stride_hs = tester.kStrideValues; }
    if (option == kArgStrideW) { stride_ws = tester.kStrideValues; }
    if (option == kArgDilationH) { dilation_hs = tester.kDilationSizes; }
    if (option == kArgDilationW) { dilation_ws = tester.kDilationSizes; }
    if (option == kArgBatchCount) { batch_counts = tester.kBatchCounts; }
    if (option == kArgNumKernels) { num_kernelss = tester.kNumKernels; }

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
                                                          // Cannot have more for-loops because of MSVC's C1061 error
                                                          handle_remaining_of_options<C>(regular_test_vector, r_args, tester,
                                                                                      kernel_modes,
                                                                                      channelss, heights, widths, kernel_hs, kernel_ws,
                                                                                      pad_hs, pad_ws, stride_hs, stride_ws,
                                                                                      dilation_hs, dilation_ws,
                                                                                      batch_counts, num_kernelss);
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
                i_args.batch_count = 3;
                i_args.alphas = std::vector<U>(i_args.batch_count);
                i_args.betas = std::vector<U>(i_args.batch_count);
                i_args.a_offsets = std::vector<size_t>(i_args.batch_count);
                i_args.b_offsets = std::vector<size_t>(i_args.batch_count);
                i_args.c_offsets = std::vector<size_t>(i_args.batch_count);
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
                if (C::BLASLevel() != 4) {
                  tester.TestInvalid(invalid_test_vector, case_name);
                }
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
