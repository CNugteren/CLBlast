
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file provides declarations for the common utility functions such as a command-line
// argument parser. On top of this, it serves as the 'common' header, including the C++ OpenCL
// wrapper.
//
// =================================================================================================

#ifndef CLBLAST_UTILITIES_H_
#define CLBLAST_UTILITIES_H_

#include <string>
#include <functional>
#include <complex>
#include <random>
#include <algorithm>
#include <iterator>

#ifdef OPENCL_API
  #include "clpp11.hpp"
  #include "clblast.h"
#elif CUDA_API
  #include "cupp11.hpp"
  #include "clblast_cuda.h"
#endif
#include "clblast_half.h"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/msvc.hpp"

namespace clblast {
// =================================================================================================

// Shorthands for half-precision
using half = unsigned short; // the 'cl_half' OpenCL type is actually an 'unsigned short'

// Shorthands for complex data-types
using float2 = std::complex<float>;
using double2 = std::complex<double>;

// Khronos OpenCL extensions
const std::string kKhronosAttributesAMD = "cl_amd_device_attribute_query";
const std::string kKhronosAttributesNVIDIA = "cl_nv_device_attribute_query";
const std::string kKhronosIntelSubgroups = "cl_intel_subgroups";

// Catched an unknown error
constexpr auto kUnknownError = -999;

// Canary size to add to buffers to check for buffer overflows
constexpr auto kCanarySize = 127;

// =================================================================================================

// The routine-specific arguments in string form
constexpr auto kArgM = "m";
constexpr auto kArgN = "n";
constexpr auto kArgK = "k";
constexpr auto kArgKL = "kl";
constexpr auto kArgKU = "ku";
constexpr auto kArgLayout = "layout";
constexpr auto kArgATransp = "transA";
constexpr auto kArgBTransp = "transB";
constexpr auto kArgSide = "side";
constexpr auto kArgTriangle = "triangle";
constexpr auto kArgDiagonal = "diagonal";
constexpr auto kArgKernelMode = "kernel_mode";
constexpr auto kArgXInc = "incx";
constexpr auto kArgYInc = "incy";
constexpr auto kArgXOffset = "offx";
constexpr auto kArgYOffset = "offy";
constexpr auto kArgALeadDim = "lda";
constexpr auto kArgBLeadDim = "ldb";
constexpr auto kArgCLeadDim = "ldc";
constexpr auto kArgAOffset = "offa";
constexpr auto kArgBOffset = "offb";
constexpr auto kArgCOffset = "offc";
constexpr auto kArgAPOffset = "offap";
constexpr auto kArgDotOffset = "offdot";
constexpr auto kArgNrm2Offset = "offnrm2";
constexpr auto kArgAsumOffset = "offasum";
constexpr auto kArgImaxOffset = "offimax";
constexpr auto kArgAlpha = "alpha";
constexpr auto kArgBeta = "beta";
constexpr auto kArgBatchCount = "batch_num";
constexpr auto kArgNumKernels = "num_kernels";

// Constants for im2col
constexpr auto kArgChannels = "channels";
constexpr auto kArgHeight = "height";
constexpr auto kArgWidth = "width";
constexpr auto kArgKernelH = "kernelh";
constexpr auto kArgKernelW = "kernelw";
constexpr auto kArgPadH = "padh";
constexpr auto kArgPadW = "padw";
constexpr auto kArgStrideH = "strideh";
constexpr auto kArgStrideW = "stridew";
constexpr auto kArgDilationH = "dilationh";
constexpr auto kArgDilationW = "dilationw";

// The tuner-specific arguments in string form
constexpr auto kArgFraction = "fraction";
constexpr auto kArgHeuristicSelection = "heuristic";
constexpr auto kArgMaxL2Norm = "max_l2_norm";
// PSO tuner-specific arguments in string form
constexpr auto kArgPsoSwarmSize = "pso_swarm_size";
constexpr auto kArgPsoInfGlobal = "pso_inf_global";
constexpr auto kArgPsoInfLocal = "pso_inf_local";
constexpr auto kArgPsoInfRandom = "pso_inf_random";
// Annealing tuner-specific arguments in string form
constexpr auto kArgAnnMaxTemp = "ann_max_temperature";

// The common arguments in string form
constexpr auto kArgPlatform = "platform";
constexpr auto kArgDevice = "device";
constexpr auto kArgPrecision = "precision";
constexpr auto kArgHelp = "h";
constexpr auto kArgQuiet = "q";
constexpr auto kArgNoAbbreviations = "no_abbrv";
constexpr auto kArgNumRuns = "runs";
constexpr auto kArgFullStatistics = "full_statistics";

// The buffer names
constexpr auto kBufVecX = "X";
constexpr auto kBufVecY = "Y";
constexpr auto kBufMatA = "A";
constexpr auto kBufMatB = "B";
constexpr auto kBufMatC = "C";
constexpr auto kBufMatAP = "AP";
constexpr auto kBufScalar = "Scalar";

// =================================================================================================

#ifdef VERBOSE
inline void log_debug(const std::string &log_string) {
  printf("[DEBUG] %s\n", log_string.c_str());
}
#else
inline void log_debug(const std::string&) { }
#endif


// =================================================================================================

// Converts a regular or complex type to it's base type (e.g. float2 to float)
template <typename T> struct BaseType { using Type = T; };
template <> struct BaseType<float2> { using Type = float; };
template <> struct BaseType<double2> { using Type = double; };

// =================================================================================================

// Returns a scalar with a default value
template <typename T> T GetScalar();

// Fixed value scalars
template <typename T> T ConstantZero();
template <typename T> T ConstantOne();
template <typename T> T ConstantNegOne();
template <typename T> T Constant(const double val);
template <typename T> T SmallConstant();

// Returns the absolute value of a scalar (modulus in case of complex numbers)
template <typename T> typename BaseType<T>::Type AbsoluteValue(const T value);

// =================================================================================================

// Structure containing all possible arguments for test clients, including their default values
template <typename T>
struct Arguments {
  // Routine-specific arguments
  size_t m = 1;
  size_t n = 1;
  size_t k = 1;
  size_t ku = 1;
  size_t kl = 1;
  Layout layout = Layout::kRowMajor;
  Transpose a_transpose = Transpose::kNo;
  Transpose b_transpose = Transpose::kNo;
  Side side = Side::kLeft;
  Triangle triangle = Triangle::kUpper;
  Diagonal diagonal = Diagonal::kUnit;
  KernelMode kernel_mode = KernelMode::kCrossCorrelation;
  size_t x_inc = 1;
  size_t y_inc = 1;
  size_t x_offset = 0;
  size_t y_offset = 0;
  size_t a_ld = 1;
  size_t b_ld = 1;
  size_t c_ld = 1;
  size_t a_offset = 0;
  size_t b_offset = 0;
  size_t c_offset = 0;
  size_t ap_offset = 0;
  size_t dot_offset = 0;
  size_t nrm2_offset = 0;
  size_t asum_offset = 0;
  size_t imax_offset = 0;
  T alpha = ConstantOne<T>();
  T beta = ConstantOne<T>();
  // Arguments for im2col and convgemm
  size_t channels = 1;
  size_t height = 1;
  size_t width = 1;
  size_t kernel_h = 3;
  size_t kernel_w = 3;
  size_t pad_h = 0;
  size_t pad_w = 0;
  size_t stride_h = 1;
  size_t stride_w = 1;
  size_t dilation_h = 1;
  size_t dilation_w = 1;
  size_t num_kernels = 1;
  // Batch-specific arguments
  size_t batch_count = 1;
  std::vector<size_t> x_offsets; // = {0};
  std::vector<size_t> y_offsets; // = {0};
  std::vector<size_t> a_offsets; // = {0};
  std::vector<size_t> b_offsets; // = {0};
  std::vector<size_t> c_offsets; // = {0};
  std::vector<T> alphas; // = {ConstantOne<T>()};
  std::vector<T> betas; // = {ConstantOne<T>()};
  // Sizes
  size_t x_size = 1;
  size_t y_size = 1;
  size_t a_size = 1;
  size_t b_size = 1;
  size_t c_size = 1;
  size_t ap_size = 1;
  size_t scalar_size = 1;
  // Tuner-specific arguments
  size_t heuristic_selection = 0;
  double fraction = 1.0;
  size_t pso_swarm_size = 8; 
  double pso_inf_global = 0.3;
  double pso_inf_local = 0.6;
  double pso_inf_random = 0.1;
  double ann_max_temperature = 1.0; // Is it a valid default value? 
  // Client-specific arguments
  int compare_clblas = 1;
  int compare_cblas = 1;
  int compare_cublas = 1;
  size_t step = 1;
  size_t num_steps = 0;
  size_t num_runs = 10;
  std::vector<std::string> tuner_files = {};
  bool full_statistics = false;
  #ifdef CLBLAST_REF_CUBLAS
    void* cublas_handle; // cublasHandle_t
  #endif
  // Common arguments
  size_t platform_id = 0;
  size_t device_id = 0;
  Precision precision = Precision::kSingle;
  bool print_help = false;
  bool silent = false;
  bool no_abbrv = false;
};

// =================================================================================================

// Converts a value (e.g. an integer) to a string. This also covers special cases for CLBlast
// data-types such as the Layout and Transpose data-types.
template <typename T>
std::string ToString(T value);

// =================================================================================================

// String splitting by a delimiter
template<typename Out>
void split(const std::string &s, char delimiter, Out result) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delimiter)) {
    *(result++) = item;
  }
}

// See above
inline std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> elements;
  split(s, delimiter, std::back_inserter(elements));
  return elements;
}

// String character removal
inline void remove_character(std::string &str, char to_be_removed) {
  str.erase(std::remove(str.begin(), str.end(), to_be_removed), str.end());
}

// =================================================================================================

// Parses command-line and environmental-variable arguments into a std::vector of strings
std::vector<std::string> RetrieveCommandLineArguments(int argc, char *argv[]);

// Helper for the function "GetArgument"
template <typename T>
T ConvertArgument(const char* value);

// Variant of "ConvertArgument" with default values
template <typename T>
T ConvertArgument(const char* value, T default_value);

// Basic argument parser, matching patterns in the form of "-option value" and "--option value"
template <typename T>
T GetArgument(const std::vector<std::string> &arguments, std::string &help,
              const std::string &option, const T default_value);

// Returns the precision only
Precision GetPrecision(const std::vector<std::string> &arguments,
                       const Precision default_precision = Precision::kSingle);

// As in "GetArgument", but now only checks whether an argument is given or not
bool CheckArgument(const std::vector<std::string> &arguments, std::string &help, const std::string &option);

// =================================================================================================

// Test/example data lower and upper limit
constexpr auto kTestDataLowerLimit = -2.0;
constexpr auto kTestDataUpperLimit = 2.0;

// Populates a vector with random data
template <typename T>
void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist);

// =================================================================================================

// Converts a 'real' value to a 'real argument' value to be passed to a kernel. Normally there is
// no conversion, but half-precision is not supported as kernel argument so it is converted to float.
template <typename T> struct RealArg { using Type = T; };
template <> struct RealArg<half> { using Type = float; };
template <typename T> typename RealArg<T>::Type GetRealArg(const T value);

// =================================================================================================

// Rounding functions
size_t CeilDiv(const size_t x, const size_t y);
size_t Ceil(const size_t x, const size_t y);

// Returns whether or not 'a' is a multiple of 'b'
bool IsMultiple(const size_t a, const size_t b);

// =================================================================================================

// Convert the precision enum into bytes, e.g. a double takes up 8 bytes
size_t GetBytes(const Precision precision);

// Convert the template argument into a precision value
template <typename T>
Precision PrecisionValue();

// =================================================================================================

// Returns false is this precision is not supported by the device
template <typename T>
bool PrecisionSupported(const Device &device);

// =================================================================================================

// Retrieves the squared difference, used for example for computing the L2 error
template <typename T>
double SquaredDifference(const T val1, const T val2);

// =================================================================================================

// Device information in a specific CLBlast form
std::string GetDeviceType(const Device& device);
std::string GetDeviceVendor(const Device& device);
std::string GetDeviceArchitecture(const Device& device);
std::string GetDeviceName(const Device& device);

// =================================================================================================

void SetOpenCLKernelStandard(const Device &device, std::vector<std::string> &options);

// =================================================================================================

// Solve Bezout's identity
// a * p + b * q = r = GCD(a, b)
void EuclidGCD(int a, int b, int &p, int &q, int &r);

// =================================================================================================
} // namespace clblast

// CLBLAST_UTILITIES_H_
#endif
