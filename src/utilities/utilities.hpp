
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file provides declarations for the common (test) utility functions such as a command-line
// argument parser. On top of this, it serves as the 'common' header, including the C++ OpenCL
// wrapper. These utilities are not only used for CLBlast, but also included as part of the tuners,
// the performance client and the correctness testers.
//
// =================================================================================================

#ifndef CLBLAST_UTILITIES_H_
#define CLBLAST_UTILITIES_H_

#include <string>
#include <functional>
#include <complex>
#include <random>

#include "clpp11.hpp"
#include "clblast.h"
#include "clblast_half.h"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/msvc.hpp"

namespace clblast {
// =================================================================================================

// Shorthands for complex data-types
using float2 = std::complex<float>;
using double2 = std::complex<double>;

// Khronos OpenCL extensions
const std::string kKhronosHalfPrecision = "cl_khr_fp16";
const std::string kKhronosDoublePrecision = "cl_khr_fp64";

// Catched an unknown error
constexpr auto kUnknownError = -999;

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

// The tuner-specific arguments in string form
constexpr auto kArgFraction = "fraction";

// The client-specific arguments in string form
constexpr auto kArgCompareclblas = "clblas";
constexpr auto kArgComparecblas = "cblas";
constexpr auto kArgComparecublas = "cublas";
constexpr auto kArgStepSize = "step";
constexpr auto kArgNumSteps = "num_steps";
constexpr auto kArgNumRuns = "runs";
constexpr auto kArgWarmUp = "warm_up";

// The test-specific arguments in string form
constexpr auto kArgFullTest = "full_test";
constexpr auto kArgVerbose = "verbose";

// The common arguments in string form
constexpr auto kArgPlatform = "platform";
constexpr auto kArgDevice = "device";
constexpr auto kArgPrecision = "precision";
constexpr auto kArgHelp = "h";
constexpr auto kArgQuiet = "q";
constexpr auto kArgNoAbbreviations = "no_abbrv";

// The buffer names
constexpr auto kBufVecX = "X";
constexpr auto kBufVecY = "Y";
constexpr auto kBufMatA = "A";
constexpr auto kBufMatB = "B";
constexpr auto kBufMatC = "C";
constexpr auto kBufMatAP = "AP";
constexpr auto kBufScalar = "Scalar";

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

// Returns whether a scalar is close to zero
template <typename T> bool IsCloseToZero(const T value);

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
  // Batch-specific arguments
  size_t batch_count = 1;
  std::vector<size_t> x_offsets = {0};
  std::vector<size_t> y_offsets = {0};
  std::vector<size_t> a_offsets = {0};
  std::vector<size_t> b_offsets = {0};
  std::vector<size_t> c_offsets = {0};
  std::vector<T> alphas = {ConstantOne<T>()};
  std::vector<T> betas = {ConstantOne<T>()};
  // Sizes
  size_t x_size = 1;
  size_t y_size = 1;
  size_t a_size = 1;
  size_t b_size = 1;
  size_t c_size = 1;
  size_t ap_size = 1;
  size_t scalar_size = 1;
  // Tuner-specific arguments
  double fraction = 1.0;
  // Client-specific arguments
  int compare_clblas = 1;
  int compare_cblas = 1;
  int compare_cublas = 1;
  size_t step = 1;
  size_t num_steps = 0;
  size_t num_runs = 10;
  // Common arguments
  size_t platform_id = 0;
  size_t device_id = 0;
  Precision precision = Precision::kSingle;
  bool print_help = false;
  bool silent = false;
  bool no_abbrv = false;
};

// Structure containing all possible buffers for test clients
template <typename T>
struct Buffers {
  Buffer<T> x_vec;
  Buffer<T> y_vec;
  Buffer<T> a_mat;
  Buffer<T> b_mat;
  Buffer<T> c_mat;
  Buffer<T> ap_mat;
  Buffer<T> scalar;
};
template <typename T>
struct BuffersHost {
  std::vector<T> x_vec;
  std::vector<T> y_vec;
  std::vector<T> a_mat;
  std::vector<T> b_mat;
  std::vector<T> c_mat;
  std::vector<T> ap_mat;
  std::vector<T> scalar;
};

// =================================================================================================

// Converts a value (e.g. an integer) to a string. This also covers special cases for CLBlast
// data-types such as the Layout and Transpose data-types.
template <typename T>
std::string ToString(T value);

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

// Returns a random number to be used as a seed
unsigned int GetRandomSeed();

// Test/example data lower and upper limit
constexpr auto kTestDataLowerLimit = -2.0;
constexpr auto kTestDataUpperLimit = 2.0;

// Populates a vector with random data
template <typename T>
void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist);

// =================================================================================================

// Copies buffers from the OpenCL device to the host
template <typename T, typename U>
void DeviceToHost(const Arguments<U> &args, Buffers<T> &buffers, BuffersHost<T> &buffers_host,
                  Queue &queue, const std::vector<std::string> &names);

// Copies buffers from the host to the OpenCL device
template <typename T, typename U>
void HostToDevice(const Arguments<U> &args, Buffers<T> &buffers, BuffersHost<T> &buffers_host,
                  Queue &queue, const std::vector<std::string> &names);

// =================================================================================================

// Conversion between half and single-precision
std::vector<float> HalfToFloatBuffer(const std::vector<half>& source);
void FloatToHalfBuffer(std::vector<half>& result, const std::vector<float>& source);

// As above, but now for OpenCL data-types instead of std::vectors
Buffer<float> HalfToFloatBuffer(const Buffer<half>& source, cl_command_queue queue_raw);
void FloatToHalfBuffer(Buffer<half>& result, const Buffer<float>& source, cl_command_queue queue_raw);

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
} // namespace clblast

// CLBLAST_UTILITIES_H_
#endif
