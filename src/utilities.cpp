
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the common (test) utility functions.
//
// =================================================================================================

#include "utilities.hpp"

#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

namespace clblast {
// =================================================================================================

// Returns a scalar with a default value
template <typename T>
T GetScalar() {
  return static_cast<T>(2.0);
}
template float GetScalar<float>();
template double GetScalar<double>();

// Specialized version of the above for half-precision
template <>
half GetScalar() {
  return FloatToHalf(2.0f);
}

// Specialized versions of the above for complex data-types
template <>
float2 GetScalar() {
  return {2.0f, 0.5f};
}
template <>
double2 GetScalar() {
  return {2.0, 0.5};
}

// Returns a scalar of value 1
template <typename T>
T ConstantOne() {
  return static_cast<T>(1.0);
}
template float ConstantOne<float>();
template double ConstantOne<double>();

// Specialized version of the above for half-precision
template <>
half ConstantOne() {
  return FloatToHalf(1.0f);
}

// Specialized versions of the above for complex data-types
template <>
float2 ConstantOne() {
  return {1.0f, 0.0f};
}
template <>
double2 ConstantOne() {
  return {1.0, 0.0};
}

// =================================================================================================

// Implements the string conversion using std::to_string if possible
template <typename T>
std::string ToString(T value) {
  return std::to_string(value);
}
template std::string ToString<int>(int value);
template std::string ToString<size_t>(size_t value);
template std::string ToString<float>(float value);
template std::string ToString<double>(double value);

// If not possible directly: special cases for complex data-types
template <>
std::string ToString(float2 value) {
  std::ostringstream real, imag;
  real << std::setprecision(2) << value.real();
  imag << std::setprecision(2) << value.imag();
  return real.str()+"+"+imag.str()+"i";
}
template <>
std::string ToString(double2 value) {
  std::ostringstream real, imag;
  real << std::setprecision(2) << value.real();
  imag << std::setprecision(2) << value.imag();
  return real.str()+"+"+imag.str()+"i";
}

// If not possible directly: special case for half-precision
template <>
std::string ToString(half value) {
  return std::to_string(HalfToFloat(value));
}

// If not possible directly: special cases for CLBlast data-types
template <>
std::string ToString(Layout value) {
  switch(value) {
    case Layout::kRowMajor: return ToString(static_cast<int>(value))+" (row-major)";
    case Layout::kColMajor: return ToString(static_cast<int>(value))+" (col-major)";
  }
}
template <>
std::string ToString(Transpose value) {
  switch(value) {
    case Transpose::kNo: return ToString(static_cast<int>(value))+" (regular)";
    case Transpose::kYes: return ToString(static_cast<int>(value))+" (transposed)";
    case Transpose::kConjugate: return ToString(static_cast<int>(value))+" (conjugate)";
  }
}
template <>
std::string ToString(Side value) {
  switch(value) {
    case Side::kLeft: return ToString(static_cast<int>(value))+" (left)";
    case Side::kRight: return ToString(static_cast<int>(value))+" (right)";
  }
}
template <>
std::string ToString(Triangle value) {
  switch(value) {
    case Triangle::kUpper: return ToString(static_cast<int>(value))+" (upper)";
    case Triangle::kLower: return ToString(static_cast<int>(value))+" (lower)";
  }
}
template <>
std::string ToString(Diagonal value) {
  switch(value) {
    case Diagonal::kUnit: return ToString(static_cast<int>(value))+" (unit)";
    case Diagonal::kNonUnit: return ToString(static_cast<int>(value))+" (non-unit)";
  }
}
template <>
std::string ToString(Precision value) {
  switch(value) {
    case Precision::kHalf: return ToString(static_cast<int>(value))+" (half)";
    case Precision::kSingle: return ToString(static_cast<int>(value))+" (single)";
    case Precision::kDouble: return ToString(static_cast<int>(value))+" (double)";
    case Precision::kComplexSingle: return ToString(static_cast<int>(value))+" (complex-single)";
    case Precision::kComplexDouble: return ToString(static_cast<int>(value))+" (complex-double)";
  }
}

// =================================================================================================

// Helper for the below function to convert the argument to the value type. Adds specialization for
// complex data-types. Note that complex arguments are accepted as regular values and are copied to
// both the real and imaginary parts.
template <typename T>
T ConvertArgument(const char* value) {
  return static_cast<T>(std::stoi(value));
}
template size_t ConvertArgument(const char* value);

template <> half ConvertArgument(const char* value) {
  return FloatToHalf(static_cast<float>(std::stod(value)));
}
template <> float ConvertArgument(const char* value) {
  return static_cast<float>(std::stod(value));
}
template <> double ConvertArgument(const char* value) {
  return static_cast<double>(std::stod(value));
}
template <> float2 ConvertArgument(const char* value) {
  auto val = static_cast<float>(std::stod(value));
  return float2{val, val};
}
template <> double2 ConvertArgument(const char* value) {
  auto val = static_cast<double>(std::stod(value));
  return double2{val, val};
}

// Variant of "ConvertArgument" with default values
template <typename T>
T ConvertArgument(const char* value, T default_value) {

  if (value) { return ConvertArgument<T>(value); }
  return default_value;
}
template size_t ConvertArgument(const char* value, size_t default_value);

// This function matches patterns in the form of "-option value" or "--option value". It returns a
// default value in case the option is not found in the argument string.
template <typename T>
T GetArgument(const int argc, char **argv, std::string &help,
              const std::string &option, const T default_value) {

  // Parses the argument. Note that this supports both the given option (e.g. -device) and one with
  // an extra dash in front (e.g. --device).
  auto return_value = static_cast<T>(default_value);
  for (int c=0; c<argc; ++c) {
    auto item = std::string{argv[c]};
    if (item.compare("-"+option) == 0 || item.compare("--"+option) == 0) {
      ++c;
      return_value = ConvertArgument<T>(argv[c]);
      break;
    }
  }

  // Updates the help message and returns
  help += "    -"+option+" "+ToString(return_value)+" ";
  help += (return_value == default_value) ? "[=default]\n" : "\n";
  return return_value;
}

// Compiles the above function
template int GetArgument<int>(const int, char **, std::string&, const std::string&, const int);
template size_t GetArgument<size_t>(const int, char **, std::string&, const std::string&, const size_t);
template half GetArgument<half>(const int, char **, std::string&, const std::string&, const half);
template float GetArgument<float>(const int, char **, std::string&, const std::string&, const float);
template double GetArgument<double>(const int, char **, std::string&, const std::string&, const double);
template float2 GetArgument<float2>(const int, char **, std::string&, const std::string&, const float2);
template double2 GetArgument<double2>(const int, char **, std::string&, const std::string&, const double2);
template Layout GetArgument<Layout>(const int, char **, std::string&, const std::string&, const Layout);
template Transpose GetArgument<Transpose>(const int, char **, std::string&, const std::string&, const Transpose);
template Side GetArgument<Side>(const int, char **, std::string&, const std::string&, const Side);
template Triangle GetArgument<Triangle>(const int, char **, std::string&, const std::string&, const Triangle);
template Diagonal GetArgument<Diagonal>(const int, char **, std::string&, const std::string&, const Diagonal);
template Precision GetArgument<Precision>(const int, char **, std::string&, const std::string&, const Precision);

// =================================================================================================

// Returns only the precision argument
Precision GetPrecision(const int argc, char *argv[], const Precision default_precision) {
  auto dummy = std::string{};
  return GetArgument(argc, argv, dummy, kArgPrecision, default_precision);
}

// =================================================================================================

// Checks whether an argument is given. Returns true or false.
bool CheckArgument(const int argc, char *argv[], std::string &help,
                   const std::string &option) {

  // Parses the argument. Note that this supports both the given option (e.g. -device) and one with
  // an extra dash in front (e.g. --device).
  auto return_value = false;
  for (int c=0; c<argc; ++c) {
    auto item = std::string{argv[c]};
    if (item.compare("-"+option) == 0 || item.compare("--"+option) == 0) {
      ++c;
      return_value = true;
    }
  }

  // Updates the help message and returns
  help += "    -"+option+" ";
  help += (return_value) ? "[true]\n" : "[false]\n";
  return return_value;
}

// =================================================================================================

// Returns a random seed. This used to be implemented using 'std::random_device', but that doesn't
// always work. The chrono-timers are more reliable in that sense, but perhaps less random.
unsigned int GetRandomSeed() {
  return static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
}

// Create a random number generator and populates a vector with samples from a random distribution
template <typename T>
void PopulateVector(std::vector<T> &vector, const unsigned int seed) {
  auto lower_limit = static_cast<T>(kTestDataLowerLimit);
  auto upper_limit = static_cast<T>(kTestDataUpperLimit);
  std::mt19937 mt(seed);
  std::uniform_real_distribution<T> dist(lower_limit, upper_limit);
  for (auto &element: vector) { element = dist(mt); }
}
template void PopulateVector<float>(std::vector<float>&, const unsigned int);
template void PopulateVector<double>(std::vector<double>&, const unsigned int);

// Specialized versions of the above for complex data-types
template <>
void PopulateVector(std::vector<float2> &vector, const unsigned int seed) {
  auto lower_limit = static_cast<float>(kTestDataLowerLimit);
  auto upper_limit = static_cast<float>(kTestDataUpperLimit);
  std::mt19937 mt(seed);
  std::uniform_real_distribution<float> dist(lower_limit, upper_limit);
  for (auto &element: vector) { element.real(dist(mt)); element.imag(dist(mt)); }
}
template <>
void PopulateVector(std::vector<double2> &vector, const unsigned int seed) {
  auto lower_limit = static_cast<double>(kTestDataLowerLimit);
  auto upper_limit = static_cast<double>(kTestDataUpperLimit);
  std::mt19937 mt(seed);
  std::uniform_real_distribution<double> dist(lower_limit, upper_limit);
  for (auto &element: vector) { element.real(dist(mt)); element.imag(dist(mt)); }
}

// Specialized versions of the above for half-precision
template <>
void PopulateVector(std::vector<half> &vector, const unsigned int seed) {
  const auto lower_limit = static_cast<float>(kTestDataLowerLimit);
  const auto upper_limit = static_cast<float>(kTestDataUpperLimit);
  std::mt19937 mt(seed);
  std::uniform_real_distribution<float> dist(lower_limit, upper_limit);
  for (auto &element: vector) { element = FloatToHalf(dist(mt)); }
}

// =================================================================================================

// Conversion between half and single-precision
std::vector<float> HalfToFloatBuffer(const std::vector<half>& source) {
  auto result = std::vector<float>(source.size());
  for (auto i = size_t(0); i < source.size(); ++i) { result[i] = HalfToFloat(source[i]); }
  return result;
}
void FloatToHalfBuffer(std::vector<half>& result, const std::vector<float>& source) {
  for (auto i = size_t(0); i < source.size(); ++i) { result[i] = FloatToHalf(source[i]); }
}

// As above, but now for OpenCL data-types instead of std::vectors
Buffer<float> HalfToFloatBuffer(const Buffer<half>& source, cl_command_queue queue_raw) {
  const auto size = source.GetSize() / sizeof(half);
  auto queue = Queue(queue_raw);
  auto context = queue.GetContext();
  auto source_cpu = std::vector<half>(size);
  source.Read(queue, size, source_cpu);
  auto result_cpu = HalfToFloatBuffer(source_cpu);
  auto result = Buffer<float>(context, size);
  result.Write(queue, size, result_cpu);
  return result;
}
void FloatToHalfBuffer(Buffer<half>& result, const Buffer<float>& source, cl_command_queue queue_raw) {
  const auto size = source.GetSize() / sizeof(float);
  auto queue = Queue(queue_raw);
  auto context = queue.GetContext();
  auto source_cpu = std::vector<float>(size);
  source.Read(queue, size, source_cpu);
  auto result_cpu = std::vector<half>(size);
  FloatToHalfBuffer(result_cpu, source_cpu);
  result.Write(queue, size, result_cpu);
}

// Converts a 'real' value to a 'real argument' value to be passed to a kernel. Normally there is
// no conversion, but half-precision is not supported as kernel argument so it is converted to float.
template <> typename RealArg<half>::Type GetRealArg(const half value) { return HalfToFloat(value); }
template <> typename RealArg<float>::Type GetRealArg(const float value) { return value; }
template <> typename RealArg<double>::Type GetRealArg(const double value) { return value; }
template <> typename RealArg<float2>::Type GetRealArg(const float2 value) { return value; }
template <> typename RealArg<double2>::Type GetRealArg(const double2 value) { return value; }

// =================================================================================================

// Rounding functions performing ceiling and division operations
size_t CeilDiv(const size_t x, const size_t y) {
  return 1 + ((x - 1) / y);
}
size_t Ceil(const size_t x, const size_t y) {
  return CeilDiv(x,y)*y;
}

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(const size_t a, const size_t b) {
  return ((a/b)*b == a) ? true : false;
};

// =================================================================================================

// Convert the precision enum (as integer) into bytes
size_t GetBytes(const Precision precision) {
  switch(precision) {
    case Precision::kHalf: return 2;
    case Precision::kSingle: return 4;
    case Precision::kDouble: return 8;
    case Precision::kComplexSingle: return 8;
    case Precision::kComplexDouble: return 16;
  }
}

// Convert the template argument into a precision value
template <> Precision PrecisionValue<half>() { return Precision::kHalf; }
template <> Precision PrecisionValue<float>() { return Precision::kSingle; }
template <> Precision PrecisionValue<double>() { return Precision::kDouble; }
template <> Precision PrecisionValue<float2>() { return Precision::kComplexSingle; }
template <> Precision PrecisionValue<double2>() { return Precision::kComplexDouble; }

// =================================================================================================

// Returns false is this precision is not supported by the device
template <> bool PrecisionSupported<float>(const Device &) { return true; }
template <> bool PrecisionSupported<float2>(const Device &) { return true; }
template <> bool PrecisionSupported<double>(const Device &device) {
  auto extensions = device.Capabilities();
  return (extensions.find(kKhronosDoublePrecision) == std::string::npos) ? false : true;
}
template <> bool PrecisionSupported<double2>(const Device &device) {
  auto extensions = device.Capabilities();
  return (extensions.find(kKhronosDoublePrecision) == std::string::npos) ? false : true;
}
template <> bool PrecisionSupported<half>(const Device &device) {
  auto extensions = device.Capabilities();
  if (device.Name() == "Mali-T628") { return true; } // supports fp16 but not cl_khr_fp16 officially
  return (extensions.find(kKhronosHalfPrecision) == std::string::npos) ? false : true;
}

// =================================================================================================
} // namespace clblast
