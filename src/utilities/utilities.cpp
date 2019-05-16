
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the common utility functions.
//
// =================================================================================================

#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

#include "utilities/utilities.hpp"

#include "utilities/device_mapping.hpp"

namespace clblast {
// =================================================================================================

// Returns a scalar with a default value
template <typename T> T GetScalar() { return static_cast<T>(2.0); }
template float GetScalar<float>();
template double GetScalar<double>();
template <> half GetScalar() { return FloatToHalf(2.0f); }
template <> float2 GetScalar() { return {2.0f, 0.5f}; }
template <> double2 GetScalar() { return {2.0, 0.5}; }

// Returns a scalar of value 0
template <typename T> T ConstantZero() { return static_cast<T>(0.0); }
template float ConstantZero<float>();
template double ConstantZero<double>();
template <> half ConstantZero() { return FloatToHalf(0.0f); }
template <> float2 ConstantZero() { return {0.0f, 0.0f}; }
template <> double2 ConstantZero() { return {0.0, 0.0}; }

// Returns a scalar of value 1
template <typename T> T ConstantOne() { return static_cast<T>(1.0); }
template float ConstantOne<float>();
template double ConstantOne<double>();
template <> half ConstantOne() { return FloatToHalf(1.0f); }
template <> float2 ConstantOne() { return {1.0f, 0.0f}; }
template <> double2 ConstantOne() { return {1.0, 0.0}; }

// Returns a scalar of value -1
template <typename T> T ConstantNegOne() { return static_cast<T>(-1.0); }
template float ConstantNegOne<float>();
template double ConstantNegOne<double>();
template <> half ConstantNegOne() { return FloatToHalf(-1.0f); }
template <> float2 ConstantNegOne() { return {-1.0f, 0.0f}; }
template <> double2 ConstantNegOne() { return {-1.0, 0.0}; }

// Returns a scalar of some value
template <typename T> T Constant(const double val) { return static_cast<T>(val); }
template float Constant<float>(const double);
template double Constant<double>(const double);
template <> half Constant(const double val) { return FloatToHalf(static_cast<float>(val)); }
template <> float2 Constant(const double val) { return {static_cast<float>(val), 0.0f}; }
template <> double2 Constant(const double val) { return {val, 0.0}; }

// Returns a small scalar value just larger than 0
template <typename T> T SmallConstant() { return static_cast<T>(1e-4); }
template float SmallConstant<float>();
template double SmallConstant<double>();
template <> half SmallConstant() { return FloatToHalf(1e-4f); }
template <> float2 SmallConstant() { return {1e-4f, 0.0f}; }
template <> double2 SmallConstant() { return {1e-4, 0.0}; }

// Returns the absolute value of a scalar (modulus in case of a complex number)
template <typename T> typename BaseType<T>::Type AbsoluteValue(const T value) { return std::fabs(value); }
template float AbsoluteValue<float>(const float);
template double AbsoluteValue<double>(const double);
template <> half AbsoluteValue(const half value) { return FloatToHalf(std::fabs(HalfToFloat(value))); }
template <> float AbsoluteValue(const float2 value) {
  if (value.real() == 0.0f && value.imag() == 0.0f) { return 0.0f; }
  return std::sqrt(value.real() * value.real() + value.imag() * value.imag());
}
template <> double AbsoluteValue(const double2 value) {
  if (value.real() == 0.0 && value.imag() == 0.0) { return 0.0; }
  return std::sqrt(value.real() * value.real() + value.imag() * value.imag());
}

// =================================================================================================

// Implements the string conversion using std::to_string if possible
template <typename T>
std::string ToString(T value) {
  return std::to_string(value);
}
template std::string ToString<int>(int value);
template std::string ToString<size_t>(size_t value);
template <>
std::string ToString(float value) {
  std::ostringstream result;
  result << std::fixed << std::setprecision(2) << value;
  return result.str();
}
template <>
std::string ToString(double value) {
  std::ostringstream result;
  result << std::fixed << std::setprecision(2) << value;
  return result.str();
}
template <> std::string ToString<std::string>(std::string value) { return value; }

// If not possible directly: special cases for complex data-types
template <>
std::string ToString(float2 value) {
  return ToString(value.real())+"+"+ToString(value.imag())+"i";
}
template <>
std::string ToString(double2 value) {
  return ToString(value.real())+"+"+ToString(value.imag())+"i";
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
    case Precision::kAny: return ToString(static_cast<int>(value))+" (any)";
  }
}
template <>
std::string ToString(KernelMode value) {
  switch(value) {
    case KernelMode::kCrossCorrelation: return ToString(static_cast<int>(value))+" (cross-correlation)";
    case KernelMode::kConvolution: return ToString(static_cast<int>(value))+" (convolution)";
  }
}
template <>
std::string ToString(StatusCode value) {
  return std::to_string(static_cast<int>(value));
}

// =================================================================================================

// Retrieves the command-line arguments in a C++ fashion. Also adds command-line arguments from
// pre-defined environmental variables
std::vector<std::string> RetrieveCommandLineArguments(int argc, char *argv[]) {

  // Regular command-line arguments
  auto command_line_args = std::vector<std::string>();
  for (auto i=0; i<argc; ++i) {
    command_line_args.push_back(std::string{argv[i]});
  }

  // Extra CLBlast arguments
  const auto extra_args = ConvertArgument(std::getenv("CLBLAST_ARGUMENTS"), std::string{""});
  std::stringstream extra_args_stream;
  extra_args_stream.str(extra_args);
  std::string extra_arg;
  while (std::getline(extra_args_stream, extra_arg, ' ')) {
    command_line_args.push_back(extra_arg);
  }
  return command_line_args;
}

// Helper for the below function to convert the argument to the value type. Adds specialization for
// complex data-types. Note that complex arguments are accepted as regular values and are copied to
// both the real and imaginary parts.
template <typename T>
T ConvertArgument(const char* value) {
  return static_cast<T>(std::stoi(value));
}
template size_t ConvertArgument(const char* value);

template <> std::string ConvertArgument(const char* value) {
  return std::string{value};
}
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
template std::string ConvertArgument(const char* value, std::string default_value);

// This function matches patterns in the form of "-option value" or "--option value". It returns a
// default value in case the option is not found in the argument string.
template <typename T>
T GetArgument(const std::vector<std::string> &arguments, std::string &help,
              const std::string &option, const T default_value) {

  // Parses the argument. Note that this supports both the given option (e.g. -device) and one with
  // an extra dash in front (e.g. --device).
  auto return_value = static_cast<T>(default_value);
  for (auto c=size_t{0}; c<arguments.size(); ++c) {
    auto item = arguments[c];
    if (item.compare("-"+option) == 0 || item.compare("--"+option) == 0) {
      ++c;
      return_value = ConvertArgument<T>(arguments[c].c_str());
      break;
    }
  }

  // Updates the help message and returns
  help += "    -"+option+" "+ToString(return_value)+" ";
  help += (return_value == default_value) ? "[=default]\n" : "\n";
  return return_value;
}

// Compiles the above function
template int GetArgument<int>(const std::vector<std::string>&, std::string&, const std::string&, const int);
template size_t GetArgument<size_t>(const std::vector<std::string>&, std::string&, const std::string&, const size_t);
template half GetArgument<half>(const std::vector<std::string>&, std::string&, const std::string&, const half);
template float GetArgument<float>(const std::vector<std::string>&, std::string&, const std::string&, const float);
template double GetArgument<double>(const std::vector<std::string>&, std::string&, const std::string&, const double);
template float2 GetArgument<float2>(const std::vector<std::string>&, std::string&, const std::string&, const float2);
template double2 GetArgument<double2>(const std::vector<std::string>&, std::string&, const std::string&, const double2);
template std::string GetArgument<std::string>(const std::vector<std::string>&, std::string&, const std::string&, const std::string);
template Layout GetArgument<Layout>(const std::vector<std::string>&, std::string&, const std::string&, const Layout);
template Transpose GetArgument<Transpose>(const std::vector<std::string>&, std::string&, const std::string&, const Transpose);
template Side GetArgument<Side>(const std::vector<std::string>&, std::string&, const std::string&, const Side);
template Triangle GetArgument<Triangle>(const std::vector<std::string>&, std::string&, const std::string&, const Triangle);
template Diagonal GetArgument<Diagonal>(const std::vector<std::string>&, std::string&, const std::string&, const Diagonal);
template Precision GetArgument<Precision>(const std::vector<std::string>&, std::string&, const std::string&, const Precision);
template KernelMode GetArgument<KernelMode>(const std::vector<std::string>&, std::string&, const std::string&, const KernelMode);

// =================================================================================================

// Returns only the precision argument
Precision GetPrecision(const std::vector<std::string> &arguments, const Precision default_precision) {
  auto dummy = std::string{};
  return GetArgument(arguments, dummy, kArgPrecision, default_precision);
}

// =================================================================================================

// Checks whether an argument is given. Returns true or false.
bool CheckArgument(const std::vector<std::string> &arguments, std::string &help,
                   const std::string &option) {

  // Parses the argument. Note that this supports both the given option (e.g. -device) and one with
  // an extra dash in front (e.g. --device).
  auto return_value = false;
  for (auto c=size_t{0}; c<arguments.size(); ++c) {
    auto item = arguments[c];
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

// Create a random number generator and populates a vector with samples from a random distribution
template <typename T>
void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
  for (auto &element: vector) { element = static_cast<T>(dist(mt)); }
}
template void PopulateVector<float>(std::vector<float>&, std::mt19937&, std::uniform_real_distribution<double>&);
template void PopulateVector<double>(std::vector<double>&, std::mt19937&, std::uniform_real_distribution<double>&);

// Specialized versions of the above for complex data-types
template <>
void PopulateVector(std::vector<float2> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
  for (auto &element: vector) {
    element.real(static_cast<float>(dist(mt)));
    element.imag(static_cast<float>(dist(mt)));
  }
}
template <>
void PopulateVector(std::vector<double2> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
  for (auto &element: vector) { element.real(dist(mt)); element.imag(dist(mt)); }
}

// Specialized versions of the above for half-precision
template <>
void PopulateVector(std::vector<half> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
  for (auto &element: vector) { element = FloatToHalf(static_cast<float>(dist(mt))); }
}

// =================================================================================================

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
}

// =================================================================================================

// Convert the precision enum (as integer) into bytes
size_t GetBytes(const Precision precision) {
  switch(precision) {
    case Precision::kHalf: return 2;
    case Precision::kSingle: return 4;
    case Precision::kDouble: return 8;
    case Precision::kComplexSingle: return 8;
    case Precision::kComplexDouble: return 16;
    case Precision::kAny: return -1;
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
template <> bool PrecisionSupported<double>(const Device &device) { return device.SupportsFP64(); }
template <> bool PrecisionSupported<double2>(const Device &device) { return device.SupportsFP64(); }
template <> bool PrecisionSupported<half>(const Device &device) { return device.SupportsFP16(); }

// =================================================================================================

// Retrieves the squared difference, used for example for computing the L2 error
template <typename T>
double SquaredDifference(const T val1, const T val2) {
  const auto difference = (val1 - val2);
  return static_cast<double>(difference * difference);
}

// Compiles the default case for standard data-types
template double SquaredDifference<float>(const float, const float);
template double SquaredDifference<double>(const double, const double);

// Specialisations for non-standard data-types
template <>
double SquaredDifference(const float2 val1, const float2 val2) {
  const auto real = SquaredDifference(val1.real(), val2.real());
  const auto imag = SquaredDifference(val1.imag(), val2.imag());
  return real + imag;
}
template <>
double SquaredDifference(const double2 val1, const double2 val2) {
  const auto real = SquaredDifference(val1.real(), val2.real());
  const auto imag = SquaredDifference(val1.imag(), val2.imag());
  return real + imag;
}
template <>
double SquaredDifference(const half val1, const half val2) {
  return SquaredDifference(HalfToFloat(val1), HalfToFloat(val2));
}

// =================================================================================================

// High-level info
std::string GetDeviceType(const Device& device) {
  return device.Type();
}
std::string GetDeviceVendor(const Device& device) {
  auto device_vendor = device.Vendor();

  for (auto &find_and_replace : device_mapping::kVendorNames) { // replacing to common names
    if (device_vendor == find_and_replace.first) { device_vendor = find_and_replace.second; }
  }
  return device_vendor;
}

// Mid-level info
std::string GetDeviceArchitecture(const Device& device) {
  auto device_architecture = std::string{""};
  #ifdef CUDA_API
    device_architecture = device.NVIDIAComputeCapability();
  #else
    if (device.HasExtension(kKhronosAttributesNVIDIA)) {
      device_architecture = device.NVIDIAComputeCapability();
    }
    else if (device.HasExtension(kKhronosAttributesAMD)) {
      device_architecture = device.Name(); // Name is architecture for AMD APP and AMD ROCm
    }
    // Note: no else - 'device_architecture' might be the empty string
  #endif

  for (auto &find_and_replace : device_mapping::kArchitectureNames) { // replacing to common names
    if (device_architecture == find_and_replace.first) { device_architecture = find_and_replace.second; }
  }
  return device_architecture;
}

// Lowest-level
std::string GetDeviceName(const Device& device) {
  auto device_name = std::string{""};
  if (device.HasExtension(kKhronosAttributesAMD)) {
    device_name = device.AMDBoardName();
  }
  else {
    device_name = device.Name();
  }

  for (auto &find_and_replace : device_mapping::kDeviceNames) { // replacing to common names
    if (device_name == find_and_replace.first) { device_name = find_and_replace.second; }
  }

  for (auto &removal : device_mapping::kDeviceRemovals) { // removing certain things
    if (device_name.find(removal) != std::string::npos) {
      auto start_position_to_erase = device_name.find(removal);
      device_name.erase(start_position_to_erase, removal.length());
    }
  }

  return device_name;
}

// =================================================================================================

void SetOpenCLKernelStandard(const Device &device, std::vector<std::string> &options) {
  // Inclusion of one of the following extensions needs OpenCL 1.2 kernels
  if (device.HasExtension(kKhronosIntelSubgroups)) {
    options.push_back("-cl-std=CL1.2");
  }
    // Otherwise we fall-back to the default CLBlast OpenCL 1.1
  else {
    options.push_back("-cl-std=CL1.1");
  }
}

// =================================================================================================

// Solve Bezout's identity
// a * p + b * q = r = GCD(a, b)
void EuclidGCD(int a, int b, int &p, int &q, int &r) {
  p = 0;
  q = 1;
  int p_1 = 1;
  int q_1 = 0;
  for (;;) {
    const int c = a % b;
    if (c == 0) {
      break;
    }
    const int p_2 = p_1;
    const int q_2 = q_1;
    p_1 = p;
    q_1 = q;
    p = p_2 - p_1 * (a / b);
    q = q_2 - q_1 * (a / b);
    a = b;
    b = c;
  }
  r = b;
}

// =================================================================================================
} // namespace clblast
