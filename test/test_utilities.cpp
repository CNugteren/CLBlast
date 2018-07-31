
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the test utility functions.
//
// =================================================================================================

#include <string>
#include <vector>
#include <cctype>
#include <algorithm>

#include "test/test_utilities.hpp"

namespace clblast {
// =================================================================================================

// Returns whether a scalar is close to zero
template <typename T> bool IsCloseToZero(const T value) { return (value > -SmallConstant<T>()) && (value < SmallConstant<T>()); }
template bool IsCloseToZero<float>(const float);
template bool IsCloseToZero<double>(const double);
template <> bool IsCloseToZero(const half value) { return IsCloseToZero(HalfToFloat(value)); }
template <> bool IsCloseToZero(const float2 value) { return IsCloseToZero(value.real()) || IsCloseToZero(value.imag()); }
template <> bool IsCloseToZero(const double2 value) { return IsCloseToZero(value.real()) || IsCloseToZero(value.imag()); }

// =================================================================================================

// Performs a complex conjugate if complex
template <typename T> T ComplexConjugate(const T value) { return value; }
template half ComplexConjugate(const half);
template float ComplexConjugate(const float);
template double ComplexConjugate(const double);
template <> float2 ComplexConjugate(const float2 value) { return float2{value.real(), -value.imag()}; }
template <> double2 ComplexConjugate(const double2 value) { return double2{value.real(), -value.imag()}; }

// =================================================================================================

template <typename T, typename U>
void DeviceToHost(const Arguments<U> &args, Buffers<T> &buffers, BuffersHost<T> &buffers_host,
                  Queue &queue, const std::vector<std::string> &names) {
  for (auto &name: names) {
    if (name == kBufVecX) {buffers_host.x_vec = std::vector<T>(args.x_size, static_cast<T>(0)); buffers.x_vec.Read(queue, args.x_size, buffers_host.x_vec); }
    else if (name == kBufVecY) { buffers_host.y_vec = std::vector<T>(args.y_size, static_cast<T>(0)); buffers.y_vec.Read(queue, args.y_size, buffers_host.y_vec); }
    else if (name == kBufMatA) { buffers_host.a_mat = std::vector<T>(args.a_size, static_cast<T>(0)); buffers.a_mat.Read(queue, args.a_size, buffers_host.a_mat); }
    else if (name == kBufMatB) { buffers_host.b_mat = std::vector<T>(args.b_size, static_cast<T>(0)); buffers.b_mat.Read(queue, args.b_size, buffers_host.b_mat); }
    else if (name == kBufMatC) { buffers_host.c_mat = std::vector<T>(args.c_size, static_cast<T>(0)); buffers.c_mat.Read(queue, args.c_size, buffers_host.c_mat); }
    else if (name == kBufMatAP) { buffers_host.ap_mat = std::vector<T>(args.ap_size, static_cast<T>(0)); buffers.ap_mat.Read(queue, args.ap_size, buffers_host.ap_mat); }
    else if (name == kBufScalar) { buffers_host.scalar = std::vector<T>(args.scalar_size, static_cast<T>(0)); buffers.scalar.Read(queue, args.scalar_size, buffers_host.scalar); }
    else { throw std::runtime_error("Invalid buffer name"); }
  }
}

template <typename T, typename U>
void HostToDevice(const Arguments<U> &args, Buffers<T> &buffers, BuffersHost<T> &buffers_host,
                  Queue &queue, const std::vector<std::string> &names) {
  for (auto &name: names) {
    if (name == kBufVecX) { buffers.x_vec.Write(queue, args.x_size, buffers_host.x_vec); }
    else if (name == kBufVecY) { buffers.y_vec.Write(queue, args.y_size, buffers_host.y_vec); }
    else if (name == kBufMatA) { buffers.a_mat.Write(queue, args.a_size, buffers_host.a_mat); }
    else if (name == kBufMatB) { buffers.b_mat.Write(queue, args.b_size, buffers_host.b_mat); }
    else if (name == kBufMatC) { buffers.c_mat.Write(queue, args.c_size, buffers_host.c_mat); }
    else if (name == kBufMatAP) { buffers.ap_mat.Write(queue, args.ap_size, buffers_host.ap_mat); }
    else if (name == kBufScalar) { buffers.scalar.Write(queue, args.scalar_size, buffers_host.scalar); }
    else { throw std::runtime_error("Invalid buffer name"); }
  }
}

// Compiles the above functions
template void DeviceToHost(const Arguments<half>&, Buffers<half>&, BuffersHost<half>&, Queue&, const std::vector<std::string>&);
template void DeviceToHost(const Arguments<float>&, Buffers<float>&, BuffersHost<float>&, Queue&, const std::vector<std::string>&);
template void DeviceToHost(const Arguments<double>&, Buffers<double>&, BuffersHost<double>&, Queue&, const std::vector<std::string>&);
template void DeviceToHost(const Arguments<float>&, Buffers<float2>&, BuffersHost<float2>&, Queue&, const std::vector<std::string>&);
template void DeviceToHost(const Arguments<double>&, Buffers<double2>&, BuffersHost<double2>&, Queue&, const std::vector<std::string>&);
template void DeviceToHost(const Arguments<float2>&, Buffers<float2>&, BuffersHost<float2>&, Queue&, const std::vector<std::string>&);
template void DeviceToHost(const Arguments<double2>&, Buffers<double2>&, BuffersHost<double2>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<half>&, Buffers<half>&, BuffersHost<half>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<float>&, Buffers<float>&, BuffersHost<float>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<double>&, Buffers<double>&, BuffersHost<double>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<float>&, Buffers<float2>&, BuffersHost<float2>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<double>&, Buffers<double2>&, BuffersHost<double2>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<float2>&, Buffers<float2>&, BuffersHost<float2>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<double2>&, Buffers<double2>&, BuffersHost<double2>&, Queue&, const std::vector<std::string>&);

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
#ifdef OPENCL_API
  Buffer<float> HalfToFloatBuffer(const Buffer<half>& source, RawCommandQueue queue_raw) {
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
  void FloatToHalfBuffer(Buffer<half>& result, const Buffer<float>& source, RawCommandQueue queue_raw) {
    const auto size = source.GetSize() / sizeof(float);
    auto queue = Queue(queue_raw);
    auto context = queue.GetContext();
    auto source_cpu = std::vector<float>(size);
    source.Read(queue, size, source_cpu);
    auto result_cpu = std::vector<half>(size);
    FloatToHalfBuffer(result_cpu, source_cpu);
    result.Write(queue, size, result_cpu);
  }
#endif

// =================================================================================================

void OverrideParametersFromJSONFiles(const std::vector<std::string>& file_names,
                                     const RawDeviceID device, const Precision precision) {

  // Retrieves the best parameters for each file from disk
  BestParametersCollection all_parameters;
  for (const auto json_file_name : file_names) {
    GetBestParametersFromJSONFile(json_file_name, all_parameters, precision);
  }

  // Applies the parameter override
  for (const auto &best_parameters : all_parameters) {
    const auto kernel_family = best_parameters.first;
    const auto parameters = best_parameters.second;
    const auto status = OverrideParameters(device, kernel_family, precision, parameters);
    if (status == StatusCode::kSuccess) {
      fprintf(stdout, "* Applying parameter override successfully for '%s'\n",
              kernel_family.c_str());
    } else {
      fprintf(stdout, "* Error while applying parameter override for '%s'\n",
              kernel_family.c_str());
    }
  }

  if (file_names.size() > 0) {
    fprintf(stdout, "\n");
  }
}

void GetBestParametersFromJSONFile(const std::string& file_name,
                                   BestParametersCollection& all_parameters,
                                   const Precision precision) {

  std::ifstream json_file(file_name);
  if (!json_file) {
    fprintf(stdout, "* Could not open file '%s'\n", file_name.c_str());
    return;
  }

  fprintf(stdout, "* Reading override-parameters from '%s'\n", file_name.c_str());
  std::string line;
  auto kernel_family = std::string{};
  while (std::getline(json_file, line)) {
    const auto line_split = split(line, ':');
    if (line_split.size() != 2) { continue; }

    // Retrieves the kernel name
    if (line_split[0] == "  \"kernel_family\"") {
      const auto value_split = split(line_split[1], '\"');
      if (value_split.size() != 3) { break; }
      kernel_family = value_split[1];
      kernel_family[0] = toupper(kernel_family[0]);  // because of a tuner - database naming mismatch
      kernel_family.erase(std::remove(kernel_family.begin(), kernel_family.end(), '_'), kernel_family.end());
      kernel_family.erase(std::remove(kernel_family.begin(), kernel_family.end(), '1'), kernel_family.end());
      kernel_family.erase(std::remove(kernel_family.begin(), kernel_family.end(), '2'), kernel_family.end());
      kernel_family.erase(std::remove(kernel_family.begin(), kernel_family.end(), '3'), kernel_family.end());
      if (kernel_family == "Xgemmdirect") { kernel_family = "XgemmDirect"; } // more kinds of mismatches
    }

    // Retrieves the best-parameters and sets the override
    if (line_split[0] == "  \"best_parameters\"" && kernel_family != std::string{""}) {
      const auto value_split = split(line_split[1], '\"');
      if (value_split.size() != 3) { break; }
      const auto config_split = split(value_split[1], ' ');
      if (config_split.size() == 0) { break; }

      // Loads an existing list of parameters for this kernel family (if present)
      BestParameters parameters;
      if (all_parameters.count(kernel_family) == 1) {
        parameters = all_parameters.at(kernel_family);
      }

      // Creates the list of parameters
      fprintf(stdout, "* Found parameters for kernel '%s': { ", kernel_family.c_str());
      for (const auto config : config_split) {
        const auto params_split = split(config, '=');
        if (params_split.size() != 2) { break; }
        const auto parameter_name = params_split[0];
        const auto parameter_value = static_cast<size_t>(std::stoi(params_split[1].c_str()));
        if (parameter_name != "PRECISION") {
          printf("%s=%zu ", parameter_name.c_str(), parameter_value);
          parameters[parameter_name] = parameter_value;
        }
        else {
          if (static_cast<size_t>(precision) != parameter_value) {
            fprintf(stdout, "ERROR! }\n");
            fprintf(stdout, "* Precision is not matching, continuing\n");
            json_file.close();
            return;
          }
        }
      }
      fprintf(stdout, "}\n");

      // Sets the new (possibly extended) parameter map as the final result
      all_parameters[kernel_family] = parameters;
      json_file.close();
      return;
    }
  }

  // Ends this function (failure)
  fprintf(stdout, "* Failed to extract parameters from this file, continuing\n");
  json_file.close();
}

// =================================================================================================
} // namespace clblast
