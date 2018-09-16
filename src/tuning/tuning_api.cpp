
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the parameter configurations for the CLBlast auto-tuner (taken from CLTune).
// This is only used for the optional tuner binaries and not part of the core of CLBlast.
//
// =================================================================================================

#include <vector>
#include <string>
#include <random>
#include <utility>
#include <algorithm>

#include "tuning/tuning.hpp"
#include "tuning/kernels/xaxpy.hpp"
#include "tuning/kernels/xdot.hpp"
#include "tuning/kernels/xgemv.hpp"
#include "tuning/kernels/xger.hpp"
#include "tuning/kernels/xgemm.hpp"
#include "tuning/kernels/xgemm_direct.hpp"
#include "tuning/kernels/copy_fast.hpp"
#include "tuning/kernels/copy_pad.hpp"
#include "tuning/kernels/transpose_fast.hpp"
#include "tuning/kernels/transpose_pad.hpp"
#include "tuning/kernels/invert.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
StatusCode TuneXaxpy(RawCommandQueue * queue, const size_t n,
                     const double fraction, std::unordered_map<std::string,size_t> &parameters) {
  auto args = Arguments<T>(); args.fraction = fraction; args.n = n;
  auto queue_cpp = Queue(*queue);
  return TunerAPI<T>(queue_cpp, args, 0, XaxpyGetTunerDefaults, XaxpyGetTunerSettings<T>,
                     XaxpyTestValidArguments<T>, XaxpySetConstraints, XaxpyComputeLocalMemSize<T>, XaxpySetArguments<T>, parameters);
}
template StatusCode PUBLIC_API TuneXaxpy<half>(RawCommandQueue*, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXaxpy<float>(RawCommandQueue*, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXaxpy<double>(RawCommandQueue*, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXaxpy<float2>(RawCommandQueue*, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXaxpy<double2>(RawCommandQueue*, const size_t, const double, std::unordered_map<std::string,size_t>&);

template <typename T>
StatusCode TuneXdot(RawCommandQueue * queue, const size_t n,
                    const double fraction, std::unordered_map<std::string,size_t> &parameters) {
  auto args = Arguments<T>(); args.fraction = fraction; args.n = n;
  auto queue_cpp = Queue(*queue);
  auto status = TunerAPI<T>(queue_cpp, args, 1, XdotGetTunerDefaults, XdotGetTunerSettings<T>,
                            XdotTestValidArguments<T>, XdotSetConstraints, XdotComputeLocalMemSize<T>, XdotSetArguments<T>, parameters);
  if (status != StatusCode::kSuccess) { return status; }
  return TunerAPI<T>(queue_cpp, args, 2, XdotGetTunerDefaults, XdotGetTunerSettings<T>,
                     XdotTestValidArguments<T>, XdotSetConstraints, XdotComputeLocalMemSize<T>, XdotSetArguments<T>, parameters);
}
template StatusCode PUBLIC_API TuneXdot<half>(RawCommandQueue*, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXdot<float>(RawCommandQueue*, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXdot<double>(RawCommandQueue*, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXdot<float2>(RawCommandQueue*, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXdot<double2>(RawCommandQueue*, const size_t, const double, std::unordered_map<std::string,size_t>&);

template <typename T>
StatusCode TuneXgemv(RawCommandQueue * queue, const size_t m, const size_t n,
                     const double fraction, std::unordered_map<std::string,size_t> &parameters) {
  auto args = Arguments<T>(); args.fraction = fraction; args.m = m; args.n = n;
  auto queue_cpp = Queue(*queue);
  auto status = TunerAPI<T>(queue_cpp, args, 1, XgemvGetTunerDefaults, XgemvGetTunerSettings<T>,
                            XgemvTestValidArguments<T>, XgemvSetConstraints, XgemvComputeLocalMemSize<T>, XgemvSetArguments<T>, parameters);
  if (status != StatusCode::kSuccess) { return status; }
  status = TunerAPI<T>(queue_cpp, args, 2, XgemvGetTunerDefaults, XgemvGetTunerSettings<T>,
                       XgemvTestValidArguments<T>, XgemvSetConstraints, XgemvComputeLocalMemSize<T>, XgemvSetArguments<T>, parameters);
  if (status != StatusCode::kSuccess) { return status; }
  return TunerAPI<T>(queue_cpp, args, 3, XgemvGetTunerDefaults, XgemvGetTunerSettings<T>,
                     XgemvTestValidArguments<T>, XgemvSetConstraints, XgemvComputeLocalMemSize<T>, XgemvSetArguments<T>, parameters);
}
template StatusCode PUBLIC_API TuneXgemv<half>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXgemv<float>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXgemv<double>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXgemv<float2>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXgemv<double2>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);

template <typename T>
StatusCode TuneXger(RawCommandQueue * queue, const size_t m, const size_t n,
                    const double fraction, std::unordered_map<std::string,size_t> &parameters) {
  auto args = Arguments<T>(); args.fraction = fraction; args.m = m; args.n = n;
  auto queue_cpp = Queue(*queue);
  return TunerAPI<T>(queue_cpp, args, 0, XgerGetTunerDefaults, XgerGetTunerSettings<T>,
                     XgerTestValidArguments<T>, XgerSetConstraints, XgerComputeLocalMemSize<T>, XgerSetArguments<T>, parameters);
}
template StatusCode PUBLIC_API TuneXger<half>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXger<float>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXger<double>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXger<float2>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXger<double2>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);

template <typename T>
StatusCode TuneXgemm(RawCommandQueue * queue, const size_t m, const size_t n, const size_t k,
                     const double fraction, std::unordered_map<std::string,size_t> &parameters) {
  auto args = Arguments<T>(); args.fraction = fraction; args.m = m; args.n = n; args.k = k;
  auto queue_cpp = Queue(*queue);
  auto status = TunerAPI<T>(queue_cpp, args, 2, XgemmGetTunerDefaults, XgemmGetTunerSettings<T>,
                            XgemmTestValidArguments<T>, XgemmSetConstraints, XgemmComputeLocalMemSize<T>, XgemmSetArguments<T>, parameters);
  if (status != StatusCode::kSuccess) { return status; }
  return TunerAPI<T>(queue_cpp, args, 12, XgemmGetTunerDefaults, XgemmGetTunerSettings<T>,
                     XgemmTestValidArguments<T>, XgemmSetConstraints, XgemmComputeLocalMemSize<T>, XgemmSetArguments<T>, parameters);
}
template StatusCode PUBLIC_API TuneXgemm<half>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXgemm<float>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXgemm<double>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXgemm<float2>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXgemm<double2>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);

template <typename T>
StatusCode TuneXgemmDirect(RawCommandQueue * queue, const size_t m, const size_t n, const size_t k,
                           const double fraction, std::unordered_map<std::string,size_t> &parameters) {
  auto args = Arguments<T>(); args.fraction = fraction; args.m = m; args.n = n; args.k = k;
  auto queue_cpp = Queue(*queue);
  return TunerAPI<T>(queue_cpp, args, 2, XgemmDirectGetTunerDefaults, XgemmDirectGetTunerSettings<T>,
                     XgemmDirectTestValidArguments<T>, XgemmDirectSetConstraints, XgemmDirectComputeLocalMemSize<T>, XgemmDirectSetArguments<T>, parameters);
}
template StatusCode PUBLIC_API TuneXgemmDirect<half>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXgemmDirect<float>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXgemmDirect<double>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXgemmDirect<float2>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneXgemmDirect<double2>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);

template <typename T>
StatusCode TuneCopy(RawCommandQueue * queue, const size_t m, const size_t n,
                    const double fraction, std::unordered_map<std::string,size_t> &parameters) {
  auto args = Arguments<T>(); args.fraction = fraction; args.m = m; args.n = n;
  auto queue_cpp = Queue(*queue);
  return TunerAPI<T>(queue_cpp, args, 0, CopyGetTunerDefaults, CopyGetTunerSettings<T>,
                     CopyTestValidArguments<T>, CopySetConstraints, CopyComputeLocalMemSize<T>, CopySetArguments<T>, parameters);
}
template StatusCode PUBLIC_API TuneCopy<half>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneCopy<float>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneCopy<double>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneCopy<float2>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneCopy<double2>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);

template <typename T>
StatusCode TunePad(RawCommandQueue * queue, const size_t m, const size_t n,
                   const double fraction, std::unordered_map<std::string,size_t> &parameters) {
  auto args = Arguments<T>(); args.fraction = fraction; args.m = m; args.n = n;
  auto queue_cpp = Queue(*queue);
  return TunerAPI<T>(queue_cpp, args, 0, PadGetTunerDefaults, PadGetTunerSettings<T>,
                     PadTestValidArguments<T>, PadSetConstraints, PadComputeLocalMemSize<T>, PadSetArguments<T>, parameters);
}
template StatusCode PUBLIC_API TunePad<half>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TunePad<float>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TunePad<double>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TunePad<float2>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TunePad<double2>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);

template <typename T>
StatusCode TuneTranspose(RawCommandQueue * queue, const size_t m, const size_t n,
                         const double fraction, std::unordered_map<std::string,size_t> &parameters) {
  auto args = Arguments<T>(); args.fraction = fraction; args.m = m; args.n = n;
  auto queue_cpp = Queue(*queue);
  return TunerAPI<T>(queue_cpp, args, 0, TransposeGetTunerDefaults, TransposeGetTunerSettings<T>,
                     TransposeTestValidArguments<T>, TransposeSetConstraints, TransposeComputeLocalMemSize<T>, TransposeSetArguments<T>, parameters);
}
template StatusCode PUBLIC_API TuneTranspose<half>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneTranspose<float>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneTranspose<double>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneTranspose<float2>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneTranspose<double2>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);

template <typename T>
StatusCode TunePadtranspose(RawCommandQueue * queue, const size_t m, const size_t n,
                            const double fraction, std::unordered_map<std::string,size_t> &parameters) {
  auto args = Arguments<T>(); args.fraction = fraction; args.m = m; args.n = n;
  auto queue_cpp = Queue(*queue);
  return TunerAPI<T>(queue_cpp, args, 0, PadtransposeGetTunerDefaults, PadtransposeGetTunerSettings<T>,
                     PadtransposeTestValidArguments<T>, PadtransposeSetConstraints, PadtransposeComputeLocalMemSize<T>, PadtransposeSetArguments<T>, parameters);
}
template StatusCode PUBLIC_API TunePadtranspose<half>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TunePadtranspose<float>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TunePadtranspose<double>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TunePadtranspose<float2>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TunePadtranspose<double2>(RawCommandQueue*, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);

template <typename T>
StatusCode TuneInvert(RawCommandQueue * queue, const size_t m, const size_t n, const size_t k,
                      const double fraction, std::unordered_map<std::string,size_t> &parameters) {
  auto args = Arguments<T>(); args.fraction = fraction; args.m = m; args.n = n; args.k = k;
  auto queue_cpp = Queue(*queue);
  return TunerAPI<T>(queue_cpp, args, 0, InvertGetTunerDefaults, InvertGetTunerSettings<T>,
                     InvertTestValidArguments<T>, InvertSetConstraints, InvertComputeLocalMemSize<T>, InvertSetArguments<T>, parameters);
}
template StatusCode PUBLIC_API TuneInvert<half>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneInvert<float>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneInvert<double>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneInvert<float2>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);
template StatusCode PUBLIC_API TuneInvert<double2>(RawCommandQueue*, const size_t, const size_t, const size_t, const double, std::unordered_map<std::string,size_t>&);

// =================================================================================================

// The main tuner API, similar to the one in tuning.cpp, but without I/O
template <typename T>
StatusCode TunerAPI(Queue &queue, const Arguments<T> &args, const int V,
                    const GetTunerDefaultsFunc GetTunerDefaults,
                    const GetTunerSettingsFunc<T> GetTunerSettings,
                    const TestValidArgumentsFunc<T> TestValidArguments,
                    const SetConstraintsFunc SetConstraints,
                    const ComputeLocalMemSizeFunc<T> ComputeLocalMemSize,
                    const SetArgumentsFunc<T> SetArguments,
                    std::unordered_map<std::string,size_t> &parameters) {

  // Sets the parameters and platform/device for which to tune (command-line options)
  const TunerDefaults defaults = GetTunerDefaults(V);
  const TunerSettings settings = GetTunerSettings(V, args);

  // Tests validity of the given arguments
  TestValidArguments(V, args);

  // Retrieves OpenCL classes
  const auto device = queue.GetDevice();
  const auto context = queue.GetContext();

  // Inspects whether or not FP64 is supported in case of double precision
  if ((PrecisionValue<T>() == Precision::kDouble && !PrecisionSupported<double>(device)) ||
      (PrecisionValue<T>() == Precision::kComplexDouble && !PrecisionSupported<double2>(device))) {
    return StatusCode::kNoDoublePrecision;
  }

  // As above, but for FP16 (half precision)
  if (PrecisionValue<T>() == Precision::kHalf && !PrecisionSupported<half>(device)) {
    return StatusCode::kNoHalfPrecision;
  }

  // Retrieves properties
  const auto device_type = GetDeviceType(device);
  const auto device_vendor = GetDeviceVendor(device);
  const auto device_architecture = GetDeviceArchitecture(device);
  const auto device_name = GetDeviceName(device);

  // Creates input buffers with random data. Adds a 'canary' region to detect buffer overflows.
  const auto buffer_sizes = std::vector<size_t>{
      settings.size_x + kCanarySize, settings.size_y + kCanarySize,
      settings.size_a + kCanarySize, settings.size_b + kCanarySize, settings.size_c + kCanarySize,
      settings.size_temp + kCanarySize
  };
  const auto seed = static_cast<unsigned long>(time(nullptr));
  std::mt19937 mt(seed);
  std::uniform_real_distribution<double> dist(kTestDataLowerLimit, kTestDataUpperLimit);
  auto source_buffers = std::vector<std::vector<T>>();
  auto reference_buffers = std::vector<std::vector<T>>();
  auto result_buffers = std::vector<std::vector<T>>();
  auto device_buffers = std::vector<Buffer<T>>();
  for (const auto size : buffer_sizes) {
    auto host_buffer = std::vector<T>(size);
    PopulateVector(host_buffer, mt, dist);
    source_buffers.push_back(host_buffer);
    reference_buffers.push_back(std::vector<T>(size));
    result_buffers.push_back(std::vector<T>(size));
    device_buffers.push_back(Buffer<T>(context, size));
  }

  // Sets the tunable parameters and their possible values
  auto configurations = SetConfigurations(device, settings.parameters, settings.local_size,
                                          settings.mul_local, settings.div_local,
                                          SetConstraints(V), ComputeLocalMemSize(V));

  // Select the search method (full search or a random fraction)
  if (args.fraction != 0.0 && args.fraction != 1.0) {
    const auto new_size = static_cast<size_t>(configurations.size() * args.fraction);
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(configurations), std::end(configurations), rng);
    configurations.resize(new_size);
  }

  // First runs a reference example to compare against
  try {

    // Sets the input
    for (const auto id : settings.inputs) {
      device_buffers[id].Write(queue, buffer_sizes[id], source_buffers[id]);
    }

    // Compiles the kernel
    auto compiler_options = std::vector<std::string>();
    const auto program = CompileFromSource(settings.sources, args.precision, settings.kernel_name,
                                           device, context, compiler_options, 0);
    auto kernel = Kernel(program, settings.kernel_name);
    SetArguments(V, kernel, args, device_buffers);

    // Runs the kernel
    const auto time_ms = TimeKernel(args.num_runs, kernel, queue, device,
                                    settings.global_size_ref, settings.local_size_ref, true);
    if (time_ms == -1.0) { throw std::runtime_error("Error in reference implementation"); }

    // Saves the result
    for (const auto id : settings.outputs) {
      device_buffers[id].Read(queue, buffer_sizes[id], reference_buffers[id]);
    }
  }
  catch (...) {
    const auto status_code = DispatchExceptionCatchAll(true);
    return status_code;
  }

  // Starts the tuning process
  auto results = std::vector<TuningResult>();
  for (auto config_id = size_t{0}; config_id < configurations.size(); ++config_id) {
    try {
      auto configuration = configurations[config_id];

      // Sets the input
      for (const auto id : settings.inputs) {
        device_buffers[id].Write(queue, buffer_sizes[id], source_buffers[id]);
      }

      // Sets the thread configuration
      const auto global = SetThreadConfiguration(configuration, settings.global_size,
                                                 settings.mul_global, settings.div_global);
      const auto local = SetThreadConfiguration(configuration, settings.local_size,
                                                settings.mul_local, settings.div_local);

      // Sets the parameters for this configuration
      auto kernel_source = std::string{""};
      for (const auto &parameter : configuration) {
        kernel_source += "#define " + parameter.first + " " + ToString(parameter.second) + "\n";
      }
      kernel_source += settings.sources;

      // Compiles the kernel
      auto compiler_options = std::vector<std::string>();
      const auto program = CompileFromSource(kernel_source, args.precision, settings.kernel_name,
                                             device, context, compiler_options, 0, true);
      auto kernel = Kernel(program, settings.kernel_name);

      // Runs the kernel
      SetArguments(V, kernel, args, device_buffers);
      const auto time_ms = TimeKernel(args.num_runs, kernel, queue, device, global, local, true);

      // Kernel run was not successful
      if (time_ms == -1.0) {
        continue;
      }

      // Compares the results
      auto l2_error = 0.0;
      for (const auto id : settings.outputs) {
        device_buffers[id].Read(queue, buffer_sizes[id], result_buffers[id]);
        for (auto index = size_t{0}; index<buffer_sizes[id]; ++index) {
          const auto diff = SquaredDifference(result_buffers[id][index], reference_buffers[id][index]);
          l2_error += diff;
        }
        l2_error /= static_cast<double>(buffer_sizes[id]);
        if (std::isnan(l2_error) || l2_error > 1.0e-4) {
          throw std::runtime_error("L2 error too large");
        }
      }
      results.push_back(TuningResult{settings.kernel_name, time_ms, configuration});
    }
    catch (...) {
    }
  }

  // Completed the tuning process
  if (results.size() == 0) { return StatusCode::kUnexpectedError; }

  // Computes the best results
  auto comparison = [](const TuningResult& lhs, const TuningResult& rhs) { return lhs.score < rhs.score; };
  const auto best_configuration = std::min_element(results.begin(), results.end(), comparison);
  const auto best_time_ms = best_configuration->score;
  if (best_time_ms == 0.0) { return StatusCode::kUnexpectedError; }

  // Stores the best parameters
  for (const auto config : best_configuration->config) {
    parameters[config.first] = config.second;
  }
  return StatusCode::kSuccess;
}

// Compiles the above function
template StatusCode TunerAPI<half>(Queue &queue, const Arguments<half> &args, const int V, const GetTunerDefaultsFunc GetTunerDefaults, const GetTunerSettingsFunc<half> GetTunerSettings, const TestValidArgumentsFunc<half> TestValidArguments, const SetConstraintsFunc SetConstraints, const ComputeLocalMemSizeFunc<half> ComputeLocalMemSize, const SetArgumentsFunc<half> SetArguments, std::unordered_map<std::string,size_t>&);
template StatusCode TunerAPI<float>(Queue &queue, const Arguments<float> &args, const int V, const GetTunerDefaultsFunc GetTunerDefaults, const GetTunerSettingsFunc<float> GetTunerSettings, const TestValidArgumentsFunc<float> TestValidArguments, const SetConstraintsFunc SetConstraints, const ComputeLocalMemSizeFunc<float> ComputeLocalMemSize, const SetArgumentsFunc<float> SetArguments, std::unordered_map<std::string,size_t>&);
template StatusCode TunerAPI<double>(Queue &queue, const Arguments<double> &args, const int V, const GetTunerDefaultsFunc GetTunerDefaults, const GetTunerSettingsFunc<double> GetTunerSettings, const TestValidArgumentsFunc<double> TestValidArguments, const SetConstraintsFunc SetConstraints, const ComputeLocalMemSizeFunc<double> ComputeLocalMemSize, const SetArgumentsFunc<double> SetArguments, std::unordered_map<std::string,size_t>&);
template StatusCode TunerAPI<float2>(Queue &queue, const Arguments<float2> &args, const int V, const GetTunerDefaultsFunc GetTunerDefaults, const GetTunerSettingsFunc<float2> GetTunerSettings, const TestValidArgumentsFunc<float2> TestValidArguments, const SetConstraintsFunc SetConstraints, const ComputeLocalMemSizeFunc<float2> ComputeLocalMemSize, const SetArgumentsFunc<float2> SetArguments, std::unordered_map<std::string,size_t>&);
template StatusCode TunerAPI<double2>(Queue &queue, const Arguments<double2> &args, const int V, const GetTunerDefaultsFunc GetTunerDefaults, const GetTunerSettingsFunc<double2> GetTunerSettings, const TestValidArgumentsFunc<double2> TestValidArguments, const SetConstraintsFunc SetConstraints, const ComputeLocalMemSizeFunc<double2> ComputeLocalMemSize, const SetArgumentsFunc<double2> SetArguments, std::unordered_map<std::string,size_t>&);

// =================================================================================================
} // namespace clblast
