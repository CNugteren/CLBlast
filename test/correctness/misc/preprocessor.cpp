
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the tests for the simple integrated OpenCL pre-processor
//
// =================================================================================================

#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <iostream>

#include "utilities/utilities.hpp"
#include "utilities/compile.hpp"
#include "kernel_preprocessor.hpp"

namespace clblast {
// =================================================================================================

bool TestDefines() {
  const auto source1 =
  R"(
  #define VAR1
  #define VAR2 32
  #if VAR2 == 32
    #ifndef VAR1
      #define ERROR
      #ifdef VAR1
        #define ERROR
      #endif
    #else
      #if VAR2 == 32 || VAR3 == 4
        #define SUCCESS
      #else
        #define ERROR
      #endif
      #define SUCCESS
    #endif
  #endif
  #ifndef VAR3
    #define SUCCESS
  #else
    #define ERROR
  #endif
  )";
  const auto expected1 =
  "  #define VAR1\n"
  "  #define VAR2 32\n"
  "        #define SUCCESS\n"
  "      #define SUCCESS\n"
  "    #define SUCCESS\n"
  "  \n";
  const auto result1 = PreprocessKernelSource(source1);
  return result1 == expected1;
}
// =================================================================================================

bool TestArrayToRegisterPromotion() {
  const auto source1 =
      R"(#define WPT 2
inline void SetValues(int float, float values[WPT],
                      const float k) {
  #pragma unroll
  for (int i = 0; i < WPT; i += 1) {
    values[i] = k + j;
  }
}
__kernel void ExampleKernel() {
  #pragma promote_to_registers
  float values[WPT];
  #pragma unroll
  for (int i = 0; i < WPT; i += 1) {
    values[i] = 0.0f;
  }
  SetValues(12.3f, values, -3.9f);
}
)";
  const auto expected1 =
      R"(#define WPT 2
inline void SetValues(int float, float values_0, float values_1,
                      const float k) {
  {
    values_0 = k + j;
  }
  {
    values_1 = k + j;
  }
}
__kernel void ExampleKernel() {
  float values_0;
  float values_1;
  {
    values_0 = 0.0f;
  }
  {
    values_1 = 0.0f;
  }
  SetValues(12.3f, values_0, values_1, -3.9f);
}
)";
  const auto result1 = PreprocessKernelSource(source1);
  return result1 == expected1;
}

// =================================================================================================

bool TestKernel(const Device& device, const Context& context,
                const std::string &kernel_name, const std::string &kernel_source,
                const Precision precision) {

  fprintf(stdout, "* Testing simple OpenCL pre-processor for '%s'\n", kernel_name.c_str());

  // Verifies that the current kernel compiles properly (assumes so, otherwise throws an error)
  auto compiler_options_ref = std::vector<std::string>();
  const auto program_ref = CompileFromSource(kernel_source, precision, kernel_name,
                                             device, context, compiler_options_ref, 2);

  // Compiles the same kernel, but now with the pre-processor enabled
  try {
    auto compiler_options = std::vector<std::string>();
    const auto program = CompileFromSource(kernel_source, precision, kernel_name,
                                           device, context, compiler_options, 1);
    return true;
  } catch (const CLCudaAPIBuildError &e) {
    fprintf(stdout, "* ERROR: Compilation warnings/errors with pre-processed kernel, status %d\n",
            e.status());
    return false;
  } catch (const Error<std::runtime_error> &e) {
    fprintf(stdout, "* ERROR: Pre-processor error, message:\n%s\n", e.what());
    return false;
  }
}

// =================================================================================================

size_t RunPreprocessor(int argc, char *argv[], const bool silent, const Precision precision) {
  auto errors = size_t{0};
  auto passed = size_t{0};

  // Retrieves the arguments
  auto help = std::string{"Options given/available:\n"};
  auto arguments = RetrieveCommandLineArguments(argc, argv);
  const auto platform_id = GetArgument(arguments, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  const auto device_id = GetArgument(arguments, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  if (!silent) { fprintf(stdout, "\n* %s\n", help.c_str()); }

  // Initializes OpenCL
  const auto platform = Platform(platform_id);
  const auto device = Device(platform, device_id);
  const auto context = Context(device);

  // Basic tests
  if (TestDefines()) { passed++; } else { errors++; }
  if (TestArrayToRegisterPromotion()) { passed++; } else { errors++; }

  // XAXPY
  const auto xaxpy_sources =
    "#define WPT 2\n"
    #include "../src/kernels/level1/level1.opencl"
    #include "../src/kernels/level1/xaxpy.opencl"
  ;
  if (TestKernel(device, context, "XaxpyFastest", xaxpy_sources, precision)) { passed++; } else { errors++; }

  // XGER
  const auto xger_sources =
    "#define WPT 2\n"
    #include "../src/kernels/level2/level2.opencl"
    #include "../src/kernels/level2/xger.opencl"
  ;
  if (TestKernel(device, context, "Xger", xger_sources, precision)) { passed++; } else { errors++; }

  // XGEMV
  const auto xgemv_sources =
    "#define WPT1 2\n"
    "#define WPT2 2\n"
    "#define WPT3 2\n"
    "#define UNROLL1 4\n"
    "#define VW2 2\n"
    #include "../src/kernels/level2/xgemv.opencl"
    #include "../src/kernels/level2/xgemv_fast.opencl"
  ;
  if (TestKernel(device, context, "XgemvFast", xgemv_sources, precision)) { passed++; } else { errors++; }

  // CopyFast
  const auto copy_fast_sources =
    "#define COPY_WPT 2\n"
    #include "../src/kernels/level3/level3.opencl"
    #include "../src/kernels/level3/copy_fast.opencl"
  ;
  if (TestKernel(device, context, "CopyMatrixFast", copy_fast_sources, precision)) { passed++; } else { errors++; }

  // CopyPad
  const auto copy_pad_sources =
    "#define PAD_WPTX 2\n"
    "#define PAD_WPTY 2\n"
    #include "../src/kernels/level3/level3.opencl"
    #include "../src/kernels/level3/copy_pad.opencl"
  ;
  if (TestKernel(device, context, "CopyPadMatrix", copy_pad_sources, precision)) { passed++; } else { errors++; }

  // TransposeFast
  const auto transpose_fast_sources =
    "#define TRA_WPT 2\n"
    #include "../src/kernels/level3/level3.opencl"
    #include "../src/kernels/level3/transpose_fast.opencl"
  ;
  if (TestKernel(device, context, "TransposeMatrixFast", transpose_fast_sources, precision)) { passed++; } else { errors++; }

  // TransposePad
  const auto transpose_pad_sources =
    "#define PADTRA_WPT 2\n"
    #include "../src/kernels/level3/level3.opencl"
    #include "../src/kernels/level3/transpose_pad.opencl"
  ;
  if (TestKernel(device, context, "TransposePadMatrix", transpose_pad_sources, precision)) { passed++; } else { errors++; }

  // GEMM (in-direct) GEMMK==0
  const auto gemm_sources =
    "#define KWI 2\n"
    "#define MWG 16\n"
    "#define NWG 16\n"
    "#define SA 1\n"
    #include "../src/kernels/level3/xgemm_part1.opencl"
    #include "../src/kernels/level3/xgemm_part2.opencl"
    #include "../src/kernels/level3/xgemm_part3.opencl"
    #include "../src/kernels/level3/xgemm_part4.opencl"
  ;
  if (TestKernel(device, context, "Xgemm", gemm_sources, precision)) { passed++; } else { errors++; }

  // GEMM (in-direct) GEMMK==1
  const auto gemm_sources_gemmk1 =
    "#define MWG 16\n"
    "#define NWG 16\n"
    "#define GEMMK 1\n"
    #include "../src/kernels/level3/xgemm_part1.opencl"
    #include "../src/kernels/level3/xgemm_part2.opencl"
    #include "../src/kernels/level3/xgemm_part3.opencl"
    #include "../src/kernels/level3/xgemm_part4.opencl"
  ;
  if (TestKernel(device, context, "Xgemm", gemm_sources_gemmk1, precision)) { passed++; } else { errors++; }

  // GEMM (direct)
  const auto gemm_direct_sources =
    "#define KWID 2\n"
    "#define WGD 16\n"
    #include "../src/kernels/level3/xgemm_direct_part1.opencl"
    #include "../src/kernels/level3/xgemm_direct_part2.opencl"
    #include "../src/kernels/level3/xgemm_direct_part3.opencl"
  ;
  if (TestKernel(device, context, "XgemmDirectTN", gemm_direct_sources, precision)) { passed++; } else { errors++; }

  // HEMM
  if (precision == Precision::kComplexSingle || precision == Precision::kComplexDouble) {
    const auto herm_sources =
      "#define ROUTINE_HEMM\n"
      #include "../src/kernels/level3/level3.opencl"
      #include "../src/kernels/level3/convert_hermitian.opencl"
    ;
    if (TestKernel(device, context, "HermLowerToSquared", herm_sources, precision)) { passed++; } else { errors++; }
  }

  // Prints and returns the statistics
  std::cout << std::endl;
  std::cout << "    " << passed << " test(s) passed" << std::endl;
  std::cout << "    " << errors << " test(s) failed" << std::endl;
  std::cout << std::endl;
  return errors;
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunPreprocessor(argc, argv, false, clblast::Precision::kSingle);
  errors += clblast::RunPreprocessor(argc, argv, true, clblast::Precision::kComplexDouble);
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
