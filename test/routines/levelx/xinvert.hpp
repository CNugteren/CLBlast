
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a class with static methods to describe the Xinvert routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XINVERT_H_
#define CLBLAST_TEST_ROUTINES_XINVERT_H_

#include "test/routines/common.hpp"
#include "src/routines/levelx/xinvert.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
StatusCode RunReference(const Arguments<T> &args, BuffersHost<T> &buffers_host) {
  const bool is_upper = ((args.triangle == Triangle::kUpper && args.layout != Layout::kRowMajor) ||
                         (args.triangle == Triangle::kLower && args.layout == Layout::kRowMajor));

  // Helper variables
  const auto block_size = args.m;
  const auto num_blocks = CeilDiv(args.n, block_size);
  const auto a_ld = args.a_ld;
  const auto b_ld = block_size;

  // Checks for valid arguments
  if ((block_size == 0) || (args.n == 0)) {
    return StatusCode::kInvalidDimension;
  }
  if ((block_size % 16 != 0) || (block_size > 128)) {
    return StatusCode::kUnknownError;
  }

  // Start at zero
  for (size_t i =0; i < args.m; ++i) {
    for (size_t j = 0; j < args.n; ++j) {
      buffers_host.b_mat[j * args.m + i] = T{0.0};
    }
  }

  // Loops over the amount of diagonal blocks of size args.m by args.m each
  for (auto block_id = size_t{0}; block_id < num_blocks; ++block_id) {
    const auto a_offset = block_id * (block_size + a_ld * block_size) + args.a_offset;
    const auto b_offset = block_id * block_size * block_size;

    // Inverts the diagonal elements of the matrix
    for (auto i = size_t{0}; i < block_size; ++i) {
      auto a_value = T{1.0};
      if (args.diagonal == Diagonal::kNonUnit) {
        if (i + block_id * block_size < args.n) {
          if (buffers_host.a_mat[i * a_ld + i + a_offset] == T{0.0}) { return StatusCode::kUnknownError; }
          a_value = T{1.0} / buffers_host.a_mat[i * a_ld + i + a_offset];
        }
      }
      buffers_host.b_mat[i * b_ld + i + b_offset] = a_value;
    }

    // Inverts the upper triangle row by row
    if (is_upper) {
      for (int i = static_cast<int>(block_size) - 2; i >= 0; --i) {
        for (auto j = static_cast<int>(block_size) - 1; j > i; --j) {
          auto sum = T{0.0};
          for (auto k = i + 1; k <= j; ++k) {
            auto a_value = T{0.0};
            if ((i + block_id * block_size < args.n) && (k + block_id * block_size < args.n)) {
              a_value = buffers_host.a_mat[k * a_ld + i + a_offset];
            }
            sum += a_value * buffers_host.b_mat[j * b_ld + k + b_offset];
          }
          buffers_host.b_mat[j * b_ld + i + b_offset] = - sum * buffers_host.b_mat[i * b_ld + i + b_offset];
        }
      }
    }

    // Inverts the lower triangle row by row
    else {
      for (auto i = size_t{1}; i < block_size; ++i) {
        for (auto j = size_t{0}; j < i; ++j) {
          auto sum = T{0.0};
          for (auto k = j; k < i; ++k) {
            auto a_value = T{0.0};
            if ((i + block_id * block_size < args.n) && (k + block_id * block_size < args.n)) {
              a_value = buffers_host.a_mat[k * a_ld + i + a_offset];
            }
            sum += a_value * buffers_host.b_mat[j * b_ld + k + b_offset];
          }
          buffers_host.b_mat[j * b_ld + i + b_offset] = - sum * buffers_host.b_mat[i * b_ld + i + b_offset];
        }
      }
    }
  }
  return StatusCode::kSuccess;
}

// Half-precision version calling the above reference implementation after conversions
template <>
StatusCode RunReference<half>(const Arguments<half> &args, BuffersHost<half> &buffers_host) {
  auto a_buffer2 = HalfToFloatBuffer(buffers_host.a_mat);
  auto b_buffer2 = HalfToFloatBuffer(buffers_host.b_mat);
  auto dummy = std::vector<float>(0);
  auto buffers2 = BuffersHost<float>{dummy, dummy, a_buffer2, b_buffer2, dummy, dummy, dummy};
  auto args2 = Arguments<float>();
  args2.a_size = args.a_size; args2.b_size = args.b_size;
  args2.a_ld = args.a_ld; args2.m = args.m; args2.n = args.n;
  args2.a_offset = args.a_offset;
  args2.layout = args.layout; args2.triangle = args.triangle; args2.diagonal = args.diagonal;
  auto status = RunReference(args2, buffers2);
  FloatToHalfBuffer(buffers_host.b_mat, b_buffer2);
  return status;
}

// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestXinvert {
 public:

  // The BLAS level: 4 for the extra routines
  static size_t BLASLevel() { return 4; }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgN, kArgM,
            kArgLayout, kArgTriangle, kArgDiagonal,
            kArgALeadDim, kArgAOffset};
  }
  static std::vector<std::string> BuffersIn() { return {kBufMatA, kBufMatB}; }
  static std::vector<std::string> BuffersOut() { return {kBufMatB}; }

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeA(const Arguments<T> &args) {
    return args.n * args.a_ld + args.a_offset;
  }
  static size_t GetSizeB(const Arguments<T> &args) {
    const auto block_size = args.m;
    const auto num_blocks = CeilDiv(args.n, block_size);
    return num_blocks * block_size * block_size;
  }

  // Describes how to set the sizes of all the buffers
  static void SetSizes(Arguments<T> &args, Queue&) {
    args.a_size = GetSizeA(args);
    args.b_size = GetSizeB(args);
  }

  // Describes what the default values of the leading dimensions of the matrices are
  static size_t DefaultLDA(const Arguments<T> &args) { return args.n; }
  static size_t DefaultLDB(const Arguments<T> &) { return 1; } // N/A for this routine
  static size_t DefaultLDC(const Arguments<T> &) { return 1; } // N/A for this routine

  // Describes which omatcopyose options are relevant for this routine
  using Transposes = std::vector<Transpose>;
  static Transposes GetATransposes(const Transposes &) { return {}; } // N/A for this routine
  static Transposes GetBTransposes(const Transposes &) { return {}; } // N/A for this routine

  // Describes how to prepare the input data
  static void PrepareData(const Arguments<T>&, Queue&, const int, std::vector<T>&,
                          std::vector<T>&, std::vector<T>&, std::vector<T>&, std::vector<T>&,
                          std::vector<T>&, std::vector<T>&) {} // N/A for this routine

  // Describes how to run the CLBlast routine
  static StatusCode RunRoutine(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
    try {
      #ifdef OPENCL_API
        auto event = cl_event{};
        auto inverter = Xinvert<T>(queue, &event);
        inverter.InvertMatrixDiagonalBlocks(args.layout, args.triangle, args.diagonal,
                                            args.n, args.m,
                                            buffers.a_mat, args.a_offset, args.a_ld,
                                            buffers.b_mat);
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
      #elif CUDA_API
        auto inverter = Xinvert<T>(queue, nullptr);
        inverter.InvertMatrixDiagonalBlocks(args.layout, args.triangle, args.diagonal,
                                            args.n, args.m,
                                            buffers.a_mat, args.a_offset, args.a_ld,
                                            buffers.b_mat);
        cuStreamSynchronize(queue());
      #endif
    } catch (...) { return DispatchException(); }
    return StatusCode::kSuccess;
  }

  // Describes how to run a naive version of the routine (for correctness/performance comparison).
  // Note that a proper clBLAS or CPU BLAS comparison is not available for non-BLAS routines.
  static StatusCode RunReference1(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
    auto buffers_host = BuffersHost<T>();
    DeviceToHost(args, buffers, buffers_host, queue, BuffersIn());
    const auto status = RunReference(args, buffers_host);
    HostToDevice(args, buffers, buffers_host, queue, BuffersOut());
    return status;
  }

  static StatusCode RunReference2(const Arguments<T> &args, BuffersHost<T> &buffers_host, Queue&) {
    return RunReference(args, buffers_host);
  }
  static StatusCode RunReference3(const Arguments<T> &args, BuffersCUDA<T> &buffers, Queue &) {
    return StatusCode::kUnknownError;
  }

  // Describes how to download the results of the computation (more importantly: which buffer)
  static std::vector<T> DownloadResult(const Arguments<T> &args, Buffers<T> &buffers, Queue &queue) {
    std::vector<T> result(args.b_size, static_cast<T>(0));
    buffers.b_mat.Read(queue, args.b_size, result);
    return result;
  }

  // Describes how to compute the indices of the result buffer
  static size_t ResultID1(const Arguments<T> &args) { return args.m; }
  static size_t ResultID2(const Arguments<T> &args) { return Ceil(args.n, args.m); }
  static size_t GetResultIndex(const Arguments<T> &args, const size_t id1, const size_t id2) {
    return id1 * Ceil(args.n, args.m) + id2;
  }

  // Describes how to compute performance metrics
  static size_t GetFlops(const Arguments<T> &args) {
    const auto block_size = args.m;
    const auto num_blocks = CeilDiv(args.n, block_size);
    return num_blocks * (block_size * (block_size / 2) * (block_size / 2));
  }
  static size_t GetBytes(const Arguments<T> &args) {
    return (args.a_size * args.b_size) * sizeof(T);
  }
};

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_ROUTINES_XINVERT_H_
#endif
