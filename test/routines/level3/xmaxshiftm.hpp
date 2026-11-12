
// =================================================================================================
// This file is part of the CLBlast project.
// =================================================================================================

// Correctness and performance description for Xmaxshiftm.
#ifndef CLBLAST_TEST_ROUTINES_XMAXSHIFTM_H_
#define CLBLAST_TEST_ROUTINES_XMAXSHIFTM_H_

#include "test/routines/maxshift_common.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
class TestXmaxshiftm {
 public:
  static size_t BLASLevel() { return 3; }
  static std::vector<std::string> GetOptions() { return {kArgM, kArgN, kArgLayout, kArgALeadDim, kArgAOffset}; }
  static std::vector<std::string> BuffersIn() { return {kBufMatA}; }
  static std::vector<std::string> BuffersOut() { return {kBufMatA}; }

  static size_t GetSizeA(const Arguments<T>& args) {
    return args.a_ld * ((args.layout == Layout::kRowMajor) ? args.m : args.n) + args.a_offset;
  }
  static void SetSizes(Arguments<T>& args, Queue&) { args.a_size = GetSizeA(args); }
  static size_t DefaultLDA(const Arguments<T>& args) { return (args.layout == Layout::kRowMajor) ? args.n : args.m; }
  static size_t DefaultLDB(const Arguments<T>&) { return 1; }
  static size_t DefaultLDC(const Arguments<T>&) { return 1; }

  using Transposes = std::vector<Transpose>;
  static Transposes GetATransposes(const Transposes&) { return {}; }
  static Transposes GetBTransposes(const Transposes&) { return {}; }
  static void PrepareData(const Arguments<T>& args, Queue&, const int, std::vector<T>&, std::vector<T>&,
                          std::vector<T>& a, std::vector<T>&, std::vector<T>&, std::vector<T>&, std::vector<T>&) {
    // Different row maxima catch accidental whole-matrix or per-column reductions.
    for (size_t row = 0; row < args.m; ++row) {
      const auto max = (row % 2 != 0) ? -static_cast<double>(row) - 1 : static_cast<double>(row) + 2;
      for (size_t col = 0; col < args.n; ++col) {
        a[GetResultIndex(args, row, col)] = MaxshiftWrite<T>(max - static_cast<double>((col + 1) % 17));
      }
      if (args.n > 0) {
        a[GetResultIndex(args, row, args.n - 1)] = MaxshiftWrite<T>(max);
      }
    }
  }

  static StatusCode RunRoutine(const Arguments<T>& args, Buffers<T>& buffers, Queue& queue) {
#ifdef OPENCL_API
    auto queue_plain = queue();
    auto event = cl_event{};
    const auto status =
        Maxshiftm<T>(args.layout, args.m, args.n, buffers.a_mat(), args.a_offset, args.a_ld, &queue_plain, &event);
    if (status == StatusCode::kSuccess) {
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    }
#elif CUDA_API
    const auto status = Maxshiftm<T>(args.layout, args.m, args.n, buffers.a_mat(), args.a_offset, args.a_ld,
                                     queue.GetContext()(), queue.GetDevice()());
    cuStreamSynchronize(queue());
#endif
    return status;
  }

  // CPU reference: these operations have no BLAS counterpart.
  static StatusCode RunReference(const Arguments<T>& args, BuffersHost<T>& buffers_host) {
    if (args.m == 0 || args.n == 0) {
      return StatusCode::kInvalidDimension;
    }
    const auto rowmajor = (args.layout == Layout::kRowMajor);
    const auto one = rowmajor ? args.n : args.m;
    const auto two = rowmajor ? args.m : args.n;
    if (args.a_ld < one) {
      return StatusCode::kInvalidLeadDimA;
    }
    if (buffers_host.a_mat.size() < args.a_offset + (two - 1) * args.a_ld + one) {
      return StatusCode::kInsufficientMemoryA;
    }
    for (size_t row = 0; row < args.m; ++row) {
      auto max = -std::numeric_limits<double>::infinity();
      for (size_t col = 0; col < args.n; ++col) {
        max = std::max(max, MaxshiftRead(buffers_host.a_mat[GetResultIndex(args, row, col)]));
      }
      for (size_t col = 0; col < args.n; ++col) {
        const auto index = GetResultIndex(args, row, col);
        buffers_host.a_mat[index] = MaxshiftWrite<T>(MaxshiftRead(buffers_host.a_mat[index]) - max);
      }
    }
    return StatusCode::kSuccess;
  }
  static StatusCode RunReference1(const Arguments<T>& args, Buffers<T>& buffers, Queue& queue) {
    try {
      if (args.m == 0 || args.n == 0) {
        return StatusCode::kInvalidDimension;
      }
      const auto rowmajor = (args.layout == Layout::kRowMajor);
      TestMatrixA(rowmajor ? args.n : args.m, rowmajor ? args.m : args.n, buffers.a_mat, args.a_offset, args.a_ld);
      auto buffers_host = BuffersHost<T>();
      DeviceToHost(args, buffers, buffers_host, queue, BuffersIn());
      const auto status = RunReference(args, buffers_host);
      HostToDevice(args, buffers, buffers_host, queue, BuffersOut());
      return status;
    } catch (...) {
      return DispatchException();
    }
  }
  static StatusCode RunReference2(const Arguments<T>& args, BuffersHost<T>& buffers_host, Queue&) {
    return RunReference(args, buffers_host);
  }
  static StatusCode RunReference3(const Arguments<T>&, BuffersCUDA<T>&, Queue&) { return StatusCode::kUnknownError; }

  static std::vector<T> DownloadResult(const Arguments<T>& args, Buffers<T>& buffers, Queue& queue) {
    auto result = std::vector<T>(args.a_size);
    buffers.a_mat.Read(queue, args.a_size, result);
    return result;
  }
  static size_t ResultID1(const Arguments<T>& args) { return args.m; }
  static size_t ResultID2(const Arguments<T>& args) { return args.n; }
  static size_t GetResultIndex(const Arguments<T>& args, const size_t row, const size_t col) {
    return args.a_offset + ((args.layout == Layout::kRowMajor) ? row * args.a_ld + col : col * args.a_ld + row);
  }

  static size_t GetFlops(const Arguments<T>& args) { return args.m * args.n; }
  static size_t GetBytes(const Arguments<T>& args) { return (3 * args.m * args.n) * sizeof(T); }
};

// =================================================================================================
}  // namespace clblast
#endif
