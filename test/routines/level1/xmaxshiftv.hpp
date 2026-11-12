
// =================================================================================================
// This file is part of the CLBlast project.
// =================================================================================================

// Correctness and performance description for Xmaxshiftv.
#ifndef CLBLAST_TEST_ROUTINES_XMAXSHIFTV_H_
#define CLBLAST_TEST_ROUTINES_XMAXSHIFTV_H_

#include "test/routines/maxshift_common.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
class TestXmaxshiftv {
 public:
  static size_t BLASLevel() { return 1; }
  static std::vector<std::string> GetOptions() { return {kArgN, kArgXInc, kArgXOffset}; }
  static std::vector<std::string> BuffersIn() { return {kBufVecX}; }
  static std::vector<std::string> BuffersOut() { return {kBufVecX}; }

  static size_t GetSizeX(const Arguments<T>& args) { return args.n * args.x_inc + args.x_offset; }
  static void SetSizes(Arguments<T>& args, Queue&) { args.x_size = GetSizeX(args); }
  static size_t DefaultLDA(const Arguments<T>&) { return 1; }
  static size_t DefaultLDB(const Arguments<T>&) { return 1; }
  static size_t DefaultLDC(const Arguments<T>&) { return 1; }

  using Transposes = std::vector<Transpose>;
  static Transposes GetATransposes(const Transposes&) { return {}; }
  static Transposes GetBTransposes(const Transposes&) { return {}; }
  static void PrepareData(const Arguments<T>& args, Queue&, const int, std::vector<T>& x, std::vector<T>&,
                          std::vector<T>&, std::vector<T>&, std::vector<T>&, std::vector<T>&, std::vector<T>&) {
    // Cover all-negative vectors as well as mixed signs, with the maximum at the end.
    const auto negative = (args.n % 2 != 0);
    for (size_t i = 0; i < args.n; ++i) {
      const auto value = negative ? -static_cast<double>(i % 29) - 2 : static_cast<double>(i % 29) - 14;
      x[args.x_offset + i * args.x_inc] = MaxshiftWrite<T>(value);
    }
    if (args.n > 0) {
      x[args.x_offset + (args.n - 1) * args.x_inc] = MaxshiftWrite<T>(negative ? -1 : 15);
    }
  }

  static StatusCode RunRoutine(const Arguments<T>& args, Buffers<T>& buffers, Queue& queue) {
#ifdef OPENCL_API
    auto queue_plain = queue();
    auto event = cl_event{};
    const auto status = Maxshiftv<T>(args.n, buffers.x_vec(), args.x_offset, args.x_inc, &queue_plain, &event);
    if (status == StatusCode::kSuccess) {
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    }
#elif CUDA_API
    const auto status =
        Maxshiftv<T>(args.n, buffers.x_vec(), args.x_offset, args.x_inc, queue.GetContext()(), queue.GetDevice()());
    cuStreamSynchronize(queue());
#endif
    return status;
  }

  // CPU reference: these operations have no BLAS counterpart.
  static StatusCode RunReference(const Arguments<T>& args, BuffersHost<T>& buffers_host) {
    if (args.n == 0) {
      return StatusCode::kInvalidDimension;
    }
    if (args.x_inc == 0) {
      return StatusCode::kInvalidIncrementX;
    }
    if (buffers_host.x_vec.size() < args.x_offset + (args.n - 1) * args.x_inc + 1) {
      return StatusCode::kInsufficientMemoryX;
    }
    auto max = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < args.n; ++i) {
      max = std::max(max, MaxshiftRead(buffers_host.x_vec[args.x_offset + i * args.x_inc]));
    }
    for (size_t i = 0; i < args.n; ++i) {
      const auto index = args.x_offset + i * args.x_inc;
      buffers_host.x_vec[index] = MaxshiftWrite<T>(MaxshiftRead(buffers_host.x_vec[index]) - max);
    }
    return StatusCode::kSuccess;
  }
  static StatusCode RunReference1(const Arguments<T>& args, Buffers<T>& buffers, Queue& queue) {
    try {
      if (args.n == 0) {
        return StatusCode::kInvalidDimension;
      }
      TestVectorX(args.n, buffers.x_vec, args.x_offset, args.x_inc);
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
    auto result = std::vector<T>(args.x_size);
    buffers.x_vec.Read(queue, args.x_size, result);
    return result;
  }
  static size_t ResultID1(const Arguments<T>& args) { return args.n; }
  static size_t ResultID2(const Arguments<T>&) { return 1; }
  static size_t GetResultIndex(const Arguments<T>& args, const size_t id1, const size_t) {
    return args.x_offset + id1 * args.x_inc;
  }

  static size_t GetFlops(const Arguments<T>& args) { return args.n; }
  static size_t GetBytes(const Arguments<T>& args) { return (3 * args.n) * sizeof(T); }
};

// =================================================================================================
}  // namespace clblast
#endif
