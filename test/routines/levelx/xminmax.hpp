
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Ekansh Jain
//
// This file implements a class with static methods to describe the Xminmax routine. Examples of
// such 'descriptions' are how to calculate the size a of buffer or how to run the routine. These
// static methods are used by the correctness tester and the performance tester.
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XMINMAX_H_
#define CLBLAST_TEST_ROUTINES_XMINMAX_H_

#include <cstddef>
#include <string>
#include <vector>

#include "test/test_utilities.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestXminmax {
 public:
  // Although it is a non-BLAS routine, it can still be tested against the Xmin and Xmax provided by CLBlast
  static size_t BLASLevel() { return 1; }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgN, kArgImaxOffset, kArgIminOffset, kArgXOffset, kArgXInc};
  }
  static std::vector<std::string> BuffersIn() { return {kBufVecX, kBufScalarUint, kBufSecondScalarUint}; }
  static std::vector<std::string> BuffersOut() { return {kBufScalarUint, kBufSecondScalarUint}; }

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T>& args) { return args.n * args.x_inc + args.x_offset; }
  static size_t GetSizeImax(const Arguments<T>& args) { return args.imax_offset + 1; }
  static size_t GetSizeImin(const Arguments<T>& args) { return args.imin_offset + 1; }

  // Describes how to set the sizes of all the buffers
  static void SetSizes(Arguments<T>& args, Queue&) {
    args.x_size = GetSizeX(args);
    args.scalar_size = GetSizeImax(args) + GetSizeImin(args);
  }

  // Describes what the default values of the leading dimensions of the matrices are
  static size_t DefaultLDA(const Arguments<T>&) { return 1; }  // N/A for this routine
  static size_t DefaultLDB(const Arguments<T>&) { return 1; }  // N/A for this routine
  static size_t DefaultLDC(const Arguments<T>&) { return 1; }  // N/A for this routine

  // Describes which transpose options are relevant for this routine
  using Transposes = std::vector<Transpose>;
  static Transposes GetATransposes(const Transposes&) { return {}; }  // N/A for this routine
  static Transposes GetBTransposes(const Transposes&) { return {}; }  // N/A for this routine

  // Describes how to prepare the input data
  static void PrepareData(const Arguments<T>&, Queue&, const int, std::vector<T>&, std::vector<T>&, std::vector<T>&,
                          std::vector<T>&, std::vector<T>&, std::vector<T>&, std::vector<T>&) {
  }  // N/A for this routine

  // Describes how to run the CLBlast routine
  static StatusCode RunRoutine(const Arguments<T>& args, Buffers<T>& buffers, Queue& queue) {
#ifdef OPENCL_API
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = Minmax<T>(args.n, buffers.scalar_uint(), args.imax_offset, buffers.second_scalar_uint(),
                            args.imin_offset, buffers.x_vec(), args.x_offset, args.x_inc, &queue_plain, &event);
    if (status == StatusCode::kSuccess) {
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    }
#elif CUDA_API
    auto status =
        Minmax<T>(args.n, buffers.scalar_uint(), args.imax_offset, buffers.second_scalar_uint(), args.imin_offset,
                  buffers.x_vec(), args.x_offset, args.x_inc, queue.GetContext()(), queue.getDevice()());
    cuStreamSynchronize(queue());
#endif
    return status;
  }

  // Note: No CBlas or CLBlas routine exists to compare against so instead it compares against the results of Max and
  // Min routines in CLBlast.
  static StatusCode RunReference1(const Arguments<T>& args, Buffers<T>& buffers, Queue& queue) {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = Max<T>(args.n, buffers.scalar_uint(), args.imax_offset, buffers.x_vec(), args.x_offset, args.x_inc,
                         &queue_plain, &event);
    if (status != StatusCode::kSuccess) {
      return status;
    }
    clWaitForEvents(1, &event);
    clReleaseEvent(event);

    status = Min<T>(args.n, buffers.second_scalar_uint(), args.imin_offset, buffers.x_vec(), args.x_offset, args.x_inc,
                    &queue_plain, &event);
    if (status != StatusCode::kSuccess) {
      return status;
    }
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
    return status;
  }

// Note: No CBlas routine exists to compare against so intead uses the RunReference1 method which works for both CBlas
// and CLBlas
#ifdef CLBLAST_REF_CBLAS
  static StatusCode RunReference2(const Arguments<T>& args, BuffersHost<T>& buffers_host, Queue& queue) {
    auto context = queue.GetContext();
    Buffers<T> buffers{Buffer<T>{queue.GetContext(), buffers_host.x_vec.size()},
                       CreateInvalidBuffer<T>(context, 0),
                       CreateInvalidBuffer<T>(context, 0),
                       CreateInvalidBuffer<T>(context, 0),
                       CreateInvalidBuffer<T>(context, 0),
                       CreateInvalidBuffer<T>(context, 0),
                       CreateInvalidBuffer<T>(context, 0),
                       Buffer<unsigned int>{queue.GetContext(), buffers_host.scalar_uint.size()},
                       Buffer<unsigned int>{queue.GetContext(), buffers_host.second_scalar_uint.size()}};
    HostToDevice(args, buffers, buffers_host, queue, {kBufVecX, kBufScalarUint, kBufSecondScalarUint});
    auto status = RunReference1(args, buffers, queue);
    DeviceToHost(args, buffers, buffers_host, queue, {kBufVecX, kBufScalarUint, kBufSecondScalarUint});
    return status;
  }
#endif

// Describes how to run the cuBLAS routine (for correctness/performance comparison)
#ifdef CLBLAST_REF_CUBLAS
  static StatusCode RunReference3(const Arguments<T>& args, BuffersCUDA<T>& buffers, Queue&) {
    return StatusCode::kUnknownError;
  }
#endif

  // Describes how to download the results of the computation (more importantly: which buffer)
  static std::vector<T> DownloadResult(const Arguments<T>& args, Buffers<T>& buffers, Queue& queue) {
    std::vector<unsigned int> result_uint(args.scalar_size, 0);
    std::vector<unsigned int> second_result_uint(args.second_scalar_size, 0);
    buffers.scalar_uint.Read(queue, args.scalar_size, result_uint);
    buffers.second_scalar_uint.Read(queue, args.second_scalar_size, second_result_uint);
    // The result is an integer. However, since the test infrastructure assumes results of
    // type 'T' (float/double/float2/double2/half), we store the results into T instead.
    // The values might then become meaningless, but a comparison for testing should still
    // be valid to verify correctness.
    auto result_as_T = static_cast<T>(result_uint[0]);
    std::vector<T> result(args.scalar_size + args.second_scalar_size);
    result[0] = result_as_T;
    result[1] = static_cast<T>(second_result_uint[0]);
    return result;
  }

  // Describes how to compute the indices of the result buffer
  static size_t ResultID1(const Arguments<T>&) { return 1; }  // N/A for this routine
  static size_t ResultID2(const Arguments<T>&) { return 1; }  // N/A for this routine
  static size_t GetResultIndex(const Arguments<T>& args, const size_t, const size_t) { return args.imax_offset; }

  // Describes how to compute performance metrics
  static size_t GetFlops(const Arguments<T>& args) { return args.n * 2; }
  static size_t GetBytes(const Arguments<T>& args) { return ((args.n) + 1) * sizeof(T); }
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_TEST_ROUTINES_XAXPYBATCHED_H_
#endif
