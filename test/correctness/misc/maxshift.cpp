// =================================================================================================
// This file is part of the CLBlast project.
//
// End-to-end tests for the maximum-shift C APIs, including padding and out-of-order queues.
// =================================================================================================

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "clblast_c.h"
#include "test/routines/maxshift_common.hpp"

namespace clblast {

template <typename T>
CLBlastStatusCode CallMaxshiftv(const size_t n, cl_mem x, const size_t offset, const size_t inc,
                                cl_command_queue* queue, cl_event* event);
template <typename T>
CLBlastStatusCode CallMaxshiftm(const CLBlastLayout layout, const size_t m, const size_t n, cl_mem a,
                                const size_t offset, const size_t ld, cl_command_queue* queue, cl_event* event);

template <>
CLBlastStatusCode CallMaxshiftv<half>(const size_t n, cl_mem x, const size_t offset, const size_t inc,
                                      cl_command_queue* queue, cl_event* event) {
  return CLBlastHmaxshiftv(n, x, offset, inc, queue, event);
}
template <>
CLBlastStatusCode CallMaxshiftm<half>(const CLBlastLayout layout, const size_t m, const size_t n, cl_mem a,
                                      const size_t offset, const size_t ld, cl_command_queue* queue, cl_event* event) {
  return CLBlastHmaxshiftm(layout, m, n, a, offset, ld, queue, event);
}

template <>
CLBlastStatusCode CallMaxshiftv<float>(const size_t n, cl_mem x, const size_t offset, const size_t inc,
                                       cl_command_queue* queue, cl_event* event) {
  return CLBlastSmaxshiftv(n, x, offset, inc, queue, event);
}
template <>
CLBlastStatusCode CallMaxshiftm<float>(const CLBlastLayout layout, const size_t m, const size_t n, cl_mem a,
                                       const size_t offset, const size_t ld, cl_command_queue* queue, cl_event* event) {
  return CLBlastSmaxshiftm(layout, m, n, a, offset, ld, queue, event);
}

template <>
CLBlastStatusCode CallMaxshiftv<double>(const size_t n, cl_mem x, const size_t offset, const size_t inc,
                                        cl_command_queue* queue, cl_event* event) {
  return CLBlastDmaxshiftv(n, x, offset, inc, queue, event);
}
template <>
CLBlastStatusCode CallMaxshiftm<double>(const CLBlastLayout layout, const size_t m, const size_t n, cl_mem a,
                                        const size_t offset, const size_t ld, cl_command_queue* queue,
                                        cl_event* event) {
  return CLBlastDmaxshiftm(layout, m, n, a, offset, ld, queue, event);
}

void ExpectStatus(const CLBlastStatusCode actual, const StatusCode expected = StatusCode::kSuccess) {
  if (static_cast<StatusCode>(actual) != expected) {
    throw std::runtime_error("Unexpected CLBlast status: " + ToString(static_cast<int>(actual)));
  }
}

// Wait for the completion event, or synchronize explicitly when no event was requested.
void WaitForMaxshift(Queue& queue, Event& event, const bool request_event) {
  if (request_event) {
    CheckError(clWaitForEvents(1, event.pointer()));
  } else {
    queue.Finish();
  }
}

template <typename T>
void ExpectBuffer(Queue& queue, const Buffer<T>& buffer, const std::vector<T>& expected) {
  auto actual = std::vector<T>(expected.size());
  buffer.Read(queue, actual.size(), actual);
  for (size_t i = 0; i < actual.size(); ++i) {
    const auto a = MaxshiftRead(actual[i]);
    const auto b = MaxshiftRead(expected[i]);
    if (a != b && !(std::isnan(a) && std::isnan(b))) {
      throw std::runtime_error("Incorrect maximum shift or changed padding at index " + ToString(i));
    }
  }
}

template <typename T>
void TestVector(Queue& queue, const std::vector<double>& values, const size_t offset, const size_t inc,
                const bool request_event = true) {
  auto input = std::vector<T>(offset + values.size() * inc + 11, MaxshiftWrite<T>(123));
  auto max = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < values.size(); ++i) {
    input[offset + i * inc] = MaxshiftWrite<T>(values[i]);
    max = std::max(max, MaxshiftRead(input[offset + i * inc]));
  }
  auto expected = input;
  for (size_t i = 0; i < values.size(); ++i) {
    expected[offset + i * inc] = MaxshiftWrite<T>(MaxshiftRead(input[offset + i * inc]) - max);
  }
  auto buffer = Buffer<T>(queue.GetContext(), input.size());
  buffer.Write(queue, input.size(), input);
  auto event = Event();
  auto raw_queue = queue();
  ExpectStatus(
      CallMaxshiftv<T>(values.size(), buffer(), offset, inc, &raw_queue, request_event ? event.pointer() : nullptr));
  WaitForMaxshift(queue, event, request_event);
  ExpectBuffer(queue, buffer, expected);
}

template <typename T>
void TestMatrix(Queue& queue, const size_t m, const size_t n, const bool rowmajor, const size_t offset,
                const size_t padding) {
  const auto ld = (rowmajor ? n : m) + padding;
  auto input = std::vector<T>(offset + ld * (rowmajor ? m : n) + 11, MaxshiftWrite<T>(123));
  const auto index = [=](const size_t row, const size_t col) {
    return offset + (rowmajor ? row * ld + col : col * ld + row);
  };
  for (size_t row = 0; row < m; ++row) {
    const auto max = row % 2 ? -static_cast<double>(row) - 1 : static_cast<double>(row) + 2;
    for (size_t col = 0; col < n; ++col) {
      input[index(row, col)] = MaxshiftWrite<T>(max - static_cast<double>((col + 1) % 17));
    }
    input[index(row, n - 1)] = MaxshiftWrite<T>(max);
  }
  auto expected = input;
  for (size_t row = 0; row < m; ++row) {
    auto max = -std::numeric_limits<double>::infinity();
    for (size_t col = 0; col < n; ++col) {
      max = std::max(max, MaxshiftRead(input[index(row, col)]));
    }
    for (size_t col = 0; col < n; ++col) {
      expected[index(row, col)] = MaxshiftWrite<T>(MaxshiftRead(input[index(row, col)]) - max);
    }
  }
  auto buffer = Buffer<T>(queue.GetContext(), input.size());
  buffer.Write(queue, input.size(), input);
  auto event = Event();
  auto raw_queue = queue();
  ExpectStatus(CallMaxshiftm<T>(rowmajor ? CLBlastLayoutRowMajor : CLBlastLayoutColMajor, m, n, buffer(), offset, ld,
                                &raw_queue, event.pointer()));
  WaitForMaxshift(queue, event, true);
  ExpectBuffer(queue, buffer, expected);
}

template <typename T>
size_t RunMaxshiftTests(Queue& queue) {
  if (!PrecisionSupported<T>(queue.GetDevice())) {
    return 0;
  }
  auto passed = size_t{0};
  for (const auto n : std::vector<size_t>{1, 2, 7, 63, 64, 65, 129, 4097, 131075}) {
    auto values = std::vector<double>(n);
    for (size_t i = 0; i < n; ++i) {
      values[i] = -static_cast<double>(i % 37) - 4;
    }
    values.back() = -1;
    TestVector<T>(queue, values, 0, 1);
    TestVector<T>(queue, values, 5, 3);
    passed += 2;
  }
  for (const auto& values : std::vector<std::vector<double>>{{4, 4, 4}, {-5, -1, -1, -7}, {3, -7, 12, 0}}) {
    TestVector<T>(queue, values, 0, 1, false);
    ++passed;
  }
  // Finite values below the old SMALLEST reduction sentinel, plus IEEE infinity/NaN cases.
  const auto limit = std::is_same<T, half>::value ? 65504.0 : static_cast<double>(std::numeric_limits<T>::max());
  TestVector<T>(queue, {-limit, -limit / 2, -limit * 0.75}, 0, 1);
  const auto inf = std::numeric_limits<double>::infinity();
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  TestVector<T>(queue, {-inf, -inf}, 0, 1);
  TestVector<T>(queue, {-1, inf, 3}, 0, 1);
  TestVector<T>(queue, {nan, -5, -1}, 0, 1);
  passed += 4;

  for (const auto& dims :
       std::vector<std::pair<size_t, size_t>>{{1, 1}, {1, 129}, {7, 1}, {7, 3}, {7, 63}, {65, 65}, {7, 513}}) {
    for (const auto rowmajor : {false, true}) {
      TestMatrix<T>(queue, dims.first, dims.second, rowmajor, 0, 0);
      TestMatrix<T>(queue, dims.first, dims.second, rowmajor, 5, 7);
      passed += 2;
    }
  }

  // Exercise the public C API's argument validation, independently of external BLAS libraries.
  auto buffer = Buffer<T>(queue.GetContext(), 8);
  auto raw_queue = queue();
  ExpectStatus(CallMaxshiftv<T>(0, buffer(), 0, 1, &raw_queue, nullptr), StatusCode::kInvalidDimension);
  ExpectStatus(CallMaxshiftv<T>(1, buffer(), 0, 0, &raw_queue, nullptr), StatusCode::kInvalidIncrementX);
  ExpectStatus(CallMaxshiftv<T>(16, buffer(), 0, 1, &raw_queue, nullptr), StatusCode::kInsufficientMemoryX);
  ExpectStatus(CallMaxshiftv<T>(1, nullptr, 0, 1, &raw_queue, nullptr), StatusCode::kInvalidVectorX);
  ExpectStatus(CallMaxshiftm<T>(CLBlastLayoutRowMajor, 0, 4, buffer(), 0, 4, &raw_queue, nullptr),
               StatusCode::kInvalidDimension);
  ExpectStatus(CallMaxshiftm<T>(CLBlastLayoutRowMajor, 2, 4, buffer(), 0, 3, &raw_queue, nullptr),
               StatusCode::kInvalidLeadDimA);
  ExpectStatus(CallMaxshiftm<T>(CLBlastLayoutColMajor, 2, 4, buffer(), 0, 1, &raw_queue, nullptr),
               StatusCode::kInvalidLeadDimA);
  ExpectStatus(CallMaxshiftm<T>(CLBlastLayoutRowMajor, 3, 4, buffer(), 0, 4, &raw_queue, nullptr),
               StatusCode::kInsufficientMemoryA);
  passed += 8;
  return passed;
}

}  // namespace clblast

int main(int argc, char* argv[]) {
  using namespace clblast;
  try {
    const auto arguments = RetrieveCommandLineArguments(argc, argv);
    auto help = std::string{};
    const auto platform_id =
        GetArgument(arguments, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
    const auto device_id =
        GetArgument(arguments, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
    const auto device = Device(Platform(platform_id), device_id);
    const auto context = Context(device);
    cl_command_queue_properties properties = 0;
    CheckError(clGetDeviceInfo(device(), CL_DEVICE_QUEUE_PROPERTIES, sizeof(properties), &properties, nullptr));
    auto passed = size_t{0};
    for (const auto out_of_order : {false, true}) {
      if (out_of_order && !(properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)) {
        continue;
      }
      auto error = cl_int{0};
      const auto raw_queue =
          clCreateCommandQueue(context(), device(), out_of_order ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0, &error);
      CheckError(error);
      auto queue = Queue(raw_queue);
      CheckError(clReleaseCommandQueue(raw_queue));
      passed += RunMaxshiftTests<half>(queue);
      passed += RunMaxshiftTests<float>(queue);
      passed += RunMaxshiftTests<double>(queue);
    }
    std::cout << passed << " maximum-shift C API cases passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
