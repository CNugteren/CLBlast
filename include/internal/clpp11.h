
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a C++11 wrapper around some OpenCL C data-types, similar to Khronos' cl.hpp.
// The main differences are modern C++11 support and a straightforward implemenation of the basic
// needs (as required for this project). It also includes some extra functionality not available
// in cl.hpp, such as including the sources with a Program object and querying a Kernel's validity
// in terms of local memory usage.
//
// This file is adapted from the C++ bindings from the CLTune project and therefore contains the
// following copyright notice:
//
// =================================================================================================
//
// Copyright 2014 SURFsara
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//  http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =================================================================================================

#ifndef CLBLAST_CLPP11_H_
#define CLBLAST_CLPP11_H_

#include <utility> // std::swap
#include <algorithm> // std::copy
#include <string> // std::string
#include <vector> // std::vector
#include <stdexcept> // std::runtime_error

// Includes the normal OpenCL C header
#if defined(__APPLE__) || defined(__MACOSX)
  #include <OpenCL/opencl.h>
#else
  #include <CL/opencl.h>
#endif

namespace clblast {
// =================================================================================================

// Base class for any object
class Object {
 protected:

  // Error handling (NOTE: these functions are [[noreturn]])
  void Error(const std::string &message) const {
    throw std::runtime_error("Internal OpenCL error: "+message);
  }
  void Error(const cl_int status) const {
    throw std::runtime_error("Internal OpenCL error with status: "+std::to_string(status));
  }
};

// =================================================================================================

// Base class for objects which require memory management
class ObjectWithState: public Object {

};

// =================================================================================================

// C++11 version of cl_event
class Event: public Object {
 public:

  // Constructor based on the plain C data-type
  explicit Event(const cl_event event): event_(event) { }

  // New event
  Event(): event_() {}

  // Public functions
  size_t GetProfilingStart() const {
    auto bytes = size_t{0};
    clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_START, 0, nullptr, &bytes);
    auto result = size_t{0};
    clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_START, bytes, &result, nullptr);
    return result;
  }
  size_t GetProfilingEnd() const {
    auto bytes = size_t{0};
    clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_END, 0, nullptr, &bytes);
    auto result = size_t{0};
    clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_END, bytes, &result, nullptr);
    return result;
  }
  cl_int Wait() const {
    return clWaitForEvents(1, &event_);
  }

  // Accessors to the private data-member
  cl_event operator()() const { return event_; }
  cl_event& operator()() { return event_; }
 private:
  cl_event event_;
};

// =================================================================================================

// C++11 version of cl_platform_id
class Platform: public Object {
 public:

  // Constructor based on the plain C data-type
  explicit Platform(const cl_platform_id platform): platform_(platform) { }

  // Initialize the platform. Note that this constructor can throw exceptions!
  explicit Platform(const size_t platform_id) {
    auto num_platforms = cl_uint{0};
    auto status = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (status != CL_SUCCESS) { Error(status); }
    if (num_platforms == 0) { Error("no platforms found"); }
    auto platforms = std::vector<cl_platform_id>(num_platforms);
    status = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (status != CL_SUCCESS) { Error(status); }
    if (platform_id >= num_platforms) { Error("invalid platform ID "+std::to_string(platform_id)); }
    platform_ = platforms[platform_id];
  }

  // Accessors to the private data-member
  cl_platform_id operator()() const { return platform_; }
  cl_platform_id& operator()() { return platform_; }
 private:
  cl_platform_id platform_;
};

// =================================================================================================

// C++11 version of cl_device_id
class Device: public Object {
 public:

  // Constructor based on the plain C data-type
  explicit Device(const cl_device_id device): device_(device) { }

  // Initialize the device. Note that this constructor can throw exceptions!
  explicit Device(const Platform &platform, const cl_device_type type, const size_t device_id) {
    auto num_devices = cl_uint{0};
    auto status = clGetDeviceIDs(platform(), type, 0, nullptr, &num_devices);
    if (status != CL_SUCCESS) { Error(status); }
    if (num_devices == 0) { Error("no devices found"); }
    auto devices = std::vector<cl_device_id>(num_devices);
    status = clGetDeviceIDs(platform(), type, num_devices, devices.data(), nullptr);
    if (status != CL_SUCCESS) { Error(status); }
    if (device_id >= num_devices) { Error("invalid device ID "+std::to_string(device_id)); }
    device_ = devices[device_id];
  }

  // Public functions
  std::string Version()     const { return GetInfoString(CL_DEVICE_VERSION); }
  cl_device_type Type()     const { return GetInfo<cl_device_type>(CL_DEVICE_TYPE); }
  std::string Vendor()      const { return GetInfoString(CL_DEVICE_VENDOR); }
  std::string Name()        const { return GetInfoString(CL_DEVICE_NAME); }
  std::string Extensions()  const { return GetInfoString(CL_DEVICE_EXTENSIONS); }
  size_t MaxWorkGroupSize() const { return GetInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE); }
  cl_ulong LocalMemSize()   const { return GetInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE); }
  cl_uint MaxWorkItemDimensions() const {
    return GetInfo<cl_uint>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
  }
  std::vector<size_t> MaxWorkItemSizes() const {
    return GetInfoVector<size_t>(CL_DEVICE_MAX_WORK_ITEM_SIZES);
  }

  // Configuration-validity checks
  bool IsLocalMemoryValid(const size_t local_mem_usage) const {
    return (local_mem_usage <= LocalMemSize());
  }
  bool IsThreadConfigValid(const std::vector<size_t> &local) const {
    auto local_size = size_t{1};
    for (auto &item: local) { local_size *= item; }
    for (auto i=size_t{0}; i<local.size(); ++i) {
      if (local[i] > MaxWorkItemSizes()[i]) { return false; }
    }
    if (local_size > MaxWorkGroupSize()) { return false; }
    if (local.size() > MaxWorkItemDimensions()) { return false; }
    return true;
  }

  // Accessors to the private data-member
  cl_device_id operator()() const { return device_; }
  cl_device_id& operator()() { return device_; }
 private:

  // Helper functions
  template <typename T>
  T GetInfo(const cl_device_info info) const {
    auto bytes = size_t{0};
    clGetDeviceInfo(device_, info, 0, nullptr, &bytes);
    auto result = T(0);
    clGetDeviceInfo(device_, info, bytes, &result, nullptr);
    return result;
  }
  template <typename T>
  std::vector<T> GetInfoVector(const cl_device_info info) const {
    auto bytes = size_t{0};
    clGetDeviceInfo(device_, info, 0, nullptr, &bytes);
    auto result = std::vector<T>(bytes/sizeof(T));
    clGetDeviceInfo(device_, info, bytes, result.data(), nullptr);
    return result;
  }
  std::string GetInfoString(const cl_device_info info) const {
    auto bytes = size_t{0};
    clGetDeviceInfo(device_, info, 0, nullptr, &bytes);
    auto result = std::vector<char>(bytes);
    clGetDeviceInfo(device_, info, bytes, result.data(), nullptr);
    return std::string(result.data());
  }

  cl_device_id device_;
};

// =================================================================================================

// C++11 version of cl_context
class Context: public ObjectWithState {
 public:

  // Constructor based on the plain C data-type
  explicit Context(const cl_context context): context_(context) {
    clRetainContext(context_);
  }

  // Memory management
  explicit Context(const Device &device) {
    auto status = CL_SUCCESS;
    const cl_device_id dev = device();
    context_ = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &status);
    if (status != CL_SUCCESS) { Error(status); }
  }
  ~Context() {
    clReleaseContext(context_);
  }
  Context(const Context &other):
    context_(other.context_) {
    clRetainContext(context_);
  }
  Context& operator=(Context other) {
    swap(*this, other);
    return *this;
  }
  friend void swap(Context &first, Context &second) {
    std::swap(first.context_, second.context_);
  }

  // Accessors to the private data-member
  cl_context operator()() const { return context_; }
  cl_context& operator()() { return context_; }
 private:
  cl_context context_;
};

// =================================================================================================

// C++11 version of cl_program. Additionally holds the program's source code.
class Program: public ObjectWithState {
 public:

  // Note that there is no constructor based on the plain C data-type because of extra state

  // Memory management
  explicit Program(const Context &context, const std::string &source):
    length_(source.length()) {
      std::copy(source.begin(), source.end(), back_inserter(source_));
      source_ptr_ = source_.data();
      auto status = CL_SUCCESS;
      program_ = clCreateProgramWithSource(context(), 1, &source_ptr_, &length_, &status);
      if (status != CL_SUCCESS) { Error(status); }
    }
  ~Program() {
    clReleaseProgram(program_);
  }
  Program(const Program &other):
      length_(other.length_),
      source_(other.source_),
      source_ptr_(other.source_ptr_),
      program_(other.program_) {
    clRetainProgram(program_);
  }
  Program& operator=(Program other) {
    swap(*this, other);
    return *this;
  }
  /*
  TODO: Implement move construction/assignment?
  Program(Program &&other) {
    clRetainProgram(program_);
    swap(*this, other);
  }
  Program& operator=(Program &&other) {
    swap(*this, other);
    return *this;
  }*/
  friend void swap(Program &first, Program &second) {
    std::swap(first.length_, second.length_);
    std::swap(first.source_, second.source_);
    std::swap(first.source_ptr_, second.source_ptr_);
    std::swap(first.program_, second.program_);
  }

  // Public functions
  cl_int Build(const Device &device, const std::string &options) {
    const cl_device_id dev = device();
    return clBuildProgram(program_, 1, &dev, options.c_str(), nullptr, nullptr);
  }
  std::string GetBuildInfo(const Device &device) const {
    auto bytes = size_t{0};
    clGetProgramBuildInfo(program_, device(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &bytes);
    auto result = std::vector<char>(bytes);
    clGetProgramBuildInfo(program_, device(), CL_PROGRAM_BUILD_LOG, bytes, result.data(), nullptr);
    return std::string(result.data());
  }

  // Accessors to the private data-member
  cl_program operator()() const { return program_; }
  cl_program& operator()() { return program_; }
 private:
  size_t length_;
  std::vector<char> source_;
  const char* source_ptr_;
  cl_program program_;
};

// =================================================================================================

// C++11 version of cl_kernel
class Kernel: public ObjectWithState {
 public:

  // Constructor based on the plain C data-type
  explicit Kernel(const cl_kernel kernel): kernel_(kernel) {
    clRetainKernel(kernel_);
  }

  // Memory management
  explicit Kernel(const Program &program, const std::string &name) {
    auto status = CL_SUCCESS;
    kernel_ = clCreateKernel(program(), name.c_str(), &status);
    if (status != CL_SUCCESS) { Error(status); }
  }
  ~Kernel() {
    clReleaseKernel(kernel_);
  }
  Kernel(const Kernel &other):
    kernel_(other.kernel_) {
    clRetainKernel(kernel_);
  }
  Kernel& operator=(Kernel other) {
    swap(*this, other);
    return *this;
  }
  friend void swap(Kernel &first, Kernel &second) {
    std::swap(first.kernel_, second.kernel_);
  }

  // Public functions
  template <typename T> // Note: doesn't work with T=Buffer
  cl_int SetArgument(const cl_uint index, const T &value) {
    return clSetKernelArg(kernel_, index, sizeof(T), &value);
  }
  size_t LocalMemUsage(const Device &device) const {
    auto bytes = size_t{0};
    clGetKernelWorkGroupInfo(kernel_, device(), CL_KERNEL_LOCAL_MEM_SIZE, 0, nullptr, &bytes);
    auto result = size_t{0};
    clGetKernelWorkGroupInfo(kernel_, device(), CL_KERNEL_LOCAL_MEM_SIZE, bytes, &result, nullptr);
    return result;
  }

  // Accessors to the private data-member
  cl_kernel operator()() const { return kernel_; }
  cl_kernel& operator()() { return kernel_; }
 private:
  cl_kernel kernel_;
};

// =================================================================================================

// C++11 version of cl_command_queue
class CommandQueue: public ObjectWithState {
 public:

  // Constructor based on the plain C data-type
  explicit CommandQueue(const cl_command_queue queue): queue_(queue) {
    clRetainCommandQueue(queue_);
  }

  // Memory management
  explicit CommandQueue(const Context &context, const Device &device) {
    auto status = CL_SUCCESS;
    queue_ = clCreateCommandQueue(context(), device(), CL_QUEUE_PROFILING_ENABLE, &status);
    if (status != CL_SUCCESS) { Error(status); }
  }
  ~CommandQueue() {
    clReleaseCommandQueue(queue_);
  }
  CommandQueue(const CommandQueue &other):
    queue_(other.queue_) {
    clRetainCommandQueue(queue_);
  }
  CommandQueue& operator=(CommandQueue other) {
    swap(*this, other);
    return *this;
  }
  friend void swap(CommandQueue &first, CommandQueue &second) {
    std::swap(first.queue_, second.queue_);
  }

  // Public functions
  cl_int EnqueueKernel(const Kernel &kernel, const std::vector<size_t> &global,
                       const std::vector<size_t> &local, Event &event) {
    return clEnqueueNDRangeKernel(queue_, kernel(), static_cast<cl_uint>(global.size()), nullptr,
                                  global.data(), local.data(), 0, nullptr, &(event()));
  }
  Context GetContext() const {
    auto bytes = size_t{0};
    clGetCommandQueueInfo(queue_, CL_QUEUE_CONTEXT, 0, nullptr, &bytes);
    cl_context result;
    clGetCommandQueueInfo(queue_, CL_QUEUE_CONTEXT, bytes, &result, nullptr);
    return Context(result);
  }
  Device GetDevice() const {
    auto bytes = size_t{0};
    clGetCommandQueueInfo(queue_, CL_QUEUE_DEVICE, 0, nullptr, &bytes);
    cl_device_id result;
    clGetCommandQueueInfo(queue_, CL_QUEUE_DEVICE, bytes, &result, nullptr);
    return Device(result);
  }
  cl_int Finish() {
    return clFinish(queue_);
  }

  // Accessors to the private data-member
  cl_command_queue operator()() const { return queue_; }
  cl_command_queue& operator()() { return queue_; }
 private:
  cl_command_queue queue_;
};

// =================================================================================================

// C++11 version of cl_mem
class Buffer: public ObjectWithState {
 public:

  // Constructor based on the plain C data-type
  explicit Buffer(const cl_mem buffer): buffer_(buffer) {
    clRetainMemObject(buffer_);
  }

  // Memory management
  explicit Buffer(const Context &context, const cl_mem_flags flags, const size_t bytes) {
    auto status = CL_SUCCESS;
    buffer_ = clCreateBuffer(context(), flags, bytes, nullptr, &status);
    if (status != CL_SUCCESS) { Error(status); }
  }
  ~Buffer() {
    clReleaseMemObject(buffer_);
  }
  Buffer(const Buffer &other):
    buffer_(other.buffer_) {
    clRetainMemObject(buffer_);
  }
  Buffer& operator=(Buffer other) {
    swap(*this, other);
    return *this;
  }
  friend void swap(Buffer &first, Buffer &second) {
    std::swap(first.buffer_, second.buffer_);
  }

  // Public functions
  template <typename T>
  cl_int ReadBuffer(const CommandQueue &queue, const size_t bytes, T* host) {
    return clEnqueueReadBuffer(queue(), buffer_, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr);
  }
  template <typename T>
  cl_int ReadBuffer(const CommandQueue &queue, const size_t bytes, std::vector<T> &host) {
    return ReadBuffer(queue, bytes, host.data());
  }
  template <typename T>
  cl_int WriteBuffer(const CommandQueue &queue, const size_t bytes, const T* host) {
    return clEnqueueWriteBuffer(queue(), buffer_, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr);
  }
  template <typename T>
  cl_int WriteBuffer(const CommandQueue &queue, const size_t bytes, const std::vector<T> &host) {
    return WriteBuffer(queue, bytes, &host[0]);
  }
  size_t GetSize() const {
    auto bytes = size_t{0};
    auto status = clGetMemObjectInfo(buffer_, CL_MEM_SIZE, 0, nullptr, &bytes);
    if (status != CL_SUCCESS) { Error(status); }
    auto result = size_t{0};
    status = clGetMemObjectInfo(buffer_, CL_MEM_SIZE, bytes, &result, nullptr);
    if (status != CL_SUCCESS) { Error(status); }
    return result;
  }

  // Accessors to the private data-member
  cl_mem operator()() const { return buffer_; }
  cl_mem& operator()() { return buffer_; }
 private:
  cl_mem buffer_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_CLPP11_H_
#endif
