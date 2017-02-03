
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Ivan Shapovalov <intelfx@intelfx.name>
//
// This file contains definitions for the plugin API.
//
// =================================================================================================

#include "utilities/plugin.hpp"
#include "routines/common.hpp"
#include "clblast_plugin.h"

namespace clblast {

namespace plugin {
// =================================================================================================

void RunKernel(cl_kernel kernel, cl_command_queue queue, cl_device_id device,
               const std::vector<size_t> &global, const std::vector<size_t> &local,
               EventPointer event, const std::vector<cl_event> &waitForEvents)
{
  auto waitForEventsCpp = std::vector<Event>{};
  for (auto e: waitForEvents) {
    waitForEventsCpp.push_back(Event{e});
  }

  RunKernel(Kernel(kernel), Queue(queue), Device(device),
            global, local,
            event, waitForEventsCpp);
}

// =================================================================================================

// This namespace declaration seems necessary to implement the PadCopyTransposeMatrix<T>() template
inline namespace version_1 {

template <typename T>
void PadCopyTransposeMatrix(cl_command_queue queue, cl_device_id device,
                            const Database &db,
                            EventPointer event, const std::vector<cl_event> &waitForEvents,
                            const size_t src_one, const size_t src_two,
                            const size_t src_ld, const size_t src_offset,
                            cl_mem src,
                            const size_t dest_one, const size_t dest_two,
                            const size_t dest_ld, const size_t dest_offset,
                            cl_mem dest,
                            const T alpha,
                            cl_program program, const bool do_pad,
                            const bool do_transpose, const bool do_conjugate,
                            const bool upper, const bool lower,
                            const bool diagonal_imag_zero)
{
  auto waitForEventsCpp = std::vector<Event>{};
  for (auto e: waitForEvents) {
    waitForEventsCpp.push_back(Event{e});
  }

  PadCopyTransposeMatrix<T>(Queue{queue}, Device{device},
                            dynamic_cast<const clblast::Database &>(db),
                            event, waitForEventsCpp,
                            src_one, src_two,
                            src_ld, src_offset,
                            Buffer<T>{src},
                            dest_one, dest_two,
                            dest_ld, dest_offset,
                            Buffer<T>{dest},
                            alpha,
                            Program{program}, do_pad,
                            do_transpose, do_conjugate,
                            upper, lower,
                            diagonal_imag_zero);
}

template
void PadCopyTransposeMatrix<half>(cl_command_queue queue, cl_device_id device,
                                  const Database &db,
                                  EventPointer event, const std::vector<cl_event> &waitForEvents,
                                  const size_t src_one, const size_t src_two,
                                  const size_t src_ld, const size_t src_offset,
                                  cl_mem src,
                                  const size_t dest_one, const size_t dest_two,
                                  const size_t dest_ld, const size_t dest_offset,
                                  cl_mem dest,
                                  const half alpha,
                                  cl_program program, const bool do_pad,
                                  const bool do_transpose, const bool do_conjugate,
                                  const bool upper, const bool lower,
                                  const bool diagonal_imag_zero);

template
void PadCopyTransposeMatrix<float>(cl_command_queue queue, cl_device_id device,
                                   const Database &db,
                                   EventPointer event, const std::vector<cl_event> &waitForEvents,
                                   const size_t src_one, const size_t src_two,
                                   const size_t src_ld, const size_t src_offset,
                                   cl_mem src,
                                   const size_t dest_one, const size_t dest_two,
                                   const size_t dest_ld, const size_t dest_offset,
                                   cl_mem dest,
                                   const float alpha,
                                   cl_program program, const bool do_pad,
                                   const bool do_transpose, const bool do_conjugate,
                                   const bool upper, const bool lower,
                                   const bool diagonal_imag_zero);

template
void PadCopyTransposeMatrix<float2>(cl_command_queue queue, cl_device_id device,
                                    const Database &db,
                                    EventPointer event, const std::vector<cl_event> &waitForEvents,
                                    const size_t src_one, const size_t src_two,
                                    const size_t src_ld, const size_t src_offset,
                                    cl_mem src,
                                    const size_t dest_one, const size_t dest_two,
                                    const size_t dest_ld, const size_t dest_offset,
                                    cl_mem dest,
                                    const float2 alpha,
                                    cl_program program, const bool do_pad,
                                    const bool do_transpose, const bool do_conjugate,
                                    const bool upper, const bool lower,
                                    const bool diagonal_imag_zero);

template
void PadCopyTransposeMatrix<double>(cl_command_queue queue, cl_device_id device,
                                    const Database &db,
                                    EventPointer event, const std::vector<cl_event> &waitForEvents,
                                    const size_t src_one, const size_t src_two,
                                    const size_t src_ld, const size_t src_offset,
                                    cl_mem src,
                                    const size_t dest_one, const size_t dest_two,
                                    const size_t dest_ld, const size_t dest_offset,
                                    cl_mem dest,
                                    const double alpha,
                                    cl_program program, const bool do_pad,
                                    const bool do_transpose, const bool do_conjugate,
                                    const bool upper, const bool lower,
                                    const bool diagonal_imag_zero);

template
void PadCopyTransposeMatrix<double2>(cl_command_queue queue, cl_device_id device,
                                     const Database &db,
                                     EventPointer event, const std::vector<cl_event> &waitForEvents,
                                     const size_t src_one, const size_t src_two,
                                     const size_t src_ld, const size_t src_offset,
                                     cl_mem src,
                                     const size_t dest_one, const size_t dest_two,
                                     const size_t dest_ld, const size_t dest_offset,
                                     cl_mem dest,
                                     const double2 alpha,
                                     cl_program program, const bool do_pad,
                                     const bool do_transpose, const bool do_conjugate,
                                     const bool upper, const bool lower,
                                     const bool diagonal_imag_zero);

} // namespace version_X

// =================================================================================================

Database::~Database() = default;

Base::~Base() = default;

Interface::~Interface() = default;

Routine::~Routine() = default;

// =================================================================================================

Routine::Routine():
    kernel_mode(KernelMode::Default)
{}

// =================================================================================================

template <typename T>
void RoutineXgemm<T>::DoGemm(cl_event *, const Database *,
                             const Layout,
                             const Transpose, const Transpose,
                             const size_t, const size_t, const size_t,
                             const T,
                             const cl_mem, const size_t, const size_t,
                             const cl_mem, const size_t, const size_t,
                             const T,
                             const cl_mem, const size_t, const size_t) const {

  throw LogicError("plugin::RoutineXgemm: stub host routine called");
}

// =================================================================================================

const Routine *Plugin::PickStubRoutine() {

  // This class must inherit from all device-specific entry subclasses
  struct RoutineStub : RoutineXgemm<half>, RoutineXgemm<float>, RoutineXgemm<double>,
                       RoutineXgemm<float2>, RoutineXgemm<double2> {
  } static stub_routine;

  return &stub_routine;
}

// =================================================================================================
} // namespace plugin

} // namespace clblast

