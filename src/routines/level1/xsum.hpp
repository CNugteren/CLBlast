
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsum routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSUM_H_
#define CLBLAST_ROUTINES_XSUM_H_

#include "routine.hpp"
#include "routines/level1/xasum.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xsum: public Xasum<T> {
 public:

  // Members and methods from the base class
  using Xasum<T>::DoAsum;

  // Constructor
  Xsum(Queue &queue, EventPointer event, const std::string &name = "SUM"):
    Xasum<T>(queue, event, name) {
  }

  // Forwards to the regular absolute version. The implementation difference is realised in the
  // kernel through a pre-processor macro based on the name of the routine.
  void DoSum(const size_t n,
             const Buffer<T> &sum_buffer, const size_t sum_offset,
             const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc) {
    DoAsum(n, sum_buffer, sum_offset, x_buffer, x_offset, x_inc);
  }
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XSUM_H_
#endif
