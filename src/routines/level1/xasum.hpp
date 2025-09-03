
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xasum routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XASUM_H_
#define CLBLAST_ROUTINES_XASUM_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xasum : public Routine {
 public:
  // Constructor
  Xasum(Queue& queue, EventPointer event, const std::string& name = "ASUM");

  // Templated-precision implementation of the routine
  void DoAsum(const size_t n, const Buffer<T>& asum_buffer, const size_t asum_offset, const Buffer<T>& x_buffer,
              const size_t x_offset, const size_t x_inc);
};

extern template class Xasum<half>;
extern template class Xasum<float>;
extern template class Xasum<double>;
extern template class Xasum<float2>;
extern template class Xasum<double2>;
// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XASUM_H_
#endif
