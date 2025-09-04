
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xamin routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XAMIN_H_
#define CLBLAST_ROUTINES_XAMIN_H_

#include "routine.hpp"
#include "routines/level1/xamax.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xamin : public Xamax<T> {
 public:
  // Members and methods from the base class
  using Xamax<T>::DoAmax;

  // Constructor
  Xamin(Queue& queue, EventPointer event, const std::string& name = "AMIN") : Xamax<T>(queue, event, name) {}

  // Forwards to the regular max-absolute version. The implementation difference is realised in the
  // kernel through a pre-processor macro based on the name of the routine.
  void DoAmin(const size_t n, const Buffer<unsigned int>& imin_buffer, const size_t imin_offset,
              const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc) {
    DoAmax(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc);
  }
};
extern template class Xamin<half>;
extern template class Xamin<float>;
extern template class Xamin<double>;
extern template class Xamin<float2>;
extern template class Xamin<double2>;

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XAMIN_H_
#endif
