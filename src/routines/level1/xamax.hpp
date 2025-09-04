
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xamax routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XAMAX_H_
#define CLBLAST_ROUTINES_XAMAX_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xamax : public Routine {
 public:
  // Constructor
  Xamax(Queue& queue, EventPointer event, const std::string& name = "AMAX");

  // Templated-precision implementation of the routine
  void DoAmax(const size_t n, const Buffer<unsigned int>& imax_buffer, const size_t imax_offset,
              const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc);
};

extern template class Xamax<half>;
extern template class Xamax<float>;
extern template class Xamax<double>;
extern template class Xamax<float2>;
extern template class Xamax<double2>;

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XAMAX_H_
#endif
