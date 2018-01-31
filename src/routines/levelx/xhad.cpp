
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhad class (see the header for information about the class).
//
// =================================================================================================

#include "routines/levelx/xhad.hpp"

#include <string>
#include <vector>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xhad<T>::Xhad(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
#include "../../kernels/level1/level1.opencl"
#include "../../kernels/level1/xaxpy.opencl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xhad<T>::DoHad(const size_t n, const T alpha,
                    const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                    const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc, const T beta,
                    const Buffer<T> &z_buffer, const size_t z_offset, const size_t z_inc) {

  // Makes sure all dimensions are larger than zero
  if (n == 0) { throw BLASError(StatusCode::kInvalidDimension); }

  // Tests the vectors for validity
  TestVectorX(n, x_buffer, x_offset, x_inc);
  TestVectorY(n, y_buffer, y_offset, y_inc);
  TestVectorY(n, z_buffer, z_offset, z_inc); // TODO: Make a TestVectorZ function with error codes

}

// =================================================================================================

// Compiles the templated class
template class Xhad<half>;
template class Xhad<float>;
template class Xhad<double>;
template class Xhad<float2>;
template class Xhad<double2>;

// =================================================================================================
} // namespace clblast
