#include "routines/level1/xmin.hpp"

#include "utilities/utilities.hpp"

namespace clblast {
template <typename T>
void Xmin<T>::DoMin(const size_t n, const Buffer<unsigned int>& imin_buffer, const size_t imin_offset,
                    const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc) {
  DoAmax(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc);
}

template class Xmin<half>;
template class Xmin<float>;
template class Xmin<double>;
template class Xmin<float2>;
template class Xmin<double2>;
}  // namespace clblast