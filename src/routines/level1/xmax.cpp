#include "routines/level1/xmax.hpp"

#include "utilities/utilities.hpp"

namespace clblast {
template <typename T>
void Xmax<T>::DoMax(const size_t n, const Buffer<unsigned int>& imax_buffer, const size_t imax_offset,
                    const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc) {
  DoAmax(n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc);
}

template class Xmax<half>;
template class Xmax<float>;
template class Xmax<double>;
template class Xmax<float2>;
template class Xmax<double2>;
}  // namespace clblast