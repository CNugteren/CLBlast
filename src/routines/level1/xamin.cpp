#include "xamin.hpp"

#include "utilities/utilities.hpp"

namespace clblast {
template <typename T>
void Xamin<T>::DoAmin(const size_t n, const Buffer<unsigned int>& imin_buffer, const size_t imin_offset,
                      const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc) {
  DoAmax(n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc);
}

template class Xamin<half>;
template class Xamin<float>;
template class Xamin<double>;
template class Xamin<float2>;
template class Xamin<double2>;
}  // namespace clblast