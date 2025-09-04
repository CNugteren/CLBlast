#include "routines/level1/xsum.hpp"

#include "utilities/utilities.hpp"

namespace clblast {
template <typename T>
void Xsum<T>::DoSum(const size_t n, const Buffer<T>& sum_buffer, const size_t sum_offset, const Buffer<T>& x_buffer,
                    const size_t x_offset, const size_t x_inc) {
  DoAsum(n, sum_buffer, sum_offset, x_buffer, x_offset, x_inc);
}

template class Xsum<half>;
template class Xsum<float>;
template class Xsum<double>;
template class Xsum<float2>;
template class Xsum<double2>;
}  // namespace clblast