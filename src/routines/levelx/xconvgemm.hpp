
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xconvgemm routine. The precision is implemented as a template argument.
// This implements batched convolution of a 4D input 'image' tensor, a 3D input 'kernel' matrix,
// resulting in a 4D output 'result' tensor.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XCONVGEMM_H_
#define CLBLAST_ROUTINES_XCONVGEMM_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xconvgemm : public Routine {
 public:
  // Constructor
  enum class ConvGemmMethod { kWithIm2Col, kSingleKernel };
  Xconvgemm(Queue& queue, EventPointer event, const std::string& name = "CONVGEMM",
            ConvGemmMethod method = ConvGemmMethod::kSingleKernel);

  // Templated-precision implementation of the routine
  void DoConvgemm(KernelMode kernel_mode, size_t channels, size_t height, size_t width, size_t kernel_h,
                  size_t kernel_w, size_t pad_h, size_t pad_w, size_t stride_h, size_t stride_w, size_t dilation_h,
                  size_t dilation_w, size_t num_kernels, size_t batch_count, const Buffer<T>& im_buffer,
                  size_t im_offset, const Buffer<T>& kernel_buffer, size_t kernel_offset,
                  const Buffer<T>& result_buffer, size_t result_offset);

 private:
  ConvGemmMethod method_;
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XCONVGEMM_H_
#endif
