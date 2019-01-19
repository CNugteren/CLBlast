
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xconvgemm routine. The precision is implemented as a template argument.
// This implements batched convolution of a 4D input 'image' tensor, a 3D input 'kernel' matrix,
// resulting in a 4D output 'result' tensor.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XCONVGEMM_H_
#define CLBLAST_ROUTINES_XCONVGEMM_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xconvgemm: public Routine {
 public:

  // Constructor
  enum class ConvGemmMethod {kWithIm2Col, kSingleKernel};
  Xconvgemm(Queue &queue, EventPointer event, const std::string &name = "CONVGEMM",
            const ConvGemmMethod method = ConvGemmMethod::kSingleKernel);

  // Templated-precision implementation of the routine
  void DoConvgemm(const KernelMode kernel_mode,
                  const size_t channels, const size_t height, const size_t width,
                  const size_t kernel_h, const size_t kernel_w,
                  const size_t pad_h, const size_t pad_w,
                  const size_t stride_h, const size_t stride_w,
                  const size_t dilation_h, const size_t dilation_w,
                  const size_t num_kernels, const size_t batch_count,
                  const Buffer<T> &im_buffer, const size_t im_offset,
                  const Buffer<T> &kernel_buffer, const size_t kernel_offset,
                  const Buffer<T> &result_buffer, const size_t result_offset);

 private:
  const ConvGemmMethod method_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XCONVGEMM_H_
#endif
