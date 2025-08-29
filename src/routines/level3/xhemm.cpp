
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhemm class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xhemm.hpp"

#include <string>
#include <vector>

#include "routines/common.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xhemm<T>::Xhemm(Queue& queue, EventPointer event, const std::string& name) : Xgemm<T>(queue, event, name) {}

// =================================================================================================

// The main routine
template <typename T>
void Xhemm<T>::DoHemm(const Layout layout, const Side side, const Triangle triangle, const size_t m, const size_t n,
                      const T alpha, const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
                      const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld, const T beta,
                      const Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld) {
  // Makes sure all dimensions are larger than zero
  if ((m == 0) || (n == 0)) {
    throw BLASError(StatusCode::kInvalidDimension);
  }

  // Computes the k dimension. This is based on whether or not the hermitian matrix is A (on the
  // left) or B (on the right) in the Xgemm routine.
  auto k = (side == Side::kLeft) ? m : n;

  // Checks for validity of the squared A matrix
  TestMatrixA(k, k, a_buffer, a_offset, a_ld);

  // Determines which kernel to run based on the layout (the Xgemm kernel assumes column-major as
  // default) and on whether we are dealing with an upper or lower triangle of the hermitian matrix
  bool is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
                   (triangle == Triangle::kLower && layout == Layout::kRowMajor));
  auto kernel_name = (is_upper) ? "HermUpperToSquared" : "HermLowerToSquared";

  // Temporary buffer for a copy of the hermitian matrix
  auto temp_herm = Buffer<T>(context_, k * k);

  // Creates a general matrix from the hermitian matrix to be able to run the regular Xgemm
  // routine afterwards
  auto kernel = Kernel(program_, kernel_name);

  // Sets the arguments for the hermitian-to-squared kernel
  kernel.SetArgument(0, static_cast<int>(k));
  kernel.SetArgument(1, static_cast<int>(a_ld));
  kernel.SetArgument(2, static_cast<int>(a_offset));
  kernel.SetArgument(3, a_buffer());
  kernel.SetArgument(4, static_cast<int>(k));
  kernel.SetArgument(5, static_cast<int>(k));
  kernel.SetArgument(6, 0);
  kernel.SetArgument(7, temp_herm());

  // Uses the common padding kernel's thread configuration. This is allowed, since the
  // hermitian-to-squared kernel uses the same parameters.
  auto global = std::vector<size_t>{Ceil(CeilDiv(k, db_["PAD_WPTX"]), db_["PAD_DIMX"]),
                                    Ceil(CeilDiv(k, db_["PAD_WPTY"]), db_["PAD_DIMY"])};
  auto local = std::vector<size_t>{db_["PAD_DIMX"], db_["PAD_DIMY"]};
  auto kernelEvent = Event();
  RunKernel(kernel, queue_, device_, global, local, kernelEvent.pointer());

  // Synchronize now: 'DoGemm' does not accept a list of events to wait for
  kernelEvent.WaitForCompletion();

  // Runs the regular Xgemm code with either "C := AB+C" or ...
  if (side == Side::kLeft) {
    DoGemm(layout, Transpose::kNo, Transpose::kNo, m, n, k, alpha, temp_herm, 0, k, b_buffer, b_offset, b_ld, beta,
           c_buffer, c_offset, c_ld);
  }

  // ... with "C := BA+C". Note that A and B are now reversed.
  else {
    try {
      DoGemm(layout, Transpose::kNo, Transpose::kNo, m, n, k, alpha, b_buffer, b_offset, b_ld, temp_herm, 0, k, beta,
             c_buffer, c_offset, c_ld);
    } catch (BLASError& e) {
      // A and B are now reversed, so also reverse the error codes returned from the Xgemm routine
      switch (e.status()) {
        case StatusCode::kInvalidMatrixA:
          throw BLASError(StatusCode::kInvalidMatrixB, e.details());
        case StatusCode::kInvalidMatrixB:
          throw BLASError(StatusCode::kInvalidMatrixA, e.details());
        case StatusCode::kInvalidLeadDimA:
          throw BLASError(StatusCode::kInvalidLeadDimB, e.details());
        case StatusCode::kInvalidLeadDimB:
          throw BLASError(StatusCode::kInvalidLeadDimA, e.details());
        case StatusCode::kInsufficientMemoryA:
          throw BLASError(StatusCode::kInsufficientMemoryB, e.details());
        case StatusCode::kInsufficientMemoryB:
          throw BLASError(StatusCode::kInsufficientMemoryA, e.details());
        default:
          throw;
      }
    }
  }
}

// =================================================================================================

// Compiles the templated class
template class Xhemm<float2>;
template class Xhemm<double2>;

// =================================================================================================
}  // namespace clblast
