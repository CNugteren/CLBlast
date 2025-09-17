
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains all the common code to perform (partial) matrix inverting.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XINVERT_H_
#define CLBLAST_ROUTINES_XINVERT_H_

#include <cstddef>
#include <string>


#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
class Xinvert : public Routine {
 public:
  // Constructor
  Xinvert(Queue& queue, EventPointer event, const std::string& name = "INVERT");

  // Inverts diagonal square blocks of a matrix
  void InvertMatrixDiagonalBlocks(const Layout layout, const Triangle triangle, const Diagonal diag, const size_t n,
                                  const size_t block_size, const Buffer<T>& src, const size_t offset,
                                  const size_t ld_src, Buffer<T>& dest);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XINVERT_H_
#endif
