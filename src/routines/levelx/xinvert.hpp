
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains all the common code to perform (partial) matrix inverting.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XINVERT_H_
#define CLBLAST_ROUTINES_XINVERT_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
class Xinvert: public Routine {
 public:

  // Constructor
  Xinvert(Queue &queue, EventPointer event, const std::string &name = "INVERT");

  // Inverts diagonal square blocks of a matrix
  void InvertMatrixDiagonalBlocks(const Layout layout, const Triangle triangle, const Diagonal diag,
                                  const size_t n, const size_t block_size,
                                  const Buffer<T> &src, const size_t offset, const size_t ld_src,
                                  Buffer<T> &dest);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XINVERT_H_
#endif
