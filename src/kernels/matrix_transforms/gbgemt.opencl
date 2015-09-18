
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the general banded (gb) to general (ge) matrix transforms.
//
// This kernel uses the matrix-transforms common tuning parameters.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if defined(ROUTINE_GBMV)

// Kernel to transform a general banded matrix into a general matrix
__attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
__kernel void GeneralBandedToGeneral(const int src_one, const int src_two,
                                     const int src_ld, const int src_offset,
                                     __global const real* restrict src,
                                     const int dest_one, const int dest_two,
                                     const int dest_ld, const int dest_offset,
                                     __global real* dest,
                                     const int layout,
                                     const int kl, const int ku) {

  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int w_one=0; w_one<PAD_WPTX; ++w_one) {
    const int id_one = (get_group_id(0)*PAD_WPTX + w_one) * PAD_DIMX + get_local_id(0);
    #pragma unroll
    for (int w_two=0; w_two<PAD_WPTY; ++w_two) {
      const int id_two = (get_group_id(1)*PAD_WPTY + w_two) * PAD_DIMY + get_local_id(1);
      if (id_two < dest_two && id_one < dest_one) {
        real result;
        SetToZero(result);
        const int k = ku - id_two + id_one;
        if ((id_one >= id_two - ku) && (id_one < id_two + kl + 1)) {
          result = src[id_two*src_ld + k + src_offset];
        }
        dest[id_two*dest_ld + id_one + dest_offset] = result;
      }
    }
  }
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
