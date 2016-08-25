
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains kernels to convert triangular matrices to/from general matrices.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if defined(ROUTINE_TRMM)

// Kernel to populate a squared triangular matrix, given that the triangle which holds the data is
// stored as the lower-triangle of the input matrix. This uses the padding kernel's parameters.
__kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
void TriaLowerToSquared(const int src_dim,
                        const int src_ld, const int src_offset,
                        __global const real* restrict src,
                        const int dest_dim,
                        const int dest_ld, const int dest_offset,
                        __global real* dest,
                        const int unit_diagonal) {

  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int w_one=0; w_one<PAD_WPTX; ++w_one) {
    const int id_one = (get_group_id(0)*PAD_WPTX + w_one) * PAD_DIMX + get_local_id(0);
    #pragma unroll
    for (int w_two=0; w_two<PAD_WPTY; ++w_two) {
      const int id_two = (get_group_id(1)*PAD_WPTY + w_two) * PAD_DIMY + get_local_id(1);
      if (id_two < dest_dim && id_one < dest_dim) {

        // Loads data from the lower-triangular matrix
        real result;
        SetToZero(result);
        if (id_two < src_dim && id_one < src_dim) {
          if (id_two <= id_one) { result = src[id_two*src_ld + id_one + src_offset]; }
          if (id_two == id_one && unit_diagonal) { SetToOne(result); }
          // Else: result is zero
        }

        // Stores the result in the destination matrix
        dest[id_two*dest_ld + id_one + dest_offset] = result;
      }
    }
  }
}

// Same as above, but now the matrix' data is stored in the upper-triangle
__kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
void TriaUpperToSquared(const int src_dim,
                        const int src_ld, const int src_offset,
                        __global const real* restrict src,
                        const int dest_dim,
                        const int dest_ld, const int dest_offset,
                        __global real* dest,
                        const int unit_diagonal) {

  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int w_one=0; w_one<PAD_WPTX; ++w_one) {
    const int id_one = (get_group_id(0)*PAD_WPTX + w_one) * PAD_DIMX + get_local_id(0);
    #pragma unroll
    for (int w_two=0; w_two<PAD_WPTY; ++w_two) {
      const int id_two = (get_group_id(1)*PAD_WPTY + w_two) * PAD_DIMY + get_local_id(1);
      if (id_two < dest_dim && id_one < dest_dim) {

        // Loads data from the upper-triangular matrix
        real result;
        SetToZero(result);
        if (id_two < src_dim && id_one < src_dim) {
          if (id_one <= id_two) { result = src[id_two*src_ld + id_one + src_offset]; }
          if (id_one == id_two && unit_diagonal) { SetToOne(result); }
          // Else: result is zero
        }

        // Stores the result in the destination matrix
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
