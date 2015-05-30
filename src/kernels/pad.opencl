
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common kernels shared among different BLAS routines. This file contains
// kernels to copy and pad matrices in various ways, including:
// 1) copying into a larger matrix by adding padding
// 2) copying into a smaller matrix by removing padding
// 3) from upper/lower triangle into a full matrix
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef PAD_DIMX
  #define PAD_DIMX 8      // Local workgroup size in the first dimension (x)
#endif
#ifndef PAD_DIMY
  #define PAD_DIMY 8      // Local workgroup size in the second dimension (y)
#endif
#ifndef PAD_WPTX
  #define PAD_WPTX 1      // Work per thread in the first dimension (x)
#endif
#ifndef PAD_WPTY
  #define PAD_WPTY 1      // Work per thread in the second dimension (y)
#endif

// =================================================================================================

// Copies a matrix from source to destination. The output is padded with zero values in case the
// destination matrix dimensions are larger than the source matrix dimensions. Additionally, the ld
// value and offset can be different.
__attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
__kernel void PadMatrix(const int src_one, const int src_two,
                        const int src_ld, const int src_offset,
                        __global const real* restrict src,
                        const int dest_one, const int dest_two,
                        const int dest_ld, const int dest_offset,
                        __global real* dest) {

  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int w_one=0; w_one<PAD_WPTX; ++w_one) {
    const int id_one = (get_group_id(0)*PAD_WPTX + w_one) * PAD_DIMX + get_local_id(0);
    #pragma unroll
    for (int w_two=0; w_two<PAD_WPTY; ++w_two) {
      const int id_two = (get_group_id(1)*PAD_WPTY + w_two) * PAD_DIMY + get_local_id(1);
      if (id_two < dest_two && id_one < dest_one) {

        // Loads data if the thread IDs are within bounds of the source matrix. Otherwise, set the
        // value to be written to zero.
        real value;
        SetToZero(value);
        if (id_two < src_two && id_one < src_one) {
          value = src[id_two*src_ld + id_one + src_offset];
        }

        // Stores the value in the destination matrix
        dest[id_two*dest_ld + id_one + dest_offset] = value;
      }
    }
  }
}

// =================================================================================================

// Same as above, but now un-pads a matrix. This kernel reads data from a padded source matrix, but
// writes only the actual data back to the destination matrix. Again, the ld value and offset can
// be different.
__attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
__kernel void UnPadMatrix(const int src_one, const int src_two,
                          const int src_ld, const int src_offset,
                          __global const real* restrict src,
                          const int dest_one, const int dest_two,
                          const int dest_ld, const int dest_offset,
                          __global real* dest) {

  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int w_one=0; w_one<PAD_WPTX; ++w_one) {
    const int id_one = (get_group_id(0)*PAD_WPTX + w_one) * PAD_DIMX + get_local_id(0);
    #pragma unroll
    for (int w_two=0; w_two<PAD_WPTY; ++w_two) {
      const int id_two = (get_group_id(1)*PAD_WPTY + w_two) * PAD_DIMY + get_local_id(1);
      if (id_two < dest_two && id_one < dest_one) {

        // Copies the value into the destination matrix. This is always within bounds of the source
        // matrix, as we know that the destination matrix is smaller than the source.
        dest[id_two*dest_ld + id_one + dest_offset] = src[id_two*src_ld + id_one + src_offset];
      }
    }
  }
}

// =================================================================================================

// Kernel to populate a squared symmetric matrix, given that the triangle which holds the data is
// stored as the lower-triangle of the input matrix. This uses the padding kernel's parameters.
__attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
__kernel void SymmLowerToSquared(const int src_dim,
                                 const int src_ld, const int src_offset,
                                 __global const real* restrict src,
                                 const int dest_dim,
                                 const int dest_ld, const int dest_offset,
                                 __global real* dest) {

  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int w_one=0; w_one<PAD_WPTX; ++w_one) {
    const int id_one = (get_group_id(0)*PAD_WPTX + w_one) * PAD_DIMX + get_local_id(0);
    #pragma unroll
    for (int w_two=0; w_two<PAD_WPTY; ++w_two) {
      const int id_two = (get_group_id(1)*PAD_WPTY + w_two) * PAD_DIMY + get_local_id(1);
      if (id_two < dest_dim && id_one < dest_dim) {

        // Loads data from the lower-symmetric matrix
        real value;
        SetToZero(value);
        if (id_two < src_dim && id_one < src_dim) {
          if (id_two <= id_one) { value = src[id_two*src_ld + id_one + src_offset]; }
          else                  { value = src[id_one*src_ld + id_two + src_offset]; }
        }

        // Stores the value in the destination matrix
        dest[id_two*dest_ld + id_one + dest_offset] = value;
      }
    }
  }
}

// Same as above, but now the matrix' data is stored in the upper-triangle
__attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
__kernel void SymmUpperToSquared(const int src_dim,
                                 const int src_ld, const int src_offset,
                                 __global const real* restrict src,
                                 const int dest_dim,
                                 const int dest_ld, const int dest_offset,
                                 __global real* dest) {

  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int w_one=0; w_one<PAD_WPTX; ++w_one) {
    const int id_one = (get_group_id(0)*PAD_WPTX + w_one) * PAD_DIMX + get_local_id(0);
    #pragma unroll
    for (int w_two=0; w_two<PAD_WPTY; ++w_two) {
      const int id_two = (get_group_id(1)*PAD_WPTY + w_two) * PAD_DIMY + get_local_id(1);
      if (id_two < dest_dim && id_one < dest_dim) {

        // Loads data from the upper-symmetric matrix
        real value;
        SetToZero(value);
        if (id_two < src_dim && id_one < src_dim) {
          if (id_one <= id_two) { value = src[id_two*src_ld + id_one + src_offset]; }
          else                  { value = src[id_one*src_ld + id_two + src_offset]; }
        }

        // Stores the value in the destination matrix
        dest[id_two*dest_ld + id_one + dest_offset] = value;
      }
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)";

// =================================================================================================
