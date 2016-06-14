
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common kernels shared among different BLAS functions. This file contains
// kernels to copy and pad matrices in various ways, including:
// 1) copying into a larger matrix by adding padding
// 2) copying into a smaller matrix by optionally removing padding. This is the general version
//    without restrictions, see the 'copy.opencl' file for a faster but more restricted copy kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Copies a matrix from source to destination. The output is padded with zero values in case the
// destination matrix dimensions are larger than the source matrix dimensions. Additionally, the ld
// value and offset can be different.
__attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
__kernel void CopyPadMatrix(const int src_one, const int src_two,
                            const int src_ld, const int src_offset,
                            __global const real* restrict src,
                            const int dest_one, const int dest_two,
                            const int dest_ld, const int dest_offset,
                            __global real* dest,
                            const int do_conjugate) {

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
        if (do_conjugate == 1) { COMPLEX_CONJUGATE(value); }
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
__kernel void CopyMatrix(const int src_one, const int src_two,
                         const int src_ld, const int src_offset,
                         __global const real* restrict src,
                         const int dest_one, const int dest_two,
                         const int dest_ld, const int dest_offset,
                         __global real* dest,
                         const int upper, const int lower,
                         const int diagonal_imag_zero) {

  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int w_one=0; w_one<PAD_WPTX; ++w_one) {
    const int id_one = (get_group_id(0)*PAD_WPTX + w_one) * PAD_DIMX + get_local_id(0);
    #pragma unroll
    for (int w_two=0; w_two<PAD_WPTY; ++w_two) {
      const int id_two = (get_group_id(1)*PAD_WPTY + w_two) * PAD_DIMY + get_local_id(1);

      // Masking in case of triangular matrices: updates only the upper or lower part
      bool condition = true;
      #if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)
        if (upper == 1) { condition = (id_two >= id_one); }
        else if (lower == 1) { condition = (id_two <= id_one); }
      #endif
      if (condition) {

        // Copies the value into the destination matrix. This is always within bounds of the source
        // matrix, as we know that the destination matrix is smaller or equal to the source.
        if (id_two < dest_two && id_one < dest_one) {
          real value = src[id_two*src_ld + id_one + src_offset];
          if (diagonal_imag_zero == 1 && id_one == id_two) { ImagToZero(value); }
          dest[id_two*dest_ld + id_one + dest_offset] = value;
        }
      }
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
