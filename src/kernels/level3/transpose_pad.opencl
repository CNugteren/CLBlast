
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common kernels shared among different BLAS functions. This file contains
// kernels to transpose matrices in various ways, including:
// 1) transposing into a larger matrix by adding padding
// 2) transposing into a smaller matrix by optionally removing padding. This is the general version
//    without restrictions, see the 'transpose.opencl' file for a faster but more restricted
//    transpose kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Transposes a matrix from source to destination. The output is padded with zero values in case the
// destination matrix dimensions are larger than the transposed source matrix dimensions.
__kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))
void TransposePadMatrix(const int src_one, const int src_two,
                        const int src_ld, const int src_offset,
                        __global const real* restrict src,
                        const int dest_one, const int dest_two,
                        const int dest_ld, const int dest_offset,
                        __global real* dest,
                        const real_arg arg_alpha,
                        const int do_conjugate) {
  const real alpha = GetRealArg(arg_alpha);

  // Local memory to store a tile of the matrix (for coalescing)
  __local real tile[PADTRA_WPT*PADTRA_TILE][PADTRA_WPT*PADTRA_TILE + PADTRA_PAD];

  // Loop over the work per thread
  #pragma unroll
  for (int w_one=0; w_one<PADTRA_WPT; ++w_one) {
    #pragma unroll
    for (int w_two=0; w_two<PADTRA_WPT; ++w_two) {

      // Computes the identifiers for the source matrix. Note that the local and global dimensions
      // do not correspond to each other!
      const int id_src_one = (get_group_id(1)*PADTRA_WPT + w_two) * PADTRA_TILE + get_local_id(0);
      const int id_src_two = (get_group_id(0)*PADTRA_WPT + w_one) * PADTRA_TILE + get_local_id(1);

      // Loads data into the local memory if the thread IDs are within bounds of the source matrix.
      // Otherwise, set the local memory value to zero.
      real value;
      SetToZero(value);
      if (id_src_two < src_two && id_src_one < src_one) {
        value = src[id_src_two*src_ld + id_src_one + src_offset];
      }
      tile[get_local_id(1)*PADTRA_WPT + w_two][get_local_id(0)*PADTRA_WPT + w_one] = value;
    }
  }

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Loop over the work per thread
  #pragma unroll
  for (int w_one=0; w_one<PADTRA_WPT; ++w_one) {
    #pragma unroll
    for (int w_two=0; w_two<PADTRA_WPT; ++w_two) {

      // Computes the identifiers for the destination matrix
      const int id_dest_one = (get_group_id(0)*PADTRA_WPT + w_one) * PADTRA_TILE + get_local_id(0);
      const int id_dest_two = (get_group_id(1)*PADTRA_WPT + w_two) * PADTRA_TILE + get_local_id(1);

      // Stores the transposed value in the destination matrix
      if ((id_dest_one < dest_one) && (id_dest_two < dest_two)) {
        real value = tile[get_local_id(0)*PADTRA_WPT + w_two][get_local_id(1)*PADTRA_WPT + w_one];
        if (do_conjugate == 1) { COMPLEX_CONJUGATE(value); }
        Multiply(dest[id_dest_two*dest_ld + id_dest_one + dest_offset], alpha, value);
      }
    }
  }
}

// =================================================================================================

// Transposes a matrix, while considering possible padding in the source matrix. Data is read from a
// padded source matrix, but only the actual data is written back to the transposed destination
// matrix. This kernel optionally checks for upper/lower triangular matrices.
__kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))
void TransposeMatrix(const int src_one, const int src_two,
                     const int src_ld, const int src_offset,
                     __global const real* restrict src,
                     const int dest_one, const int dest_two,
                     const int dest_ld, const int dest_offset,
                     __global real* dest,
                     const real_arg arg_alpha,
                     const int upper, const int lower,
                     const int diagonal_imag_zero) {
  const real alpha = GetRealArg(arg_alpha);

  // Local memory to store a tile of the matrix (for coalescing)
  __local real tile[PADTRA_WPT*PADTRA_TILE][PADTRA_WPT*PADTRA_TILE + PADTRA_PAD];

  // Loop over the work per thread
  #pragma unroll
  for (int w_one=0; w_one<PADTRA_WPT; ++w_one) {
    #pragma unroll
    for (int w_two=0; w_two<PADTRA_WPT; ++w_two) {

      // Computes the identifiers for the source matrix. Note that the local and global dimensions
      // do not correspond to each other!
      const int id_src_one = (get_group_id(1)*PADTRA_WPT + w_two) * PADTRA_TILE + get_local_id(0);
      const int id_src_two = (get_group_id(0)*PADTRA_WPT + w_one) * PADTRA_TILE + get_local_id(1);

      // Loads data into the local memory if the thread IDs are within bounds of the source matrix.
      if ((id_src_one < src_one) && (id_src_two < src_two)) {
        real value = src[id_src_two*src_ld + id_src_one + src_offset];
        tile[get_local_id(1)*PADTRA_WPT + w_two][get_local_id(0)*PADTRA_WPT + w_one] = value;
      }
    }
  }

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Loop over the work per thread
  #pragma unroll
  for (int w_one=0; w_one<PADTRA_WPT; ++w_one) {
    #pragma unroll
    for (int w_two=0; w_two<PADTRA_WPT; ++w_two) {

      // Computes the identifiers for the destination matrix
      const int id_dest_one = (get_group_id(0)*PADTRA_WPT + w_one) * PADTRA_TILE + get_local_id(0);
      const int id_dest_two = (get_group_id(1)*PADTRA_WPT + w_two) * PADTRA_TILE + get_local_id(1);

      // Masking in case of triangular matrices: updates only the upper or lower part
      bool condition = true;
      #if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)
        if (upper == 1) { condition = (id_dest_one >= id_dest_two); }
        else if (lower == 1) { condition = (id_dest_one <= id_dest_two); }
      #endif
      if (condition) {

        // Stores the transposed value in the destination matrix
        if ((id_dest_one < dest_one) && (id_dest_two < dest_two)) {
          real value = tile[get_local_id(0)*PADTRA_WPT + w_two][get_local_id(1)*PADTRA_WPT + w_one];
          if (diagonal_imag_zero == 1 && id_dest_one == id_dest_two) { ImagToZero(value); }
          Multiply(dest[id_dest_two*dest_ld + id_dest_one + dest_offset], alpha, value);
        }
      }
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
