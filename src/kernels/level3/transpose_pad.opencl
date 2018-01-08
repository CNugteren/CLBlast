
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
INLINE_FUNC void _TransposePadMatrix(LOCAL_PTR real* tile,
                                     const int src_one, const int src_two,
                                     const int src_ld, const int src_offset,
                                     __global const real* restrict src,
                                     const int dest_one, const int dest_two,
                                     const int dest_ld, const int dest_offset,
                                     __global real* dest,
                                     const real alpha,
                                     const int do_conjugate) {

  // Loop over the work per thread
  #pragma unroll
  for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
    #pragma unroll
    for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

      // Computes the identifiers for the source matrix. Note that the local and global dimensions
      // do not correspond to each other!
      const int id_src_one = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(0);
      const int id_src_two = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(1);

      // Loads data into the local memory if the thread IDs are within bounds of the source matrix.
      // Otherwise, set the local memory value to zero.
      real value;
      SetToZero(value);
      if (id_src_two < src_two && id_src_one < src_one) {
        value = src[id_src_two*src_ld + id_src_one + src_offset];
      }
      const int tile_id0 = get_local_id(0)*PADTRA_WPT + _w_one;
      const int tile_id1 = get_local_id(1)*PADTRA_WPT + _w_two;
      tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0] = value;
    }
  }

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Loop over the work per thread
  #pragma unroll
  for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
    #pragma unroll
    for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

      // Computes the identifiers for the destination matrix
      const int id_dest_one = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(0);
      const int id_dest_two = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(1);

      // Stores the transposed value in the destination matrix
      if ((id_dest_one < dest_one) && (id_dest_two < dest_two)) {
        const int tile_id0 = get_local_id(1)*PADTRA_WPT + _w_one;
        const int tile_id1 = get_local_id(0)*PADTRA_WPT + _w_two;
        real value = tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0];
        if (do_conjugate == 1) { COMPLEX_CONJUGATE(value); }
        Multiply(dest[id_dest_two*dest_ld + id_dest_one + dest_offset], alpha, value);
      }
    }
  }
}

// Interface to the above function
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
  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
  _TransposePadMatrix(tile, src_one, src_two, src_ld, src_offset, src,
                      dest_one, dest_two, dest_ld, dest_offset, dest,
                      alpha, do_conjugate);
}

// =================================================================================================

// Transposes a matrix, while considering possible padding in the source matrix. Data is read from a
// padded source matrix, but only the actual data is written back to the transposed destination
// matrix. This kernel optionally checks for upper/lower triangular matrices.
INLINE_FUNC void _TransposeMatrix(LOCAL_PTR real* tile,
                                  const int src_one, const int src_two,
                                  const int src_ld, const int src_offset,
                                  __global const real* restrict src,
                                  const int dest_one, const int dest_two,
                                  const int dest_ld, const int dest_offset,
                                  __global real* dest,
                                  const real alpha,
                                  const int upper, const int lower,
                                  const int diagonal_imag_zero) {

  // Loop over the work per thread
  #pragma unroll
  for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
    #pragma unroll
    for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

      // Computes the identifiers for the source matrix. Note that the local and global dimensions
      // do not correspond to each other!
      const int id_src_one = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(0);
      const int id_src_two = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(1);

      // Loads data into the local memory if the thread IDs are within bounds of the source matrix.
      if ((id_src_one < src_one) && (id_src_two < src_two)) {
        real value = src[id_src_two*src_ld + id_src_one + src_offset];
        const int tile_id0 = get_local_id(0)*PADTRA_WPT + _w_one;
        const int tile_id1 = get_local_id(1)*PADTRA_WPT + _w_two;
        tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0] = value;
      }
    }
  }

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Loop over the work per thread
  #pragma unroll
  for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
    #pragma unroll
    for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

      // Computes the identifiers for the destination matrix
      const int id_dest_one = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(0);
      const int id_dest_two = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(1);

      // Masking in case of triangular matrices: updates only the upper or lower part
      bool condition = true;
      #if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)
        if (upper == 1) { condition = (id_dest_one >= id_dest_two); }
        else if (lower == 1) { condition = (id_dest_one <= id_dest_two); }
      #endif
      if (condition) {

        // Stores the transposed value in the destination matrix
        if ((id_dest_one < dest_one) && (id_dest_two < dest_two)) {
          const int tile_id0 = get_local_id(1)*PADTRA_WPT + _w_one;
          const int tile_id1 = get_local_id(0)*PADTRA_WPT + _w_two;
          real value = tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0];
          if (diagonal_imag_zero == 1 && id_dest_one == id_dest_two) { ImagToZero(value); }
          Multiply(dest[id_dest_two*dest_ld + id_dest_one + dest_offset], alpha, value);
        }
      }
    }
  }
}

// Interface to the above function
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
  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
  _TransposeMatrix(tile, src_one, src_two, src_ld, src_offset, src,
                   dest_one, dest_two, dest_ld, dest_offset, dest,
                   alpha, upper, lower, diagonal_imag_zero);
}

// =================================================================================================
#if defined(ROUTINE_GEMMBATCHED)

// Batched version of the above
__kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))
void TransposePadMatrixBatched(const int src_one, const int src_two,
                               const int src_ld, const __constant int* src_offsets,
                               __global const real* restrict src,
                               const int dest_one, const int dest_two,
                               const int dest_ld, const __constant int* dest_offsets,
                               __global real* dest,
                               const int do_conjugate) {
  const int batch = get_group_id(2);
  const int src_offset = src_offsets[batch];
  const int dest_offset = dest_offsets[batch];
  real alpha; SetToOne(alpha);
  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
  _TransposePadMatrix(tile, src_one, src_two, src_ld, src_offset, src,
                      dest_one, dest_two, dest_ld, dest_offset, dest,
                      alpha, do_conjugate);
}

// Batched version of the above
__kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))
void TransposeMatrixBatched(const int src_one, const int src_two,
                            const int src_ld, const __constant int* src_offsets,
                            __global const real* restrict src,
                            const int dest_one, const int dest_two,
                            const int dest_ld, const __constant int* dest_offsets,
                            __global real* dest) {
  const int batch = get_group_id(2);
  const int src_offset = src_offsets[batch];
  const int dest_offset = dest_offsets[batch];
  real alpha; SetToOne(alpha);
  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
  _TransposeMatrix(tile, src_one, src_two, src_ld, src_offset, src,
                   dest_one, dest_two, dest_ld, dest_offset, dest,
                   alpha, 0, 0, 0);
}

#endif
// =================================================================================================
#if defined(ROUTINE_GEMMSTRIDEDBATCHED)

// Strided-batched version of the above
__kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))
void TransposePadMatrixStridedBatched(const int src_one, const int src_two,
                                      const int src_ld, const int src_offset,
                                      const int src_stride, __global const real* restrict src,
                                      const int dest_one, const int dest_two,
                                      const int dest_ld, const int dest_offset,
                                      const int dest_stride, __global real* dest,
                                      const int do_conjugate) {
  const int batch = get_group_id(2);
  const int src_offset_batch = src_offset + src_stride * batch;
  const int dest_offset_batch = dest_offset + dest_stride * batch;
  real alpha; SetToOne(alpha);
  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
  _TransposePadMatrix(tile, src_one, src_two, src_ld, src_offset_batch, src,
                      dest_one, dest_two, dest_ld, dest_offset_batch, dest,
                      alpha, do_conjugate);
}

// Strided-batched version of the above
__kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))
void TransposeMatrixStridedBatched(const int src_one, const int src_two,
                                   const int src_ld, const int src_offset,
                                   const int src_stride, __global const real* restrict src,
                                   const int dest_one, const int dest_two,
                                   const int dest_ld, const int dest_offset,
                                   const int dest_stride, __global real* dest) {
  const int batch = get_group_id(2);
  const int src_offset_batch = src_offset + src_stride * batch;
  const int dest_offset_batch = dest_offset + dest_stride * batch;
  real alpha; SetToOne(alpha);
  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
  _TransposeMatrix(tile, src_one, src_two, src_ld, src_offset_batch, src,
                   dest_one, dest_two, dest_ld, dest_offset_batch, dest,
                   alpha, 0, 0, 0);
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
