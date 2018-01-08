
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
INLINE_FUNC void _CopyPadMatrix(const int src_one, const int src_two,
                                const int src_ld, const int src_offset,
                                __global const real* restrict src,
                                const int dest_one, const int dest_two,
                                const int dest_ld, const int dest_offset,
                                __global real* dest,
                                const real alpha,
                                const int do_conjugate) {

  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
    const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
    #pragma unroll
    for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
      const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);
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
        Multiply(dest[id_two*dest_ld + id_one + dest_offset], alpha, value);
      }
    }
  }
}

// Interface to the above function
__kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
void CopyPadMatrix(const int src_one, const int src_two,
                   const int src_ld, const int src_offset,
                   __global const real* restrict src,
                   const int dest_one, const int dest_two,
                   const int dest_ld, const int dest_offset,
                   __global real* dest,
                   const real_arg arg_alpha,
                   const int do_conjugate) {
  const real alpha = GetRealArg(arg_alpha);
  _CopyPadMatrix(src_one, src_two, src_ld, src_offset, src,
                 dest_one, dest_two, dest_ld, dest_offset, dest,
                 alpha, do_conjugate);
}

// =================================================================================================

// Same as above, but now un-pads a matrix. This kernel reads data from a padded source matrix, but
// writes only the actual data back to the destination matrix. Again, the ld value and offset can
// be different.
INLINE_FUNC void _CopyMatrix(const int src_one, const int src_two,
                             const int src_ld, const int src_offset,
                             __global const real* restrict src,
                             const int dest_one, const int dest_two,
                             const int dest_ld, const int dest_offset,
                             __global real* dest,
                             const real alpha,
                             const int upper, const int lower,
                             const int diagonal_imag_zero) {

  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
    const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
    #pragma unroll
    for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
      const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);

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
          Multiply(dest[id_two*dest_ld + id_one + dest_offset], alpha, value);
        }
      }
    }
  }
}

// Interface to the above function
__kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
void CopyMatrix(const int src_one, const int src_two,
                const int src_ld, const int src_offset,
                __global const real* restrict src,
                const int dest_one, const int dest_two,
                const int dest_ld, const int dest_offset,
                __global real* dest,
                const real_arg arg_alpha,
                const int upper, const int lower,
                const int diagonal_imag_zero) {
  const real alpha = GetRealArg(arg_alpha);
  _CopyMatrix(src_one, src_two, src_ld, src_offset, src,
              dest_one, dest_two, dest_ld, dest_offset, dest,
              alpha, upper, lower, diagonal_imag_zero);
}

// =================================================================================================
#if defined(ROUTINE_GEMMBATCHED)

// Batched version of the above
__kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
void CopyPadMatrixBatched(const int src_one, const int src_two,
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
  _CopyPadMatrix(src_one, src_two, src_ld, src_offset, src,
                 dest_one, dest_two, dest_ld, dest_offset, dest,
                 alpha, do_conjugate);
}

// Batched version of the above
__kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
void CopyMatrixBatched(const int src_one, const int src_two,
                       const int src_ld, const __constant int* src_offsets,
                       __global const real* restrict src,
                       const int dest_one, const int dest_two,
                       const int dest_ld, const __constant int* dest_offsets,
                       __global real* dest) {
  const int batch = get_group_id(2);
  const int src_offset = src_offsets[batch];
  const int dest_offset = dest_offsets[batch];
  real alpha; SetToOne(alpha);
  _CopyMatrix(src_one, src_two, src_ld, src_offset, src,
              dest_one, dest_two, dest_ld, dest_offset, dest,
              alpha, 0, 0, 0);
}

#endif
// =================================================================================================
#if defined(ROUTINE_GEMMSTRIDEDBATCHED)

// Strided-batched version of the above
__kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
void CopyPadMatrixStridedBatched(const int src_one, const int src_two,
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
  _CopyPadMatrix(src_one, src_two, src_ld, src_offset_batch, src,
                 dest_one, dest_two, dest_ld, dest_offset_batch, dest,
                 alpha, do_conjugate);
}

// Strided-batched version of the above
__kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
void CopyMatrixStridedBatched(const int src_one, const int src_two,
                              const int src_ld, const int src_offset,
                              const int src_stride, __global const real* restrict src,
                              const int dest_one, const int dest_two,
                              const int dest_ld, const int dest_offset,
                              const int dest_stride, __global real* dest) {
  const int batch = get_group_id(2);
  const int src_offset_batch = src_offset + src_stride * batch;
  const int dest_offset_batch = dest_offset + dest_stride * batch;
  real alpha; SetToOne(alpha);
  _CopyMatrix(src_one, src_two, src_ld, src_offset_batch, src,
              dest_one, dest_two, dest_ld, dest_offset_batch, dest,
              alpha, 0, 0, 0);
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
