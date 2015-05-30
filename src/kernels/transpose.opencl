
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common kernels shared among different BLAS functions. This file contains
// kernels to transpose matrices.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef TRA_DIM
  #define TRA_DIM 8    // Number of local threads in the two dimensions (x,y)
#endif
#ifndef TRA_WPT
  #define TRA_WPT 1    // Work per thread in one dimension and vector-width in the other
#endif
#ifndef TRA_PAD
  #define TRA_PAD 0    // Padding of the local memory to avoid bank-conflicts
#endif

// =================================================================================================

// Data-widths
#if TRA_WPT == 1
  typedef real realT;
#elif TRA_WPT == 2
  typedef real2 realT;
#elif TRA_WPT == 4
  typedef real4 realT;
#elif TRA_WPT == 8
  typedef real8 realT;
#elif TRA_WPT == 16
  typedef real16 realT;
#endif

// =================================================================================================

// Transposes and copies a matrix. Requires both matrices to be of the same dimensions and without
// offset. A more general version is available in 'padtranspose.opencl'.
__attribute__((reqd_work_group_size(TRA_DIM, TRA_DIM, 1)))
__kernel void TransposeMatrix(const int ld,
                              __global const realT* restrict src,
                              __global realT* dest) {

  // Local memory to store a tile of the matrix (for coalescing)
  __local real tile[TRA_WPT*TRA_DIM][TRA_WPT*TRA_DIM + TRA_PAD];

  // Loop over the work per thread
  #pragma unroll
  for (int w_one=0; w_one<TRA_WPT; ++w_one) {

    // Computes the identifiers for the source matrix. Note that the local and global dimensions
    // do not correspond to each other!
    const int id_one = get_group_id(1) * TRA_DIM + get_local_id(0);
    const int id_two = (get_group_id(0) * TRA_DIM + get_local_id(1))*TRA_WPT + w_one;

    // Loads data into the local memory
    realT value = src[id_two*(ld/TRA_WPT) + id_one];
    #if TRA_WPT == 1
      tile[get_local_id(1)*TRA_WPT + 0][get_local_id(0)*TRA_WPT + w_one] = value;
    #elif TRA_WPT == 2
      tile[get_local_id(1)*TRA_WPT + 0][get_local_id(0)*TRA_WPT + w_one] = value.x;
      tile[get_local_id(1)*TRA_WPT + 1][get_local_id(0)*TRA_WPT + w_one] = value.y;
    #elif TRA_WPT == 4
      tile[get_local_id(1)*TRA_WPT + 0][get_local_id(0)*TRA_WPT + w_one] = value.x;
      tile[get_local_id(1)*TRA_WPT + 1][get_local_id(0)*TRA_WPT + w_one] = value.y;
      tile[get_local_id(1)*TRA_WPT + 2][get_local_id(0)*TRA_WPT + w_one] = value.z;
      tile[get_local_id(1)*TRA_WPT + 3][get_local_id(0)*TRA_WPT + w_one] = value.w;
    #elif TRA_WPT == 8
      tile[get_local_id(1)*TRA_WPT + 0][get_local_id(0)*TRA_WPT + w_one] = value.s0;
      tile[get_local_id(1)*TRA_WPT + 1][get_local_id(0)*TRA_WPT + w_one] = value.s1;
      tile[get_local_id(1)*TRA_WPT + 2][get_local_id(0)*TRA_WPT + w_one] = value.s2;
      tile[get_local_id(1)*TRA_WPT + 3][get_local_id(0)*TRA_WPT + w_one] = value.s3;
      tile[get_local_id(1)*TRA_WPT + 4][get_local_id(0)*TRA_WPT + w_one] = value.s4;
      tile[get_local_id(1)*TRA_WPT + 5][get_local_id(0)*TRA_WPT + w_one] = value.s5;
      tile[get_local_id(1)*TRA_WPT + 6][get_local_id(0)*TRA_WPT + w_one] = value.s6;
      tile[get_local_id(1)*TRA_WPT + 7][get_local_id(0)*TRA_WPT + w_one] = value.s7;
    #elif TRA_WPT == 16
      tile[get_local_id(1)*TRA_WPT +  0][get_local_id(0)*TRA_WPT + w_one] = value.s0;
      tile[get_local_id(1)*TRA_WPT +  1][get_local_id(0)*TRA_WPT + w_one] = value.s1;
      tile[get_local_id(1)*TRA_WPT +  2][get_local_id(0)*TRA_WPT + w_one] = value.s2;
      tile[get_local_id(1)*TRA_WPT +  3][get_local_id(0)*TRA_WPT + w_one] = value.s3;
      tile[get_local_id(1)*TRA_WPT +  4][get_local_id(0)*TRA_WPT + w_one] = value.s4;
      tile[get_local_id(1)*TRA_WPT +  5][get_local_id(0)*TRA_WPT + w_one] = value.s5;
      tile[get_local_id(1)*TRA_WPT +  6][get_local_id(0)*TRA_WPT + w_one] = value.s6;
      tile[get_local_id(1)*TRA_WPT +  7][get_local_id(0)*TRA_WPT + w_one] = value.s7;
      tile[get_local_id(1)*TRA_WPT +  8][get_local_id(0)*TRA_WPT + w_one] = value.s8;
      tile[get_local_id(1)*TRA_WPT +  9][get_local_id(0)*TRA_WPT + w_one] = value.s9;
      tile[get_local_id(1)*TRA_WPT + 10][get_local_id(0)*TRA_WPT + w_one] = value.sA;
      tile[get_local_id(1)*TRA_WPT + 11][get_local_id(0)*TRA_WPT + w_one] = value.sB;
      tile[get_local_id(1)*TRA_WPT + 12][get_local_id(0)*TRA_WPT + w_one] = value.sC;
      tile[get_local_id(1)*TRA_WPT + 13][get_local_id(0)*TRA_WPT + w_one] = value.sD;
      tile[get_local_id(1)*TRA_WPT + 14][get_local_id(0)*TRA_WPT + w_one] = value.sE;
      tile[get_local_id(1)*TRA_WPT + 15][get_local_id(0)*TRA_WPT + w_one] = value.sF;
    #endif
  }

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Loop over the work per thread
  #pragma unroll
  for (int w_two=0; w_two<TRA_WPT; ++w_two) {

    // Computes the identifiers for the destination matrix
    const int id_one = get_global_id(0);
    const int id_two = get_global_id(1)*TRA_WPT + w_two;

    // Stores the transposed value in the destination matrix
    realT value;
    #if TRA_WPT == 1
      value = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 0];
    #elif TRA_WPT == 2
      value.x = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 0];
      value.y = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 1];
    #elif TRA_WPT == 4
      value.x = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 0];
      value.y = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 1];
      value.z = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 2];
      value.w = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 3];
    #elif TRA_WPT == 8
      value.s0 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 0];
      value.s1 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 1];
      value.s2 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 2];
      value.s3 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 3];
      value.s4 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 4];
      value.s5 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 5];
      value.s6 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 6];
      value.s7 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 7];
    #elif TRA_WPT == 16
      value.s0 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT +  0];
      value.s1 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT +  1];
      value.s2 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT +  2];
      value.s3 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT +  3];
      value.s4 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT +  4];
      value.s5 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT +  5];
      value.s6 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT +  6];
      value.s7 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT +  7];
      value.s8 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT +  8];
      value.s9 = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT +  9];
      value.sA = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 10];
      value.sB = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 11];
      value.sC = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 12];
      value.sD = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 13];
      value.sE = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 14];
      value.sF = tile[get_local_id(0)*TRA_WPT + w_two][get_local_id(1)*TRA_WPT + 15];
    #endif
    dest[id_two*(ld/TRA_WPT) + id_one] = value;
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)";

// =================================================================================================
