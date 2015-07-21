
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
  __local realT tile[TRA_WPT*TRA_DIM][TRA_DIM + TRA_PAD];

  // Loop over the work per thread
  #pragma unroll
  for (int w_one=0; w_one<TRA_WPT; ++w_one) {

    // Computes the identifiers for the source matrix. Note that the local and global dimensions
    // do not correspond to each other!
    const int id_one = get_group_id(1) * TRA_DIM + get_local_id(0);
    const int id_two = (get_group_id(0) * TRA_DIM + get_local_id(1))*TRA_WPT + w_one;

    // Loads data into the local memory
    realT value = src[id_two*(ld/TRA_WPT) + id_one];
    tile[get_local_id(0)*TRA_WPT + w_one][get_local_id(1)] = value;
  }

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Loads transposed data from the local memory
  realT v[TRA_WPT];
  #pragma unroll
  for (int w_one=0; w_one<TRA_WPT; ++w_one) {
    v[w_one] = tile[get_local_id(1)*TRA_WPT + w_one][get_local_id(0)];
  }

  // Performs the register-level transpose of the vectorized data
  realT results[TRA_WPT];
  #if TRA_WPT == 1
    results[0] = v[0];
  #elif TRA_WPT == 2
    results[0] = (realT) (v[0].x, v[1].x);
    results[1] = (realT) (v[0].y, v[1].y);
  #elif TRA_WPT == 4
    results[0] = (realT) (v[0].x, v[1].x, v[2].x, v[3].x);
    results[1] = (realT) (v[0].y, v[1].y, v[2].y, v[3].y);
    results[2] = (realT) (v[0].z, v[1].z, v[2].z, v[3].z);
    results[3] = (realT) (v[0].w, v[1].w, v[2].w, v[3].w);
  #elif TRA_WPT == 8
    results[0] = (realT) (v[0].s0, v[1].s0, v[2].s0, v[3].s0, v[4].s0, v[5].s0, v[6].s0, v[7].s0);
    results[1] = (realT) (v[0].s1, v[1].s1, v[2].s1, v[3].s1, v[4].s1, v[5].s1, v[6].s1, v[7].s1);
    results[2] = (realT) (v[0].s2, v[1].s2, v[2].s2, v[3].s2, v[4].s2, v[5].s2, v[6].s2, v[7].s2);
    results[3] = (realT) (v[0].s3, v[1].s3, v[2].s3, v[3].s3, v[4].s3, v[5].s3, v[6].s3, v[7].s3);
    results[4] = (realT) (v[0].s4, v[1].s4, v[2].s4, v[3].s4, v[4].s4, v[5].s4, v[6].s4, v[7].s4);
    results[5] = (realT) (v[0].s5, v[1].s5, v[2].s5, v[3].s5, v[4].s5, v[5].s5, v[6].s5, v[7].s5);
    results[6] = (realT) (v[0].s6, v[1].s6, v[2].s6, v[3].s6, v[4].s6, v[5].s6, v[6].s6, v[7].s6);
    results[7] = (realT) (v[0].s7, v[1].s7, v[2].s7, v[3].s7, v[4].s7, v[5].s7, v[6].s7, v[7].s7);
  #elif TRA_WPT == 16
    results[ 0] = (realT) (v[0].s0, v[1].s0, v[2].s0, v[3].s0, v[4].s0, v[5].s0, v[6].s0, v[7].s0, v[8].s0, v[9].s0, v[10].s0, v[11].s0, v[12].s0, v[13].s0, v[14].s0, v[15].s0);
    results[ 1] = (realT) (v[0].s1, v[1].s1, v[2].s1, v[3].s1, v[4].s1, v[5].s1, v[6].s1, v[7].s1, v[8].s1, v[9].s1, v[10].s1, v[11].s1, v[12].s1, v[13].s1, v[14].s1, v[15].s1);
    results[ 2] = (realT) (v[0].s2, v[1].s2, v[2].s2, v[3].s2, v[4].s2, v[5].s2, v[6].s2, v[7].s2, v[8].s2, v[9].s2, v[10].s2, v[11].s2, v[12].s2, v[13].s2, v[14].s2, v[15].s2);
    results[ 3] = (realT) (v[0].s3, v[1].s3, v[2].s3, v[3].s3, v[4].s3, v[5].s3, v[6].s3, v[7].s3, v[8].s3, v[9].s3, v[10].s3, v[11].s3, v[12].s3, v[13].s3, v[14].s3, v[15].s3);
    results[ 4] = (realT) (v[0].s4, v[1].s4, v[2].s4, v[3].s4, v[4].s4, v[5].s4, v[6].s4, v[7].s4, v[8].s4, v[9].s4, v[10].s4, v[11].s4, v[12].s4, v[13].s4, v[14].s4, v[15].s4);
    results[ 5] = (realT) (v[0].s5, v[1].s5, v[2].s5, v[3].s5, v[4].s5, v[5].s5, v[6].s5, v[7].s5, v[8].s5, v[9].s5, v[10].s5, v[11].s5, v[12].s5, v[13].s5, v[14].s5, v[15].s5);
    results[ 6] = (realT) (v[0].s6, v[1].s6, v[2].s6, v[3].s6, v[4].s6, v[5].s6, v[6].s6, v[7].s6, v[8].s6, v[9].s6, v[10].s6, v[11].s6, v[12].s6, v[13].s6, v[14].s6, v[15].s6);
    results[ 7] = (realT) (v[0].s7, v[1].s7, v[2].s7, v[3].s7, v[4].s7, v[5].s7, v[6].s7, v[7].s7, v[8].s7, v[9].s7, v[10].s7, v[11].s7, v[12].s7, v[13].s7, v[14].s7, v[15].s7);
    results[ 8] = (realT) (v[0].s8, v[1].s8, v[2].s8, v[3].s8, v[4].s8, v[5].s8, v[6].s8, v[7].s8, v[8].s8, v[9].s8, v[10].s8, v[11].s8, v[12].s8, v[13].s8, v[14].s8, v[15].s8);
    results[ 9] = (realT) (v[0].s9, v[1].s9, v[2].s9, v[3].s9, v[4].s9, v[5].s9, v[6].s9, v[7].s9, v[8].s9, v[9].s9, v[10].s9, v[11].s9, v[12].s9, v[13].s9, v[14].s9, v[15].s9);
    results[10] = (realT) (v[0].sA, v[1].sA, v[2].sA, v[3].sA, v[4].sA, v[5].sA, v[6].sA, v[7].sA, v[8].sA, v[9].sA, v[10].sA, v[11].sA, v[12].sA, v[13].sA, v[14].sA, v[15].sA);
    results[11] = (realT) (v[0].sB, v[1].sB, v[2].sB, v[3].sB, v[4].sB, v[5].sB, v[6].sB, v[7].sB, v[8].sB, v[9].sB, v[10].sB, v[11].sB, v[12].sB, v[13].sB, v[14].sB, v[15].sB);
    results[12] = (realT) (v[0].sC, v[1].sC, v[2].sC, v[3].sC, v[4].sC, v[5].sC, v[6].sC, v[7].sC, v[8].sC, v[9].sC, v[10].sC, v[11].sC, v[12].sC, v[13].sC, v[14].sC, v[15].sC);
    results[13] = (realT) (v[0].sD, v[1].sD, v[2].sD, v[3].sD, v[4].sD, v[5].sD, v[6].sD, v[7].sD, v[8].sD, v[9].sD, v[10].sD, v[11].sD, v[12].sD, v[13].sD, v[14].sD, v[15].sD);
    results[14] = (realT) (v[0].sE, v[1].sE, v[2].sE, v[3].sE, v[4].sE, v[5].sE, v[6].sE, v[7].sE, v[8].sE, v[9].sE, v[10].sE, v[11].sE, v[12].sE, v[13].sE, v[14].sE, v[15].sE);
    results[15] = (realT) (v[0].sF, v[1].sF, v[2].sF, v[3].sF, v[4].sF, v[5].sF, v[6].sF, v[7].sF, v[8].sF, v[9].sF, v[10].sF, v[11].sF, v[12].sF, v[13].sF, v[14].sF, v[15].sF);
  #endif

  // Stores the results into the destination matrix
  #pragma unroll
  for (int w_two=0; w_two<TRA_WPT; ++w_two) {
    const int id_one = get_global_id(0);
    const int id_two = get_global_id(1)*TRA_WPT + w_two;
    dest[id_two*(ld/TRA_WPT) + id_one] = results[w_two];
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
