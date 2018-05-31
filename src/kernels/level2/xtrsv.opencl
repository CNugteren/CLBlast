
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains kernels to perform forward or backward substition, as used in the TRSV routine
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if defined(ROUTINE_TRSV)

__kernel
void FillVector(const int n, const int inc, const int offset,
                __global real* restrict dest, const real_arg arg_value) {
  const real value = GetRealArg(arg_value);
  const int tid = get_global_id(0);
  if (tid < n) {
    dest[tid*inc + offset] = value;
  }
}

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

#ifndef TRSV_BLOCK_SIZE
  #define TRSV_BLOCK_SIZE 32    // The block size for forward or backward substition
#endif

// =================================================================================================

__kernel __attribute__((reqd_work_group_size(TRSV_BLOCK_SIZE, 1, 1)))
void trsv_forward(int n,
                  const __global real *A, const int a_offset, int a_ld,
                  __global real *b, const int b_offset, int b_inc,
                  __global real *x, const int x_offset, int x_inc,
                  const int is_transposed, const int is_unit_diagonal, const int do_conjugate) {
  __local real alm[TRSV_BLOCK_SIZE][TRSV_BLOCK_SIZE];
  __local real xlm[TRSV_BLOCK_SIZE];
  const int tid = get_local_id(0);

  // Pre-loads the data into local memory
  if (tid < n) {
    Subtract(xlm[tid], b[tid*b_inc + b_offset], x[tid*x_inc + x_offset]);
    if (is_transposed == 0) {
      for (int i = 0; i < n; ++i) {
        alm[i][tid] = A[i + tid*a_ld + a_offset];
      }
    }
    else {
      for (int i = 0; i < n; ++i) {
        alm[i][tid] = A[tid + i*a_ld + a_offset];
      }
    }
    if (do_conjugate) {
      for (int i = 0; i < n; ++i) {
        COMPLEX_CONJUGATE(alm[i][tid]);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Computes the result (single-threaded for now)
  if (tid == 0) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < i; ++j) {
        MultiplySubtract(xlm[i], alm[i][j], xlm[j]);
      }
      if (is_unit_diagonal == 0) { DivideFull(xlm[i], xlm[i], alm[i][i]); }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Stores the results
  if (tid < n) {
    x[tid*x_inc + x_offset] = xlm[tid];
  }
}

__kernel __attribute__((reqd_work_group_size(TRSV_BLOCK_SIZE, 1, 1)))
void trsv_backward(int n,
                   const __global real *A, const int a_offset, int a_ld,
                   __global real *b, const int b_offset, int b_inc,
                   __global real *x, const int x_offset, int x_inc,
                   const int is_transposed, const int is_unit_diagonal, const int do_conjugate) {
  __local real alm[TRSV_BLOCK_SIZE][TRSV_BLOCK_SIZE];
  __local real xlm[TRSV_BLOCK_SIZE];
  const int tid = get_local_id(0);

  // Pre-loads the data into local memory
  if (tid < n) {
    Subtract(xlm[tid], b[tid*b_inc + b_offset], x[tid*x_inc + x_offset]);
    if (is_transposed == 0) {
      for (int i = 0; i < n; ++i) {
        alm[i][tid] = A[i + tid*a_ld + a_offset];
      }
    }
    else {
      for (int i = 0; i < n; ++i) {
        alm[i][tid] = A[tid + i*a_ld + a_offset];
      }
    }
    if (do_conjugate) {
      for (int i = 0; i < n; ++i) {
        COMPLEX_CONJUGATE(alm[i][tid]);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Computes the result (single-threaded for now)
  if (tid == 0) {
    for (int i = n - 1; i >= 0; --i) {
      for (int j = i + 1; j < n; ++j) {
        MultiplySubtract(xlm[i], alm[i][j], xlm[j]);
      }
      if (is_unit_diagonal == 0) { DivideFull(xlm[i], xlm[i], alm[i][i]); }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Stores the results
  if (tid < n) {
    x[tid*x_inc + x_offset] = xlm[tid];
  }
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
