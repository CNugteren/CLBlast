
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

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void FillVector(const int n, const int inc, const int offset,
                __global real* restrict dest, const real_arg arg_value) {
  const real value = GetRealArg(arg_value);
  const int tid = get_global_id(0);
  if (tid < n) {
    dest[tid*inc + offset] = value;
  }
}

// =================================================================================================

// TODO: Put variable in database
#define TRSV_BLOCK_SIZE 256

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void trsv_forward(int n,
                  const __global float *A, const int a_offset, int lda,
                  __global float *b, const int b_offset, int b_inc,
                  __global float *x, const int x_offset, int x_inc,
                  const int is_transposed, const int is_unit_diagonal) {
  __local float sx[TRSV_BLOCK_SIZE];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = 0; i < n; ++i) {
    real sum = b[i*b_inc + b_offset];
    for (int j = 0; j < i; ++j) {
      real a_value;
      if (is_transposed == 0) { a_value = A[i + j*lda + a_offset]; }
      else { a_value = A[j + i*lda + a_offset]; }
      sum -= a_value * sx[j];
    }
    sum -= x[i*x_inc + x_offset];
    if (is_unit_diagonal == 0) { sum /= A[i + i*lda + a_offset]; }
    sx[i] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int i = 0; i < n; ++i) {
    x[i*x_inc + x_offset] = sx[i];
  }
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void trsv_backward(int n,
                   const __global float *A, const int a_offset, int lda,
                   __global float *b, const int b_offset, int b_inc,
                   __global float *x, const int x_offset, int x_inc,
                   const int is_trans, const int is_unit_diagonal) {
  __local float sx[TRSV_BLOCK_SIZE];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = n - 1; i >= 0; --i) {
    real sum = b[i*b_inc + b_offset];
    for (int j = i + 1; j < n; ++j) {
      real a_value;
      if (is_trans == 0) { a_value = A[i + j*lda + a_offset]; }
      else { a_value = A[j + i*lda + a_offset]; }
      sum -= a_value * sx[j];
    }
    sum -= x[i*x_inc + x_offset];
    if (is_unit_diagonal == 0) { sum /= A[i + i*lda + a_offset]; }
    sx[i] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int i = 0; i < n; ++i) {
    x[i*x_inc + x_offset] = sx[i];
  }
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
