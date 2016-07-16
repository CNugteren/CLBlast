
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is a generic GEMM kernel that works for all sizes and configurations: it doesn't require any
// pre and and post-processing kernels.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Main entry point of the kernel. This is the direct version.
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void XgemmDirect(const int kSizeM, const int kSizeN, const int kSizeK,
                          const real_arg arg_alpha,
                          const real_arg arg_beta,
                          const __global real* restrict agm, const int a_offset, const int a_ld,
                          const __global real* restrict bgm, const int b_offset, const int b_ld,
                          __global real* cgm, const int c_offset, const int c_ld,
                          const int a_transpose, const int b_transpose, const int c_transpose,
                          const int a_conjugate, const int b_conjugate) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Thread identifiers
  const int mid = get_global_id(0); // Row ID of cgm
  const int nid = get_global_id(1); // Col ID of cgm

  // Allows for incomplete workgroups
  if (mid < kSizeM && nid < kSizeN) {

    // Computes a single element
    real acc;
    SetToZero(acc);
    for (int k=0; k<kSizeK; ++k) {
      const int a_index = (a_transpose) ? mid*a_ld + k : k*a_ld + mid;
      const int b_index = (b_transpose) ? nid*b_ld + k : k*b_ld + nid;
      real a_val = agm[a_index + a_offset];
      real b_val = bgm[b_index + b_offset];
      if (a_conjugate) { COMPLEX_CONJUGATE(a_val); }
      if (b_conjugate) { COMPLEX_CONJUGATE(b_val); }
      MultiplyAdd(acc, a_val, b_val);
    }

    // Determines the destination index
    const int c_index = (c_transpose) ? mid*c_ld + nid : nid*c_ld + mid;

    // Computes the result
    real result;
    AXPBY(result, alpha, acc, beta, cgm[c_index + c_offset]);

    // Stores the result
    cgm[c_index + c_offset] = result;
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
