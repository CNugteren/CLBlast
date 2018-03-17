
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xhad kernel. It contains one fast vectorized version in case of unit
// strides (incx=incy=incz=1) and no offsets (offx=offy=offz=0). Another version is more general,
// but doesn't support vector data-types. Based on the XAXPY kernels.
//
// This kernel uses the level-1 BLAS common tuning parameters.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// A vector-vector multiply function. See also level1.opencl for a vector-scalar version
INLINE_FUNC realV MultiplyVectorVector(realV cvec, const realV aval, const realV bvec) {
  #if VW == 1
    Multiply(cvec, aval, bvec);
  #elif VW == 2
    Multiply(cvec.x, aval.x, bvec.x);
    Multiply(cvec.y, aval.y, bvec.y);
  #elif VW == 4
    Multiply(cvec.x, aval.x, bvec.x);
    Multiply(cvec.y, aval.y, bvec.y);
    Multiply(cvec.z, aval.z, bvec.z);
    Multiply(cvec.w, aval.w, bvec.w);
  #elif VW == 8
    Multiply(cvec.s0, aval.s0, bvec.s0);
    Multiply(cvec.s1, aval.s1, bvec.s1);
    Multiply(cvec.s2, aval.s2, bvec.s2);
    Multiply(cvec.s3, aval.s3, bvec.s3);
    Multiply(cvec.s4, aval.s4, bvec.s4);
    Multiply(cvec.s5, aval.s5, bvec.s5);
    Multiply(cvec.s6, aval.s6, bvec.s6);
    Multiply(cvec.s7, aval.s7, bvec.s7);
  #elif VW == 16
    Multiply(cvec.s0, aval.s0, bvec.s0);
    Multiply(cvec.s1, aval.s1, bvec.s1);
    Multiply(cvec.s2, aval.s2, bvec.s2);
    Multiply(cvec.s3, aval.s3, bvec.s3);
    Multiply(cvec.s4, aval.s4, bvec.s4);
    Multiply(cvec.s5, aval.s5, bvec.s5);
    Multiply(cvec.s6, aval.s6, bvec.s6);
    Multiply(cvec.s7, aval.s7, bvec.s7);
    Multiply(cvec.s8, aval.s8, bvec.s8);
    Multiply(cvec.s9, aval.s9, bvec.s9);
    Multiply(cvec.sA, aval.sA, bvec.sA);
    Multiply(cvec.sB, aval.sB, bvec.sB);
    Multiply(cvec.sC, aval.sC, bvec.sC);
    Multiply(cvec.sD, aval.sD, bvec.sD);
    Multiply(cvec.sE, aval.sE, bvec.sE);
    Multiply(cvec.sF, aval.sF, bvec.sF);
  #endif
  return cvec;
}

// =================================================================================================

// Full version of the kernel with offsets and strided accesses
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xhad(const int n, const real_arg arg_alpha, const real_arg arg_beta,
          const __global real* restrict xgm, const int x_offset, const int x_inc,
          const __global real* restrict ygm, const int y_offset, const int y_inc,
          __global real* zgm, const int z_offset, const int z_inc) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Loops over the work that needs to be done (allows for an arbitrary number of threads)
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real xvalue = xgm[id*x_inc + x_offset];
    real yvalue = ygm[id*y_inc + y_offset];
    real zvalue = zgm[id*z_inc + z_offset];
    real result;
    real alpha_times_x;
    Multiply(alpha_times_x, alpha, xvalue);
    Multiply(result, alpha_times_x, yvalue);
    MultiplyAdd(result, beta, zvalue);
    zgm[id*z_inc + z_offset] = result;
  }
}

// Faster version of the kernel without offsets and strided accesses but with if-statement. Also
// assumes that 'n' is dividable by 'VW' and 'WPT'.
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XhadFaster(const int n, const real_arg arg_alpha, const real_arg arg_beta,
                const __global realV* restrict xgm, const __global realV* restrict ygm,
                __global realV* zgm) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  if (get_global_id(0) < n / (VW)) {
    #pragma unroll
    for (int _w = 0; _w < WPT; _w += 1) {
      const int id = _w*get_global_size(0) + get_global_id(0);
      realV xvalue = xgm[id];
      realV yvalue = ygm[id];
      realV zvalue = zgm[id];
      realV result;
      realV alpha_times_x;
      alpha_times_x = MultiplyVector(alpha_times_x, alpha, xvalue);
      result = MultiplyVectorVector(result, alpha_times_x, yvalue);
      zgm[id] = MultiplyAddVector(result, beta, zvalue);
    }
  }
}

// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XhadFastest(const int n, const real_arg arg_alpha, const real_arg arg_beta,
                 const __global realV* restrict xgm, const __global realV* restrict ygm,
                 __global realV* zgm) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  #pragma unroll
  for (int _w = 0; _w < WPT; _w += 1) {
    const int id = _w*get_global_size(0) + get_global_id(0);
    realV xvalue = xgm[id];
    realV yvalue = ygm[id];
    realV zvalue = zgm[id];
    realV result;
    realV alpha_times_x;
    alpha_times_x = MultiplyVector(alpha_times_x, alpha, xvalue);
    result = MultiplyVectorVector(result, alpha_times_x, yvalue);
    zgm[id] = MultiplyAddVector(result, beta, zvalue);
  }
}


// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
