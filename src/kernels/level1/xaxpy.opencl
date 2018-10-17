
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xaxpy kernel. It contains one fast vectorized version in case of unit
// strides (incx=incy=1) and no offsets (offx=offy=0). Another version is more general, but doesn't
// support vector data-types. The general version has a batched implementation as well.
//
// This kernel uses the level-1 BLAS common tuning parameters.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Full version of the kernel with offsets and strided accesses
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xaxpy(const int n, const real_arg arg_alpha,
           const __global real* restrict xgm, const int x_offset, const int x_inc,
           __global real* ygm, const int y_offset, const int y_inc) {
  const real alpha = GetRealArg(arg_alpha);

  // Loops over the work that needs to be done (allows for an arbitrary number of threads)
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real xvalue = xgm[id*x_inc + x_offset];
    MultiplyAdd(ygm[id*y_inc + y_offset], alpha, xvalue);
  }
}

// Faster version of the kernel without offsets and strided accesses but with if-statement. Also
// assumes that 'n' is dividable by 'VW' and 'WPT'.
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XaxpyFaster(const int n, const real_arg arg_alpha,
                 const __global realV* restrict xgm,
                 __global realV* ygm) {
  const real alpha = GetRealArg(arg_alpha);

  const int num_usefull_threads = n / (VW * WPT);
  if (get_global_id(0) < num_usefull_threads) {
    #pragma unroll
    for (int _w = 0; _w < WPT; _w += 1) {
      const int id = _w*num_usefull_threads + get_global_id(0);
      realV xvalue = xgm[id];
      realV yvalue = ygm[id];
      ygm[id] = MultiplyAddVector(yvalue, alpha, xvalue);
    }
  }
}

// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XaxpyFastest(const int n, const real_arg arg_alpha,
                  const __global realV* restrict xgm,
                  __global realV* ygm) {
  const real alpha = GetRealArg(arg_alpha);

  #pragma unroll
  for (int _w = 0; _w < WPT; _w += 1) {
    const int id = _w*get_global_size(0) + get_global_id(0);
    realV xvalue = xgm[id];
    realV yvalue = ygm[id];
    ygm[id] = MultiplyAddVector(yvalue, alpha, xvalue);
  }
}

// =================================================================================================

// Full version of the kernel with offsets and strided accesses: batched version
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XaxpyBatched(const int n, const __constant real_arg* arg_alphas,
                  const __global real* restrict xgm, const __constant int* x_offsets, const int x_inc,
                  __global real* ygm, const __constant int* y_offsets, const int y_inc) {
  const int batch = get_group_id(1);
  const real alpha = GetRealArg(arg_alphas[batch]);

  // Loops over the work that needs to be done (allows for an arbitrary number of threads)
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real xvalue = xgm[id*x_inc + x_offsets[batch]];
    MultiplyAdd(ygm[id*y_inc + y_offsets[batch]], alpha, xvalue);
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
