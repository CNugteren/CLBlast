
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xswap kernel. It contains one fast vectorized version in case of unit
// strides (incx=incy=1) and no offsets (offx=offy=0). Another version is more general, but doesn't
// support vector data-types.
//
// This kernel uses the level-1 BLAS common tuning parameters.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Full version of the kernel with offsets and strided accesses
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
#endif
void Xswap(const int n,
           __global real* xgm, const int x_offset, const int x_inc,
           __global real* ygm, const int y_offset, const int y_inc) {

  // Loops over the work that needs to be done (allows for an arbitrary number of threads)
  for (int id = get_global_id(0); id<n; id += get_global_size(0)) {
    real temp = xgm[id*x_inc + x_offset];
    xgm[id*x_inc + x_offset] = ygm[id*y_inc + y_offset];
    ygm[id*y_inc + y_offset] = temp;
  }
}

// =================================================================================================

// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
#endif
void XswapFast(const int n,
               __global realV* xgm,
               __global realV* ygm) {
#if __has_builtin(__builtin_assume)
  __builtin_assume(n % VW == 0);
  __builtin_assume(n % WPT == 0);
  __builtin_assume(n % WGS == 0);
#endif
  #pragma unroll
  for (int _w = 0; _w < WPT; _w += 1) {
    const int id = _w*get_global_size(0) + get_global_id(0);
    realV temp = xgm[id];
    xgm[id] = ygm[id];
    ygm[id] = temp;
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
