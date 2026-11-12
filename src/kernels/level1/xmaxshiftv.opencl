
// =================================================================================================
// This file is part of the CLBlast project.
//
// This file contains the Xmaxshiftv kernels. They subtract the maximum value from every element
// of a real vector in-place. As in Xamax, reduction is split into a main and an epilogue kernel.
// A final element-wise kernel applies the shift after the reduction has completed.
//
// Launch Xmaxshiftv with local size WGS1 and allocate one temporary real per workgroup. Launch
// XmaxshiftvEpilogue with one workgroup of WGS2 threads, passing the number of main workgroups,
// and store its result in a separate one-element buffer. Finally launch XmaxshiftvShift with
// local size WGS2. The launches must be ordered through the queue or event dependencies.
// WGS1 and WGS2 must be powers of two, including when RELAX_WORKGROUP_SIZE is enabled.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef WGS1
  #define WGS1 64     // The local work-group size of the main kernel
#endif
#ifndef WGS2
  #define WGS2 64     // The local work-group size of the epilogue kernel
#endif

// Maximum ordering is only defined for real data-types.
#if PRECISION == 16 || PRECISION == 32 || PRECISION == 64

// =================================================================================================

// The main reduction kernel, storing one maximum per workgroup
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
#endif
void Xmaxshiftv(const int n,
               const __global real* restrict xgm, const int x_offset, const int x_inc,
               __global real* maxgm) {
  __local real maxlm[WGS1];
  const int lid = get_local_id(0);
  const int wgid = get_group_id(0);
  const int num_groups = get_num_groups(0);

  // Performs loading and the first steps of the reduction. Negative infinity also handles
  // all-negative vectors and values below the finite SMALLEST constant.
  real max = -INFINITY;
  int id = wgid*WGS1 + lid;
  while (id < n) {
    const real x = xgm[id*x_inc + x_offset];
    if (x > max) { max = x; }
    id += WGS1*num_groups;
  }
  maxlm[lid] = max;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Performs reduction in local memory
  for (int s=WGS1/2; s>0; s=s>>1) {
    if (lid < s) {
      if (maxlm[lid + s] > maxlm[lid]) { maxlm[lid] = maxlm[lid + s]; }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores the per-workgroup result
  if (lid == 0) { maxgm[wgid] = maxlm[0]; }
}

// =================================================================================================

// The epilogue reduction kernel. Launch with a single workgroup; maxgm and max must not overlap.
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
#endif
void XmaxshiftvEpilogue(const int num_groups,
                       const __global real* restrict maxgm, __global real* max) {
  __local real maxlm[WGS2];
  const int lid = get_local_id(0);

  // Loads an arbitrary number of partial maxima, including fewer than WGS2
  real acc = -INFINITY;
  for (int id = lid; id < num_groups; id += WGS2) {
    if (maxgm[id] > acc) { acc = maxgm[id]; }
  }
  maxlm[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Performs reduction in local memory
  for (int s=WGS2/2; s>0; s=s>>1) {
    if (lid < s) {
      if (maxlm[lid + s] > maxlm[lid]) { maxlm[lid] = maxlm[lid + s]; }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores the final maximum
  if (lid == 0) { max[0] = maxlm[0]; }
}

// =================================================================================================

// Subtracts the final maximum in-place. The maximum buffer must not overlap the vector.
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
#endif
void XmaxshiftvShift(const int n,
                    __global real* xgm, const int x_offset, const int x_inc,
                    const __global real* restrict maxgm) {
  const real max = maxgm[0];
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    const int x_index = id*x_inc + x_offset;
    xgm[x_index] = xgm[x_index] - max;
  }
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
