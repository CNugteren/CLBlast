
// =================================================================================================
// This file is part of the CLBlast project.
//
// This file contains the Xmaxshiftm kernel. It subtracts each row's maximum from every element
// of that row in-place. Each workgroup reduces one row in local memory before applying the shift.
// Both row-major and column-major storage are supported, including offsets and leading dimensions.
//
// Launch with global size (WGS1, m) and local size (WGS1, 1). WGS1 must be a power of two, including
// when RELAX_WORKGROUP_SIZE is enabled. The matrix has m rows and n columns; a_ld is at least n
// for row-major storage and at least m for column-major storage.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef WGS1
  #define WGS1 64     // The local work-group size for each row
#endif

// Maximum ordering is only defined for real data-types.
#if PRECISION == 16 || PRECISION == 32 || PRECISION == 64

// =================================================================================================

// Computes and subtracts the row maximum. All threads finish reading the row before it is modified.
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
#endif
void Xmaxshiftm(const int m, const int n,
               __global real* agm, const int a_offset, const int a_ld,
               const int is_rowmajor) {
  __local real maxlm[WGS1];
  const int lid = get_local_id(0);
  const int row = get_group_id(1);

  // This condition is uniform across the workgroup, so no threads skip a barrier independently
  if (row >= m) { return; }
  const int row_offset = a_offset + (is_rowmajor ? row*a_ld : row);
  const int col_inc = is_rowmajor ? 1 : a_ld;

  // Performs loading and the first steps of the reduction
  real max = -INFINITY;
  for (int col = lid; col < n; col += WGS1) {
    const real x = agm[row_offset + col*col_inc];
    if (x > max) { max = x; }
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

  // Subtracts the row maximum without modifying padding or elements outside the matrix
  max = maxlm[0];
  for (int col = lid; col < n; col += WGS1) {
    const int a_index = row_offset + col*col_inc;
    agm[a_index] = agm[a_index] - max;
  }
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
