
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Ekansh Jain
//
// This file contains the Xminmax kernel. It implements index of (absolute) min/max computation using
// reduction kernels. Reduction is split in two parts. In the first (main) kernel the X vector is
// loaded, followed by a per-thread and a per-workgroup reduction. The second (epilogue) kernel
// is executed with a single workgroup only, computing the final result.
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

// =================================================================================================

// The main reduction kernel, performing the loading and the majority of the operation
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
#endif
void Xminmax(const int n,
           const __global real* restrict xgm, const int x_offset, const int x_inc,
           __global real* restrict mgm, __global unsigned int* igm) {
  __local singlereal maxlm[WGS1];
  __local signlereal minlm[WGS1];
  __local unsigned int imaxlm[WGS1];
  __local unsigned int iminlm[WGS1];

  const int lid = get_local_id(0);
  const int wgid = get_group_id(0);
  const int num_groups = get_num_groups(0);

  // Performs loading and the first steps of the reduction
  #ifdef ABSOLUTE
    singlereal max = ZERO;
    singlereal min = SMALLEST;
  #else
    singlereal max = SMALLEST;
    singlereal min = SMALLEST;
  #endif

  unsigned int imax = 0;
  unsigned int imin = 0;

  int id = wgid*WGS1 + lid;
  while (id < n) {
    const int x_index = id*x_inc + x_offset;
    #if PRECISION == 3232 || PRECISION == 6464
      singlereal x = fabs(xgm[x_index].x) + fabs(xgm[x_index].y);
    #else
      singlereal x = xgm[x_index];
    #endif

    #ifdef ABSOLUTE
      singlereal xmin = -fabs(x);
      x = fabs(x);
    #else defined(ROUTINE_MIN) // non-absolute minimum version
      singlereal xmin = -x;
    #endif

    if (x > max) {
      max = x;
      imax = id;
    }
    if (xmin > min) {
      min = xmin;
      imin = id;
    }
    id += WGS1*num_groups;
  }
  maxlm[lid] = max;
  imaxlm[lid] = imax;
  minlm[lid] = min;
  iminlm[lid] = imin;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Performs reduction in local memory
  for (int s=WGS1/2; s>0; s=s>>1) {
    if (lid < s) {
      if (maxlm[lid + s] > maxlm[lid]) {
        maxlm[lid] = maxlm[lid + s];
        imaxlm[lid] = imaxlm[lid + s];
      }
      if (minlm[lid + s] > minlm[lid]) {
        minlm[lid] = minlm[lid + s];
        iminlm[lid] = iminlm[lid + s];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores the per-workgroup result
  if (lid == 0) {
    mgm[wgid] = maxlm[0];
    mgm[wgid + (WGS2 * 2)] = minlm[0];
    igm[wgid] = imaxlm[0];
    igm[wgid + (WGS2 * 2)] = iminlm[0];
  }
}

// =================================================================================================

// The epilogue reduction kernel, performing the final bit of the operation. This kernel has to
// be launched with a single workgroup only.
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
#endif
void XminmaxEpilogue(const __global singlereal* restrict mgm,
                   const __global unsigned int* restrict igm,
                   __global unsigned int* imax, const int imax_offset,
                   __global unsinged int* imin, const int imin_offset) {
  __local singlereal maxlm[WGS2];
  __local singlereal minlm[WGS2];
  __local unsigned int imaxlm[WGS2];
  __local unsigned int iminlm[WGS2];
  const int lid = get_local_id(0);

  // Performs the first step of the reduction while loading the data
  if (mgm[lid + WGS2] > mgm[lid]) {
    maxlm[lid] = mgm[lid + WGS2];
    imaxlm[lid] = igm[lid + WGS2];
  }
  else {
    maxlm[lid] = mgm[lid];
    imaxlm[lid] = igm[lid];
  }
  if (mgm[lid + (WGS2 * 3)] > mgm[lid + (WGS2 * 2)]) {
    maxlm[lid] = mgm[lid + (WGS2 * 3)];
    imaxlm[lid] = igm[lid + (WGS2 * 3)];
  }
  else {
    maxlm[lid] = mgm[lid + (WGS2 * 2)];
    imaxlm[lid] = igm[lid + (WGS2 * 2)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Performs reduction in local memory
  for (int s=WGS2/2; s>0; s=s>>1) {
    if (lid < s) {
      if (maxlm[lid + s] > maxlm[lid]) {
        maxlm[lid] = maxlm[lid + s];
        imaxlm[lid] = imaxlm[lid + s];
      }
      if (minlm[lid + s] > minlm[lid]) {
        minlm[lid] = minlm[lid + s];
        iminlm[lid] = iminlm[lid + s];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores the final result
  if (lid == 0) {
    imax[imax_offset] = imaxlm[0];
    imin[imin_offset] = iminlm[0];
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
