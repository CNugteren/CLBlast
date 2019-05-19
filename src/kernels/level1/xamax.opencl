
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xamax kernel. It implements index of (absolute) min/max computation using
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
__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
void Xamax(const int n,
           const __global real* restrict xgm, const int x_offset, const int x_inc,
           __global singlereal* maxgm, __global unsigned int* imaxgm) {
  __local singlereal maxlm[WGS1];
  __local unsigned int imaxlm[WGS1];
  const int lid = get_local_id(0);
  const int wgid = get_group_id(0);
  const int num_groups = get_num_groups(0);

  // Performs loading and the first steps of the reduction
  #if defined(ROUTINE_MAX) || defined(ROUTINE_MIN) || defined(ROUTINE_AMIN)
    singlereal max = SMALLEST;
  #else
    singlereal max = ZERO;
  #endif
  unsigned int imax = 0;
  int id = wgid*WGS1 + lid;
  while (id < n) {
    const int x_index = id*x_inc + x_offset;
    #if PRECISION == 3232 || PRECISION == 6464
      singlereal x = xgm[x_index].x;
    #else
      singlereal x = xgm[x_index];
    #endif
    #if defined(ROUTINE_MAX) // non-absolute maximum version
      // nothing special here
    #elif defined(ROUTINE_MIN) // non-absolute minimum version
      x = -x;
    #elif defined(ROUTINE_AMIN) // absolute minimum version
      x = -fabs(x);
    #else
      x = fabs(x);
    #endif
    if (x >= max) {
      max = x;
      imax = id*x_inc + x_offset;
    }
    id += WGS1*num_groups;
  }
  maxlm[lid] = max;
  imaxlm[lid] = imax;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Performs reduction in local memory
  for (int s=WGS1/2; s>0; s=s>>1) {
    if (lid < s) {
      if (maxlm[lid + s] >= maxlm[lid]) {
        maxlm[lid] = maxlm[lid + s];
        imaxlm[lid] = imaxlm[lid + s];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores the per-workgroup result
  if (lid == 0) {
    maxgm[wgid] = maxlm[0];
    imaxgm[wgid] = imaxlm[0];
  }
}

// =================================================================================================

// The epilogue reduction kernel, performing the final bit of the operation. This kernel has to
// be launched with a single workgroup only.
__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
void XamaxEpilogue(const __global singlereal* restrict maxgm,
                   const __global unsigned int* restrict imaxgm,
                   __global unsigned int* imax, const int imax_offset) {
  __local singlereal maxlm[WGS2];
  __local unsigned int imaxlm[WGS2];
  const int lid = get_local_id(0);

  // Performs the first step of the reduction while loading the data
  if (maxgm[lid + WGS2] >= maxgm[lid]) {
    maxlm[lid] = maxgm[lid + WGS2];
    imaxlm[lid] = imaxgm[lid + WGS2];
  }
  else {
    maxlm[lid] = maxgm[lid];
    imaxlm[lid] = imaxgm[lid];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Performs reduction in local memory
  for (int s=WGS2/2; s>0; s=s>>1) {
    if (lid < s) {
      if (maxlm[lid + s] >= maxlm[lid]) {
        maxlm[lid] = maxlm[lid + s];
        imaxlm[lid] = imaxlm[lid + s];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores the final result
  if (lid == 0) {
    imax[imax_offset] = imaxlm[0];
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
