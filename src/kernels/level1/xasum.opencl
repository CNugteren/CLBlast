
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xasum kernel. It implements a absolute sum computation using reduction
// kernels. Reduction is split in two parts. In the first (main) kernel the X vector is loaded,
// followed by a per-thread and a per-workgroup reduction. The second (epilogue) kernel
// is executed with a single workgroup only, computing the final result.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#if defined(cl_khr_work_group_uniform_arithmetic)
#pragma OPENCL EXTENSION cl_khr_work_group_uniform_arithmetic : enable
#elif defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

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
void Xasum(const int n,
           const __global real* restrict xgm, const int x_offset, const int x_inc,
           __global real* output) {
  __local real lm[WGS1];
  const int lid = get_local_id(0);
  const int wgid = get_group_id(0);
  const int num_groups = get_num_groups(0);

  // Performs loading and the first steps of the reduction
  real acc;
  SetToZero(acc);
  int id = wgid*WGS1 + lid;
  while (id < n) {
    real x = xgm[id*x_inc + x_offset];
    #if defined(ROUTINE_SUM) // non-absolute version
    #else
      AbsoluteValue(x);
    #endif
    Add(acc, acc, x);
    id += WGS1*num_groups;
  }
  lm[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Performs reduction in local memory and stores the per work group result
  #if defined(cl_khr_work_group_uniform_arithmetic) || defined(__opencl_c_work_group_collective_functions)
    real result = work_group_reduce_add(lm[lid])

    if (lid == 0) {
      output[wgid] = result;
    }
  #else
    #if defined(cl_khr_subgroups) || defined(__opencl_c_subgroups)
      lm[get_sub_group_local_id()] = sub_group_reduce_add(lm[lid]);
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int s = get_num_sub_groups() >> 1; s > 0; s >>= 1) {
        if (lid < s) {
          Add(lm[lid], lm[lid], lm[lid + s]);
        }
      }
    #else
      for (int s=WGS1/2; s>0; s=s>>1) {
        if (lid < s) {
          Add(lm[lid], lm[lid], lm[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    #endif

    if (lid == 0) {
      output[wgid] = lm[0];
    }
  #endif
}

// =================================================================================================

// The epilogue reduction kernel, performing the final bit of the operation. This kernel has to
// be launched with a single workgroup only.
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
#endif
void XasumEpilogue(const __global real* restrict input,
                   __global real* asum, const int asum_offset) {
  __local real lm[WGS2];
  const int lid = get_local_id(0);

  // Performs the first step of the reduction while loading the data
  Add(lm[lid], input[lid], input[lid + WGS2]);
  barrier(CLK_LOCAL_MEM_FENCE);

  // Performs reduction in local memory and stores the final result of the absolute value
  #if defined(cl_khr_work_group_uniform_arithmetic) || defined(__opencl_c_work_group_collective_functions)
    real result = work_group_reduce_add(lm[lid])

    if (lid == 0) {
      #if (PRECISION == 3232 || PRECISION == 6464) && defined(ROUTINE_ASUM)
        asum[asum_offset].x = real.x + real.y; // the result is a non-complex number
      #else
        asum[asum_offset] = real;
      #endif
    }
  #else
    #if defined(cl_khr_subgroups) || defined(__opencl_c_subgroups)
      lm[get_sub_group_local_id()] = sub_group_reduce_add(lm[lid]);
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int s = get_num_sub_groups() >> 1; s > 0; s >>= 1) {
        if (lid < s) {
          Add(lm[lid], lm[lid], lm[lid + s]);
        }
      }
    #else
      for (int s=WGS1/2; s>0; s=s>>1) {
        if (lid < s) {
          Add(lm[lid], lm[lid], lm[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    #endif

    if (lid == 0) {
      #if (PRECISION == 3232 || PRECISION == 6464) && defined(ROUTINE_ASUM)
        asum[asum_offset].x = lm[0].x + lm[0].y; // the result is a non-complex number
      #else
        asum[asum_offset] = lm[0];
      #endif
    }
  #endif

  // Computes the absolute value and stores the final result
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
