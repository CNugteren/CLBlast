
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xnrm2 kernel. It implements a squared norm computation using reduction
// kernels. Reduction is split in two parts. In the first (main) kernel the X vector is squared,
// followed by a per-thread and a per-workgroup reduction. The second (epilogue) kernel
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

// The main reduction kernel, performing the multiplication and the majority of the operation
__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
void Xnrm2(const int n,
           const __global real* restrict xgm, const int x_offset, const int x_inc,
           __global real* output) {
  __local real lm[WGS1];
  const int lid = get_local_id(0);
  const int wgid = get_group_id(0);
  const int num_groups = get_num_groups(0);

  // Performs multiplication and the first steps of the reduction
  real acc;
  SetToZero(acc);
  int id = wgid*WGS1 + lid;
  while (id < n) {
    real x1 = xgm[id*x_inc + x_offset];
    real x2 = x1;
    COMPLEX_CONJUGATE(x2);
    MultiplyAdd(acc, x1, x2);
    id += WGS1*num_groups;
  }
  lm[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Performs reduction in local memory
  #pragma unroll
  for (int s=WGS1/2; s>0; s=s>>1) {
    if (lid < s) {
      Add(lm[lid], lm[lid], lm[lid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores the per-workgroup result
  if (lid == 0) {
    output[wgid] = lm[0];
  }
}

// =================================================================================================

// The epilogue reduction kernel, performing the final bit of the operation. This kernel has to
// be launched with a single workgroup only.
__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
void Xnrm2Epilogue(const __global real* restrict input,
                   __global real* nrm2, const int nrm2_offset) {
  __local real lm[WGS2];
  const int lid = get_local_id(0);

  // Performs the first step of the reduction while loading the data
  Add(lm[lid], input[lid], input[lid + WGS2]);
  barrier(CLK_LOCAL_MEM_FENCE);

  // Performs reduction in local memory
  #pragma unroll
  for (int s=WGS2/2; s>0; s=s>>1) {
    if (lid < s) {
      Add(lm[lid], lm[lid], lm[lid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Computes the square root and stores the final result
  if (lid == 0) {
    #if PRECISION == 3232 || PRECISION == 6464
      nrm2[nrm2_offset].x = sqrt(lm[0].x); // the result is a non-complex number
    #else
      nrm2[nrm2_offset] = sqrt(lm[0]);
    #endif
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
