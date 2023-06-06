
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 2, see part 1 of the invert kernel for a description
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if defined(ROUTINE_INVERT)

// B21 = A21 * B11
__kernel
void TripleMatMul16Part1Lower(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart1(16, false, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B21 = -B22 * B21
__kernel
void TripleMatMul16Part2Lower(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart2(16, false, lm, n, dest, current_size, num_pages, block_size);
}

// B21 = A21 * B11
__kernel
void TripleMatMul32Part1Lower(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart1(32, false, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B21 = -B22 * B21
__kernel
void TripleMatMul32Part2Lower(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart2(32, false, lm, n, dest, current_size, num_pages, block_size);
}

// B21 = A21 * B11
__kernel
void TripleMatMul64Part1Lower(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart1(64, false, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B21 = -B22 * B21
__kernel
void TripleMatMul64Part2Lower(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart2(64, false, lm, n, dest, current_size, num_pages, block_size);
}

// =================================================================================================

// B12 =  A12 * B22
__kernel
void TripleMatMul16Part1Upper(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart1(16, true, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B12 = -B11 * B12
__kernel
void TripleMatMul16Part2Upper(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart2(16, true, lm, n, dest, current_size, num_pages, block_size);
}

// B12 =  A12 * B22
__kernel
void TripleMatMul32Part1Upper(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart1(32, true, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B12 = -B11 * B12
__kernel
void TripleMatMul32Part2Upper(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart2(32, true, lm, n, dest, current_size, num_pages, block_size);
}

// B12 =  A12 * B22
__kernel
void TripleMatMul64Part1Upper(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart1(64, true, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B12 = -B11 * B12
__kernel
void TripleMatMul64Part2Upper(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart2(64, true, lm, n, dest, current_size, num_pages, block_size);
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
