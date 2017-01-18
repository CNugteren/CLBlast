
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains kernels to invert squared diagonal blocks of a matrix. These kernels are based
// on the TRSM implementation in the CUDA version of Magma version 2.2.0 and the poster "Triangular
// Linear System Solver for GPU with CUDA and OpenCL" by Peng Du, Stanimire Tomov, Piotr Luszczek,
// and Jack Dongarra.
//
// =================================================================================================
//
//  Let A be an block_size*block_size lower triangular matrix, and B its inverse.
//  Then the block decomposition
//  
//      [ A11   0  ] * [ B11   0  ] = [ I 0 ]
//      [ A21  A22 ]   [ B21  B22 ]   [ 0 I ]
//  
//  yields
//  
//      A11*B11 = I            ==>  B11 =  A11^{-1},
//      A22*B22 = I            ==>  B22 =  A22^{-1},
//      A21*B11 + A22*B21 = 0  ==>  B21 = -A22^{-1}*A21*B11 = -B22*A21*B11.
//  
//  The InvertDiagonalBlock kernel inverts A11 and A22.
//  The TripleMatMul routines multiply:
//  part 1:  B21 =  A21 * B11,
//  part 2:  B21 = -B22 * B21.
//  
//  At this level, inner block is current_size=16, with one 4 x 4 work-group per inner block. Each
//  submatrix Aij and Bij is current_size x current_size. The submatrix dimension is multiplied by 2
//  at each level, so the next level is current_size*2 = 32. A 'page' is the next bigger block,
//  here current_size*2=32,
//                 [ B11   0  ]
//  which contains [ B21  B22 ].
//  Outer blocks are block_size x block_size.
//  
//  A21 may have < current_size rows, but is guaranteed to have current_size cols since A22 is on
//  the right. This makes a single check easy to do.
//  
//  B is stored in workspace that is a full multiple of block_size x block_size; no checks needed.
//  
//  We split this into part1 & part2 to synchronize all blocks and make sure
//  that writes to B12 are observed by all blocks.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if defined(ROUTINE_INVERT)

#define LOCALX 17 // 16 + 1 to avoid bank conflicts
#define LOCALY 16

// =================================================================================================

// Inverts a diagonal block of INTERNAL_BLOCK_SIZE by INTERNAL_BLOCK_SIZE elements in a larger matrix
__kernel __attribute__((reqd_work_group_size(INTERNAL_BLOCK_SIZE, 1, 1)))
void InvertDiagonalBlock(int n, __global const real* restrict src, const int src_offset, const int src_ld,
                         __global real* restrict dest, const int outer_block_size,
                         const int unit_diagonal, const int is_upper)
{
  const int thread_index = get_local_id(0);
  const int block_index = get_group_id(0);

  // Sets the offset for this particular block in the source and destination matrices
  const int src_block_offset = block_index * (INTERNAL_BLOCK_SIZE + src_ld * INTERNAL_BLOCK_SIZE) + src_offset;
  const int num_inner_blocks = outer_block_size / INTERNAL_BLOCK_SIZE;
  const int dest_block_offset = (block_index / num_inner_blocks) * outer_block_size * outer_block_size + // go to the (block_index / num_inner_blocks) outer outer_block_size*outer_block_size block,
                                (block_index % num_inner_blocks) * (outer_block_size*INTERNAL_BLOCK_SIZE + INTERNAL_BLOCK_SIZE); // then to the (block_index % num_inner_blocks) inner INTERNAL_BLOCK_SIZE*INTERNAL_BLOCK_SIZE block inside that

  // Local memory to store the inverted block of INTERNAL_BLOCK_SIZE by INTERNAL_BLOCK_SIZE
  __local real lm[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];

  // Loads the source lower triangle into local memory. Any values in the upper triangle or
  // outside of the matrix are set to zero
  #pragma unroll
  for (int j = 0; j < INTERNAL_BLOCK_SIZE; ++j) {
    const bool condition = (is_upper) ? (thread_index <= j && block_index*INTERNAL_BLOCK_SIZE + j < n) :
                                        (thread_index >= j && block_index*INTERNAL_BLOCK_SIZE + thread_index < n);
    if (condition) {
      lm[thread_index][j] = src[j*src_ld + thread_index + src_block_offset];
    }
    else {
      SetToZero(lm[thread_index][j]);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // Inverts the diagonal
  real inverted_diagonal;
  SetToOne(inverted_diagonal);
  if (unit_diagonal == 0) {
    const real diagonal_value = lm[thread_index][thread_index];
    if (!IsZero(diagonal_value)) { // Only for non-singular values and values inside the matrix
      DivideReal(inverted_diagonal, inverted_diagonal, diagonal_value);
    }
  }
  lm[thread_index][thread_index] = inverted_diagonal;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Upper-triangular
  if (is_upper) {

    // Computes the elements 0:j-1 of the j-th column
    for (int j = 1; j < INTERNAL_BLOCK_SIZE; ++j) {
      if (thread_index < j) {
        real sum;
        SetToZero(sum);
        #pragma unroll
        for (int k = 0; k < j; ++k) {
          MultiplyAdd(sum, lm[thread_index][k], lm[k][j]);
        }
        real diagonal_value = lm[j][j];
        Negate(diagonal_value);
        Multiply(lm[thread_index][j], diagonal_value, sum);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  // Lower triangular
  else {

    // Computes the elements j+1:INTERNAL_BLOCK_SIZE-1 of the j-th column
    for (int j = INTERNAL_BLOCK_SIZE - 2; j >= 0; --j) {
      if (thread_index > j) {
        real sum;
        SetToZero(sum);
        #pragma unroll
        for (int k = j + 1; k < INTERNAL_BLOCK_SIZE; ++k) {
          MultiplyAdd(sum, lm[thread_index][k], lm[k][j]);
        }
        Multiply(lm[thread_index][j], -lm[j][j], sum);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  
  // Writes the result to global memory
  #pragma unroll
  for (int j = 0; j < INTERNAL_BLOCK_SIZE; ++j) {
    dest[j*outer_block_size + thread_index + dest_block_offset] = lm[thread_index][j];
  }
}

// =================================================================================================

// Triple matrix-multiplication kernel: C = A * B
inline void TripleMatMul(const int size, const bool upper, const int part, __local real* blm, int n,
                         __global const real* agm, __global const real* bgm, __global real* cgm,
                         const int lda, const int ldb, const int ldc,
                         int current_size, int num_pages, const int block_size) {

  // Emulates a 3D grid: NX * (NY * num_pages)
  const int by   = get_group_id(1) / num_pages;
  const int page = get_group_id(1) % num_pages;
  const int lidx = get_local_id(0);
  const int lidy = get_local_id(1);
  const int ibx  = get_group_id(0) * (get_local_size(0)*get_local_size(1));
  const int iby  = by*16;
  const int id   = lidx + lidy*get_local_size(0);
  const int row  = page*current_size*2 + current_size + ibx + id;
  int col        = page*current_size*2 + current_size;

  // Sets the offsets for this specific thread
  agm += ibx + id;
  bgm += lidx + (iby + lidy)*ldb;
  cgm += ibx + id + iby*ldc;

  // Initializes the result registers
  real cpm[16];
  #pragma unroll
  for (int j = 0; j < 16; ++j) {
    SetToZero(cpm[j]);
  }

  // Computes NT x 16 block of C, each thread computes one 1 x 16 row
  for (int k = 0; k < current_size; k += 16) {

    // Loads a 16 x 16 block of B into local memory using NX x 4 threads
    #pragma unroll
    for( int i=0; i < 16; i += (size/4) ) {  // += get_local_size(0)
      #pragma unroll
      for( int j=0; j < 16; j += 4 ) {  // += get_local_size(1)
        blm[(lidx + i) * LOCALX + (lidy + j)] = bgm[k + i + j*ldb];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Upper triangular
    if (upper) {

      // Performs 16 x 16 multiply-add operations
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        if (part == 2 || col++ < n) {
          #pragma unroll
          for (int j = 0; j < 16; ++j) {
            MultiplyAdd(cpm[j], agm[(i + k) * lda], blm[i * LOCALX + j]);
          }
        }
      }
    }

    // Lower triangular
    else {
      if (row < n) {

        // Performs 16 x 16 multiply-add operations
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
          #pragma unroll
          for (int j = 0; j < 16; ++j) {
            MultiplyAdd(cpm[j], agm[(i + k) * lda], blm[i * LOCALX + j]);
          }
        }
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores NT x 16 results: each thread writes one 16 x 1 row
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    if (part == 2) { Negate(cpm[i]); }
    cgm[0] = cpm[i];
    cgm += ldc;
  }
}

// =================================================================================================

// Triple matrix-multiplication kernel part 1: B12 = A12 * B22 (upper) or B21 = A21 * B11 (lower)
inline void TripleMatMulPart1(const int size, const bool upper, __local real* blm, int n,
                              __global const real* src, const int a_offset, const int lda,
                              __global real* dest, int current_size, int num_pages, const int block_size) {

  // Emulates a 3D grid: NX * (NY * num_pages)
  const int page = get_group_id(1) % num_pages;

  // Computes the destination block offset:
  // - go to the (page / pages_per_block) outer block_size * block_size block
  // - then the (page % pages_per_block) inner (current_size*2) * (current_size*2) page inside that
  const int pages_per_block = block_size / (current_size*2);
  dest += (page / pages_per_block) * block_size * block_size +
          (page % pages_per_block) * (current_size*2*block_size + current_size*2);

  // Using the GEMM notation: C = A*B
  __global const real* agm;
  __global const real* bgm;
  __global real* cgm;
  if (upper) { // upper triangular: B12 = A12 * B22
    agm = src + a_offset + page*current_size*2*lda + page*current_size*2 + current_size*lda;  // A12
    bgm = dest + current_size*block_size + current_size;                                      // B22
    cgm = dest + current_size*block_size;                                                     // B12
  }
  else { // lower triangular: B21 = A21 * B11
    agm = src + a_offset + page*current_size*2*lda + page*current_size*2 + current_size;  // A21
    bgm = dest;                                                                           // B11
    cgm = dest + current_size;                                                            // B21
  }

  // Runs the generic C = A * B matrix multiplication
  const int ldb = block_size;
  const int ldc = block_size;
  TripleMatMul(size, upper, 1, blm, n, agm, bgm, cgm, lda, ldb, ldc, current_size, num_pages, block_size);
}

// Triple matrix-multiplication kernel part 1: B12 = -B11 * B12 (upper) or B21 = -B22 * B21 (lower)
inline void TripleMatMulPart2(const int size, const bool upper, __local real* blm, const int n,
                              __global real* dest, int current_size, int num_pages, const int block_size) {

  // Emulates a 3D grid: NX * (NY * num_pages)
  const int page = get_group_id(1) % num_pages;

  // Computes the destination block offset:
  // - go to the (page / pages_per_block) outer block_size * block_size block
  // - then the (page % pages_per_block) inner (current_size*2) * (current_size*2) page inside that
  const int pages_per_block = block_size / (current_size*2);
  dest += (page / pages_per_block) * block_size * block_size +
          (page % pages_per_block) * (current_size*2*block_size + current_size*2);

  // Using the GEMM notation: C = A*B
  __global const real* agm;
  __global const real* bgm;
  __global real* cgm;
  if (upper) { // upper triangular: B12 = -B11 * B12
    agm = dest;                            // B11
    cgm = dest + current_size*block_size;  // B12
    bgm = cgm;                             // B12, okay to overwrite
  }

  else { // lower triangular: B21 = -B22 * B21
    agm = dest + current_size*block_size + current_size;  // B22
    cgm = dest + current_size;                            // B21
    bgm = cgm;                                            // B21, okay to overwrite
  }

  // Runs the generic C = A * B matrix multiplication
  const int lda = block_size;
  const int ldb = block_size;
  const int ldc = block_size;
  TripleMatMul(size, upper, 2, blm, n, agm, bgm, cgm, lda, ldb, ldc, current_size, num_pages, block_size);
}

// =================================================================================================

// B21 = A21 * B11
__kernel __attribute__((reqd_work_group_size(4, 4, 1)))
void TripleMatMul16Part1Lower(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart1(16, false, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B21 = -B22 * B21
__kernel __attribute__((reqd_work_group_size(4, 4, 1)))
void TripleMatMul16Part2Lower(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart2(16, false, lm, n, dest, current_size, num_pages, block_size);
}

// B21 = A21 * B11
__kernel __attribute__((reqd_work_group_size(8, 4, 1)))
void TripleMatMul32Part1Lower(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart1(32, false, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B21 = -B22 * B21
__kernel __attribute__((reqd_work_group_size(8, 4, 1)))
void TripleMatMul32Part2Lower(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart2(32, false, lm, n, dest, current_size, num_pages, block_size);
}

// B21 = A21 * B11
__kernel __attribute__((reqd_work_group_size(16, 4, 1)))
void TripleMatMul64Part1Lower(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart1(64, false, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B21 = -B22 * B21
__kernel __attribute__((reqd_work_group_size(16, 4, 1)))
void TripleMatMul64Part2Lower(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart2(64, false, lm, n, dest, current_size, num_pages, block_size);
}

// =================================================================================================

// B12 =  A12 * B22
__kernel __attribute__((reqd_work_group_size(4, 4, 1)))
void TripleMatMul16Part1Upper(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart1(16, true, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B12 = -B11 * B12
__kernel __attribute__((reqd_work_group_size(4, 4, 1)))
void TripleMatMul16Part2Upper(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart2(16, true, lm, n, dest, current_size, num_pages, block_size);
}

// B12 =  A12 * B22
__kernel __attribute__((reqd_work_group_size(8, 4, 1)))
void TripleMatMul32Part1Upper(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart1(32, true, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B12 = -B11 * B12
__kernel __attribute__((reqd_work_group_size(8, 4, 1)))
void TripleMatMul32Part2Upper(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart2(32, true, lm, n, dest, current_size, num_pages, block_size);
}

// B12 =  A12 * B22
__kernel __attribute__((reqd_work_group_size(16, 4, 1)))
void TripleMatMul64Part1Upper(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TripleMatMulPart1(64, true, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B12 = -B11 * B12
__kernel __attribute__((reqd_work_group_size(16, 4, 1)))
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
