#include <iostream>
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
//#include <clBLAS.h>
#include <CL/cl.h>
#include <stdlib.h>
using namespace std;

extern "C" {
  #include "clblast_c.h"
}

//using namespace clblast;

  cl_int err;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx = 0;
  cl_command_queue queue = 0;
  cl_mem bufA, bufB, bufC;
  cl_event event = NULL;
  int ret = 0;

void clgemm(int colmaj, char transAchar, char transBchar, int M, int N, int K, float alpha, float *A, int lda,
     float *B, int ldb, float beta, float *C, int ldc, float *result) {
Transpose transA = transAchar == 'n' ? kNo : kYes;
Transpose transB = transBchar == 'n' ? kNo : kYes;

size_t off = 0;
size_t offA = 0;
size_t offB = 0;
size_t offC = 0;

Layout order;
if(colmaj == 1 ) {
  order = kColMajor;
} else {
  order = kRowMajor;
}

  bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * K * sizeof(*A),
                        NULL, &err);
  bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * N * sizeof(*B),
                        NULL, &err);
  bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * N * sizeof(*C),
                        NULL, &err);

  err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
      M * K * sizeof(*A), A, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
      K * N * sizeof(*B), B, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0,
      M * N * sizeof(*C), C, 0, NULL, NULL);

//template <typename T>
//StatusCode Gemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
//                const size_t m, const size_t n, const size_t k,
//                const T alpha,
//                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
//                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
//                const T beta,
//                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
//                cl_command_queue* queue, cl_event* event);

//StatusCode CLBlastSgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
//                        const size_t m, const size_t n, const size_t k,
//                        const float alpha,
//                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
//                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
//                        const float beta,
//                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
//                        cl_command_queue* queue, cl_event* event);

StatusCode blash_ret = CLBlastSgemm(order, transA, transB,
                M, N, K,
                alpha,
                bufA, offA, lda,
                bufB, offB, ldb,
                beta,
                bufC, offC, ldc,
                &queue, &event);

//  err = clblasSgemm(order, transA, transB, M - off, N - off, K - off,
//                       alpha, bufA, offA, lda,
//                       bufB, offB, ldb, beta,
//                       bufC, offC, ldc,
//                       1, &queue, 0, NULL, &event);
  if (err != CL_SUCCESS) {
      printf("clblasSgemmEx() failed with %d\n", err);
      ret = 1;
      exit(1);
  }
  else {
      err = clWaitForEvents(1, &event);
      err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                                M * N * sizeof(*result),
                                result, 0, NULL, NULL);
  }

  clReleaseMemObject(bufC);
  clReleaseMemObject(bufB);
  clReleaseMemObject(bufA);

}

void copy(float *target, float *source, int numels ) {
  for(int i = 0; i < numels; i++) {
    target[i] = source[i];
  }
}

// assumes row major
void transpose(float *A, int rows, int cols) {
  float *A_ = new float[rows * cols];
  for(int i=0; i < rows; i++ ) {
    for(int j = 0; j< cols; j++) {
      A_[j * rows + i] = A[i * cols + j];
    }
  }
  copy(A, A_, rows * cols);
  delete[] A_;
}

// assumes row major
void mult(float *C, float *A, float *B, int M, int K, int N) {
  for(int m = 0; m < M; m++ ) {
    for(int n = 0; n < N; n++ ) {
      float sum = 0;
      for(int k = 0; k < K; k++ ) {
        sum = sum + A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum;
    }
  }
}

bool test1(int colmaj, int M, int N, int K, int transAint, int transBint) {
  char transa = transAint == 1 ? 't' : 'n';
  char transb = transBint == 1 ? 't' : 'n';
//  cout << "colmaj=" << colmaj << " " << transa << " " << transb << " M=" << M << " K=" << K << " N=" << N << endl;

  float alpha = 1;
  int lda, ldb, ldc;

  if(colmaj == 1) {
    if(transAint == 1) {
       lda = K;
    } else {
       lda = M;
    }
    if(transBint == 1) {
       ldb = N;
    } else {
       ldb = K;
    }
  } else {
    if(transAint == 1) {
       lda = M;    
    } else {
       lda = K;
    }
    if(transBint == 1) {
       ldb = K;    
    } else {
       ldb = N;
    }
  }

  if(colmaj == 1) {
    ldc = M;
  } else {
    ldc = N;
  }

  float beta = 0;
  // assume these are row major, untransposed
  float *A = new float[M * K];
  float *B = new float[K * N];
  float *C = new float[M * N];
  for(int i = 0; i < M * K; i++) {
     A[i] = rand() / (float)RAND_MAX - 0.5f;
  }
  for(int i = 0; i < N * K; i++) {
    B[i] = rand() / (float)RAND_MAX - 0.5f;
  }
  for(int i = 0; i < M * N; i++) {
    C[i] = 0.0f;
  }

//  cout << "op(A):" << endl;
//    for(int m=0; m < M; m++) {
//      for(int k = 0; k < K; k++) {
//        cout << A[m * K + k] << " ";
//      }
//      cout << endl;
//    }

//  cout << "op(B):" << endl;
//    for(int k = 0; k < K; k++) {
//      for(int n=0; n < N; n++) {
//        cout << B[k * N + n] << " ";
//      }
//      cout << endl;
//    }

   float *Aforblas = new float[M*K];
   float *Bforblas = new float[K * N];
   float *Cforblas = new float[M*N];
   copy(Aforblas, A, M*K);
   copy(Bforblas, B, N*K);

  float *Cours = new float[M * N];

  float *Aour = new float[M * K];
  float *Bour = new float[K * N];
  copy(Aour, A, M * K);
  copy(Bour, B, N * K);
  bool flipAforblas = !(colmaj == 1) != !(transAint == 1);
  bool flipBforblas = !(colmaj == 1) != !(transBint == 1);
  if(flipAforblas) {
    transpose(Aforblas, M, K);
   }
  if(flipBforblas) {
    transpose(Bforblas, K, N);
   }
  mult(Cours, A, B, M, K, N);

//  cout<< "result from CPU: " << endl;
//  for(int m = 0; m < M; m++) {
//    for(int n = 0; n < N; n++) {
//      int i = m + n * M;
//      cout << Cours[i] << " ";
//    }
//    cout << endl;
//  }

  float *clout = new float[M * N];
  clgemm(colmaj, transa, transb, M, N, K, alpha, Aforblas, lda,
     Bforblas, ldb, beta, C, ldc, clout);
  if(colmaj == 1 ) {
    transpose(clout, N, M);
  }
  bool ok = true;
  for(int m = 0; m < M; m++) {
    for(int n = 0; n < N; n++) {
      int i = m + n * M;
//      cout << "  " << i << " " << Cours[i] << " " << clout[i] << endl;
      float diff = clout[i] - Cours[i];
      diff = diff < 0 ? - diff : diff;
      if(diff > 0.0001) {
//         cout << "ERROR " << M << " " << N << " " << K << " " << transa << " " << transb << endl;
         ok = false;
//         exit(1);
      }
    }
  }
  if(!ok) {
   cout << "ERROR colmaj=" << colmaj << " M=" << M << " N=" << N << " K=" << K << " transa=" << transa << " transb=" << transb << endl;
  }
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] Cours;
  delete[] clout;
  return ok;
}

int main(int argc, char *argv[]) {
//  clewInit();

  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS) {
      printf( "clGetPlatformIDs() failed with %d\n", err );
      return 1;
  }
  cout << "got platforms" << endl;

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) {
      printf( "clGetDeviceIDs() failed with %d\n", err );
      return 1;
  }

  props[1] = (cl_context_properties)platform;
  ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
      printf( "clCreateContext() failed with %d\n", err );
      return 1;
  }

  queue = clCreateCommandQueue(ctx, device, 0, &err);
  if (err != CL_SUCCESS) {
      printf( "clCreateCommandQueue() failed with %d\n", err );
      clReleaseContext(ctx);
      return 1;
  }

  test1(1, 1, 7, 1, 0, 0);
  test1(1, 1, 7, 1, 1, 1);
//  for(int colmaj = 0; colmaj <= 1; colmaj++ ) {
  int numTestsDone = 0;
  int colmaj = 1; {
    for(int m=1; m <= 16; m++) {
      for(int n=1; n <= 16; n++) {
        for(int k=1; k <= 16; k++) {
          for(int transA =0; transA <= 1; transA++) {
  //        int transA = 1; {
            for(int transB =0; transB <= 1; transB++) {
//            int transB = 0; {
  //            test1();
              test1(colmaj, m, n, k, transA, transB);
              numTestsDone++;
              if(numTestsDone % 1000 == 0) {
                cout << numTestsDone << " tests done" << endl;
              }
            }
          }
        }
      }
    }
  }
  cout << "All done! numTestsDone: " << numTestsDone << endl;
  // check:
  // colmaj transa transb res
  // 0 0 0 ok
  // 0 1 0 ok
  // 0 0 1 FAIL 7 1 1 n t
  // 0 1 1 ok
  // 1 0 0 ok
  // 1 1 0 FAIL 1 7 1 t n
  // 1 0 1 ok
  // 1 1 1 ok

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}

