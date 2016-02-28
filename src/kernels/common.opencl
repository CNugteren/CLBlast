
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common defines and type-defs for the CLBlast OpenCL kernels.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(
// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this file is used outside of the CLBlast library.
#ifndef PRECISION
  #define PRECISION 32      // Data-types: single or double precision, complex or regular
#endif

// =================================================================================================

// Enable support for double-precision
#if PRECISION == 64 || PRECISION == 6464
  #if __OPENCL_VERSION__ <= CL_VERSION_1_1
     #pragma OPENCL EXTENSION cl_khr_fp64: enable
  #endif
#endif

// Single-precision
#if PRECISION == 32
  typedef float real;
  typedef float2 real2;
  typedef float4 real4;
  typedef float8 real8;
  typedef float16 real16;
  #define ZERO 0.0f
  #define ONE 1.0f

// Double-precision 
#elif PRECISION == 64
  typedef double real;
  typedef double2 real2;
  typedef double4 real4;
  typedef double8 real8;
  typedef double16 real16;
  #define ZERO 0.0
  #define ONE 1.0

// Complex single-precision
#elif PRECISION == 3232
  typedef struct cfloat {float x; float y;} real;
  typedef struct cfloat2 {real x; real y;} real2;
  typedef struct cfloat4 {real x; real y; real z; real w;} real4;
  typedef struct cfloat8 {real s0; real s1; real s2; real s3;
                          real s4; real s5; real s6; real s7;} real8;
  typedef struct cfloat16 {real s0; real s1; real s2; real s3;
                           real s4; real s5; real s6; real s7;
                           real s8; real s9; real sA; real sB;
                           real sC; real sD; real sE; real sF;} real16;
  #define ZERO 0.0f
  #define ONE 1.0f

// Complex Double-precision
#elif PRECISION == 6464
  typedef struct cdouble {double x; double y;} real;
  typedef struct cdouble2 {real x; real y;} real2;
  typedef struct cdouble4 {real x; real y; real z; real w;} real4;
  typedef struct cdouble8 {real s0; real s1; real s2; real s3;
                           real s4; real s5; real s6; real s7;} real8;
  typedef struct cdouble16 {real s0; real s1; real s2; real s3;
                            real s4; real s5; real s6; real s7;
                            real s8; real s9; real sA; real sB;
                            real sC; real sD; real sE; real sF;} real16;
  #define ZERO 0.0
  #define ONE 1.0
#endif

// =================================================================================================

// Don't use the non-IEEE754 compliant OpenCL built-in mad() instruction per default. For specific
// devices, this is enabled (see src/routine.cc).
#ifndef USE_CL_MAD
  #define USE_CL_MAD 0
#endif

// Sets a variable to zero
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToZero(a) a.x = ZERO; a.y = ZERO
#else
  #define SetToZero(a) a = ZERO
#endif

// Sets a variable to zero (only the imaginary part)
#if PRECISION == 3232 || PRECISION == 6464
  #define ImagToZero(a) a.y = ZERO
#else
  #define ImagToZero(a) 
#endif

// Sets a variable to one
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToOne(a) a.x = ONE; a.y = ZERO
#else
  #define SetToOne(a) a = ONE
#endif

// Adds two complex variables
#if PRECISION == 3232 || PRECISION == 6464
  #define Add(c, a, b) c.x = a.x + b.x; c.y = a.y + b.y
#else
  #define Add(c, a, b) c = a + b
#endif

// Multiply two complex variables (used in the defines below)
#if PRECISION == 3232 || PRECISION == 6464
  #define MulReal(a, b) a.x*b.x - a.y*b.y
  #define MulImag(a, b) a.x*b.y + a.y*b.x
#endif

// The scalar multiply function
#if PRECISION == 3232 || PRECISION == 6464
  #define Multiply(c, a, b) c.x = MulReal(a,b); c.y = MulImag(a,b)
#else
  #define Multiply(c, a, b) c = a * b
#endif

// The scalar multiply-add function
#if PRECISION == 3232 || PRECISION == 6464
  #define MultiplyAdd(c, a, b) c.x += MulReal(a,b); c.y += MulImag(a,b)
#else
  #if USE_CL_MAD == 1
    #define MultiplyAdd(c, a, b) c = mad(a, b, c)
  #else
    #define MultiplyAdd(c, a, b) c += a * b
  #endif
#endif

// The scalar AXPBY function
#if PRECISION == 3232 || PRECISION == 6464
  #define AXPBY(e, a, b, c, d) e.x = MulReal(a,b) + MulReal(c,d); e.y = MulImag(a,b) + MulImag(c,d)
#else
  #define AXPBY(e, a, b, c, d) e = a*b + c*d
#endif

// The scalar GER function
#if PRECISION == 3232 || PRECISION == 6464
  #define GER(e, a, b, c, d) real ab; ab.x = MulReal(a,b); ab.y = MulImag(a,b); e.x = MulReal(ab,c) + d.x; e.y = MulImag(ab,c) + d.y
#else
  #define GER(e, a, b, c, d) e = a*b*c + d
#endif

// The complex conjugate operation for complex transforms
#if PRECISION == 3232 || PRECISION == 6464
  #define COMPLEX_CONJUGATE(value) value.x = value.x; value.y = -value.y
#else
  #define COMPLEX_CONJUGATE(value) value = value
#endif

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
