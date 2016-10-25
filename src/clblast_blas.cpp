
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Netlib CBLAS implementations to the CLBlast BLAS routines, performing buffer
// copies automatically and running on the default OpenCL platform and device. For full control over
// performance, it is advised to use the regular clblast.h or clblast_c.h headers instead.
//
// =================================================================================================

#include <cstdlib>

#include "clblast_blas.h"
#include "clblast.h"
#include "utilities/utilities.hpp"

namespace clblast {

// =================================================================================================

// Helper function to get a default OpenCL platform and device
Device get_device() {
  auto platform_id = ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0});
  auto device_id = ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0});
  auto platform = Platform(platform_id);
  return Device(platform, device_id);
}

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// ROTG
void cblas_srotg(float* sa,
                 float* sb,
                 float* sc,
                 float* ss) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto sa_size = 1;
  auto sa_buffer = Buffer<float>(context, sa_size);
  const auto sb_size = 1;
  auto sb_buffer = Buffer<float>(context, sb_size);
  const auto sc_size = 1;
  auto sc_buffer = Buffer<float>(context, sc_size);
  const auto ss_size = 1;
  auto ss_buffer = Buffer<float>(context, ss_size);
  sa_buffer.Write(queue, sa_size, reinterpret_cast<float*>(sa));
  sb_buffer.Write(queue, sb_size, reinterpret_cast<float*>(sb));
  sc_buffer.Write(queue, sc_size, reinterpret_cast<float*>(sc));
  ss_buffer.Write(queue, ss_size, reinterpret_cast<float*>(ss));
  auto queue_cl = queue();
  auto s = Rotg<float>(sa_buffer(), 0,
                       sb_buffer(), 0,
                       sc_buffer(), 0,
                       ss_buffer(), 0,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  sa_buffer.Read(queue, sa_size, reinterpret_cast<float*>(sa));
  sb_buffer.Read(queue, sb_size, reinterpret_cast<float*>(sb));
  sc_buffer.Read(queue, sc_size, reinterpret_cast<float*>(sc));
  ss_buffer.Read(queue, ss_size, reinterpret_cast<float*>(ss));
}
void cblas_drotg(double* sa,
                 double* sb,
                 double* sc,
                 double* ss) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto sa_size = 1;
  auto sa_buffer = Buffer<double>(context, sa_size);
  const auto sb_size = 1;
  auto sb_buffer = Buffer<double>(context, sb_size);
  const auto sc_size = 1;
  auto sc_buffer = Buffer<double>(context, sc_size);
  const auto ss_size = 1;
  auto ss_buffer = Buffer<double>(context, ss_size);
  sa_buffer.Write(queue, sa_size, reinterpret_cast<double*>(sa));
  sb_buffer.Write(queue, sb_size, reinterpret_cast<double*>(sb));
  sc_buffer.Write(queue, sc_size, reinterpret_cast<double*>(sc));
  ss_buffer.Write(queue, ss_size, reinterpret_cast<double*>(ss));
  auto queue_cl = queue();
  auto s = Rotg<double>(sa_buffer(), 0,
                        sb_buffer(), 0,
                        sc_buffer(), 0,
                        ss_buffer(), 0,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  sa_buffer.Read(queue, sa_size, reinterpret_cast<double*>(sa));
  sb_buffer.Read(queue, sb_size, reinterpret_cast<double*>(sb));
  sc_buffer.Read(queue, sc_size, reinterpret_cast<double*>(sc));
  ss_buffer.Read(queue, ss_size, reinterpret_cast<double*>(ss));
}

// ROTMG
void cblas_srotmg(float* sd1,
                  float* sd2,
                  float* sx1,
                  const float* sy1,
                  float* sparam) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto sy1_size = 1;
  auto sy1_buffer = Buffer<float>(context, sy1_size);
  const auto sd1_size = 1;
  auto sd1_buffer = Buffer<float>(context, sd1_size);
  const auto sd2_size = 1;
  auto sd2_buffer = Buffer<float>(context, sd2_size);
  const auto sx1_size = 1;
  auto sx1_buffer = Buffer<float>(context, sx1_size);
  const auto sparam_size = 1;
  auto sparam_buffer = Buffer<float>(context, sparam_size);
  sy1_buffer.Write(queue, sy1_size, reinterpret_cast<const float*>(sy1));
  sd1_buffer.Write(queue, sd1_size, reinterpret_cast<float*>(sd1));
  sd2_buffer.Write(queue, sd2_size, reinterpret_cast<float*>(sd2));
  sx1_buffer.Write(queue, sx1_size, reinterpret_cast<float*>(sx1));
  sparam_buffer.Write(queue, sparam_size, reinterpret_cast<float*>(sparam));
  auto queue_cl = queue();
  auto s = Rotmg<float>(sd1_buffer(), 0,
                        sd2_buffer(), 0,
                        sx1_buffer(), 0,
                        sy1_buffer(), 0,
                        sparam_buffer(), 0,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  sd1_buffer.Read(queue, sd1_size, reinterpret_cast<float*>(sd1));
  sd2_buffer.Read(queue, sd2_size, reinterpret_cast<float*>(sd2));
  sx1_buffer.Read(queue, sx1_size, reinterpret_cast<float*>(sx1));
  sparam_buffer.Read(queue, sparam_size, reinterpret_cast<float*>(sparam));
}
void cblas_drotmg(double* sd1,
                  double* sd2,
                  double* sx1,
                  const double* sy1,
                  double* sparam) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto sy1_size = 1;
  auto sy1_buffer = Buffer<double>(context, sy1_size);
  const auto sd1_size = 1;
  auto sd1_buffer = Buffer<double>(context, sd1_size);
  const auto sd2_size = 1;
  auto sd2_buffer = Buffer<double>(context, sd2_size);
  const auto sx1_size = 1;
  auto sx1_buffer = Buffer<double>(context, sx1_size);
  const auto sparam_size = 1;
  auto sparam_buffer = Buffer<double>(context, sparam_size);
  sy1_buffer.Write(queue, sy1_size, reinterpret_cast<const double*>(sy1));
  sd1_buffer.Write(queue, sd1_size, reinterpret_cast<double*>(sd1));
  sd2_buffer.Write(queue, sd2_size, reinterpret_cast<double*>(sd2));
  sx1_buffer.Write(queue, sx1_size, reinterpret_cast<double*>(sx1));
  sparam_buffer.Write(queue, sparam_size, reinterpret_cast<double*>(sparam));
  auto queue_cl = queue();
  auto s = Rotmg<double>(sd1_buffer(), 0,
                         sd2_buffer(), 0,
                         sx1_buffer(), 0,
                         sy1_buffer(), 0,
                         sparam_buffer(), 0,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  sd1_buffer.Read(queue, sd1_size, reinterpret_cast<double*>(sd1));
  sd2_buffer.Read(queue, sd2_size, reinterpret_cast<double*>(sd2));
  sx1_buffer.Read(queue, sx1_size, reinterpret_cast<double*>(sx1));
  sparam_buffer.Read(queue, sparam_size, reinterpret_cast<double*>(sparam));
}

// ROT
void cblas_srot(const int n,
                float* x, const int x_inc,
                float* y, const int y_inc,
                const float cos,
                const float sin) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<float>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = Rot(n,
               x_buffer(), 0, x_inc,
               y_buffer(), 0, y_inc,
               cos,
               sin,
               &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_drot(const int n,
                double* x, const int x_inc,
                double* y, const int y_inc,
                const double cos,
                const double sin) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<double>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = Rot(n,
               x_buffer(), 0, x_inc,
               y_buffer(), 0, y_inc,
               cos,
               sin,
               &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}

// ROTM
void cblas_srotm(const int n,
                 float* x, const int x_inc,
                 float* y, const int y_inc,
                 float* sparam) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<float>(context, y_size);
  const auto sparam_size = 1;
  auto sparam_buffer = Buffer<float>(context, sparam_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  sparam_buffer.Write(queue, sparam_size, reinterpret_cast<float*>(sparam));
  auto queue_cl = queue();
  auto s = Rotm<float>(n,
                       x_buffer(), 0, x_inc,
                       y_buffer(), 0, y_inc,
                       sparam_buffer(), 0,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
  sparam_buffer.Read(queue, sparam_size, reinterpret_cast<float*>(sparam));
}
void cblas_drotm(const int n,
                 double* x, const int x_inc,
                 double* y, const int y_inc,
                 double* sparam) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<double>(context, y_size);
  const auto sparam_size = 1;
  auto sparam_buffer = Buffer<double>(context, sparam_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  sparam_buffer.Write(queue, sparam_size, reinterpret_cast<double*>(sparam));
  auto queue_cl = queue();
  auto s = Rotm<double>(n,
                        x_buffer(), 0, x_inc,
                        y_buffer(), 0, y_inc,
                        sparam_buffer(), 0,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
  sparam_buffer.Read(queue, sparam_size, reinterpret_cast<double*>(sparam));
}

// SWAP
void cblas_sswap(const int n,
                 float* x, const int x_inc,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<float>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = Swap<float>(n,
                       x_buffer(), 0, x_inc,
                       y_buffer(), 0, y_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dswap(const int n,
                 double* x, const int x_inc,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<double>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = Swap<double>(n,
                        x_buffer(), 0, x_inc,
                        y_buffer(), 0, y_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}
void cblas_cswap(const int n,
                 void* x, const int x_inc,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<float2>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = Swap<float2>(n,
                        x_buffer(), 0, x_inc,
                        y_buffer(), 0, y_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zswap(const int n,
                 void* x, const int x_inc,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<double2>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = Swap<double2>(n,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// SCAL
void cblas_sscal(const int n,
                 const float alpha,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n;
  auto x_buffer = Buffer<float>(context, x_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = Scal(n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dscal(const int n,
                 const double alpha,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n;
  auto x_buffer = Buffer<double>(context, x_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = Scal(n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_cscal(const int n,
                 const void* alpha,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto x_size = n;
  auto x_buffer = Buffer<float2>(context, x_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = Scal(n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_zscal(const int n,
                 const void* alpha,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto x_size = n;
  auto x_buffer = Buffer<double2>(context, x_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = Scal(n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// COPY
void cblas_scopy(const int n,
                 const float* x, const int x_inc,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<float>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = Copy<float>(n,
                       x_buffer(), 0, x_inc,
                       y_buffer(), 0, y_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dcopy(const int n,
                 const double* x, const int x_inc,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<double>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = Copy<double>(n,
                        x_buffer(), 0, x_inc,
                        y_buffer(), 0, y_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}
void cblas_ccopy(const int n,
                 const void* x, const int x_inc,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<float2>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = Copy<float2>(n,
                        x_buffer(), 0, x_inc,
                        y_buffer(), 0, y_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zcopy(const int n,
                 const void* x, const int x_inc,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<double2>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = Copy<double2>(n,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// AXPY
void cblas_saxpy(const int n,
                 const float alpha,
                 const float* x, const int x_inc,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<float>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = Axpy(n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_daxpy(const int n,
                 const double alpha,
                 const double* x, const int x_inc,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<double>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = Axpy(n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}
void cblas_caxpy(const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto x_size = n;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<float2>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = Axpy(n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zaxpy(const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto x_size = n;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<double2>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = Axpy(n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// DOT
void cblas_sdot(const int n,
                float* dot,
                const float* x, const int x_inc,
                const float* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<float>(context, y_size);
  const auto dot_size = 1;
  auto dot_buffer = Buffer<float>(context, dot_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float*>(y));
  dot_buffer.Write(queue, dot_size, reinterpret_cast<float*>(dot));
  auto queue_cl = queue();
  auto s = Dot<float>(n,
                      dot_buffer(), 0,
                      x_buffer(), 0, x_inc,
                      y_buffer(), 0, y_inc,
                      &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  dot_buffer.Read(queue, dot_size, reinterpret_cast<float*>(dot));
}
void cblas_ddot(const int n,
                double* dot,
                const double* x, const int x_inc,
                const double* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<double>(context, y_size);
  const auto dot_size = 1;
  auto dot_buffer = Buffer<double>(context, dot_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double*>(y));
  dot_buffer.Write(queue, dot_size, reinterpret_cast<double*>(dot));
  auto queue_cl = queue();
  auto s = Dot<double>(n,
                       dot_buffer(), 0,
                       x_buffer(), 0, x_inc,
                       y_buffer(), 0, y_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  dot_buffer.Read(queue, dot_size, reinterpret_cast<double*>(dot));
}

// DOTU
void cblas_cdotu(const int n,
                 void* dot,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<float2>(context, y_size);
  const auto dot_size = 1;
  auto dot_buffer = Buffer<float2>(context, dot_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float2*>(y));
  dot_buffer.Write(queue, dot_size, reinterpret_cast<float2*>(dot));
  auto queue_cl = queue();
  auto s = Dotu<float2>(n,
                        dot_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        y_buffer(), 0, y_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  dot_buffer.Read(queue, dot_size, reinterpret_cast<float2*>(dot));
}
void cblas_zdotu(const int n,
                 void* dot,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<double2>(context, y_size);
  const auto dot_size = 1;
  auto dot_buffer = Buffer<double2>(context, dot_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double2*>(y));
  dot_buffer.Write(queue, dot_size, reinterpret_cast<double2*>(dot));
  auto queue_cl = queue();
  auto s = Dotu<double2>(n,
                         dot_buffer(), 0,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  dot_buffer.Read(queue, dot_size, reinterpret_cast<double2*>(dot));
}

// DOTC
void cblas_cdotc(const int n,
                 void* dot,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<float2>(context, y_size);
  const auto dot_size = 1;
  auto dot_buffer = Buffer<float2>(context, dot_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float2*>(y));
  dot_buffer.Write(queue, dot_size, reinterpret_cast<float2*>(dot));
  auto queue_cl = queue();
  auto s = Dotc<float2>(n,
                        dot_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        y_buffer(), 0, y_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  dot_buffer.Read(queue, dot_size, reinterpret_cast<float2*>(dot));
}
void cblas_zdotc(const int n,
                 void* dot,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = n;
  auto y_buffer = Buffer<double2>(context, y_size);
  const auto dot_size = 1;
  auto dot_buffer = Buffer<double2>(context, dot_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double2*>(y));
  dot_buffer.Write(queue, dot_size, reinterpret_cast<double2*>(dot));
  auto queue_cl = queue();
  auto s = Dotc<double2>(n,
                         dot_buffer(), 0,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  dot_buffer.Read(queue, dot_size, reinterpret_cast<double2*>(dot));
}

// NRM2
void cblas_snrm2(const int n,
                 float* nrm2,
                 const float* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto nrm2_size = 1;
  auto nrm2_buffer = Buffer<float>(context, nrm2_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  nrm2_buffer.Write(queue, nrm2_size, reinterpret_cast<float*>(nrm2));
  auto queue_cl = queue();
  auto s = Nrm2<float>(n,
                       nrm2_buffer(), 0,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  nrm2_buffer.Read(queue, nrm2_size, reinterpret_cast<float*>(nrm2));
}
void cblas_dnrm2(const int n,
                 double* nrm2,
                 const double* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto nrm2_size = 1;
  auto nrm2_buffer = Buffer<double>(context, nrm2_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  nrm2_buffer.Write(queue, nrm2_size, reinterpret_cast<double*>(nrm2));
  auto queue_cl = queue();
  auto s = Nrm2<double>(n,
                        nrm2_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  nrm2_buffer.Read(queue, nrm2_size, reinterpret_cast<double*>(nrm2));
}
void cblas_scnrm2(const int n,
                 void* nrm2,
                 const void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto nrm2_size = 1;
  auto nrm2_buffer = Buffer<float2>(context, nrm2_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  nrm2_buffer.Write(queue, nrm2_size, reinterpret_cast<float2*>(nrm2));
  auto queue_cl = queue();
  auto s = Nrm2<float2>(n,
                        nrm2_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  nrm2_buffer.Read(queue, nrm2_size, reinterpret_cast<float2*>(nrm2));
}
void cblas_dznrm2(const int n,
                 void* nrm2,
                 const void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto nrm2_size = 1;
  auto nrm2_buffer = Buffer<double2>(context, nrm2_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  nrm2_buffer.Write(queue, nrm2_size, reinterpret_cast<double2*>(nrm2));
  auto queue_cl = queue();
  auto s = Nrm2<double2>(n,
                         nrm2_buffer(), 0,
                         x_buffer(), 0, x_inc,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  nrm2_buffer.Read(queue, nrm2_size, reinterpret_cast<double2*>(nrm2));
}

// ASUM
void cblas_sasum(const int n,
                 float* asum,
                 const float* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto asum_size = 1;
  auto asum_buffer = Buffer<float>(context, asum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  asum_buffer.Write(queue, asum_size, reinterpret_cast<float*>(asum));
  auto queue_cl = queue();
  auto s = Asum<float>(n,
                       asum_buffer(), 0,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  asum_buffer.Read(queue, asum_size, reinterpret_cast<float*>(asum));
}
void cblas_dasum(const int n,
                 double* asum,
                 const double* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto asum_size = 1;
  auto asum_buffer = Buffer<double>(context, asum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  asum_buffer.Write(queue, asum_size, reinterpret_cast<double*>(asum));
  auto queue_cl = queue();
  auto s = Asum<double>(n,
                        asum_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  asum_buffer.Read(queue, asum_size, reinterpret_cast<double*>(asum));
}
void cblas_scasum(const int n,
                 void* asum,
                 const void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto asum_size = 1;
  auto asum_buffer = Buffer<float2>(context, asum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  asum_buffer.Write(queue, asum_size, reinterpret_cast<float2*>(asum));
  auto queue_cl = queue();
  auto s = Asum<float2>(n,
                        asum_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  asum_buffer.Read(queue, asum_size, reinterpret_cast<float2*>(asum));
}
void cblas_dzasum(const int n,
                 void* asum,
                 const void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto asum_size = 1;
  auto asum_buffer = Buffer<double2>(context, asum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  asum_buffer.Write(queue, asum_size, reinterpret_cast<double2*>(asum));
  auto queue_cl = queue();
  auto s = Asum<double2>(n,
                         asum_buffer(), 0,
                         x_buffer(), 0, x_inc,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  asum_buffer.Read(queue, asum_size, reinterpret_cast<double2*>(asum));
}

// SUM
void cblas_ssum(const int n,
                float* sum,
                const float* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto sum_size = 1;
  auto sum_buffer = Buffer<float>(context, sum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  sum_buffer.Write(queue, sum_size, reinterpret_cast<float*>(sum));
  auto queue_cl = queue();
  auto s = Sum<float>(n,
                      sum_buffer(), 0,
                      x_buffer(), 0, x_inc,
                      &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  sum_buffer.Read(queue, sum_size, reinterpret_cast<float*>(sum));
}
void cblas_dsum(const int n,
                double* sum,
                const double* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto sum_size = 1;
  auto sum_buffer = Buffer<double>(context, sum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  sum_buffer.Write(queue, sum_size, reinterpret_cast<double*>(sum));
  auto queue_cl = queue();
  auto s = Sum<double>(n,
                       sum_buffer(), 0,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  sum_buffer.Read(queue, sum_size, reinterpret_cast<double*>(sum));
}
void cblas_scsum(const int n,
                void* sum,
                const void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto sum_size = 1;
  auto sum_buffer = Buffer<float2>(context, sum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  sum_buffer.Write(queue, sum_size, reinterpret_cast<float2*>(sum));
  auto queue_cl = queue();
  auto s = Sum<float2>(n,
                       sum_buffer(), 0,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  sum_buffer.Read(queue, sum_size, reinterpret_cast<float2*>(sum));
}
void cblas_dzsum(const int n,
                void* sum,
                const void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto sum_size = 1;
  auto sum_buffer = Buffer<double2>(context, sum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  sum_buffer.Write(queue, sum_size, reinterpret_cast<double2*>(sum));
  auto queue_cl = queue();
  auto s = Sum<double2>(n,
                        sum_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  sum_buffer.Read(queue, sum_size, reinterpret_cast<double2*>(sum));
}

// AMAX
void cblas_isamax(const int n,
                 float* imax,
                 const float* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto imax_size = 1;
  auto imax_buffer = Buffer<float>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<float*>(imax));
  auto queue_cl = queue();
  auto s = Amax<float>(n,
                       imax_buffer(), 0,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<float*>(imax));
}
void cblas_idamax(const int n,
                 double* imax,
                 const double* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto imax_size = 1;
  auto imax_buffer = Buffer<double>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<double*>(imax));
  auto queue_cl = queue();
  auto s = Amax<double>(n,
                        imax_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<double*>(imax));
}
void cblas_icamax(const int n,
                 void* imax,
                 const void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto imax_size = 1;
  auto imax_buffer = Buffer<float2>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<float2*>(imax));
  auto queue_cl = queue();
  auto s = Amax<float2>(n,
                        imax_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<float2*>(imax));
}
void cblas_izamax(const int n,
                 void* imax,
                 const void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto imax_size = 1;
  auto imax_buffer = Buffer<double2>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<double2*>(imax));
  auto queue_cl = queue();
  auto s = Amax<double2>(n,
                         imax_buffer(), 0,
                         x_buffer(), 0, x_inc,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<double2*>(imax));
}

// MAX
void cblas_ismax(const int n,
                float* imax,
                const float* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto imax_size = 1;
  auto imax_buffer = Buffer<float>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<float*>(imax));
  auto queue_cl = queue();
  auto s = Max<float>(n,
                      imax_buffer(), 0,
                      x_buffer(), 0, x_inc,
                      &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<float*>(imax));
}
void cblas_idmax(const int n,
                double* imax,
                const double* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto imax_size = 1;
  auto imax_buffer = Buffer<double>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<double*>(imax));
  auto queue_cl = queue();
  auto s = Max<double>(n,
                       imax_buffer(), 0,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<double*>(imax));
}
void cblas_icmax(const int n,
                void* imax,
                const void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto imax_size = 1;
  auto imax_buffer = Buffer<float2>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<float2*>(imax));
  auto queue_cl = queue();
  auto s = Max<float2>(n,
                       imax_buffer(), 0,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<float2*>(imax));
}
void cblas_izmax(const int n,
                void* imax,
                const void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto imax_size = 1;
  auto imax_buffer = Buffer<double2>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<double2*>(imax));
  auto queue_cl = queue();
  auto s = Max<double2>(n,
                        imax_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<double2*>(imax));
}

// MIN
void cblas_ismin(const int n,
                float* imin,
                const float* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto imin_size = 1;
  auto imin_buffer = Buffer<float>(context, imin_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  imin_buffer.Write(queue, imin_size, reinterpret_cast<float*>(imin));
  auto queue_cl = queue();
  auto s = Min<float>(n,
                      imin_buffer(), 0,
                      x_buffer(), 0, x_inc,
                      &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  imin_buffer.Read(queue, imin_size, reinterpret_cast<float*>(imin));
}
void cblas_idmin(const int n,
                double* imin,
                const double* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto imin_size = 1;
  auto imin_buffer = Buffer<double>(context, imin_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  imin_buffer.Write(queue, imin_size, reinterpret_cast<double*>(imin));
  auto queue_cl = queue();
  auto s = Min<double>(n,
                       imin_buffer(), 0,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  imin_buffer.Read(queue, imin_size, reinterpret_cast<double*>(imin));
}
void cblas_icmin(const int n,
                void* imin,
                const void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto imin_size = 1;
  auto imin_buffer = Buffer<float2>(context, imin_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  imin_buffer.Write(queue, imin_size, reinterpret_cast<float2*>(imin));
  auto queue_cl = queue();
  auto s = Min<float2>(n,
                       imin_buffer(), 0,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  imin_buffer.Read(queue, imin_size, reinterpret_cast<float2*>(imin));
}
void cblas_izmin(const int n,
                void* imin,
                const void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto x_size = n;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto imin_size = 1;
  auto imin_buffer = Buffer<double2>(context, imin_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  imin_buffer.Write(queue, imin_size, reinterpret_cast<double2*>(imin));
  auto queue_cl = queue();
  auto s = Min<double2>(n,
                        imin_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  imin_buffer.Read(queue, imin_size, reinterpret_cast<double2*>(imin));
}

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// GEMV
void cblas_sgemv(const Layout layout, const Transpose a_transpose,
                 const int m, const int n,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float* x, const int x_inc,
                 const float beta,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto x_size = (a_transpose != Transpose::kNo) ? m * x_inc : n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = (a_transpose != Transpose::kNo) ? n * y_inc : m * y_inc;
  auto y_buffer = Buffer<float>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = Gemv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Transpose>(a_transpose),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dgemv(const Layout layout, const Transpose a_transpose,
                 const int m, const int n,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double* x, const int x_inc,
                 const double beta,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto x_size = (a_transpose != Transpose::kNo) ? m * x_inc : n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = (a_transpose != Transpose::kNo) ? n * y_inc : m * y_inc;
  auto y_buffer = Buffer<double>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = Gemv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Transpose>(a_transpose),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}
void cblas_cgemv(const Layout layout, const Transpose a_transpose,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto x_size = (a_transpose != Transpose::kNo) ? m * x_inc : n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = (a_transpose != Transpose::kNo) ? n * y_inc : m * y_inc;
  auto y_buffer = Buffer<float2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = Gemv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Transpose>(a_transpose),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zgemv(const Layout layout, const Transpose a_transpose,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto x_size = (a_transpose != Transpose::kNo) ? m * x_inc : n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = (a_transpose != Transpose::kNo) ? n * y_inc : m * y_inc;
  auto y_buffer = Buffer<double2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = Gemv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Transpose>(a_transpose),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// GBMV
void cblas_sgbmv(const Layout layout, const Transpose a_transpose,
                 const int m, const int n, const int kl, const int ku,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float* x, const int x_inc,
                 const float beta,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto x_size = (a_transpose != Transpose::kNo) ? m * x_inc : n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = (a_transpose != Transpose::kNo) ? n * y_inc : m * y_inc;
  auto y_buffer = Buffer<float>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = Gbmv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Transpose>(a_transpose),
                m, n, kl, ku,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dgbmv(const Layout layout, const Transpose a_transpose,
                 const int m, const int n, const int kl, const int ku,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double* x, const int x_inc,
                 const double beta,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto x_size = (a_transpose != Transpose::kNo) ? m * x_inc : n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = (a_transpose != Transpose::kNo) ? n * y_inc : m * y_inc;
  auto y_buffer = Buffer<double>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = Gbmv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Transpose>(a_transpose),
                m, n, kl, ku,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}
void cblas_cgbmv(const Layout layout, const Transpose a_transpose,
                 const int m, const int n, const int kl, const int ku,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto x_size = (a_transpose != Transpose::kNo) ? m * x_inc : n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = (a_transpose != Transpose::kNo) ? n * y_inc : m * y_inc;
  auto y_buffer = Buffer<float2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = Gbmv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Transpose>(a_transpose),
                m, n, kl, ku,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zgbmv(const Layout layout, const Transpose a_transpose,
                 const int m, const int n, const int kl, const int ku,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto x_size = (a_transpose != Transpose::kNo) ? m * x_inc : n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = (a_transpose != Transpose::kNo) ? n * y_inc : m * y_inc;
  auto y_buffer = Buffer<double2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = Gbmv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Transpose>(a_transpose),
                m, n, kl, ku,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// HEMV
void cblas_chemv(const Layout layout, const Triangle triangle,
                 const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<float2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = Hemv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zhemv(const Layout layout, const Triangle triangle,
                 const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<double2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = Hemv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// HBMV
void cblas_chbmv(const Layout layout, const Triangle triangle,
                 const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<float2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = Hbmv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zhbmv(const Layout layout, const Triangle triangle,
                 const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<double2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = Hbmv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// HPMV
void cblas_chpmv(const Layout layout, const Triangle triangle,
                 const int n,
                 const void* alpha,
                 const void* ap,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<float2>(context, ap_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<float2>(context, y_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const float2*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = Hpmv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                ap_buffer(), 0,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zhpmv(const Layout layout, const Triangle triangle,
                 const int n,
                 const void* alpha,
                 const void* ap,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<double2>(context, ap_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<double2>(context, y_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const double2*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = Hpmv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                ap_buffer(), 0,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// SYMV
void cblas_ssymv(const Layout layout, const Triangle triangle,
                 const int n,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float* x, const int x_inc,
                 const float beta,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<float>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = Symv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dsymv(const Layout layout, const Triangle triangle,
                 const int n,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double* x, const int x_inc,
                 const double beta,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<double>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = Symv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}

// SBMV
void cblas_ssbmv(const Layout layout, const Triangle triangle,
                 const int n, const int k,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float* x, const int x_inc,
                 const float beta,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<float>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = Sbmv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dsbmv(const Layout layout, const Triangle triangle,
                 const int n, const int k,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double* x, const int x_inc,
                 const double beta,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<double>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = Sbmv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}

// SPMV
void cblas_sspmv(const Layout layout, const Triangle triangle,
                 const int n,
                 const float alpha,
                 const float* ap,
                 const float* x, const int x_inc,
                 const float beta,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<float>(context, ap_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<float>(context, y_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const float*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = Spmv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                ap_buffer(), 0,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dspmv(const Layout layout, const Triangle triangle,
                 const int n,
                 const double alpha,
                 const double* ap,
                 const double* x, const int x_inc,
                 const double beta,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<double>(context, ap_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<double>(context, y_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const double*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = Spmv(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                ap_buffer(), 0,
                x_buffer(), 0, x_inc,
                beta_cpp,
                y_buffer(), 0, y_inc,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}

// TRMV
void cblas_strmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const float* a, const int a_ld,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = Trmv<float>(static_cast<clblast::Layout>(layout),
                       static_cast<clblast::Triangle>(triangle),
                       static_cast<clblast::Transpose>(a_transpose),
                       static_cast<clblast::Diagonal>(diagonal),
                       n,
                       a_buffer(), 0, a_ld,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const double* a, const int a_ld,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = Trmv<double>(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        static_cast<clblast::Transpose>(a_transpose),
                        static_cast<clblast::Diagonal>(diagonal),
                        n,
                        a_buffer(), 0, a_ld,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_ctrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = Trmv<float2>(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        static_cast<clblast::Transpose>(a_transpose),
                        static_cast<clblast::Diagonal>(diagonal),
                        n,
                        a_buffer(), 0, a_ld,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_ztrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = Trmv<double2>(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         n,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// TBMV
void cblas_stbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n, const int k,
                 const float* a, const int a_ld,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = Tbmv<float>(static_cast<clblast::Layout>(layout),
                       static_cast<clblast::Triangle>(triangle),
                       static_cast<clblast::Transpose>(a_transpose),
                       static_cast<clblast::Diagonal>(diagonal),
                       n, k,
                       a_buffer(), 0, a_ld,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n, const int k,
                 const double* a, const int a_ld,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = Tbmv<double>(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        static_cast<clblast::Transpose>(a_transpose),
                        static_cast<clblast::Diagonal>(diagonal),
                        n, k,
                        a_buffer(), 0, a_ld,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_ctbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n, const int k,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = Tbmv<float2>(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        static_cast<clblast::Transpose>(a_transpose),
                        static_cast<clblast::Diagonal>(diagonal),
                        n, k,
                        a_buffer(), 0, a_ld,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_ztbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n, const int k,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = Tbmv<double2>(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         n, k,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// TPMV
void cblas_stpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const float* ap,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<float>(context, ap_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const float*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = Tpmv<float>(static_cast<clblast::Layout>(layout),
                       static_cast<clblast::Triangle>(triangle),
                       static_cast<clblast::Transpose>(a_transpose),
                       static_cast<clblast::Diagonal>(diagonal),
                       n,
                       ap_buffer(), 0,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const double* ap,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<double>(context, ap_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const double*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = Tpmv<double>(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        static_cast<clblast::Transpose>(a_transpose),
                        static_cast<clblast::Diagonal>(diagonal),
                        n,
                        ap_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_ctpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const void* ap,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<float2>(context, ap_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const float2*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = Tpmv<float2>(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        static_cast<clblast::Transpose>(a_transpose),
                        static_cast<clblast::Diagonal>(diagonal),
                        n,
                        ap_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_ztpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const void* ap,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<double2>(context, ap_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const double2*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = Tpmv<double2>(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         n,
                         ap_buffer(), 0,
                         x_buffer(), 0, x_inc,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// TRSV
void cblas_strsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const float* a, const int a_ld,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = Trsv<float>(static_cast<clblast::Layout>(layout),
                       static_cast<clblast::Triangle>(triangle),
                       static_cast<clblast::Transpose>(a_transpose),
                       static_cast<clblast::Diagonal>(diagonal),
                       n,
                       a_buffer(), 0, a_ld,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const double* a, const int a_ld,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = Trsv<double>(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        static_cast<clblast::Transpose>(a_transpose),
                        static_cast<clblast::Diagonal>(diagonal),
                        n,
                        a_buffer(), 0, a_ld,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_ctrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = Trsv<float2>(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        static_cast<clblast::Transpose>(a_transpose),
                        static_cast<clblast::Diagonal>(diagonal),
                        n,
                        a_buffer(), 0, a_ld,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_ztrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = Trsv<double2>(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         n,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// TBSV
void cblas_stbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n, const int k,
                 const float* a, const int a_ld,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = Tbsv<float>(static_cast<clblast::Layout>(layout),
                       static_cast<clblast::Triangle>(triangle),
                       static_cast<clblast::Transpose>(a_transpose),
                       static_cast<clblast::Diagonal>(diagonal),
                       n, k,
                       a_buffer(), 0, a_ld,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n, const int k,
                 const double* a, const int a_ld,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = Tbsv<double>(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        static_cast<clblast::Transpose>(a_transpose),
                        static_cast<clblast::Diagonal>(diagonal),
                        n, k,
                        a_buffer(), 0, a_ld,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_ctbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n, const int k,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = Tbsv<float2>(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        static_cast<clblast::Transpose>(a_transpose),
                        static_cast<clblast::Diagonal>(diagonal),
                        n, k,
                        a_buffer(), 0, a_ld,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_ztbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n, const int k,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = Tbsv<double2>(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         n, k,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// TPSV
void cblas_stpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const float* ap,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<float>(context, ap_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const float*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = Tpsv<float>(static_cast<clblast::Layout>(layout),
                       static_cast<clblast::Triangle>(triangle),
                       static_cast<clblast::Transpose>(a_transpose),
                       static_cast<clblast::Diagonal>(diagonal),
                       n,
                       ap_buffer(), 0,
                       x_buffer(), 0, x_inc,
                       &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const double* ap,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<double>(context, ap_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const double*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = Tpsv<double>(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        static_cast<clblast::Transpose>(a_transpose),
                        static_cast<clblast::Diagonal>(diagonal),
                        n,
                        ap_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_ctpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const void* ap,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<float2>(context, ap_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const float2*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = Tpsv<float2>(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        static_cast<clblast::Transpose>(a_transpose),
                        static_cast<clblast::Diagonal>(diagonal),
                        n,
                        ap_buffer(), 0,
                        x_buffer(), 0, x_inc,
                        &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_ztpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int n,
                 const void* ap,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<double2>(context, ap_size);
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const double2*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = Tpsv<double2>(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         n,
                         ap_buffer(), 0,
                         x_buffer(), 0, x_inc,
                         &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// GER
void cblas_sger(const Layout layout,
                const int m, const int n,
                const float alpha,
                const float* x, const int x_inc,
                const float* y, const int y_inc,
                float* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = m * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<float>(context, y_size);
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<float*>(a));
  auto queue_cl = queue();
  auto s = Ger(static_cast<clblast::Layout>(layout),
               m, n,
               alpha_cpp,
               x_buffer(), 0, x_inc,
               y_buffer(), 0, y_inc,
               a_buffer(), 0, a_ld,
               &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float*>(a));
}
void cblas_dger(const Layout layout,
                const int m, const int n,
                const double alpha,
                const double* x, const int x_inc,
                const double* y, const int y_inc,
                double* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = m * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<double>(context, y_size);
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<double*>(a));
  auto queue_cl = queue();
  auto s = Ger(static_cast<clblast::Layout>(layout),
               m, n,
               alpha_cpp,
               x_buffer(), 0, x_inc,
               y_buffer(), 0, y_inc,
               a_buffer(), 0, a_ld,
               &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double*>(a));
}

// GERU
void cblas_cgeru(const Layout layout,
                 const int m, const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto x_size = m * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<float2>(context, y_size);
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float2*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<float2*>(a));
  auto queue_cl = queue();
  auto s = Geru(static_cast<clblast::Layout>(layout),
                m, n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                a_buffer(), 0, a_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float2*>(a));
}
void cblas_zgeru(const Layout layout,
                 const int m, const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto x_size = m * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<double2>(context, y_size);
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double2*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<double2*>(a));
  auto queue_cl = queue();
  auto s = Geru(static_cast<clblast::Layout>(layout),
                m, n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                a_buffer(), 0, a_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double2*>(a));
}

// GERC
void cblas_cgerc(const Layout layout,
                 const int m, const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto x_size = m * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<float2>(context, y_size);
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float2*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<float2*>(a));
  auto queue_cl = queue();
  auto s = Gerc(static_cast<clblast::Layout>(layout),
                m, n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                a_buffer(), 0, a_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float2*>(a));
}
void cblas_zgerc(const Layout layout,
                 const int m, const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto x_size = m * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<double2>(context, y_size);
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double2*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<double2*>(a));
  auto queue_cl = queue();
  auto s = Gerc(static_cast<clblast::Layout>(layout),
                m, n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                a_buffer(), 0, a_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double2*>(a));
}

// HER
void cblas_cher(const Layout layout, const Triangle triangle,
                const int n,
                const void* alpha,
                const void* x, const int x_inc,
                void* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  a_buffer.Write(queue, a_size, reinterpret_cast<float2*>(a));
  auto queue_cl = queue();
  auto s = Her(static_cast<clblast::Layout>(layout),
               static_cast<clblast::Triangle>(triangle),
               n,
               alpha_cpp,
               x_buffer(), 0, x_inc,
               a_buffer(), 0, a_ld,
               &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float2*>(a));
}
void cblas_zher(const Layout layout, const Triangle triangle,
                const int n,
                const void* alpha,
                const void* x, const int x_inc,
                void* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  a_buffer.Write(queue, a_size, reinterpret_cast<double2*>(a));
  auto queue_cl = queue();
  auto s = Her(static_cast<clblast::Layout>(layout),
               static_cast<clblast::Triangle>(triangle),
               n,
               alpha_cpp,
               x_buffer(), 0, x_inc,
               a_buffer(), 0, a_ld,
               &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double2*>(a));
}

// HPR
void cblas_chpr(const Layout layout, const Triangle triangle,
                const int n,
                const void* alpha,
                const void* x, const int x_inc,
                void* ap) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<float2>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<float2*>(ap));
  auto queue_cl = queue();
  auto s = Hpr(static_cast<clblast::Layout>(layout),
               static_cast<clblast::Triangle>(triangle),
               n,
               alpha_cpp,
               x_buffer(), 0, x_inc,
               ap_buffer(), 0,
               &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<float2*>(ap));
}
void cblas_zhpr(const Layout layout, const Triangle triangle,
                const int n,
                const void* alpha,
                const void* x, const int x_inc,
                void* ap) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<double2>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<double2*>(ap));
  auto queue_cl = queue();
  auto s = Hpr(static_cast<clblast::Layout>(layout),
               static_cast<clblast::Triangle>(triangle),
               n,
               alpha_cpp,
               x_buffer(), 0, x_inc,
               ap_buffer(), 0,
               &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<double2*>(ap));
}

// HER2
void cblas_cher2(const Layout layout, const Triangle triangle,
                 const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<float2>(context, y_size);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float2*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<float2*>(a));
  auto queue_cl = queue();
  auto s = Her2(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                a_buffer(), 0, a_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float2*>(a));
}
void cblas_zher2(const Layout layout, const Triangle triangle,
                 const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<double2>(context, y_size);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double2*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<double2*>(a));
  auto queue_cl = queue();
  auto s = Her2(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                a_buffer(), 0, a_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double2*>(a));
}

// HPR2
void cblas_chpr2(const Layout layout, const Triangle triangle,
                 const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* ap) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<float2>(context, y_size);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<float2>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float2*>(y));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<float2*>(ap));
  auto queue_cl = queue();
  auto s = Hpr2(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                ap_buffer(), 0,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<float2*>(ap));
}
void cblas_zhpr2(const Layout layout, const Triangle triangle,
                 const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* ap) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double2>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<double2>(context, y_size);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<double2>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double2*>(y));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<double2*>(ap));
  auto queue_cl = queue();
  auto s = Hpr2(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                ap_buffer(), 0,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<double2*>(ap));
}

// SYR
void cblas_ssyr(const Layout layout, const Triangle triangle,
                const int n,
                const float alpha,
                const float* x, const int x_inc,
                float* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  a_buffer.Write(queue, a_size, reinterpret_cast<float*>(a));
  auto queue_cl = queue();
  auto s = Syr(static_cast<clblast::Layout>(layout),
               static_cast<clblast::Triangle>(triangle),
               n,
               alpha_cpp,
               x_buffer(), 0, x_inc,
               a_buffer(), 0, a_ld,
               &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float*>(a));
}
void cblas_dsyr(const Layout layout, const Triangle triangle,
                const int n,
                const double alpha,
                const double* x, const int x_inc,
                double* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  a_buffer.Write(queue, a_size, reinterpret_cast<double*>(a));
  auto queue_cl = queue();
  auto s = Syr(static_cast<clblast::Layout>(layout),
               static_cast<clblast::Triangle>(triangle),
               n,
               alpha_cpp,
               x_buffer(), 0, x_inc,
               a_buffer(), 0, a_ld,
               &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double*>(a));
}

// SPR
void cblas_sspr(const Layout layout, const Triangle triangle,
                const int n,
                const float alpha,
                const float* x, const int x_inc,
                float* ap) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<float>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<float*>(ap));
  auto queue_cl = queue();
  auto s = Spr(static_cast<clblast::Layout>(layout),
               static_cast<clblast::Triangle>(triangle),
               n,
               alpha_cpp,
               x_buffer(), 0, x_inc,
               ap_buffer(), 0,
               &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<float*>(ap));
}
void cblas_dspr(const Layout layout, const Triangle triangle,
                const int n,
                const double alpha,
                const double* x, const int x_inc,
                double* ap) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<double>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<double*>(ap));
  auto queue_cl = queue();
  auto s = Spr(static_cast<clblast::Layout>(layout),
               static_cast<clblast::Triangle>(triangle),
               n,
               alpha_cpp,
               x_buffer(), 0, x_inc,
               ap_buffer(), 0,
               &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<double*>(ap));
}

// SYR2
void cblas_ssyr2(const Layout layout, const Triangle triangle,
                 const int n,
                 const float alpha,
                 const float* x, const int x_inc,
                 const float* y, const int y_inc,
                 float* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<float>(context, y_size);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<float*>(a));
  auto queue_cl = queue();
  auto s = Syr2(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                a_buffer(), 0, a_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float*>(a));
}
void cblas_dsyr2(const Layout layout, const Triangle triangle,
                 const int n,
                 const double alpha,
                 const double* x, const int x_inc,
                 const double* y, const int y_inc,
                 double* a, const int a_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<double>(context, y_size);
  const auto a_size = n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<double*>(a));
  auto queue_cl = queue();
  auto s = Syr2(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                a_buffer(), 0, a_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double*>(a));
}

// SPR2
void cblas_sspr2(const Layout layout, const Triangle triangle,
                 const int n,
                 const float alpha,
                 const float* x, const int x_inc,
                 const float* y, const int y_inc,
                 float* ap) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<float>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<float>(context, y_size);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<float>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float*>(y));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<float*>(ap));
  auto queue_cl = queue();
  auto s = Spr2(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                ap_buffer(), 0,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<float*>(ap));
}
void cblas_dspr2(const Layout layout, const Triangle triangle,
                 const int n,
                 const double alpha,
                 const double* x, const int x_inc,
                 const double* y, const int y_inc,
                 double* ap) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  auto x_buffer = Buffer<double>(context, x_size);
  const auto y_size = n * y_inc;
  auto y_buffer = Buffer<double>(context, y_size);
  const auto ap_size = ((n*(n+1)) / 2);
  auto ap_buffer = Buffer<double>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double*>(y));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<double*>(ap));
  auto queue_cl = queue();
  auto s = Spr2(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                n,
                alpha_cpp,
                x_buffer(), 0, x_inc,
                y_buffer(), 0, y_inc,
                ap_buffer(), 0,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<double*>(ap));
}

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// GEMM
void cblas_sgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                 const int m, const int n, const int k,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float* b, const int b_ld,
                 const float beta,
                 float* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? m * a_ld : k * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && b_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && b_transpose == Transpose::kNo)) ? k * b_ld : n * b_ld;
  auto b_buffer = Buffer<float>(context, b_size);
  const auto c_size = (layout == Layout::kRowMajor) ? m * c_ld : n * c_ld;
  auto c_buffer = Buffer<float>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float*>(c));
  auto queue_cl = queue();
  auto s = Gemm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Transpose>(a_transpose),
                static_cast<clblast::Transpose>(b_transpose),
                m, n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float*>(c));
}
void cblas_dgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                 const int m, const int n, const int k,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double* b, const int b_ld,
                 const double beta,
                 double* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? m * a_ld : k * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && b_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && b_transpose == Transpose::kNo)) ? k * b_ld : n * b_ld;
  auto b_buffer = Buffer<double>(context, b_size);
  const auto c_size = (layout == Layout::kRowMajor) ? m * c_ld : n * c_ld;
  auto c_buffer = Buffer<double>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double*>(c));
  auto queue_cl = queue();
  auto s = Gemm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Transpose>(a_transpose),
                static_cast<clblast::Transpose>(b_transpose),
                m, n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double*>(c));
}
void cblas_cgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                 const int m, const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* b, const int b_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? m * a_ld : k * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && b_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && b_transpose == Transpose::kNo)) ? k * b_ld : n * b_ld;
  auto b_buffer = Buffer<float2>(context, b_size);
  const auto c_size = (layout == Layout::kRowMajor) ? m * c_ld : n * c_ld;
  auto c_buffer = Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = Gemm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Transpose>(a_transpose),
                static_cast<clblast::Transpose>(b_transpose),
                m, n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                 const int m, const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* b, const int b_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? m * a_ld : k * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && b_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && b_transpose == Transpose::kNo)) ? k * b_ld : n * b_ld;
  auto b_buffer = Buffer<double2>(context, b_size);
  const auto c_size = (layout == Layout::kRowMajor) ? m * c_ld : n * c_ld;
  auto c_buffer = Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = Gemm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Transpose>(a_transpose),
                static_cast<clblast::Transpose>(b_transpose),
                m, n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// SYMM
void cblas_ssymm(const Layout layout, const Side side, const Triangle triangle,
                 const int m, const int n,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float* b, const int b_ld,
                 const float beta,
                 float* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : ((side == Side::kLeft) ? m : n) * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? ((side == Side::kLeft) ? m : n) * b_ld : n * b_ld;
  auto b_buffer = Buffer<float>(context, b_size);
  const auto c_size = (layout == Layout::kRowMajor) ? m * c_ld : n * c_ld;
  auto c_buffer = Buffer<float>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float*>(c));
  auto queue_cl = queue();
  auto s = Symm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float*>(c));
}
void cblas_dsymm(const Layout layout, const Side side, const Triangle triangle,
                 const int m, const int n,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double* b, const int b_ld,
                 const double beta,
                 double* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : ((side == Side::kLeft) ? m : n) * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? ((side == Side::kLeft) ? m : n) * b_ld : n * b_ld;
  auto b_buffer = Buffer<double>(context, b_size);
  const auto c_size = (layout == Layout::kRowMajor) ? m * c_ld : n * c_ld;
  auto c_buffer = Buffer<double>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double*>(c));
  auto queue_cl = queue();
  auto s = Symm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double*>(c));
}
void cblas_csymm(const Layout layout, const Side side, const Triangle triangle,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* b, const int b_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : ((side == Side::kLeft) ? m : n) * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? ((side == Side::kLeft) ? m : n) * b_ld : n * b_ld;
  auto b_buffer = Buffer<float2>(context, b_size);
  const auto c_size = (layout == Layout::kRowMajor) ? m * c_ld : n * c_ld;
  auto c_buffer = Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = Symm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zsymm(const Layout layout, const Side side, const Triangle triangle,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* b, const int b_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : ((side == Side::kLeft) ? m : n) * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? ((side == Side::kLeft) ? m : n) * b_ld : n * b_ld;
  auto b_buffer = Buffer<double2>(context, b_size);
  const auto c_size = (layout == Layout::kRowMajor) ? m * c_ld : n * c_ld;
  auto c_buffer = Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = Symm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// HEMM
void cblas_chemm(const Layout layout, const Side side, const Triangle triangle,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* b, const int b_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : ((side == Side::kLeft) ? m : n) * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? ((side == Side::kLeft) ? m : n) * b_ld : n * b_ld;
  auto b_buffer = Buffer<float2>(context, b_size);
  const auto c_size = (layout == Layout::kRowMajor) ? m * c_ld : n * c_ld;
  auto c_buffer = Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = Hemm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zhemm(const Layout layout, const Side side, const Triangle triangle,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* b, const int b_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : ((side == Side::kLeft) ? m : n) * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? ((side == Side::kLeft) ? m : n) * b_ld : n * b_ld;
  auto b_buffer = Buffer<double2>(context, b_size);
  const auto c_size = (layout == Layout::kRowMajor) ? m * c_ld : n * c_ld;
  auto c_buffer = Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = Hemm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// SYRK
void cblas_ssyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                 const int n, const int k,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float beta,
                 float* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? n * a_ld : k * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto c_size = n * c_ld;
  auto c_buffer = Buffer<float>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  c_buffer.Write(queue, c_size, reinterpret_cast<float*>(c));
  auto queue_cl = queue();
  auto s = Syrk(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float*>(c));
}
void cblas_dsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                 const int n, const int k,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double beta,
                 double* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? n * a_ld : k * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto c_size = n * c_ld;
  auto c_buffer = Buffer<double>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  c_buffer.Write(queue, c_size, reinterpret_cast<double*>(c));
  auto queue_cl = queue();
  auto s = Syrk(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double*>(c));
}
void cblas_csyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                 const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? n * a_ld : k * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto c_size = n * c_ld;
  auto c_buffer = Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = Syrk(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                 const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? n * a_ld : k * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto c_size = n * c_ld;
  auto c_buffer = Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = Syrk(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// HERK
void cblas_cherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                 const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? n * a_ld : k * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto c_size = n * c_ld;
  auto c_buffer = Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = Herk(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                 const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? n * a_ld : k * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto c_size = n * c_ld;
  auto c_buffer = Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = Herk(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                n, k,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                beta_cpp,
                c_buffer(), 0, c_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// SYR2K
void cblas_ssyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                  const int n, const int k,
                  const float alpha,
                  const float* a, const int a_ld,
                  const float* b, const int b_ld,
                  const float beta,
                  float* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == Layout::kColMajor && ab_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && ab_transpose == Transpose::kNo)) ? n * a_ld : k * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && ab_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && ab_transpose == Transpose::kNo)) ? n * b_ld : k * b_ld;
  auto b_buffer = Buffer<float>(context, b_size);
  const auto c_size = n * c_ld;
  auto c_buffer = Buffer<float>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float*>(c));
  auto queue_cl = queue();
  auto s = Syr2k(static_cast<clblast::Layout>(layout),
                 static_cast<clblast::Triangle>(triangle),
                 static_cast<clblast::Transpose>(ab_transpose),
                 n, k,
                 alpha_cpp,
                 a_buffer(), 0, a_ld,
                 b_buffer(), 0, b_ld,
                 beta_cpp,
                 c_buffer(), 0, c_ld,
                 &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float*>(c));
}
void cblas_dsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                  const int n, const int k,
                  const double alpha,
                  const double* a, const int a_ld,
                  const double* b, const int b_ld,
                  const double beta,
                  double* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == Layout::kColMajor && ab_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && ab_transpose == Transpose::kNo)) ? n * a_ld : k * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && ab_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && ab_transpose == Transpose::kNo)) ? n * b_ld : k * b_ld;
  auto b_buffer = Buffer<double>(context, b_size);
  const auto c_size = n * c_ld;
  auto c_buffer = Buffer<double>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double*>(c));
  auto queue_cl = queue();
  auto s = Syr2k(static_cast<clblast::Layout>(layout),
                 static_cast<clblast::Triangle>(triangle),
                 static_cast<clblast::Transpose>(ab_transpose),
                 n, k,
                 alpha_cpp,
                 a_buffer(), 0, a_ld,
                 b_buffer(), 0, b_ld,
                 beta_cpp,
                 c_buffer(), 0, c_ld,
                 &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double*>(c));
}
void cblas_csyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                  const int n, const int k,
                  const void* alpha,
                  const void* a, const int a_ld,
                  const void* b, const int b_ld,
                  const void* beta,
                  void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = ((layout == Layout::kColMajor && ab_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && ab_transpose == Transpose::kNo)) ? n * a_ld : k * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && ab_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && ab_transpose == Transpose::kNo)) ? n * b_ld : k * b_ld;
  auto b_buffer = Buffer<float2>(context, b_size);
  const auto c_size = n * c_ld;
  auto c_buffer = Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = Syr2k(static_cast<clblast::Layout>(layout),
                 static_cast<clblast::Triangle>(triangle),
                 static_cast<clblast::Transpose>(ab_transpose),
                 n, k,
                 alpha_cpp,
                 a_buffer(), 0, a_ld,
                 b_buffer(), 0, b_ld,
                 beta_cpp,
                 c_buffer(), 0, c_ld,
                 &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                  const int n, const int k,
                  const void* alpha,
                  const void* a, const int a_ld,
                  const void* b, const int b_ld,
                  const void* beta,
                  void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = ((layout == Layout::kColMajor && ab_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && ab_transpose == Transpose::kNo)) ? n * a_ld : k * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && ab_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && ab_transpose == Transpose::kNo)) ? n * b_ld : k * b_ld;
  auto b_buffer = Buffer<double2>(context, b_size);
  const auto c_size = n * c_ld;
  auto c_buffer = Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = Syr2k(static_cast<clblast::Layout>(layout),
                 static_cast<clblast::Triangle>(triangle),
                 static_cast<clblast::Transpose>(ab_transpose),
                 n, k,
                 alpha_cpp,
                 a_buffer(), 0, a_ld,
                 b_buffer(), 0, b_ld,
                 beta_cpp,
                 c_buffer(), 0, c_ld,
                 &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// HER2K
void cblas_cher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                  const int n, const int k,
                  const void* alpha,
                  const void* a, const int a_ld,
                  const void* b, const int b_ld,
                  const void* beta,
                  void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = beta;
  const auto a_size = ((layout == Layout::kColMajor && ab_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && ab_transpose == Transpose::kNo)) ? n * a_ld : k * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && ab_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && ab_transpose == Transpose::kNo)) ? n * b_ld : k * b_ld;
  auto b_buffer = Buffer<float2>(context, b_size);
  const auto c_size = n * c_ld;
  auto c_buffer = Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = Her2k(static_cast<clblast::Layout>(layout),
                 static_cast<clblast::Triangle>(triangle),
                 static_cast<clblast::Transpose>(ab_transpose),
                 n, k,
                 alpha_cpp,
                 a_buffer(), 0, a_ld,
                 b_buffer(), 0, b_ld,
                 beta_cpp,
                 c_buffer(), 0, c_ld,
                 &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                  const int n, const int k,
                  const void* alpha,
                  const void* a, const int a_ld,
                  const void* b, const int b_ld,
                  const void* beta,
                  void* c, const int c_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = beta;
  const auto a_size = ((layout == Layout::kColMajor && ab_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && ab_transpose == Transpose::kNo)) ? n * a_ld : k * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && ab_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && ab_transpose == Transpose::kNo)) ? n * b_ld : k * b_ld;
  auto b_buffer = Buffer<double2>(context, b_size);
  const auto c_size = n * c_ld;
  auto c_buffer = Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = Her2k(static_cast<clblast::Layout>(layout),
                 static_cast<clblast::Triangle>(triangle),
                 static_cast<clblast::Transpose>(ab_transpose),
                 n, k,
                 alpha_cpp,
                 a_buffer(), 0, a_ld,
                 b_buffer(), 0, b_ld,
                 beta_cpp,
                 c_buffer(), 0, c_ld,
                 &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// TRMM
void cblas_strmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int m, const int n,
                 const float alpha,
                 const float* a, const int a_ld,
                 float* b, const int b_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto a_size = (side == Side::kLeft) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? m * b_ld : n * b_ld;
  auto b_buffer = Buffer<float>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<float*>(b));
  auto queue_cl = queue();
  auto s = Trmm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                static_cast<clblast::Diagonal>(diagonal),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<float*>(b));
}
void cblas_dtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int m, const int n,
                 const double alpha,
                 const double* a, const int a_ld,
                 double* b, const int b_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto a_size = (side == Side::kLeft) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? m * b_ld : n * b_ld;
  auto b_buffer = Buffer<double>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<double*>(b));
  auto queue_cl = queue();
  auto s = Trmm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                static_cast<clblast::Diagonal>(diagonal),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<double*>(b));
}
void cblas_ctrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 void* b, const int b_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto a_size = (side == Side::kLeft) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? m * b_ld : n * b_ld;
  auto b_buffer = Buffer<float2>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<float2*>(b));
  auto queue_cl = queue();
  auto s = Trmm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                static_cast<clblast::Diagonal>(diagonal),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<float2*>(b));
}
void cblas_ztrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 void* b, const int b_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto a_size = (side == Side::kLeft) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? m * b_ld : n * b_ld;
  auto b_buffer = Buffer<double2>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<double2*>(b));
  auto queue_cl = queue();
  auto s = Trmm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                static_cast<clblast::Diagonal>(diagonal),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<double2*>(b));
}

// TRSM
void cblas_strsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int m, const int n,
                 const float alpha,
                 const float* a, const int a_ld,
                 float* b, const int b_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto a_size = (side == Side::kLeft) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? m * b_ld : n * b_ld;
  auto b_buffer = Buffer<float>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<float*>(b));
  auto queue_cl = queue();
  auto s = Trsm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                static_cast<clblast::Diagonal>(diagonal),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<float*>(b));
}
void cblas_dtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int m, const int n,
                 const double alpha,
                 const double* a, const int a_ld,
                 double* b, const int b_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto a_size = (side == Side::kLeft) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? m * b_ld : n * b_ld;
  auto b_buffer = Buffer<double>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<double*>(b));
  auto queue_cl = queue();
  auto s = Trsm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                static_cast<clblast::Diagonal>(diagonal),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<double*>(b));
}
void cblas_ctrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 void* b, const int b_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto a_size = (side == Side::kLeft) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? m * b_ld : n * b_ld;
  auto b_buffer = Buffer<float2>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<float2*>(b));
  auto queue_cl = queue();
  auto s = Trsm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                static_cast<clblast::Diagonal>(diagonal),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<float2*>(b));
}
void cblas_ztrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 void* b, const int b_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto a_size = (side == Side::kLeft) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto b_size = (layout == Layout::kRowMajor) ? m * b_ld : n * b_ld;
  auto b_buffer = Buffer<double2>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<double2*>(b));
  auto queue_cl = queue();
  auto s = Trsm(static_cast<clblast::Layout>(layout),
                static_cast<clblast::Side>(side),
                static_cast<clblast::Triangle>(triangle),
                static_cast<clblast::Transpose>(a_transpose),
                static_cast<clblast::Diagonal>(diagonal),
                m, n,
                alpha_cpp,
                a_buffer(), 0, a_ld,
                b_buffer(), 0, b_ld,
                &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<double2*>(b));
}

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

// OMATCOPY
void cblas_somatcopy(const Layout layout, const Transpose a_transpose,
                     const int m, const int n,
                     const float alpha,
                     const float* a, const int a_ld,
                     float* b, const int b_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<float>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? n * b_ld : m * b_ld;
  auto b_buffer = Buffer<float>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<float*>(b));
  auto queue_cl = queue();
  auto s = Omatcopy(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n,
                    alpha_cpp,
                    a_buffer(), 0, a_ld,
                    b_buffer(), 0, b_ld,
                    &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<float*>(b));
}
void cblas_domatcopy(const Layout layout, const Transpose a_transpose,
                     const int m, const int n,
                     const double alpha,
                     const double* a, const int a_ld,
                     double* b, const int b_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<double>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? n * b_ld : m * b_ld;
  auto b_buffer = Buffer<double>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<double*>(b));
  auto queue_cl = queue();
  auto s = Omatcopy(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n,
                    alpha_cpp,
                    a_buffer(), 0, a_ld,
                    b_buffer(), 0, b_ld,
                    &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<double*>(b));
}
void cblas_comatcopy(const Layout layout, const Transpose a_transpose,
                     const int m, const int n,
                     const void* alpha,
                     const void* a, const int a_ld,
                     void* b, const int b_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<float2>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? n * b_ld : m * b_ld;
  auto b_buffer = Buffer<float2>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<float2*>(b));
  auto queue_cl = queue();
  auto s = Omatcopy(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n,
                    alpha_cpp,
                    a_buffer(), 0, a_ld,
                    b_buffer(), 0, b_ld,
                    &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<float2*>(b));
}
void cblas_zomatcopy(const Layout layout, const Transpose a_transpose,
                     const int m, const int n,
                     const void* alpha,
                     const void* a, const int a_ld,
                     void* b, const int b_ld) {
  auto device = get_device();
  auto context = Context(device);
  auto queue = Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto a_size = (layout == Layout::kRowMajor) ? m * a_ld : n * a_ld;
  auto a_buffer = Buffer<double2>(context, a_size);
  const auto b_size = ((layout == Layout::kColMajor && a_transpose != Transpose::kNo) || (layout == Layout::kRowMajor && a_transpose == Transpose::kNo)) ? n * b_ld : m * b_ld;
  auto b_buffer = Buffer<double2>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<double2*>(b));
  auto queue_cl = queue();
  auto s = Omatcopy(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n,
                    alpha_cpp,
                    a_buffer(), 0, a_ld,
                    b_buffer(), 0, b_ld,
                    &queue_cl);
  if (s != StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<double2*>(b));
}

// =================================================================================================
} // namespace clblast
