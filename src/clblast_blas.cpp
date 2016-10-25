
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

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// Helper function to get a default OpenCL platform and device
clblast::Device get_device() {
  auto platform_id = clblast::ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0});
  auto device_id = clblast::ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0});
  auto platform = clblast::Platform(platform_id);
  return clblast::Device(platform, device_id);
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
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto sa_size = 1;
  const auto sb_size = 1;
  const auto sc_size = 1;
  const auto ss_size = 1;
  auto sa_buffer = clblast::Buffer<float>(context, sa_size);
  auto sb_buffer = clblast::Buffer<float>(context, sb_size);
  auto sc_buffer = clblast::Buffer<float>(context, sc_size);
  auto ss_buffer = clblast::Buffer<float>(context, ss_size);
  sa_buffer.Write(queue, sa_size, reinterpret_cast<float*>(sa));
  sb_buffer.Write(queue, sb_size, reinterpret_cast<float*>(sb));
  sc_buffer.Write(queue, sc_size, reinterpret_cast<float*>(sc));
  ss_buffer.Write(queue, ss_size, reinterpret_cast<float*>(ss));
  auto queue_cl = queue();
  auto s = clblast::Rotg<float>(sa_buffer(), 0,
                                sb_buffer(), 0,
                                sc_buffer(), 0,
                                ss_buffer(), 0,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
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
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto sa_size = 1;
  const auto sb_size = 1;
  const auto sc_size = 1;
  const auto ss_size = 1;
  auto sa_buffer = clblast::Buffer<double>(context, sa_size);
  auto sb_buffer = clblast::Buffer<double>(context, sb_size);
  auto sc_buffer = clblast::Buffer<double>(context, sc_size);
  auto ss_buffer = clblast::Buffer<double>(context, ss_size);
  sa_buffer.Write(queue, sa_size, reinterpret_cast<double*>(sa));
  sb_buffer.Write(queue, sb_size, reinterpret_cast<double*>(sb));
  sc_buffer.Write(queue, sc_size, reinterpret_cast<double*>(sc));
  ss_buffer.Write(queue, ss_size, reinterpret_cast<double*>(ss));
  auto queue_cl = queue();
  auto s = clblast::Rotg<double>(sa_buffer(), 0,
                                 sb_buffer(), 0,
                                 sc_buffer(), 0,
                                 ss_buffer(), 0,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
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
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto sy1_size = 1;
  const auto sd1_size = 1;
  const auto sd2_size = 1;
  const auto sx1_size = 1;
  const auto sparam_size = 1;
  auto sy1_buffer = clblast::Buffer<float>(context, sy1_size);
  auto sd1_buffer = clblast::Buffer<float>(context, sd1_size);
  auto sd2_buffer = clblast::Buffer<float>(context, sd2_size);
  auto sx1_buffer = clblast::Buffer<float>(context, sx1_size);
  auto sparam_buffer = clblast::Buffer<float>(context, sparam_size);
  sy1_buffer.Write(queue, sy1_size, reinterpret_cast<const float*>(sy1));
  sd1_buffer.Write(queue, sd1_size, reinterpret_cast<float*>(sd1));
  sd2_buffer.Write(queue, sd2_size, reinterpret_cast<float*>(sd2));
  sx1_buffer.Write(queue, sx1_size, reinterpret_cast<float*>(sx1));
  sparam_buffer.Write(queue, sparam_size, reinterpret_cast<float*>(sparam));
  auto queue_cl = queue();
  auto s = clblast::Rotmg<float>(sd1_buffer(), 0,
                                 sd2_buffer(), 0,
                                 sx1_buffer(), 0,
                                 sy1_buffer(), 0,
                                 sparam_buffer(), 0,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
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
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto sy1_size = 1;
  const auto sd1_size = 1;
  const auto sd2_size = 1;
  const auto sx1_size = 1;
  const auto sparam_size = 1;
  auto sy1_buffer = clblast::Buffer<double>(context, sy1_size);
  auto sd1_buffer = clblast::Buffer<double>(context, sd1_size);
  auto sd2_buffer = clblast::Buffer<double>(context, sd2_size);
  auto sx1_buffer = clblast::Buffer<double>(context, sx1_size);
  auto sparam_buffer = clblast::Buffer<double>(context, sparam_size);
  sy1_buffer.Write(queue, sy1_size, reinterpret_cast<const double*>(sy1));
  sd1_buffer.Write(queue, sd1_size, reinterpret_cast<double*>(sd1));
  sd2_buffer.Write(queue, sd2_size, reinterpret_cast<double*>(sd2));
  sx1_buffer.Write(queue, sx1_size, reinterpret_cast<double*>(sx1));
  sparam_buffer.Write(queue, sparam_size, reinterpret_cast<double*>(sparam));
  auto queue_cl = queue();
  auto s = clblast::Rotmg<double>(sd1_buffer(), 0,
                                  sd2_buffer(), 0,
                                  sx1_buffer(), 0,
                                  sy1_buffer(), 0,
                                  sparam_buffer(), 0,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
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
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = clblast::Rot(n,
                        x_buffer(), 0, x_inc,
                        y_buffer(), 0, y_inc,
                        cos,
                        sin,
                        &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
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
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = clblast::Rot(n,
                        x_buffer(), 0, x_inc,
                        y_buffer(), 0, y_inc,
                        cos,
                        sin,
                        &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
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
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  const auto sparam_size = 1;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  auto sparam_buffer = clblast::Buffer<float>(context, sparam_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  sparam_buffer.Write(queue, sparam_size, reinterpret_cast<float*>(sparam));
  auto queue_cl = queue();
  auto s = clblast::Rotm<float>(n,
                                x_buffer(), 0, x_inc,
                                y_buffer(), 0, y_inc,
                                sparam_buffer(), 0,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
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
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  const auto sparam_size = 1;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  auto sparam_buffer = clblast::Buffer<double>(context, sparam_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  sparam_buffer.Write(queue, sparam_size, reinterpret_cast<double*>(sparam));
  auto queue_cl = queue();
  auto s = clblast::Rotm<double>(n,
                                 x_buffer(), 0, x_inc,
                                 y_buffer(), 0, y_inc,
                                 sparam_buffer(), 0,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
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
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = clblast::Swap<float>(n,
                                x_buffer(), 0, x_inc,
                                y_buffer(), 0, y_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dswap(const int n,
                 double* x, const int x_inc,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = clblast::Swap<double>(n,
                                 x_buffer(), 0, x_inc,
                                 y_buffer(), 0, y_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}
void cblas_cswap(const int n,
                 void* x, const int x_inc,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Swap<float2>(n,
                                 x_buffer(), 0, x_inc,
                                 y_buffer(), 0, y_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zswap(const int n,
                 void* x, const int x_inc,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Swap<double2>(n,
                                  x_buffer(), 0, x_inc,
                                  y_buffer(), 0, y_inc,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// SCAL
void cblas_sscal(const int n,
                 const float alpha,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = clblast::Scal(n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dscal(const int n,
                 const double alpha,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = clblast::Scal(n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_cscal(const int n,
                 const void* alpha,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto x_size = n;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Scal(n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_zscal(const int n,
                 const void* alpha,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto x_size = n;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Scal(n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// COPY
void cblas_scopy(const int n,
                 const float* x, const int x_inc,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = clblast::Copy<float>(n,
                                x_buffer(), 0, x_inc,
                                y_buffer(), 0, y_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dcopy(const int n,
                 const double* x, const int x_inc,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = clblast::Copy<double>(n,
                                 x_buffer(), 0, x_inc,
                                 y_buffer(), 0, y_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}
void cblas_ccopy(const int n,
                 const void* x, const int x_inc,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Copy<float2>(n,
                                 x_buffer(), 0, x_inc,
                                 y_buffer(), 0, y_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zcopy(const int n,
                 const void* x, const int x_inc,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Copy<double2>(n,
                                  x_buffer(), 0, x_inc,
                                  y_buffer(), 0, y_inc,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// AXPY
void cblas_saxpy(const int n,
                 const float alpha,
                 const float* x, const int x_inc,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = clblast::Axpy(n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_daxpy(const int n,
                 const double alpha,
                 const double* x, const int x_inc,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = clblast::Axpy(n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}
void cblas_caxpy(const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Axpy(n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zaxpy(const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto x_size = n;
  const auto y_size = n;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Axpy(n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// DOT
void cblas_sdot(const int n,
                float* dot,
                const float* x, const int x_inc,
                const float* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  const auto dot_size = 1;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  auto dot_buffer = clblast::Buffer<float>(context, dot_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float*>(y));
  dot_buffer.Write(queue, dot_size, reinterpret_cast<float*>(dot));
  auto queue_cl = queue();
  auto s = clblast::Dot<float>(n,
                               dot_buffer(), 0,
                               x_buffer(), 0, x_inc,
                               y_buffer(), 0, y_inc,
                               &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  dot_buffer.Read(queue, dot_size, reinterpret_cast<float*>(dot));
}
void cblas_ddot(const int n,
                double* dot,
                const double* x, const int x_inc,
                const double* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  const auto dot_size = 1;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  auto dot_buffer = clblast::Buffer<double>(context, dot_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double*>(y));
  dot_buffer.Write(queue, dot_size, reinterpret_cast<double*>(dot));
  auto queue_cl = queue();
  auto s = clblast::Dot<double>(n,
                                dot_buffer(), 0,
                                x_buffer(), 0, x_inc,
                                y_buffer(), 0, y_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  dot_buffer.Read(queue, dot_size, reinterpret_cast<double*>(dot));
}

// DOTU
void cblas_cdotu(const int n,
                 void* dot,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  const auto dot_size = 1;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  auto dot_buffer = clblast::Buffer<float2>(context, dot_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float2*>(y));
  dot_buffer.Write(queue, dot_size, reinterpret_cast<float2*>(dot));
  auto queue_cl = queue();
  auto s = clblast::Dotu<float2>(n,
                                 dot_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 y_buffer(), 0, y_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  dot_buffer.Read(queue, dot_size, reinterpret_cast<float2*>(dot));
}
void cblas_zdotu(const int n,
                 void* dot,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  const auto dot_size = 1;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  auto dot_buffer = clblast::Buffer<double2>(context, dot_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double2*>(y));
  dot_buffer.Write(queue, dot_size, reinterpret_cast<double2*>(dot));
  auto queue_cl = queue();
  auto s = clblast::Dotu<double2>(n,
                                  dot_buffer(), 0,
                                  x_buffer(), 0, x_inc,
                                  y_buffer(), 0, y_inc,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  dot_buffer.Read(queue, dot_size, reinterpret_cast<double2*>(dot));
}

// DOTC
void cblas_cdotc(const int n,
                 void* dot,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  const auto dot_size = 1;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  auto dot_buffer = clblast::Buffer<float2>(context, dot_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float2*>(y));
  dot_buffer.Write(queue, dot_size, reinterpret_cast<float2*>(dot));
  auto queue_cl = queue();
  auto s = clblast::Dotc<float2>(n,
                                 dot_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 y_buffer(), 0, y_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  dot_buffer.Read(queue, dot_size, reinterpret_cast<float2*>(dot));
}
void cblas_zdotc(const int n,
                 void* dot,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto y_size = n;
  const auto dot_size = 1;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  auto dot_buffer = clblast::Buffer<double2>(context, dot_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double2*>(y));
  dot_buffer.Write(queue, dot_size, reinterpret_cast<double2*>(dot));
  auto queue_cl = queue();
  auto s = clblast::Dotc<double2>(n,
                                  dot_buffer(), 0,
                                  x_buffer(), 0, x_inc,
                                  y_buffer(), 0, y_inc,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  dot_buffer.Read(queue, dot_size, reinterpret_cast<double2*>(dot));
}

// NRM2
void cblas_snrm2(const int n,
                 float* nrm2,
                 const float* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto nrm2_size = 1;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto nrm2_buffer = clblast::Buffer<float>(context, nrm2_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  nrm2_buffer.Write(queue, nrm2_size, reinterpret_cast<float*>(nrm2));
  auto queue_cl = queue();
  auto s = clblast::Nrm2<float>(n,
                                nrm2_buffer(), 0,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  nrm2_buffer.Read(queue, nrm2_size, reinterpret_cast<float*>(nrm2));
}
void cblas_dnrm2(const int n,
                 double* nrm2,
                 const double* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto nrm2_size = 1;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto nrm2_buffer = clblast::Buffer<double>(context, nrm2_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  nrm2_buffer.Write(queue, nrm2_size, reinterpret_cast<double*>(nrm2));
  auto queue_cl = queue();
  auto s = clblast::Nrm2<double>(n,
                                 nrm2_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  nrm2_buffer.Read(queue, nrm2_size, reinterpret_cast<double*>(nrm2));
}
void cblas_scnrm2(const int n,
                 void* nrm2,
                 const void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto nrm2_size = 1;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto nrm2_buffer = clblast::Buffer<float2>(context, nrm2_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  nrm2_buffer.Write(queue, nrm2_size, reinterpret_cast<float2*>(nrm2));
  auto queue_cl = queue();
  auto s = clblast::Nrm2<float2>(n,
                                 nrm2_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  nrm2_buffer.Read(queue, nrm2_size, reinterpret_cast<float2*>(nrm2));
}
void cblas_dznrm2(const int n,
                 void* nrm2,
                 const void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto nrm2_size = 1;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto nrm2_buffer = clblast::Buffer<double2>(context, nrm2_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  nrm2_buffer.Write(queue, nrm2_size, reinterpret_cast<double2*>(nrm2));
  auto queue_cl = queue();
  auto s = clblast::Nrm2<double2>(n,
                                  nrm2_buffer(), 0,
                                  x_buffer(), 0, x_inc,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  nrm2_buffer.Read(queue, nrm2_size, reinterpret_cast<double2*>(nrm2));
}

// ASUM
void cblas_sasum(const int n,
                 float* asum,
                 const float* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto asum_size = 1;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto asum_buffer = clblast::Buffer<float>(context, asum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  asum_buffer.Write(queue, asum_size, reinterpret_cast<float*>(asum));
  auto queue_cl = queue();
  auto s = clblast::Asum<float>(n,
                                asum_buffer(), 0,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  asum_buffer.Read(queue, asum_size, reinterpret_cast<float*>(asum));
}
void cblas_dasum(const int n,
                 double* asum,
                 const double* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto asum_size = 1;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto asum_buffer = clblast::Buffer<double>(context, asum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  asum_buffer.Write(queue, asum_size, reinterpret_cast<double*>(asum));
  auto queue_cl = queue();
  auto s = clblast::Asum<double>(n,
                                 asum_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  asum_buffer.Read(queue, asum_size, reinterpret_cast<double*>(asum));
}
void cblas_scasum(const int n,
                 void* asum,
                 const void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto asum_size = 1;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto asum_buffer = clblast::Buffer<float2>(context, asum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  asum_buffer.Write(queue, asum_size, reinterpret_cast<float2*>(asum));
  auto queue_cl = queue();
  auto s = clblast::Asum<float2>(n,
                                 asum_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  asum_buffer.Read(queue, asum_size, reinterpret_cast<float2*>(asum));
}
void cblas_dzasum(const int n,
                 void* asum,
                 const void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto asum_size = 1;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto asum_buffer = clblast::Buffer<double2>(context, asum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  asum_buffer.Write(queue, asum_size, reinterpret_cast<double2*>(asum));
  auto queue_cl = queue();
  auto s = clblast::Asum<double2>(n,
                                  asum_buffer(), 0,
                                  x_buffer(), 0, x_inc,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  asum_buffer.Read(queue, asum_size, reinterpret_cast<double2*>(asum));
}

// SUM
void cblas_ssum(const int n,
                float* sum,
                const float* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto sum_size = 1;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto sum_buffer = clblast::Buffer<float>(context, sum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  sum_buffer.Write(queue, sum_size, reinterpret_cast<float*>(sum));
  auto queue_cl = queue();
  auto s = clblast::Sum<float>(n,
                               sum_buffer(), 0,
                               x_buffer(), 0, x_inc,
                               &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  sum_buffer.Read(queue, sum_size, reinterpret_cast<float*>(sum));
}
void cblas_dsum(const int n,
                double* sum,
                const double* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto sum_size = 1;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto sum_buffer = clblast::Buffer<double>(context, sum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  sum_buffer.Write(queue, sum_size, reinterpret_cast<double*>(sum));
  auto queue_cl = queue();
  auto s = clblast::Sum<double>(n,
                                sum_buffer(), 0,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  sum_buffer.Read(queue, sum_size, reinterpret_cast<double*>(sum));
}
void cblas_scsum(const int n,
                void* sum,
                const void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto sum_size = 1;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto sum_buffer = clblast::Buffer<float2>(context, sum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  sum_buffer.Write(queue, sum_size, reinterpret_cast<float2*>(sum));
  auto queue_cl = queue();
  auto s = clblast::Sum<float2>(n,
                                sum_buffer(), 0,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  sum_buffer.Read(queue, sum_size, reinterpret_cast<float2*>(sum));
}
void cblas_dzsum(const int n,
                void* sum,
                const void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto sum_size = 1;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto sum_buffer = clblast::Buffer<double2>(context, sum_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  sum_buffer.Write(queue, sum_size, reinterpret_cast<double2*>(sum));
  auto queue_cl = queue();
  auto s = clblast::Sum<double2>(n,
                                 sum_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  sum_buffer.Read(queue, sum_size, reinterpret_cast<double2*>(sum));
}

// AMAX
void cblas_isamax(const int n,
                 float* imax,
                 const float* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto imax_size = 1;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto imax_buffer = clblast::Buffer<float>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<float*>(imax));
  auto queue_cl = queue();
  auto s = clblast::Amax<float>(n,
                                imax_buffer(), 0,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<float*>(imax));
}
void cblas_idamax(const int n,
                 double* imax,
                 const double* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto imax_size = 1;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto imax_buffer = clblast::Buffer<double>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<double*>(imax));
  auto queue_cl = queue();
  auto s = clblast::Amax<double>(n,
                                 imax_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<double*>(imax));
}
void cblas_icamax(const int n,
                 void* imax,
                 const void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto imax_size = 1;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto imax_buffer = clblast::Buffer<float2>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<float2*>(imax));
  auto queue_cl = queue();
  auto s = clblast::Amax<float2>(n,
                                 imax_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<float2*>(imax));
}
void cblas_izamax(const int n,
                 void* imax,
                 const void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto imax_size = 1;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto imax_buffer = clblast::Buffer<double2>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<double2*>(imax));
  auto queue_cl = queue();
  auto s = clblast::Amax<double2>(n,
                                  imax_buffer(), 0,
                                  x_buffer(), 0, x_inc,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<double2*>(imax));
}

// MAX
void cblas_ismax(const int n,
                float* imax,
                const float* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto imax_size = 1;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto imax_buffer = clblast::Buffer<float>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<float*>(imax));
  auto queue_cl = queue();
  auto s = clblast::Max<float>(n,
                               imax_buffer(), 0,
                               x_buffer(), 0, x_inc,
                               &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<float*>(imax));
}
void cblas_idmax(const int n,
                double* imax,
                const double* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto imax_size = 1;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto imax_buffer = clblast::Buffer<double>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<double*>(imax));
  auto queue_cl = queue();
  auto s = clblast::Max<double>(n,
                                imax_buffer(), 0,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<double*>(imax));
}
void cblas_icmax(const int n,
                void* imax,
                const void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto imax_size = 1;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto imax_buffer = clblast::Buffer<float2>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<float2*>(imax));
  auto queue_cl = queue();
  auto s = clblast::Max<float2>(n,
                                imax_buffer(), 0,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<float2*>(imax));
}
void cblas_izmax(const int n,
                void* imax,
                const void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto imax_size = 1;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto imax_buffer = clblast::Buffer<double2>(context, imax_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  imax_buffer.Write(queue, imax_size, reinterpret_cast<double2*>(imax));
  auto queue_cl = queue();
  auto s = clblast::Max<double2>(n,
                                 imax_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  imax_buffer.Read(queue, imax_size, reinterpret_cast<double2*>(imax));
}

// MIN
void cblas_ismin(const int n,
                float* imin,
                const float* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto imin_size = 1;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto imin_buffer = clblast::Buffer<float>(context, imin_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  imin_buffer.Write(queue, imin_size, reinterpret_cast<float*>(imin));
  auto queue_cl = queue();
  auto s = clblast::Min<float>(n,
                               imin_buffer(), 0,
                               x_buffer(), 0, x_inc,
                               &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  imin_buffer.Read(queue, imin_size, reinterpret_cast<float*>(imin));
}
void cblas_idmin(const int n,
                double* imin,
                const double* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto imin_size = 1;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto imin_buffer = clblast::Buffer<double>(context, imin_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  imin_buffer.Write(queue, imin_size, reinterpret_cast<double*>(imin));
  auto queue_cl = queue();
  auto s = clblast::Min<double>(n,
                                imin_buffer(), 0,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  imin_buffer.Read(queue, imin_size, reinterpret_cast<double*>(imin));
}
void cblas_icmin(const int n,
                void* imin,
                const void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto imin_size = 1;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto imin_buffer = clblast::Buffer<float2>(context, imin_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  imin_buffer.Write(queue, imin_size, reinterpret_cast<float2*>(imin));
  auto queue_cl = queue();
  auto s = clblast::Min<float2>(n,
                                imin_buffer(), 0,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  imin_buffer.Read(queue, imin_size, reinterpret_cast<float2*>(imin));
}
void cblas_izmin(const int n,
                void* imin,
                const void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto x_size = n;
  const auto imin_size = 1;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto imin_buffer = clblast::Buffer<double2>(context, imin_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  imin_buffer.Write(queue, imin_size, reinterpret_cast<double2*>(imin));
  auto queue_cl = queue();
  auto s = clblast::Min<double2>(n,
                                 imin_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  imin_buffer.Read(queue, imin_size, reinterpret_cast<double2*>(imin));
}

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// GEMV
void cblas_sgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                 const int m, const int n,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float* x, const int x_inc,
                 const float beta,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  const auto x_size = (a_transpose != CLBlastTransposeNo) ? m * x_inc : n * x_inc;
  const auto y_size = (a_transpose != CLBlastTransposeNo) ? n * y_inc : m * y_inc;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = clblast::Gemv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Transpose>(a_transpose),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                 const int m, const int n,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double* x, const int x_inc,
                 const double beta,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  const auto x_size = (a_transpose != CLBlastTransposeNo) ? m * x_inc : n * x_inc;
  const auto y_size = (a_transpose != CLBlastTransposeNo) ? n * y_inc : m * y_inc;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = clblast::Gemv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Transpose>(a_transpose),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}
void cblas_cgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  const auto x_size = (a_transpose != CLBlastTransposeNo) ? m * x_inc : n * x_inc;
  const auto y_size = (a_transpose != CLBlastTransposeNo) ? n * y_inc : m * y_inc;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Gemv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Transpose>(a_transpose),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  const auto x_size = (a_transpose != CLBlastTransposeNo) ? m * x_inc : n * x_inc;
  const auto y_size = (a_transpose != CLBlastTransposeNo) ? n * y_inc : m * y_inc;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Gemv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Transpose>(a_transpose),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// GBMV
void cblas_sgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                 const int m, const int n, const int kl, const int ku,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float* x, const int x_inc,
                 const float beta,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  const auto x_size = (a_transpose != CLBlastTransposeNo) ? m * x_inc : n * x_inc;
  const auto y_size = (a_transpose != CLBlastTransposeNo) ? n * y_inc : m * y_inc;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = clblast::Gbmv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Transpose>(a_transpose),
                         m, n, kl, ku,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                 const int m, const int n, const int kl, const int ku,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double* x, const int x_inc,
                 const double beta,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  const auto x_size = (a_transpose != CLBlastTransposeNo) ? m * x_inc : n * x_inc;
  const auto y_size = (a_transpose != CLBlastTransposeNo) ? n * y_inc : m * y_inc;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = clblast::Gbmv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Transpose>(a_transpose),
                         m, n, kl, ku,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}
void cblas_cgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                 const int m, const int n, const int kl, const int ku,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  const auto x_size = (a_transpose != CLBlastTransposeNo) ? m * x_inc : n * x_inc;
  const auto y_size = (a_transpose != CLBlastTransposeNo) ? n * y_inc : m * y_inc;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Gbmv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Transpose>(a_transpose),
                         m, n, kl, ku,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                 const int m, const int n, const int kl, const int ku,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  const auto x_size = (a_transpose != CLBlastTransposeNo) ? m * x_inc : n * x_inc;
  const auto y_size = (a_transpose != CLBlastTransposeNo) ? n * y_inc : m * y_inc;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Gbmv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Transpose>(a_transpose),
                         m, n, kl, ku,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// HEMV
void cblas_chemv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Hemv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zhemv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Hemv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// HBMV
void cblas_chbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Hbmv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zhbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Hbmv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// HPMV
void cblas_chpmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const void* alpha,
                 const void* ap,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto ap_size = ((n*(n+1)) / 2);
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  auto ap_buffer = clblast::Buffer<float2>(context, ap_size);
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const float2*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Hpmv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         ap_buffer(), 0,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float2*>(y));
}
void cblas_zhpmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const void* alpha,
                 const void* ap,
                 const void* x, const int x_inc,
                 const void* beta,
                 void* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto ap_size = ((n*(n+1)) / 2);
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  auto ap_buffer = clblast::Buffer<double2>(context, ap_size);
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const double2*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double2*>(y));
  auto queue_cl = queue();
  auto s = clblast::Hpmv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         ap_buffer(), 0,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double2*>(y));
}

// SYMV
void cblas_ssymv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float* x, const int x_inc,
                 const float beta,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = clblast::Symv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dsymv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double* x, const int x_inc,
                 const double beta,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = clblast::Symv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}

// SBMV
void cblas_ssbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n, const int k,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float* x, const int x_inc,
                 const float beta,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = clblast::Sbmv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n, const int k,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double* x, const int x_inc,
                 const double beta,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = clblast::Sbmv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}

// SPMV
void cblas_sspmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const float alpha,
                 const float* ap,
                 const float* x, const int x_inc,
                 const float beta,
                 float* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto ap_size = ((n*(n+1)) / 2);
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  auto ap_buffer = clblast::Buffer<float>(context, ap_size);
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const float*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<float*>(y));
  auto queue_cl = queue();
  auto s = clblast::Spmv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         ap_buffer(), 0,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<float*>(y));
}
void cblas_dspmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const double alpha,
                 const double* ap,
                 const double* x, const int x_inc,
                 const double beta,
                 double* y, const int y_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto ap_size = ((n*(n+1)) / 2);
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  auto ap_buffer = clblast::Buffer<double>(context, ap_size);
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const double*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<double*>(y));
  auto queue_cl = queue();
  auto s = clblast::Spmv(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         ap_buffer(), 0,
                         x_buffer(), 0, x_inc,
                         beta_cpp,
                         y_buffer(), 0, y_inc,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  y_buffer.Read(queue, y_size, reinterpret_cast<double*>(y));
}

// TRMV
void cblas_strmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const float* a, const int a_ld,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = clblast::Trmv<float>(static_cast<clblast::Layout>(layout),
                                static_cast<clblast::Triangle>(triangle),
                                static_cast<clblast::Transpose>(a_transpose),
                                static_cast<clblast::Diagonal>(diagonal),
                                n,
                                a_buffer(), 0, a_ld,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const double* a, const int a_ld,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = clblast::Trmv<double>(static_cast<clblast::Layout>(layout),
                                 static_cast<clblast::Triangle>(triangle),
                                 static_cast<clblast::Transpose>(a_transpose),
                                 static_cast<clblast::Diagonal>(diagonal),
                                 n,
                                 a_buffer(), 0, a_ld,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_ctrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Trmv<float2>(static_cast<clblast::Layout>(layout),
                                 static_cast<clblast::Triangle>(triangle),
                                 static_cast<clblast::Transpose>(a_transpose),
                                 static_cast<clblast::Diagonal>(diagonal),
                                 n,
                                 a_buffer(), 0, a_ld,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_ztrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Trmv<double2>(static_cast<clblast::Layout>(layout),
                                  static_cast<clblast::Triangle>(triangle),
                                  static_cast<clblast::Transpose>(a_transpose),
                                  static_cast<clblast::Diagonal>(diagonal),
                                  n,
                                  a_buffer(), 0, a_ld,
                                  x_buffer(), 0, x_inc,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// TBMV
void cblas_stbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n, const int k,
                 const float* a, const int a_ld,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tbmv<float>(static_cast<clblast::Layout>(layout),
                                static_cast<clblast::Triangle>(triangle),
                                static_cast<clblast::Transpose>(a_transpose),
                                static_cast<clblast::Diagonal>(diagonal),
                                n, k,
                                a_buffer(), 0, a_ld,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n, const int k,
                 const double* a, const int a_ld,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tbmv<double>(static_cast<clblast::Layout>(layout),
                                 static_cast<clblast::Triangle>(triangle),
                                 static_cast<clblast::Transpose>(a_transpose),
                                 static_cast<clblast::Diagonal>(diagonal),
                                 n, k,
                                 a_buffer(), 0, a_ld,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_ctbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n, const int k,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tbmv<float2>(static_cast<clblast::Layout>(layout),
                                 static_cast<clblast::Triangle>(triangle),
                                 static_cast<clblast::Transpose>(a_transpose),
                                 static_cast<clblast::Diagonal>(diagonal),
                                 n, k,
                                 a_buffer(), 0, a_ld,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_ztbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n, const int k,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tbmv<double2>(static_cast<clblast::Layout>(layout),
                                  static_cast<clblast::Triangle>(triangle),
                                  static_cast<clblast::Transpose>(a_transpose),
                                  static_cast<clblast::Diagonal>(diagonal),
                                  n, k,
                                  a_buffer(), 0, a_ld,
                                  x_buffer(), 0, x_inc,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// TPMV
void cblas_stpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const float* ap,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  const auto x_size = n * x_inc;
  auto ap_buffer = clblast::Buffer<float>(context, ap_size);
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const float*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tpmv<float>(static_cast<clblast::Layout>(layout),
                                static_cast<clblast::Triangle>(triangle),
                                static_cast<clblast::Transpose>(a_transpose),
                                static_cast<clblast::Diagonal>(diagonal),
                                n,
                                ap_buffer(), 0,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const double* ap,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  const auto x_size = n * x_inc;
  auto ap_buffer = clblast::Buffer<double>(context, ap_size);
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const double*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tpmv<double>(static_cast<clblast::Layout>(layout),
                                 static_cast<clblast::Triangle>(triangle),
                                 static_cast<clblast::Transpose>(a_transpose),
                                 static_cast<clblast::Diagonal>(diagonal),
                                 n,
                                 ap_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_ctpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const void* ap,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  const auto x_size = n * x_inc;
  auto ap_buffer = clblast::Buffer<float2>(context, ap_size);
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const float2*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tpmv<float2>(static_cast<clblast::Layout>(layout),
                                 static_cast<clblast::Triangle>(triangle),
                                 static_cast<clblast::Transpose>(a_transpose),
                                 static_cast<clblast::Diagonal>(diagonal),
                                 n,
                                 ap_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_ztpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const void* ap,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  const auto x_size = n * x_inc;
  auto ap_buffer = clblast::Buffer<double2>(context, ap_size);
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const double2*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tpmv<double2>(static_cast<clblast::Layout>(layout),
                                  static_cast<clblast::Triangle>(triangle),
                                  static_cast<clblast::Transpose>(a_transpose),
                                  static_cast<clblast::Diagonal>(diagonal),
                                  n,
                                  ap_buffer(), 0,
                                  x_buffer(), 0, x_inc,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// TRSV
void cblas_strsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const float* a, const int a_ld,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = clblast::Trsv<float>(static_cast<clblast::Layout>(layout),
                                static_cast<clblast::Triangle>(triangle),
                                static_cast<clblast::Transpose>(a_transpose),
                                static_cast<clblast::Diagonal>(diagonal),
                                n,
                                a_buffer(), 0, a_ld,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const double* a, const int a_ld,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = clblast::Trsv<double>(static_cast<clblast::Layout>(layout),
                                 static_cast<clblast::Triangle>(triangle),
                                 static_cast<clblast::Transpose>(a_transpose),
                                 static_cast<clblast::Diagonal>(diagonal),
                                 n,
                                 a_buffer(), 0, a_ld,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_ctrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Trsv<float2>(static_cast<clblast::Layout>(layout),
                                 static_cast<clblast::Triangle>(triangle),
                                 static_cast<clblast::Transpose>(a_transpose),
                                 static_cast<clblast::Diagonal>(diagonal),
                                 n,
                                 a_buffer(), 0, a_ld,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_ztrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Trsv<double2>(static_cast<clblast::Layout>(layout),
                                  static_cast<clblast::Triangle>(triangle),
                                  static_cast<clblast::Transpose>(a_transpose),
                                  static_cast<clblast::Diagonal>(diagonal),
                                  n,
                                  a_buffer(), 0, a_ld,
                                  x_buffer(), 0, x_inc,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// TBSV
void cblas_stbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n, const int k,
                 const float* a, const int a_ld,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tbsv<float>(static_cast<clblast::Layout>(layout),
                                static_cast<clblast::Triangle>(triangle),
                                static_cast<clblast::Transpose>(a_transpose),
                                static_cast<clblast::Diagonal>(diagonal),
                                n, k,
                                a_buffer(), 0, a_ld,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dtbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n, const int k,
                 const double* a, const int a_ld,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tbsv<double>(static_cast<clblast::Layout>(layout),
                                 static_cast<clblast::Triangle>(triangle),
                                 static_cast<clblast::Transpose>(a_transpose),
                                 static_cast<clblast::Diagonal>(diagonal),
                                 n, k,
                                 a_buffer(), 0, a_ld,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_ctbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n, const int k,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tbsv<float2>(static_cast<clblast::Layout>(layout),
                                 static_cast<clblast::Triangle>(triangle),
                                 static_cast<clblast::Transpose>(a_transpose),
                                 static_cast<clblast::Diagonal>(diagonal),
                                 n, k,
                                 a_buffer(), 0, a_ld,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_ztbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n, const int k,
                 const void* a, const int a_ld,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto a_size = n * a_ld;
  const auto x_size = n * x_inc;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tbsv<double2>(static_cast<clblast::Layout>(layout),
                                  static_cast<clblast::Triangle>(triangle),
                                  static_cast<clblast::Transpose>(a_transpose),
                                  static_cast<clblast::Diagonal>(diagonal),
                                  n, k,
                                  a_buffer(), 0, a_ld,
                                  x_buffer(), 0, x_inc,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// TPSV
void cblas_stpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const float* ap,
                 float* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  const auto x_size = n * x_inc;
  auto ap_buffer = clblast::Buffer<float>(context, ap_size);
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const float*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<float*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tpsv<float>(static_cast<clblast::Layout>(layout),
                                static_cast<clblast::Triangle>(triangle),
                                static_cast<clblast::Transpose>(a_transpose),
                                static_cast<clblast::Diagonal>(diagonal),
                                n,
                                ap_buffer(), 0,
                                x_buffer(), 0, x_inc,
                                &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float*>(x));
}
void cblas_dtpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const double* ap,
                 double* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  const auto x_size = n * x_inc;
  auto ap_buffer = clblast::Buffer<double>(context, ap_size);
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const double*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<double*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tpsv<double>(static_cast<clblast::Layout>(layout),
                                 static_cast<clblast::Triangle>(triangle),
                                 static_cast<clblast::Transpose>(a_transpose),
                                 static_cast<clblast::Diagonal>(diagonal),
                                 n,
                                 ap_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double*>(x));
}
void cblas_ctpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const void* ap,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  const auto x_size = n * x_inc;
  auto ap_buffer = clblast::Buffer<float2>(context, ap_size);
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const float2*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<float2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tpsv<float2>(static_cast<clblast::Layout>(layout),
                                 static_cast<clblast::Triangle>(triangle),
                                 static_cast<clblast::Transpose>(a_transpose),
                                 static_cast<clblast::Diagonal>(diagonal),
                                 n,
                                 ap_buffer(), 0,
                                 x_buffer(), 0, x_inc,
                                 &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<float2*>(x));
}
void cblas_ztpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int n,
                 const void* ap,
                 void* x, const int x_inc) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto ap_size = ((n*(n+1)) / 2);
  const auto x_size = n * x_inc;
  auto ap_buffer = clblast::Buffer<double2>(context, ap_size);
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  ap_buffer.Write(queue, ap_size, reinterpret_cast<const double2*>(ap));
  x_buffer.Write(queue, x_size, reinterpret_cast<double2*>(x));
  auto queue_cl = queue();
  auto s = clblast::Tpsv<double2>(static_cast<clblast::Layout>(layout),
                                  static_cast<clblast::Triangle>(triangle),
                                  static_cast<clblast::Transpose>(a_transpose),
                                  static_cast<clblast::Diagonal>(diagonal),
                                  n,
                                  ap_buffer(), 0,
                                  x_buffer(), 0, x_inc,
                                  &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  x_buffer.Read(queue, x_size, reinterpret_cast<double2*>(x));
}

// GER
void cblas_sger(const CLBlastLayout layout,
                const int m, const int n,
                const float alpha,
                const float* x, const int x_inc,
                const float* y, const int y_inc,
                float* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = m * x_inc;
  const auto y_size = n * y_inc;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<float*>(a));
  auto queue_cl = queue();
  auto s = clblast::Ger(static_cast<clblast::Layout>(layout),
                        m, n,
                        alpha_cpp,
                        x_buffer(), 0, x_inc,
                        y_buffer(), 0, y_inc,
                        a_buffer(), 0, a_ld,
                        &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float*>(a));
}
void cblas_dger(const CLBlastLayout layout,
                const int m, const int n,
                const double alpha,
                const double* x, const int x_inc,
                const double* y, const int y_inc,
                double* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = m * x_inc;
  const auto y_size = n * y_inc;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<double*>(a));
  auto queue_cl = queue();
  auto s = clblast::Ger(static_cast<clblast::Layout>(layout),
                        m, n,
                        alpha_cpp,
                        x_buffer(), 0, x_inc,
                        y_buffer(), 0, y_inc,
                        a_buffer(), 0, a_ld,
                        &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double*>(a));
}

// GERU
void cblas_cgeru(const CLBlastLayout layout,
                 const int m, const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto x_size = m * x_inc;
  const auto y_size = n * y_inc;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float2*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<float2*>(a));
  auto queue_cl = queue();
  auto s = clblast::Geru(static_cast<clblast::Layout>(layout),
                         m, n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         a_buffer(), 0, a_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float2*>(a));
}
void cblas_zgeru(const CLBlastLayout layout,
                 const int m, const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto x_size = m * x_inc;
  const auto y_size = n * y_inc;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double2*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<double2*>(a));
  auto queue_cl = queue();
  auto s = clblast::Geru(static_cast<clblast::Layout>(layout),
                         m, n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         a_buffer(), 0, a_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double2*>(a));
}

// GERC
void cblas_cgerc(const CLBlastLayout layout,
                 const int m, const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto x_size = m * x_inc;
  const auto y_size = n * y_inc;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float2*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<float2*>(a));
  auto queue_cl = queue();
  auto s = clblast::Gerc(static_cast<clblast::Layout>(layout),
                         m, n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         a_buffer(), 0, a_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float2*>(a));
}
void cblas_zgerc(const CLBlastLayout layout,
                 const int m, const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto x_size = m * x_inc;
  const auto y_size = n * y_inc;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double2*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<double2*>(a));
  auto queue_cl = queue();
  auto s = clblast::Gerc(static_cast<clblast::Layout>(layout),
                         m, n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         a_buffer(), 0, a_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double2*>(a));
}

// HER
void cblas_cher(const CLBlastLayout layout, const CLBlastTriangle triangle,
                const int n,
                const float alpha,
                const void* x, const int x_inc,
                void* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  const auto a_size = n * a_ld;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  a_buffer.Write(queue, a_size, reinterpret_cast<float2*>(a));
  auto queue_cl = queue();
  auto s = clblast::Her(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        n,
                        alpha_cpp,
                        x_buffer(), 0, x_inc,
                        a_buffer(), 0, a_ld,
                        &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float2*>(a));
}
void cblas_zher(const CLBlastLayout layout, const CLBlastTriangle triangle,
                const int n,
                const double alpha,
                const void* x, const int x_inc,
                void* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  const auto a_size = n * a_ld;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  a_buffer.Write(queue, a_size, reinterpret_cast<double2*>(a));
  auto queue_cl = queue();
  auto s = clblast::Her(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        n,
                        alpha_cpp,
                        x_buffer(), 0, x_inc,
                        a_buffer(), 0, a_ld,
                        &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double2*>(a));
}

// HPR
void cblas_chpr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                const int n,
                const float alpha,
                const void* x, const int x_inc,
                void* ap) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  const auto ap_size = ((n*(n+1)) / 2);
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto ap_buffer = clblast::Buffer<float2>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<float2*>(ap));
  auto queue_cl = queue();
  auto s = clblast::Hpr(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        n,
                        alpha_cpp,
                        x_buffer(), 0, x_inc,
                        ap_buffer(), 0,
                        &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<float2*>(ap));
}
void cblas_zhpr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                const int n,
                const double alpha,
                const void* x, const int x_inc,
                void* ap) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  const auto ap_size = ((n*(n+1)) / 2);
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto ap_buffer = clblast::Buffer<double2>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<double2*>(ap));
  auto queue_cl = queue();
  auto s = clblast::Hpr(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        n,
                        alpha_cpp,
                        x_buffer(), 0, x_inc,
                        ap_buffer(), 0,
                        &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<double2*>(ap));
}

// HER2
void cblas_cher2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  const auto a_size = n * a_ld;
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float2*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<float2*>(a));
  auto queue_cl = queue();
  auto s = clblast::Her2(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         a_buffer(), 0, a_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float2*>(a));
}
void cblas_zher2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  const auto a_size = n * a_ld;
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double2*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<double2*>(a));
  auto queue_cl = queue();
  auto s = clblast::Her2(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         a_buffer(), 0, a_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double2*>(a));
}

// HPR2
void cblas_chpr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* ap) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  const auto ap_size = ((n*(n+1)) / 2);
  auto x_buffer = clblast::Buffer<float2>(context, x_size);
  auto y_buffer = clblast::Buffer<float2>(context, y_size);
  auto ap_buffer = clblast::Buffer<float2>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float2*>(y));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<float2*>(ap));
  auto queue_cl = queue();
  auto s = clblast::Hpr2(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         ap_buffer(), 0,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<float2*>(ap));
}
void cblas_zhpr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const void* alpha,
                 const void* x, const int x_inc,
                 const void* y, const int y_inc,
                 void* ap) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  const auto ap_size = ((n*(n+1)) / 2);
  auto x_buffer = clblast::Buffer<double2>(context, x_size);
  auto y_buffer = clblast::Buffer<double2>(context, y_size);
  auto ap_buffer = clblast::Buffer<double2>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double2*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double2*>(y));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<double2*>(ap));
  auto queue_cl = queue();
  auto s = clblast::Hpr2(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         ap_buffer(), 0,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<double2*>(ap));
}

// SYR
void cblas_ssyr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                const int n,
                const float alpha,
                const float* x, const int x_inc,
                float* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  const auto a_size = n * a_ld;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  a_buffer.Write(queue, a_size, reinterpret_cast<float*>(a));
  auto queue_cl = queue();
  auto s = clblast::Syr(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        n,
                        alpha_cpp,
                        x_buffer(), 0, x_inc,
                        a_buffer(), 0, a_ld,
                        &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float*>(a));
}
void cblas_dsyr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                const int n,
                const double alpha,
                const double* x, const int x_inc,
                double* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  const auto a_size = n * a_ld;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  a_buffer.Write(queue, a_size, reinterpret_cast<double*>(a));
  auto queue_cl = queue();
  auto s = clblast::Syr(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        n,
                        alpha_cpp,
                        x_buffer(), 0, x_inc,
                        a_buffer(), 0, a_ld,
                        &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double*>(a));
}

// SPR
void cblas_sspr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                const int n,
                const float alpha,
                const float* x, const int x_inc,
                float* ap) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  const auto ap_size = ((n*(n+1)) / 2);
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto ap_buffer = clblast::Buffer<float>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<float*>(ap));
  auto queue_cl = queue();
  auto s = clblast::Spr(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        n,
                        alpha_cpp,
                        x_buffer(), 0, x_inc,
                        ap_buffer(), 0,
                        &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<float*>(ap));
}
void cblas_dspr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                const int n,
                const double alpha,
                const double* x, const int x_inc,
                double* ap) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  const auto ap_size = ((n*(n+1)) / 2);
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto ap_buffer = clblast::Buffer<double>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<double*>(ap));
  auto queue_cl = queue();
  auto s = clblast::Spr(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Triangle>(triangle),
                        n,
                        alpha_cpp,
                        x_buffer(), 0, x_inc,
                        ap_buffer(), 0,
                        &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<double*>(ap));
}

// SYR2
void cblas_ssyr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const float alpha,
                 const float* x, const int x_inc,
                 const float* y, const int y_inc,
                 float* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  const auto a_size = n * a_ld;
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<float*>(a));
  auto queue_cl = queue();
  auto s = clblast::Syr2(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         a_buffer(), 0, a_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<float*>(a));
}
void cblas_dsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const double alpha,
                 const double* x, const int x_inc,
                 const double* y, const int y_inc,
                 double* a, const int a_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  const auto a_size = n * a_ld;
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double*>(y));
  a_buffer.Write(queue, a_size, reinterpret_cast<double*>(a));
  auto queue_cl = queue();
  auto s = clblast::Syr2(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         a_buffer(), 0, a_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  a_buffer.Read(queue, a_size, reinterpret_cast<double*>(a));
}

// SPR2
void cblas_sspr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const float alpha,
                 const float* x, const int x_inc,
                 const float* y, const int y_inc,
                 float* ap) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  const auto ap_size = ((n*(n+1)) / 2);
  auto x_buffer = clblast::Buffer<float>(context, x_size);
  auto y_buffer = clblast::Buffer<float>(context, y_size);
  auto ap_buffer = clblast::Buffer<float>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const float*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const float*>(y));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<float*>(ap));
  auto queue_cl = queue();
  auto s = clblast::Spr2(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         ap_buffer(), 0,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<float*>(ap));
}
void cblas_dspr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                 const int n,
                 const double alpha,
                 const double* x, const int x_inc,
                 const double* y, const int y_inc,
                 double* ap) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto x_size = n * x_inc;
  const auto y_size = n * y_inc;
  const auto ap_size = ((n*(n+1)) / 2);
  auto x_buffer = clblast::Buffer<double>(context, x_size);
  auto y_buffer = clblast::Buffer<double>(context, y_size);
  auto ap_buffer = clblast::Buffer<double>(context, ap_size);
  x_buffer.Write(queue, x_size, reinterpret_cast<const double*>(x));
  y_buffer.Write(queue, y_size, reinterpret_cast<const double*>(y));
  ap_buffer.Write(queue, ap_size, reinterpret_cast<double*>(ap));
  auto queue_cl = queue();
  auto s = clblast::Spr2(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         n,
                         alpha_cpp,
                         x_buffer(), 0, x_inc,
                         y_buffer(), 0, y_inc,
                         ap_buffer(), 0,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  ap_buffer.Read(queue, ap_size, reinterpret_cast<double*>(ap));
}

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// GEMM
void cblas_sgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                 const int m, const int n, const int k,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float* b, const int b_ld,
                 const float beta,
                 float* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? m * a_ld : k * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && b_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && b_transpose == CLBlastTransposeNo)) ? k * b_ld : n * b_ld;
  const auto c_size = (layout == CLBlastLayoutRowMajor) ? m * c_ld : n * c_ld;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto b_buffer = clblast::Buffer<float>(context, b_size);
  auto c_buffer = clblast::Buffer<float>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float*>(c));
  auto queue_cl = queue();
  auto s = clblast::Gemm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Transpose>(b_transpose),
                         m, n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float*>(c));
}
void cblas_dgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                 const int m, const int n, const int k,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double* b, const int b_ld,
                 const double beta,
                 double* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? m * a_ld : k * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && b_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && b_transpose == CLBlastTransposeNo)) ? k * b_ld : n * b_ld;
  const auto c_size = (layout == CLBlastLayoutRowMajor) ? m * c_ld : n * c_ld;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto b_buffer = clblast::Buffer<double>(context, b_size);
  auto c_buffer = clblast::Buffer<double>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double*>(c));
  auto queue_cl = queue();
  auto s = clblast::Gemm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Transpose>(b_transpose),
                         m, n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double*>(c));
}
void cblas_cgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                 const int m, const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* b, const int b_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? m * a_ld : k * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && b_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && b_transpose == CLBlastTransposeNo)) ? k * b_ld : n * b_ld;
  const auto c_size = (layout == CLBlastLayoutRowMajor) ? m * c_ld : n * c_ld;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto b_buffer = clblast::Buffer<float2>(context, b_size);
  auto c_buffer = clblast::Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Gemm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Transpose>(b_transpose),
                         m, n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                 const int m, const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* b, const int b_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? m * a_ld : k * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && b_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && b_transpose == CLBlastTransposeNo)) ? k * b_ld : n * b_ld;
  const auto c_size = (layout == CLBlastLayoutRowMajor) ? m * c_ld : n * c_ld;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto b_buffer = clblast::Buffer<double2>(context, b_size);
  auto c_buffer = clblast::Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Gemm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Transpose>(b_transpose),
                         m, n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// SYMM
void cblas_ssymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                 const int m, const int n,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float* b, const int b_ld,
                 const float beta,
                 float* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : ((side == CLBlastSideLeft) ? m : n) * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? ((side == CLBlastSideLeft) ? m : n) * b_ld : n * b_ld;
  const auto c_size = (layout == CLBlastLayoutRowMajor) ? m * c_ld : n * c_ld;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto b_buffer = clblast::Buffer<float>(context, b_size);
  auto c_buffer = clblast::Buffer<float>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float*>(c));
  auto queue_cl = queue();
  auto s = clblast::Symm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float*>(c));
}
void cblas_dsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                 const int m, const int n,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double* b, const int b_ld,
                 const double beta,
                 double* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : ((side == CLBlastSideLeft) ? m : n) * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? ((side == CLBlastSideLeft) ? m : n) * b_ld : n * b_ld;
  const auto c_size = (layout == CLBlastLayoutRowMajor) ? m * c_ld : n * c_ld;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto b_buffer = clblast::Buffer<double>(context, b_size);
  auto c_buffer = clblast::Buffer<double>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double*>(c));
  auto queue_cl = queue();
  auto s = clblast::Symm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double*>(c));
}
void cblas_csymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* b, const int b_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : ((side == CLBlastSideLeft) ? m : n) * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? ((side == CLBlastSideLeft) ? m : n) * b_ld : n * b_ld;
  const auto c_size = (layout == CLBlastLayoutRowMajor) ? m * c_ld : n * c_ld;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto b_buffer = clblast::Buffer<float2>(context, b_size);
  auto c_buffer = clblast::Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Symm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* b, const int b_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : ((side == CLBlastSideLeft) ? m : n) * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? ((side == CLBlastSideLeft) ? m : n) * b_ld : n * b_ld;
  const auto c_size = (layout == CLBlastLayoutRowMajor) ? m * c_ld : n * c_ld;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto b_buffer = clblast::Buffer<double2>(context, b_size);
  auto c_buffer = clblast::Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Symm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// HEMM
void cblas_chemm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* b, const int b_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : ((side == CLBlastSideLeft) ? m : n) * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? ((side == CLBlastSideLeft) ? m : n) * b_ld : n * b_ld;
  const auto c_size = (layout == CLBlastLayoutRowMajor) ? m * c_ld : n * c_ld;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto b_buffer = clblast::Buffer<float2>(context, b_size);
  auto c_buffer = clblast::Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Hemm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zhemm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* b, const int b_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : ((side == CLBlastSideLeft) ? m : n) * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? ((side == CLBlastSideLeft) ? m : n) * b_ld : n * b_ld;
  const auto c_size = (layout == CLBlastLayoutRowMajor) ? m * c_ld : n * c_ld;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto b_buffer = clblast::Buffer<double2>(context, b_size);
  auto c_buffer = clblast::Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Hemm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// SYRK
void cblas_ssyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                 const int n, const int k,
                 const float alpha,
                 const float* a, const int a_ld,
                 const float beta,
                 float* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? n * a_ld : k * a_ld;
  const auto c_size = n * c_ld;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto c_buffer = clblast::Buffer<float>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  c_buffer.Write(queue, c_size, reinterpret_cast<float*>(c));
  auto queue_cl = queue();
  auto s = clblast::Syrk(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float*>(c));
}
void cblas_dsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                 const int n, const int k,
                 const double alpha,
                 const double* a, const int a_ld,
                 const double beta,
                 double* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? n * a_ld : k * a_ld;
  const auto c_size = n * c_ld;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto c_buffer = clblast::Buffer<double>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  c_buffer.Write(queue, c_size, reinterpret_cast<double*>(c));
  auto queue_cl = queue();
  auto s = clblast::Syrk(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double*>(c));
}
void cblas_csyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                 const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? n * a_ld : k * a_ld;
  const auto c_size = n * c_ld;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto c_buffer = clblast::Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Syrk(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                 const int n, const int k,
                 const void* alpha,
                 const void* a, const int a_ld,
                 const void* beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? n * a_ld : k * a_ld;
  const auto c_size = n * c_ld;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto c_buffer = clblast::Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Syrk(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// HERK
void cblas_cherk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                 const int n, const int k,
                 const float alpha,
                 const void* a, const int a_ld,
                 const float beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? n * a_ld : k * a_ld;
  const auto c_size = n * c_ld;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto c_buffer = clblast::Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Herk(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zherk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                 const int n, const int k,
                 const double alpha,
                 const void* a, const int a_ld,
                 const double beta,
                 void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? n * a_ld : k * a_ld;
  const auto c_size = n * c_ld;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto c_buffer = clblast::Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Herk(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         n, k,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         beta_cpp,
                         c_buffer(), 0, c_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// SYR2K
void cblas_ssyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                  const int n, const int k,
                  const float alpha,
                  const float* a, const int a_ld,
                  const float* b, const int b_ld,
                  const float beta,
                  float* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == CLBlastLayoutColMajor && ab_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && ab_transpose == CLBlastTransposeNo)) ? n * a_ld : k * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && ab_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && ab_transpose == CLBlastTransposeNo)) ? n * b_ld : k * b_ld;
  const auto c_size = n * c_ld;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto b_buffer = clblast::Buffer<float>(context, b_size);
  auto c_buffer = clblast::Buffer<float>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float*>(c));
  auto queue_cl = queue();
  auto s = clblast::Syr2k(static_cast<clblast::Layout>(layout),
                          static_cast<clblast::Triangle>(triangle),
                          static_cast<clblast::Transpose>(ab_transpose),
                          n, k,
                          alpha_cpp,
                          a_buffer(), 0, a_ld,
                          b_buffer(), 0, b_ld,
                          beta_cpp,
                          c_buffer(), 0, c_ld,
                          &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float*>(c));
}
void cblas_dsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                  const int n, const int k,
                  const double alpha,
                  const double* a, const int a_ld,
                  const double* b, const int b_ld,
                  const double beta,
                  double* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto beta_cpp = beta;
  const auto a_size = ((layout == CLBlastLayoutColMajor && ab_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && ab_transpose == CLBlastTransposeNo)) ? n * a_ld : k * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && ab_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && ab_transpose == CLBlastTransposeNo)) ? n * b_ld : k * b_ld;
  const auto c_size = n * c_ld;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto b_buffer = clblast::Buffer<double>(context, b_size);
  auto c_buffer = clblast::Buffer<double>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double*>(c));
  auto queue_cl = queue();
  auto s = clblast::Syr2k(static_cast<clblast::Layout>(layout),
                          static_cast<clblast::Triangle>(triangle),
                          static_cast<clblast::Transpose>(ab_transpose),
                          n, k,
                          alpha_cpp,
                          a_buffer(), 0, a_ld,
                          b_buffer(), 0, b_ld,
                          beta_cpp,
                          c_buffer(), 0, c_ld,
                          &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double*>(c));
}
void cblas_csyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                  const int n, const int k,
                  const void* alpha,
                  const void* a, const int a_ld,
                  const void* b, const int b_ld,
                  const void* beta,
                  void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = float2{reinterpret_cast<const float*>(beta)[0], reinterpret_cast<const float*>(beta)[1]};
  const auto a_size = ((layout == CLBlastLayoutColMajor && ab_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && ab_transpose == CLBlastTransposeNo)) ? n * a_ld : k * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && ab_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && ab_transpose == CLBlastTransposeNo)) ? n * b_ld : k * b_ld;
  const auto c_size = n * c_ld;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto b_buffer = clblast::Buffer<float2>(context, b_size);
  auto c_buffer = clblast::Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Syr2k(static_cast<clblast::Layout>(layout),
                          static_cast<clblast::Triangle>(triangle),
                          static_cast<clblast::Transpose>(ab_transpose),
                          n, k,
                          alpha_cpp,
                          a_buffer(), 0, a_ld,
                          b_buffer(), 0, b_ld,
                          beta_cpp,
                          c_buffer(), 0, c_ld,
                          &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                  const int n, const int k,
                  const void* alpha,
                  const void* a, const int a_ld,
                  const void* b, const int b_ld,
                  const void* beta,
                  void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = double2{reinterpret_cast<const double*>(beta)[0], reinterpret_cast<const double*>(beta)[1]};
  const auto a_size = ((layout == CLBlastLayoutColMajor && ab_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && ab_transpose == CLBlastTransposeNo)) ? n * a_ld : k * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && ab_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && ab_transpose == CLBlastTransposeNo)) ? n * b_ld : k * b_ld;
  const auto c_size = n * c_ld;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto b_buffer = clblast::Buffer<double2>(context, b_size);
  auto c_buffer = clblast::Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Syr2k(static_cast<clblast::Layout>(layout),
                          static_cast<clblast::Triangle>(triangle),
                          static_cast<clblast::Transpose>(ab_transpose),
                          n, k,
                          alpha_cpp,
                          a_buffer(), 0, a_ld,
                          b_buffer(), 0, b_ld,
                          beta_cpp,
                          c_buffer(), 0, c_ld,
                          &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// HER2K
void cblas_cher2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                  const int n, const int k,
                  const void* alpha,
                  const void* a, const int a_ld,
                  const void* b, const int b_ld,
                  const float beta,
                  void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto beta_cpp = beta;
  const auto a_size = ((layout == CLBlastLayoutColMajor && ab_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && ab_transpose == CLBlastTransposeNo)) ? n * a_ld : k * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && ab_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && ab_transpose == CLBlastTransposeNo)) ? n * b_ld : k * b_ld;
  const auto c_size = n * c_ld;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto b_buffer = clblast::Buffer<float2>(context, b_size);
  auto c_buffer = clblast::Buffer<float2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const float2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<float2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Her2k(static_cast<clblast::Layout>(layout),
                          static_cast<clblast::Triangle>(triangle),
                          static_cast<clblast::Transpose>(ab_transpose),
                          n, k,
                          alpha_cpp,
                          a_buffer(), 0, a_ld,
                          b_buffer(), 0, b_ld,
                          beta_cpp,
                          c_buffer(), 0, c_ld,
                          &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<float2*>(c));
}
void cblas_zher2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                  const int n, const int k,
                  const void* alpha,
                  const void* a, const int a_ld,
                  const void* b, const int b_ld,
                  const double beta,
                  void* c, const int c_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto beta_cpp = beta;
  const auto a_size = ((layout == CLBlastLayoutColMajor && ab_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && ab_transpose == CLBlastTransposeNo)) ? n * a_ld : k * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && ab_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && ab_transpose == CLBlastTransposeNo)) ? n * b_ld : k * b_ld;
  const auto c_size = n * c_ld;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto b_buffer = clblast::Buffer<double2>(context, b_size);
  auto c_buffer = clblast::Buffer<double2>(context, c_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<const double2*>(b));
  c_buffer.Write(queue, c_size, reinterpret_cast<double2*>(c));
  auto queue_cl = queue();
  auto s = clblast::Her2k(static_cast<clblast::Layout>(layout),
                          static_cast<clblast::Triangle>(triangle),
                          static_cast<clblast::Transpose>(ab_transpose),
                          n, k,
                          alpha_cpp,
                          a_buffer(), 0, a_ld,
                          b_buffer(), 0, b_ld,
                          beta_cpp,
                          c_buffer(), 0, c_ld,
                          &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  c_buffer.Read(queue, c_size, reinterpret_cast<double2*>(c));
}

// TRMM
void cblas_strmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int m, const int n,
                 const float alpha,
                 const float* a, const int a_ld,
                 float* b, const int b_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto a_size = (side == CLBlastSideLeft) ? m * a_ld : n * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? m * b_ld : n * b_ld;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto b_buffer = clblast::Buffer<float>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<float*>(b));
  auto queue_cl = queue();
  auto s = clblast::Trmm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<float*>(b));
}
void cblas_dtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int m, const int n,
                 const double alpha,
                 const double* a, const int a_ld,
                 double* b, const int b_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto a_size = (side == CLBlastSideLeft) ? m * a_ld : n * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? m * b_ld : n * b_ld;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto b_buffer = clblast::Buffer<double>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<double*>(b));
  auto queue_cl = queue();
  auto s = clblast::Trmm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<double*>(b));
}
void cblas_ctrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 void* b, const int b_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto a_size = (side == CLBlastSideLeft) ? m * a_ld : n * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? m * b_ld : n * b_ld;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto b_buffer = clblast::Buffer<float2>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<float2*>(b));
  auto queue_cl = queue();
  auto s = clblast::Trmm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<float2*>(b));
}
void cblas_ztrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 void* b, const int b_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto a_size = (side == CLBlastSideLeft) ? m * a_ld : n * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? m * b_ld : n * b_ld;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto b_buffer = clblast::Buffer<double2>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<double2*>(b));
  auto queue_cl = queue();
  auto s = clblast::Trmm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<double2*>(b));
}

// TRSM
void cblas_strsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int m, const int n,
                 const float alpha,
                 const float* a, const int a_ld,
                 float* b, const int b_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto a_size = (side == CLBlastSideLeft) ? m * a_ld : n * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? m * b_ld : n * b_ld;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto b_buffer = clblast::Buffer<float>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<float*>(b));
  auto queue_cl = queue();
  auto s = clblast::Trsm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<float*>(b));
}
void cblas_dtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int m, const int n,
                 const double alpha,
                 const double* a, const int a_ld,
                 double* b, const int b_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto a_size = (side == CLBlastSideLeft) ? m * a_ld : n * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? m * b_ld : n * b_ld;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto b_buffer = clblast::Buffer<double>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<double*>(b));
  auto queue_cl = queue();
  auto s = clblast::Trsm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<double*>(b));
}
void cblas_ctrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 void* b, const int b_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto a_size = (side == CLBlastSideLeft) ? m * a_ld : n * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? m * b_ld : n * b_ld;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto b_buffer = clblast::Buffer<float2>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<float2*>(b));
  auto queue_cl = queue();
  auto s = clblast::Trsm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<float2*>(b));
}
void cblas_ztrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                 const int m, const int n,
                 const void* alpha,
                 const void* a, const int a_ld,
                 void* b, const int b_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto a_size = (side == CLBlastSideLeft) ? m * a_ld : n * a_ld;
  const auto b_size = (layout == CLBlastLayoutRowMajor) ? m * b_ld : n * b_ld;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto b_buffer = clblast::Buffer<double2>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<double2*>(b));
  auto queue_cl = queue();
  auto s = clblast::Trsm(static_cast<clblast::Layout>(layout),
                         static_cast<clblast::Side>(side),
                         static_cast<clblast::Triangle>(triangle),
                         static_cast<clblast::Transpose>(a_transpose),
                         static_cast<clblast::Diagonal>(diagonal),
                         m, n,
                         alpha_cpp,
                         a_buffer(), 0, a_ld,
                         b_buffer(), 0, b_ld,
                         &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<double2*>(b));
}

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

// OMATCOPY
void cblas_somatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                     const int m, const int n,
                     const float alpha,
                     const float* a, const int a_ld,
                     float* b, const int b_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? n * b_ld : m * b_ld;
  auto a_buffer = clblast::Buffer<float>(context, a_size);
  auto b_buffer = clblast::Buffer<float>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<float*>(b));
  auto queue_cl = queue();
  auto s = clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Transpose>(a_transpose),
                             m, n,
                             alpha_cpp,
                             a_buffer(), 0, a_ld,
                             b_buffer(), 0, b_ld,
                             &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<float*>(b));
}
void cblas_domatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                     const int m, const int n,
                     const double alpha,
                     const double* a, const int a_ld,
                     double* b, const int b_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = alpha;
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? n * b_ld : m * b_ld;
  auto a_buffer = clblast::Buffer<double>(context, a_size);
  auto b_buffer = clblast::Buffer<double>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<double*>(b));
  auto queue_cl = queue();
  auto s = clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Transpose>(a_transpose),
                             m, n,
                             alpha_cpp,
                             a_buffer(), 0, a_ld,
                             b_buffer(), 0, b_ld,
                             &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<double*>(b));
}
void cblas_comatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                     const int m, const int n,
                     const void* alpha,
                     const void* a, const int a_ld,
                     void* b, const int b_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = float2{reinterpret_cast<const float*>(alpha)[0], reinterpret_cast<const float*>(alpha)[1]};
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? n * b_ld : m * b_ld;
  auto a_buffer = clblast::Buffer<float2>(context, a_size);
  auto b_buffer = clblast::Buffer<float2>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const float2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<float2*>(b));
  auto queue_cl = queue();
  auto s = clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Transpose>(a_transpose),
                             m, n,
                             alpha_cpp,
                             a_buffer(), 0, a_ld,
                             b_buffer(), 0, b_ld,
                             &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<float2*>(b));
}
void cblas_zomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                     const int m, const int n,
                     const void* alpha,
                     const void* a, const int a_ld,
                     void* b, const int b_ld) {
  auto device = get_device();
  auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);
  const auto alpha_cpp = double2{reinterpret_cast<const double*>(alpha)[0], reinterpret_cast<const double*>(alpha)[1]};
  const auto a_size = (layout == CLBlastLayoutRowMajor) ? m * a_ld : n * a_ld;
  const auto b_size = ((layout == CLBlastLayoutColMajor && a_transpose != CLBlastTransposeNo) || (layout == CLBlastLayoutRowMajor && a_transpose == CLBlastTransposeNo)) ? n * b_ld : m * b_ld;
  auto a_buffer = clblast::Buffer<double2>(context, a_size);
  auto b_buffer = clblast::Buffer<double2>(context, b_size);
  a_buffer.Write(queue, a_size, reinterpret_cast<const double2*>(a));
  b_buffer.Write(queue, b_size, reinterpret_cast<double2*>(b));
  auto queue_cl = queue();
  auto s = clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Transpose>(a_transpose),
                             m, n,
                             alpha_cpp,
                             a_buffer(), 0, a_ld,
                             b_buffer(), 0, b_ld,
                             &queue_cl);
  if (s != clblast::StatusCode::kSuccess) {
    throw std::runtime_error("CLBlast returned with error code " + clblast::ToString(s));
  }
  b_buffer.Read(queue, b_size, reinterpret_cast<double2*>(b));
}

// =================================================================================================
