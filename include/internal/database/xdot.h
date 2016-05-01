
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xdot' kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::XdotSingle = {
  "Xdot", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",128}, {"WGS2",32} } },
        { "Tahiti",                                          { {"WGS1",128}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",32} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",1024}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",1024}, {"WGS2",32} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris Pro",                                        { {"WGS1",512}, {"WGS2",64} } },
        { "default",                                         { {"WGS1",512}, {"WGS2",64} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"WGS1",512}, {"WGS2",32} } },
        { "GeForce GTX 680",                                 { {"WGS1",128}, {"WGS2",128} } },
        { "GeForce GTX 980",                                 { {"WGS1",256}, {"WGS2",32} } },
        { "GeForce GTX TITAN X",                             { {"WGS1",256}, {"WGS2",32} } },
        { "Tesla K20m",                                      { {"WGS1",1024}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",32} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",128}, {"WGS2",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XdotComplexSingle = {
  "Xdot", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WGS2",32} } },
        { "Tahiti",                                          { {"WGS1",64}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",32} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",1024}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",1024}, {"WGS2",32} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris Pro",                                        { {"WGS1",32}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",32}, {"WGS2",32} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"WGS1",512}, {"WGS2",32} } },
        { "GeForce GTX 680",                                 { {"WGS1",128}, {"WGS2",64} } },
        { "GeForce GTX 980",                                 { {"WGS1",256}, {"WGS2",64} } },
        { "GeForce GTX TITAN X",                             { {"WGS1",256}, {"WGS2",32} } },
        { "Tesla K20m",                                      { {"WGS1",512}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",32} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",32}, {"WGS2",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XdotDouble = {
  "Xdot", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WGS2",128} } },
        { "Tahiti",                                          { {"WGS1",256}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",32} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",512}, {"WGS2",64} } },
        { "default",                                         { {"WGS1",512}, {"WGS2",64} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"WGS1",512}, {"WGS2",32} } },
        { "GeForce GTX 680",                                 { {"WGS1",128}, {"WGS2",64} } },
        { "GeForce GTX 980",                                 { {"WGS1",128}, {"WGS2",32} } },
        { "GeForce GTX TITAN X",                             { {"WGS1",256}, {"WGS2",32} } },
        { "Tesla K20m",                                      { {"WGS1",512}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",32} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",64}, {"WGS2",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XdotComplexDouble = {
  "Xdot", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WGS2",32} } },
        { "Tahiti",                                          { {"WGS1",256}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",32} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",1024}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",1024}, {"WGS2",32} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"WGS1",512}, {"WGS2",32} } },
        { "GeForce GTX 680",                                 { {"WGS1",256}, {"WGS2",64} } },
        { "GeForce GTX 980",                                 { {"WGS1",64}, {"WGS2",32} } },
        { "GeForce GTX TITAN X",                             { {"WGS1",128}, {"WGS2",32} } },
        { "Tesla K20m",                                      { {"WGS1",128}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",32} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",64}, {"WGS2",32} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
