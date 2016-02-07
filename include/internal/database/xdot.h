
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
        { "Tahiti",                                          { {"VW",1}, {"WGS1",256}, {"WGS2",256} } },
        { "default",                                         { {"VW",1}, {"WGS1",256}, {"WGS2",256} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW",1}, {"WGS1",512}, {"WGS2",1024} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"VW",1}, {"WGS1",512}, {"WGS2",1024} } },
        { "default",                                         { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                            { {"VW",1}, {"WGS1",512}, {"WGS2",32} } },
        { "default",                                         { {"VW",1}, {"WGS1",512}, {"WGS2",32} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
        { "default",                                         { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"VW",1}, {"WGS1",256}, {"WGS2",128} } },
        { "GeForce GTX 680",                                 { {"VW",1}, {"WGS1",128}, {"WGS2",128} } },
        { "GeForce GTX 750 Ti",                              { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
        { "GeForce GTX 980",                                 { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
        { "GeForce GTX TITAN",                               { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
        { "GeForce GTX TITAN X",                             { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
        { "Tesla K20m",                                      { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
        { "Tesla K40m",                                      { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
        { "default",                                         { {"VW",1}, {"WGS1",64}, {"WGS2",128} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW",1}, {"WGS1",64}, {"WGS2",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XdotComplexSingle = {
  "Xdot", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Tahiti",                                          { {"VW",1}, {"WGS1",64}, {"WGS2",256} } },
        { "default",                                         { {"VW",1}, {"WGS1",64}, {"WGS2",256} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW",1}, {"WGS1",256}, {"WGS2",1024} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"VW",1}, {"WGS1",512}, {"WGS2",1024} } },
        { "default",                                         { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                            { {"VW",1}, {"WGS1",512}, {"WGS2",1024} } },
        { "default",                                         { {"VW",1}, {"WGS1",512}, {"WGS2",1024} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
        { "default",                                         { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"VW",1}, {"WGS1",512}, {"WGS2",512} } },
        { "GeForce GTX 680",                                 { {"VW",1}, {"WGS1",256}, {"WGS2",32} } },
        { "GeForce GTX 750 Ti",                              { {"VW",1}, {"WGS1",128}, {"WGS2",32} } },
        { "GeForce GTX 980",                                 { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
        { "GeForce GTX TITAN",                               { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
        { "GeForce GTX TITAN X",                             { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
        { "Tesla K20m",                                      { {"VW",1}, {"WGS1",256}, {"WGS2",512} } },
        { "Tesla K40m",                                      { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
        { "default",                                         { {"VW",1}, {"WGS1",64}, {"WGS2",32} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW",1}, {"WGS1",64}, {"WGS2",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XdotDouble = {
  "Xdot", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Tahiti",                                          { {"VW",1}, {"WGS1",64}, {"WGS2",256} } },
        { "default",                                         { {"VW",1}, {"WGS1",64}, {"WGS2",256} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW",1}, {"WGS1",512}, {"WGS2",512} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"VW",1}, {"WGS1",1024}, {"WGS2",512} } },
        { "default",                                         { {"VW",1}, {"WGS1",64}, {"WGS2",512} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW",1}, {"WGS1",512}, {"WGS2",1024} } },
        { "default",                                         { {"VW",1}, {"WGS1",512}, {"WGS2",1024} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"VW",1}, {"WGS1",512}, {"WGS2",1024} } },
        { "GeForce GTX 680",                                 { {"VW",1}, {"WGS1",64}, {"WGS2",128} } },
        { "GeForce GTX 750 Ti",                              { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
        { "GeForce GTX 980",                                 { {"VW",1}, {"WGS1",32}, {"WGS2",512} } },
        { "GeForce GTX TITAN",                               { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
        { "GeForce GTX TITAN X",                             { {"VW",1}, {"WGS1",128}, {"WGS2",128} } },
        { "Tesla K20m",                                      { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
        { "Tesla K40m",                                      { {"VW",1}, {"WGS1",256}, {"WGS2",1024} } },
        { "default",                                         { {"VW",1}, {"WGS1",32}, {"WGS2",128} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW",1}, {"WGS1",32}, {"WGS2",128} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XdotComplexDouble = {
  "Xdot", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Tahiti",                                          { {"VW",1}, {"WGS1",64}, {"WGS2",256} } },
        { "default",                                         { {"VW",1}, {"WGS1",64}, {"WGS2",256} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW",1}, {"WGS1",256}, {"WGS2",1024} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"VW",1}, {"WGS1",512}, {"WGS2",1024} } },
        { "default",                                         { {"VW",1}, {"WGS1",64}, {"WGS2",1024} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW",1}, {"WGS1",32}, {"WGS2",1024} } },
        { "default",                                         { {"VW",1}, {"WGS1",32}, {"WGS2",1024} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"VW",1}, {"WGS1",512}, {"WGS2",512} } },
        { "GeForce GTX 680",                                 { {"VW",1}, {"WGS1",256}, {"WGS2",64} } },
        { "GeForce GTX 750 Ti",                              { {"VW",1}, {"WGS1",32}, {"WGS2",64} } },
        { "GeForce GTX 980",                                 { {"VW",1}, {"WGS1",32}, {"WGS2",128} } },
        { "GeForce GTX TITAN",                               { {"VW",1}, {"WGS1",128}, {"WGS2",512} } },
        { "GeForce GTX TITAN X",                             { {"VW",1}, {"WGS1",128}, {"WGS2",128} } },
        { "Tesla K20m",                                      { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
        { "Tesla K40m",                                      { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
        { "default",                                         { {"VW",1}, {"WGS1",32}, {"WGS2",64} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW",1}, {"WGS1",32}, {"WGS2",64} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
