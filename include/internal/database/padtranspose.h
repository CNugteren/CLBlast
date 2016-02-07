
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Padtranspose' kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::PadtransposeSingle = {
  "Padtranspose", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Tahiti",                                          { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",8} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"PADTRA_PAD",0}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                            { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 680",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 750 Ti",                              { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 980",                                 { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "GeForce GTX TITAN X",                             { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "Tesla K20m",                                      { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "Tesla K40m",                                      { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::PadtransposeComplexSingle = {
  "Padtranspose", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Tahiti",                                          { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"PADTRA_PAD",1}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",8} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                            { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 680",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 750 Ti",                              { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 980",                                 { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN X",                             { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "Tesla K20m",                                      { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "Tesla K40m",                                      { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::PadtransposeDouble = {
  "Padtranspose", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Tahiti",                                          { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"PADTRA_PAD",1}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",8} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 680",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 750 Ti",                              { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 980",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN X",                             { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "Tesla K20m",                                      { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "Tesla K40m",                                      { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::PadtransposeComplexDouble = {
  "Padtranspose", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Tahiti",                                          { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",2} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"PADTRA_PAD",1}, {"PADTRA_TILE",8}, {"PADTRA_WPT",2} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"PADTRA_PAD",1}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"PADTRA_PAD",1}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",8}, {"PADTRA_WPT",2} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 680",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 750 Ti",                              { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 980",                                 { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN X",                             { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "Tesla K20m",                                      { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "Tesla K40m",                                      { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
