
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Copy' kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::CopyHalf = {
  "Copy", Precision::kHalf, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"COPY_DIMX",8}, {"COPY_DIMY",16}, {"COPY_VW",8}, {"COPY_WPT",4} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",1} } },
        { "default",                                         { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::CopySingle = {
  "Copy", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",1} } },
        { "Hawaii",                                          { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",2} } },
        { "Oland",                                           { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",2} } },
        { "Pitcairn",                                        { {"COPY_DIMX",8}, {"COPY_DIMY",16}, {"COPY_VW",4}, {"COPY_WPT",1} } },
        { "Tahiti",                                          { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",2} } },
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",2} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",4} } },
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",4} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"COPY_DIMX",32}, {"COPY_DIMY",16}, {"COPY_VW",8}, {"COPY_WPT",2} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"COPY_DIMX",32}, {"COPY_DIMY",16}, {"COPY_VW",8}, {"COPY_WPT",1} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",1} } },
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",1} } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"COPY_DIMX",32}, {"COPY_DIMY",16}, {"COPY_VW",4}, {"COPY_WPT",1} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"COPY_DIMX",32}, {"COPY_DIMY",16}, {"COPY_VW",4}, {"COPY_WPT",1} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",1} } },
        { "Iris",                                            { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
        { "Iris Pro",                                        { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",4} } },
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",1} } },
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",1} } },
        { "GeForce GTX 1070",                                { {"COPY_DIMX",8}, {"COPY_DIMY",16}, {"COPY_VW",4}, {"COPY_WPT",1} } },
        { "GeForce GTX 480",                                 { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",1} } },
        { "GeForce GTX 670",                                 { {"COPY_DIMX",16}, {"COPY_DIMY",32}, {"COPY_VW",4}, {"COPY_WPT",1} } },
        { "GeForce GTX 680",                                 { {"COPY_DIMX",32}, {"COPY_DIMY",16}, {"COPY_VW",4}, {"COPY_WPT",1} } },
        { "GeForce GTX 750",                                 { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",2} } },
        { "GeForce GTX 750 Ti",                              { {"COPY_DIMX",16}, {"COPY_DIMY",32}, {"COPY_VW",2}, {"COPY_WPT",2} } },
        { "GeForce GTX 980",                                 { {"COPY_DIMX",32}, {"COPY_DIMY",16}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",4} } },
        { "GeForce GTX TITAN X",                             { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
        { "Tesla K20m",                                      { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",4} } },
        { "Tesla K40m",                                      { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",2} } },
        { "default",                                         { {"COPY_DIMX",8}, {"COPY_DIMY",16}, {"COPY_VW",4}, {"COPY_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::CopyComplexSingle = {
  "Copy", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "Hawaii",                                          { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
        { "Oland",                                           { {"COPY_DIMX",8}, {"COPY_DIMY",16}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "Pitcairn",                                        { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
        { "Tahiti",                                          { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",2} } },
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"COPY_DIMX",16}, {"COPY_DIMY",16}, {"COPY_VW",8}, {"COPY_WPT",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",2} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",1} } },
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",1} } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"COPY_DIMX",16}, {"COPY_DIMY",16}, {"COPY_VW",2}, {"COPY_WPT",2} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",4} } },
        { "Iris",                                            { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
        { "Iris Pro",                                        { {"COPY_DIMX",32}, {"COPY_DIMY",16}, {"COPY_VW",1}, {"COPY_WPT",4} } },
        { "default",                                         { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",1} } },
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX 1070",                                { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX 480",                                 { {"COPY_DIMX",16}, {"COPY_DIMY",16}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX 670",                                 { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX 750",                                 { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
        { "GeForce GTX 750 Ti",                              { {"COPY_DIMX",16}, {"COPY_DIMY",32}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX 980",                                 { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX TITAN X",                             { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "Tesla K20m",                                      { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",4} } },
        { "Tesla K40m",                                      { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "default",                                         { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::CopyDouble = {
  "Copy", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "Hawaii",                                          { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
        { "Oland",                                           { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",8} } },
        { "Pitcairn",                                        { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "Tahiti",                                          { {"COPY_DIMX",8}, {"COPY_DIMY",32}, {"COPY_VW",2}, {"COPY_WPT",1} } },
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",2} } },
        { "default",                                         { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",2} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"COPY_DIMX",16}, {"COPY_DIMY",32}, {"COPY_VW",2}, {"COPY_WPT",1} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"COPY_DIMX",16}, {"COPY_DIMY",16}, {"COPY_VW",8}, {"COPY_WPT",1} } },
        { "default",                                         { {"COPY_DIMX",16}, {"COPY_DIMY",16}, {"COPY_VW",8}, {"COPY_WPT",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",1} } },
        { "default",                                         { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"COPY_DIMX",32}, {"COPY_DIMY",16}, {"COPY_VW",2}, {"COPY_WPT",1} } },
        { "GeForce GTX 1070",                                { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",1} } },
        { "GeForce GTX 480",                                 { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",1} } },
        { "GeForce GTX 670",                                 { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",1} } },
        { "GeForce GTX 680",                                 { {"COPY_DIMX",16}, {"COPY_DIMY",32}, {"COPY_VW",2}, {"COPY_WPT",1} } },
        { "GeForce GTX 750",                                 { {"COPY_DIMX",8}, {"COPY_DIMY",16}, {"COPY_VW",2}, {"COPY_WPT",1} } },
        { "GeForce GTX 750 Ti",                              { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",1} } },
        { "GeForce GTX 980",                                 { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"COPY_DIMX",16}, {"COPY_DIMY",32}, {"COPY_VW",2}, {"COPY_WPT",2} } },
        { "GeForce GTX TITAN X",                             { {"COPY_DIMX",32}, {"COPY_DIMY",16}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "Tesla K20m",                                      { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",1} } },
        { "Tesla K40m",                                      { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",2} } },
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",16}, {"COPY_VW",2}, {"COPY_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::CopyComplexDouble = {
  "Copy", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"COPY_DIMX",8}, {"COPY_DIMY",16}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "Hawaii",                                          { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",8} } },
        { "Oland",                                           { {"COPY_DIMX",8}, {"COPY_DIMY",16}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "Pitcairn",                                        { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "Tahiti",                                          { {"COPY_DIMX",8}, {"COPY_DIMY",16}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "default",                                         { {"COPY_DIMX",8}, {"COPY_DIMY",16}, {"COPY_VW",1}, {"COPY_WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"COPY_DIMX",32}, {"COPY_DIMY",32}, {"COPY_VW",8}, {"COPY_WPT",1} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",1} } },
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",32}, {"COPY_VW",8}, {"COPY_WPT",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",1} } },
        { "default",                                         { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",8}, {"COPY_WPT",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX 1070",                                { {"COPY_DIMX",8}, {"COPY_DIMY",32}, {"COPY_VW",1}, {"COPY_WPT",4} } },
        { "GeForce GTX 480",                                 { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX 670",                                 { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX 680",                                 { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX 750",                                 { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX 750 Ti",                              { {"COPY_DIMX",16}, {"COPY_DIMY",16}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX 980",                                 { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"COPY_DIMX",16}, {"COPY_DIMY",16}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "GeForce GTX TITAN X",                             { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "Tesla K20m",                                      { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
        { "Tesla K40m",                                      { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "default",                                         { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
