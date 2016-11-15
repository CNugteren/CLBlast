
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Pad' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry PadHalf = {
  "Pad", Precision::kHalf, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"PAD_DIMX",8}, {"PAD_DIMY",32}, {"PAD_WPTX",2}, {"PAD_WPTY",2} } },
        { "default",                                         { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadSingle = {
  "Pad", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Hawaii",                                          { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",4} } },
        { "Oland",                                           { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "Pitcairn",                                        { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "Tahiti",                                          { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "Tonga",                                           { {"PAD_DIMX",16}, {"PAD_DIMY",16}, {"PAD_WPTX",2}, {"PAD_WPTY",2} } },
        { "default",                                         { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",4} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",4} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",4} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"PAD_DIMX",16}, {"PAD_DIMY",32}, {"PAD_WPTX",4}, {"PAD_WPTY",4} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",2} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",4} } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",4} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",2} } },
        { "Iris",                                            { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "Iris Pro",                                        { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",2} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "GeForce GTX 1070",                                { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 480",                                 { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",4} } },
        { "GeForce GTX 670",                                 { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",2} } },
        { "GeForce GTX 680",                                 { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "GeForce GTX 750",                                 { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",4}, {"PAD_WPTY",2} } },
        { "GeForce GTX 750 Ti",                              { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "GeForce GTX 980",                                 { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX TITAN",                               { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "GeForce GTX TITAN Black",                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "GeForce GTX TITAN X",                             { {"PAD_DIMX",16}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tesla K20m",                                      { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "Tesla K40m",                                      { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadComplexSingle = {
  "Pad", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Hawaii",                                          { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "Oland",                                           { {"PAD_DIMX",8}, {"PAD_DIMY",32}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Pitcairn",                                        { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "Tahiti",                                          { {"PAD_DIMX",16}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tonga",                                           { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",4} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",4} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",2} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"PAD_DIMX",32}, {"PAD_DIMY",32}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Iris",                                            { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",2}, {"PAD_WPTY",4} } },
        { "Iris Pro",                                        { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",4} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"PAD_DIMX",16}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 1070",                                { {"PAD_DIMX",8}, {"PAD_DIMY",32}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 480",                                 { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "GeForce GTX 670",                                 { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "GeForce GTX 680",                                 { {"PAD_DIMX",16}, {"PAD_DIMY",32}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "GeForce GTX 750",                                 { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "GeForce GTX 750 Ti",                              { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 980",                                 { {"PAD_DIMX",16}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX TITAN",                               { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "GeForce GTX TITAN Black",                         { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "GeForce GTX TITAN X",                             { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tesla K20m",                                      { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "Tesla K40m",                                      { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadDouble = {
  "Pad", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Hawaii",                                          { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "Oland",                                           { {"PAD_DIMX",8}, {"PAD_DIMY",32}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Pitcairn",                                        { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "Tahiti",                                          { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tonga",                                           { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",2} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",2} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"PAD_DIMX",32}, {"PAD_DIMY",32}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 1070",                                { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 480",                                 { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 670",                                 { {"PAD_DIMX",16}, {"PAD_DIMY",16}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "GeForce GTX 680",                                 { {"PAD_DIMX",32}, {"PAD_DIMY",32}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "GeForce GTX 750",                                 { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 750 Ti",                              { {"PAD_DIMX",8}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 980",                                 { {"PAD_DIMX",8}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX TITAN",                               { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX TITAN Black",                         { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX TITAN X",                             { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tesla K20m",                                      { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tesla K40m",                                      { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadComplexDouble = {
  "Pad", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Hawaii",                                          { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Oland",                                           { {"PAD_DIMX",8}, {"PAD_DIMY",16}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "Pitcairn",                                        { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tahiti",                                          { {"PAD_DIMX",8}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tonga",                                           { {"PAD_DIMX",8}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",8}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"PAD_DIMX",16}, {"PAD_DIMY",32}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",4}, {"PAD_WPTY",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 1070",                                { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",2}, {"PAD_WPTY",2} } },
        { "GeForce GTX 480",                                 { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 670",                                 { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 680",                                 { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 750",                                 { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 750 Ti",                              { {"PAD_DIMX",16}, {"PAD_DIMY",32}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX 980",                                 { {"PAD_DIMX",16}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "GeForce GTX TITAN",                               { {"PAD_DIMX",8}, {"PAD_DIMY",32}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "GeForce GTX TITAN Black",                         { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",4} } },
        { "GeForce GTX TITAN X",                             { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tesla K20m",                                      { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "Tesla K40m",                                      { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "default",                                         { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
