
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Transpose' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry TransposeHalf = {
  "Transpose", Precision::kHalf, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",8} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "default",                                         { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry TransposeSingle = {
  "Transpose", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",8} } },
        { "Hawaii",                                          { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",8} } },
        { "Oland",                                           { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
        { "Pitcairn",                                        { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "Tahiti",                                          { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
        { "Tonga",                                           { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "default",                                         { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
        { "default",                                         { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",8} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",8} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",8} } },
        { "default",                                         { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",8} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
        { "Iris",                                            { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "Iris Pro",                                        { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "default",                                         { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "default",                                         { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"TRA_DIM",32}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "GeForce GTX 1070",                                { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
        { "GeForce GTX 480",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "GeForce GTX 670",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "GeForce GTX 680",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "GeForce GTX 750",                                 { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",8} } },
        { "GeForce GTX 750 Ti",                              { {"TRA_DIM",32}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "GeForce GTX 980",                                 { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "GeForce GTX TITAN X",                             { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "Tesla K20m",                                      { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "Tesla K40m",                                      { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "default",                                         { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry TransposeComplexSingle = {
  "Transpose", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"TRA_DIM",4}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
        { "Hawaii",                                          { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "Oland",                                           { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "Pitcairn",                                        { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "Tahiti",                                          { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "Tonga",                                           { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "default",                                         { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "default",                                         { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"TRA_DIM",4}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",8} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "default",                                         { {"TRA_DIM",4}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",8} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "Iris",                                            { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "Iris Pro",                                        { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "default",                                         { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "GeForce GTX 1070",                                { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "GeForce GTX 480",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "GeForce GTX 670",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "GeForce GTX 680",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "GeForce GTX 750",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "GeForce GTX 750 Ti",                              { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "GeForce GTX 980",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "GeForce GTX TITAN X",                             { {"TRA_DIM",32}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "Tesla K20m",                                      { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "Tesla K40m",                                      { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "default",                                         { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry TransposeDouble = {
  "Transpose", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
        { "Hawaii",                                          { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "Oland",                                           { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "Pitcairn",                                        { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "Tahiti",                                          { {"TRA_DIM",4}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
        { "Tonga",                                           { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
        { "default",                                         { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",4} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "default",                                         { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"TRA_DIM",4}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",8} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",8} } },
        { "default",                                         { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",8} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"TRA_DIM",32}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "default",                                         { {"TRA_DIM",32}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "GeForce GTX 1070",                                { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "GeForce GTX 480",                                 { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "GeForce GTX 670",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "GeForce GTX 680",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "GeForce GTX 750",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "GeForce GTX 750 Ti",                              { {"TRA_DIM",32}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "GeForce GTX 980",                                 { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "GeForce GTX TITAN",                               { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "GeForce GTX TITAN X",                             { {"TRA_DIM",32}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "Tesla K20m",                                      { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "Tesla K40m",                                      { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "default",                                         { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry TransposeComplexDouble = {
  "Transpose", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"TRA_DIM",4}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "Hawaii",                                          { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "Oland",                                           { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "Pitcairn",                                        { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "Tahiti",                                          { {"TRA_DIM",16}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "Tonga",                                           { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "default",                                         { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "default",                                         { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"TRA_DIM",4}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "default",                                         { {"TRA_DIM",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "GeForce GTX 1070",                                { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "GeForce GTX 480",                                 { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "GeForce GTX 670",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "GeForce GTX 680",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
        { "GeForce GTX 750",                                 { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "GeForce GTX 750 Ti",                              { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "GeForce GTX 980",                                 { {"TRA_DIM",32}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "GeForce GTX TITAN X",                             { {"TRA_DIM",32}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "Tesla K20m",                                      { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "Tesla K40m",                                      { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "default",                                         { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
