
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
namespace database {
// =================================================================================================

const Database::DatabaseEntry PadtransposeHalf = {
  "Padtranspose", Precision::kHalf, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",1} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",1} } },
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

const Database::DatabaseEntry PadtransposeSingle = {
  "Padtranspose", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
        { "Hawaii",                                          { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
        { "Oland",                                           { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
        { "Pitcairn",                                        { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
        { "Tahiti",                                          { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",2} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",8} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"PADTRA_PAD",0}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",8} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "Iris",                                            { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "Iris Pro",                                        { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
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
        { "GRID K520",                                       { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 1070",                                { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 480",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 670",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 680",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 750",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 750 Ti",                              { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 980",                                 { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "GeForce GTX TITAN X",                             { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "Tesla K20m",                                      { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "Tesla K40m",                                      { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadtransposeComplexSingle = {
  "Padtranspose", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
        { "Hawaii",                                          { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "Oland",                                           { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "Pitcairn",                                        { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "Tahiti",                                          { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
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
        { "Intel(R) HD Graphics 530",                        { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
        { "Iris",                                            { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "Iris Pro",                                        { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
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
        { "GRID K520",                                       { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 1070",                                { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 480",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 670",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 680",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 750",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 750 Ti",                              { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 980",                                 { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN X",                             { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "Tesla K20m",                                      { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "Tesla K40m",                                      { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadtransposeDouble = {
  "Padtranspose", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
        { "Hawaii",                                          { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "Oland",                                           { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
        { "Pitcairn",                                        { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "Tahiti",                                          { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",4} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"PADTRA_PAD",1}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",8} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
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
        { "GRID K520",                                       { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 1070",                                { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 480",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 670",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 680",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 750",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 750 Ti",                              { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 980",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN X",                             { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "Tesla K20m",                                      { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "Tesla K40m",                                      { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadtransposeComplexDouble = {
  "Padtranspose", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "Hawaii",                                          { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "Oland",                                           { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "Pitcairn",                                        { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "Tahiti",                                          { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",2} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"PADTRA_PAD",1}, {"PADTRA_TILE",8}, {"PADTRA_WPT",2} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"PADTRA_PAD",1}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"PADTRA_PAD",1}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",8}, {"PADTRA_WPT",4} } },
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
        { "GRID K520",                                       { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 1070",                                { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 480",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 670",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 680",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 750",                                 { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX 750 Ti",                              { {"PADTRA_PAD",1}, {"PADTRA_TILE",8}, {"PADTRA_WPT",2} } },
        { "GeForce GTX 980",                                 { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN",                               { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "GeForce GTX TITAN X",                             { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",1} } },
        { "Tesla K20m",                                      { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "Tesla K40m",                                      { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "default",                                         { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"PADTRA_PAD",0}, {"PADTRA_TILE",8}, {"PADTRA_WPT",2} } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
