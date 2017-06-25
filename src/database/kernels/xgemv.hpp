
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xgemv' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XgemvHalf = {
  "Xgemv", Precision::kHalf, {"WGS1", "WPT1"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { 256, 1 } },
        { "default",                                         { 256, 1 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 64, 1 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 256, 1 } },
        { "default",                                         { 64, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 64, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvSingle = {
  "Xgemv", Precision::kSingle, {"WGS1", "WPT1"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 128, 1 } },
        { "ATI Radeon HD 6750M",                             { 32, 1 } },
        { "Ellesmere",                                       { 256, 1 } },
        { "Fiji",                                            { 128, 1 } },
        { "Hawaii",                                          { 128, 1 } },
        { "Oland",                                           { 128, 1 } },
        { "Pitcairn",                                        { 256, 1 } },
        { "Tahiti",                                          { 256, 1 } },
        { "Tonga",                                           { 128, 2 } },
        { "Turks",                                           { 32, 1 } },
        { "default",                                         { 128, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 128, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 64, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 32, 4 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 64, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 64, 4 } },
        { "default",                                         { 64, 4 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 256, 1 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 64, 1 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 64, 1 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 256, 1 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 32, 1 } },
        { "Iris",                                            { 64, 2 } },
        { "Iris Pro",                                        { 128, 1 } },
        { "default",                                         { 128, 1 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 64, 1 } },
        { "default",                                         { 64, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 256, 1 } },
        { "GeForce GT 650M",                                 { 256, 1 } },
        { "GeForce GTX 1070",                                { 128, 1 } },
        { "GeForce GTX 1080",                                { 32, 1 } },
        { "GeForce GTX 480",                                 { 64, 1 } },
        { "GeForce GTX 670",                                 { 64, 1 } },
        { "GeForce GTX 680",                                 { 256, 1 } },
        { "GeForce GTX 750",                                 { 256, 1 } },
        { "GeForce GTX 750 Ti",                              { 32, 1 } },
        { "GeForce GTX 980",                                 { 128, 1 } },
        { "GeForce GTX TITAN",                               { 256, 1 } },
        { "GeForce GTX TITAN Black",                         { 256, 1 } },
        { "GeForce GTX TITAN X",                             { 256, 1 } },
        { "TITAN X (Pascal)",                                { 32, 1 } },
        { "Tesla K20m",                                      { 128, 1 } },
        { "Tesla K40m",                                      { 256, 1 } },
        { "default",                                         { 256, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 128, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvComplexSingle = {
  "Xgemv", Precision::kComplexSingle, {"WGS1", "WPT1"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 64, 1 } },
        { "ATI Radeon HD 6750M",                             { 64, 1 } },
        { "Ellesmere",                                       { 32, 1 } },
        { "Fiji",                                            { 32, 1 } },
        { "Hawaii",                                          { 64, 1 } },
        { "Oland",                                           { 64, 1 } },
        { "Pitcairn",                                        { 64, 1 } },
        { "Tahiti",                                          { 64, 1 } },
        { "Tonga",                                           { 32, 1 } },
        { "Turks",                                           { 64, 1 } },
        { "default",                                         { 64, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 32, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 128, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 128, 2 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 64, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 64, 4 } },
        { "default",                                         { 64, 2 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 64, 1 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 64, 1 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 128, 1 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 256, 1 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 64, 1 } },
        { "Iris",                                            { 256, 1 } },
        { "Iris Pro",                                        { 64, 1 } },
        { "default",                                         { 64, 1 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 64, 1 } },
        { "default",                                         { 64, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 256, 1 } },
        { "GeForce GTX 1070",                                { 64, 1 } },
        { "GeForce GTX 1080",                                { 32, 1 } },
        { "GeForce GTX 480",                                 { 64, 1 } },
        { "GeForce GTX 670",                                 { 64, 1 } },
        { "GeForce GTX 680",                                 { 64, 1 } },
        { "GeForce GTX 750",                                 { 128, 1 } },
        { "GeForce GTX 750 Ti",                              { 32, 1 } },
        { "GeForce GTX TITAN",                               { 256, 1 } },
        { "GeForce GTX TITAN Black",                         { 32, 1 } },
        { "TITAN X (Pascal)",                                { 32, 1 } },
        { "default",                                         { 64, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 64, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvDouble = {
  "Xgemv", Precision::kDouble, {"WGS1", "WPT1"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 64, 1 } },
        { "Ellesmere",                                       { 32, 1 } },
        { "Fiji",                                            { 32, 1 } },
        { "Hawaii",                                          { 128, 1 } },
        { "Oland",                                           { 256, 1 } },
        { "Pitcairn",                                        { 256, 1 } },
        { "Tahiti",                                          { 256, 1 } },
        { "Tonga",                                           { 32, 1 } },
        { "default",                                         { 256, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 64, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 64, 2 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 128, 1 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 64, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 64, 4 } },
        { "default",                                         { 64, 4 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 64, 1 } },
        { "default",                                         { 64, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 128, 1 } },
        { "GeForce GTX 1070",                                { 64, 1 } },
        { "GeForce GTX 1080",                                { 32, 1 } },
        { "GeForce GTX 480",                                 { 256, 1 } },
        { "GeForce GTX 670",                                 { 128, 1 } },
        { "GeForce GTX 680",                                 { 128, 1 } },
        { "GeForce GTX 750",                                 { 64, 1 } },
        { "GeForce GTX 750 Ti",                              { 32, 1 } },
        { "GeForce GTX 980",                                 { 64, 1 } },
        { "GeForce GTX TITAN",                               { 256, 1 } },
        { "GeForce GTX TITAN Black",                         { 32, 1 } },
        { "GeForce GTX TITAN X",                             { 64, 1 } },
        { "TITAN X (Pascal)",                                { 32, 1 } },
        { "Tesla K20m",                                      { 256, 1 } },
        { "Tesla K40m",                                      { 256, 1 } },
        { "default",                                         { 128, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 128, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvComplexDouble = {
  "Xgemv", Precision::kComplexDouble, {"WGS1", "WPT1"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 64, 1 } },
        { "Ellesmere",                                       { 32, 1 } },
        { "Fiji",                                            { 64, 1 } },
        { "Hawaii",                                          { 64, 1 } },
        { "Oland",                                           { 256, 1 } },
        { "Pitcairn",                                        { 256, 1 } },
        { "Tahiti",                                          { 256, 1 } },
        { "Tonga",                                           { 64, 1 } },
        { "default",                                         { 64, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 64, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 64, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 128, 2 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 64, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 32, 4 } },
        { "default",                                         { 64, 4 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 64, 1 } },
        { "default",                                         { 64, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 128, 1 } },
        { "GeForce GTX 480",                                 { 64, 1 } },
        { "GeForce GTX 670",                                 { 128, 1 } },
        { "default",                                         { 128, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 64, 1 } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
