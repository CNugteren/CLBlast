
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xgemv_Fast' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XgemvFastHalf = {
  "XgemvFast", Precision::kHalf, {"VW2", "WGS2", "WPT2"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { 1, 32, 1 } },
        { "default",                                         { 1, 32, 1 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 1, 16, 1 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 4, 64, 4 } },
        { "default",                                         { 1, 16, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 1, 16, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastSingle = {
  "XgemvFast", Precision::kSingle, {"VW2", "WGS2", "WPT2"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 1, 128, 1 } },
        { "ATI Radeon HD 6750M",                             { 2, 64, 2 } },
        { "Ellesmere",                                       { 1, 64, 1 } },
        { "Fiji",                                            { 1, 64, 2 } },
        { "Hawaii",                                          { 1, 64, 1 } },
        { "Oland",                                           { 1, 64, 1 } },
        { "Pitcairn",                                        { 1, 64, 1 } },
        { "Tahiti",                                          { 1, 64, 1 } },
        { "Tonga",                                           { 1, 16, 4 } },
        { "Turks",                                           { 1, 256, 1 } },
        { "default",                                         { 1, 64, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 1, 32, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 4, 128, 4 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 4, 32, 4 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 1, 64, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 2, 16, 4 } },
        { "default",                                         { 4, 128, 4 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 1, 256, 1 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 2, 32, 2 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 4, 128, 4 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 1, 64, 2 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 2, 32, 2 } },
        { "Iris",                                            { 1, 128, 2 } },
        { "Iris Pro",                                        { 4, 64, 4 } },
        { "default",                                         { 2, 256, 2 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 1, 64, 1 } },
        { "default",                                         { 1, 64, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 2, 256, 2 } },
        { "GeForce GT 650M",                                 { 2, 32, 2 } },
        { "GeForce GTX 1070",                                { 1, 256, 1 } },
        { "GeForce GTX 1080",                                { 1, 128, 1 } },
        { "GeForce GTX 480",                                 { 1, 128, 1 } },
        { "GeForce GTX 670",                                 { 2, 256, 2 } },
        { "GeForce GTX 680",                                 { 1, 128, 1 } },
        { "GeForce GTX 750",                                 { 1, 256, 1 } },
        { "GeForce GTX 750 Ti",                              { 2, 32, 2 } },
        { "GeForce GTX 980",                                 { 1, 256, 1 } },
        { "GeForce GTX TITAN",                               { 1, 256, 1 } },
        { "GeForce GTX TITAN Black",                         { 1, 256, 1 } },
        { "GeForce GTX TITAN X",                             { 1, 64, 1 } },
        { "TITAN X (Pascal)",                                { 1, 64, 1 } },
        { "Tesla K20m",                                      { 1, 256, 1 } },
        { "Tesla K40m",                                      { 1, 256, 1 } },
        { "default",                                         { 1, 256, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 1, 64, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastComplexSingle = {
  "XgemvFast", Precision::kComplexSingle, {"VW2", "WGS2", "WPT2"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 2, 256, 2 } },
        { "ATI Radeon HD 6750M",                             { 1, 128, 1 } },
        { "Ellesmere",                                       { 1, 64, 1 } },
        { "Fiji",                                            { 1, 16, 1 } },
        { "Hawaii",                                          { 1, 64, 1 } },
        { "Oland",                                           { 1, 64, 1 } },
        { "Pitcairn",                                        { 1, 64, 1 } },
        { "Tahiti",                                          { 1, 128, 1 } },
        { "Tonga",                                           { 2, 32, 2 } },
        { "Turks",                                           { 1, 16, 1 } },
        { "default",                                         { 1, 64, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 2, 64, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 1, 128, 2 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 2, 128, 2 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 4, 64, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 4, 16, 4 } },
        { "default",                                         { 1, 64, 2 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 2, 128, 2 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 1, 32, 2 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 2, 128, 2 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 1, 32, 4 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 1, 32, 1 } },
        { "Iris",                                            { 1, 64, 1 } },
        { "Iris Pro",                                        { 4, 128, 4 } },
        { "default",                                         { 1, 64, 1 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 1, 64, 1 } },
        { "default",                                         { 1, 64, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 1, 256, 1 } },
        { "GeForce GTX 1070",                                { 1, 64, 1 } },
        { "GeForce GTX 480",                                 { 1, 64, 1 } },
        { "GeForce GTX 670",                                 { 1, 64, 1 } },
        { "GeForce GTX 680",                                 { 1, 64, 1 } },
        { "default",                                         { 1, 64, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 1, 64, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastDouble = {
  "XgemvFast", Precision::kDouble, {"VW2", "WGS2", "WPT2"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 1, 256, 1 } },
        { "Ellesmere",                                       { 1, 128, 1 } },
        { "Fiji",                                            { 1, 32, 1 } },
        { "Hawaii",                                          { 1, 64, 1 } },
        { "Oland",                                           { 1, 64, 1 } },
        { "Pitcairn",                                        { 1, 64, 1 } },
        { "Tahiti",                                          { 1, 64, 1 } },
        { "Tonga",                                           { 2, 32, 2 } },
        { "default",                                         { 1, 64, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 1, 64, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 4, 128, 4 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 1, 16, 1 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 1, 64, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 1, 16, 4 } },
        { "default",                                         { 1, 64, 4 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 1, 64, 1 } },
        { "default",                                         { 1, 64, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 1, 256, 1 } },
        { "GeForce GTX 1070",                                { 1, 256, 1 } },
        { "GeForce GTX 1080",                                { 1, 32, 2 } },
        { "GeForce GTX 480",                                 { 1, 64, 1 } },
        { "GeForce GTX 670",                                 { 1, 128, 1 } },
        { "GeForce GTX 680",                                 { 1, 128, 1 } },
        { "GeForce GTX 750",                                 { 2, 256, 2 } },
        { "GeForce GTX 750 Ti",                              { 1, 32, 2 } },
        { "GeForce GTX 980",                                 { 1, 64, 1 } },
        { "GeForce GTX TITAN",                               { 1, 256, 1 } },
        { "GeForce GTX TITAN Black",                         { 1, 256, 1 } },
        { "GeForce GTX TITAN X",                             { 1, 128, 1 } },
        { "TITAN X (Pascal)",                                { 1, 32, 1 } },
        { "Tesla K20m",                                      { 1, 128, 1 } },
        { "Tesla K40m",                                      { 1, 256, 1 } },
        { "default",                                         { 1, 256, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 1, 64, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastComplexDouble = {
  "XgemvFast", Precision::kComplexDouble, {"VW2", "WGS2", "WPT2"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 1, 256, 1 } },
        { "Ellesmere",                                       { 1, 16, 1 } },
        { "Fiji",                                            { 1, 16, 1 } },
        { "Hawaii",                                          { 1, 64, 1 } },
        { "Oland",                                           { 1, 256, 1 } },
        { "Pitcairn",                                        { 1, 64, 1 } },
        { "Tahiti",                                          { 1, 64, 1 } },
        { "Tonga",                                           { 1, 32, 1 } },
        { "default",                                         { 1, 64, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 4, 32, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 2, 64, 4 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 1, 64, 4 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 4, 64, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 1, 16, 2 } },
        { "default",                                         { 4, 64, 4 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 1, 64, 1 } },
        { "default",                                         { 1, 64, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 1, 128, 1 } },
        { "GeForce GTX 480",                                 { 1, 64, 1 } },
        { "GeForce GTX 670",                                 { 1, 64, 1 } },
        { "default",                                         { 1, 64, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 1, 64, 1 } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
