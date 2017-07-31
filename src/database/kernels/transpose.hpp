
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
  "Transpose", Precision::kHalf, {"TRA_DIM", "TRA_PAD", "TRA_SHUFFLE", "TRA_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { 4, 0, 1, 8 } },
        { "default",                                         { 4, 0, 1, 8 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 8, 1, 1, 8 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 16, 0, 0, 4 } },
        { "default",                                         { 8, 1, 0, 8 } },
      }
    },
    { // QUALCOMM GPUs
      kDeviceTypeGPU, "QUALCOMM", {
        { "QUALCOMM Adreno(TM)",                             { 8, 0, 0, 4 } },
        { "default",                                         { 8, 0, 0, 4 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 8, 0, 1, 8 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry TransposeSingle = {
  "Transpose", Precision::kSingle, {"TRA_DIM", "TRA_PAD", "TRA_SHUFFLE", "TRA_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 4, 0, 1, 8 } },
        { "ATI Radeon HD 6750M",                             { 8, 0, 1, 2 } },
        { "Ellesmere",                                       { 16, 0, 1, 4 } },
        { "Fiji",                                            { 16, 0, 1, 2 } },
        { "Hawaii",                                          { 4, 0, 1, 8 } },
        { "Oland",                                           { 8, 0, 1, 4 } },
        { "Pitcairn",                                        { 16, 0, 1, 1 } },
        { "Tahiti",                                          { 4, 0, 1, 4 } },
        { "Tonga",                                           { 8, 1, 1, 2 } },
        { "Turks",                                           { 8, 0, 1, 2 } },
        { "default",                                         { 8, 0, 1, 2 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 8, 0, 1, 4 } },
        { "default",                                         { 8, 0, 1, 4 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 4, 1, 0, 16 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 4, 0, 0, 8 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 4, 0, 1, 8 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 4, 0, 0, 8 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 4, 1, 0, 16 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 4, 0, 0, 8 } },
        { "default",                                         { 4, 0, 0, 8 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 16, 0, 1, 4 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 16, 0, 0, 4 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 16, 0, 0, 4 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 8, 0, 1, 4 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 16, 0, 1, 2 } },
        { "Iris",                                            { 8, 1, 0, 4 } },
        { "Iris Pro",                                        { 16, 1, 0, 4 } },
        { "default",                                         { 16, 0, 0, 4 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 16, 1, 1, 1 } },
        { "default",                                         { 16, 1, 1, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 32, 1, 1, 2 } },
        { "GeForce GT 650M",                                 { 8, 1, 0, 4 } },
        { "GeForce GTX 1070",                                { 8, 0, 1, 4 } },
        { "GeForce GTX 1080",                                { 4, 0, 0, 4 } },
        { "GeForce GTX 480",                                 { 16, 1, 0, 2 } },
        { "GeForce GTX 670",                                 { 16, 1, 1, 2 } },
        { "GeForce GTX 680",                                 { 16, 1, 1, 2 } },
        { "GeForce GTX 750",                                 { 4, 0, 0, 8 } },
        { "GeForce GTX 750 Ti",                              { 32, 1, 0, 2 } },
        { "GeForce GTX 980",                                 { 16, 0, 0, 1 } },
        { "GeForce GTX TITAN",                               { 8, 1, 0, 4 } },
        { "GeForce GTX TITAN Black",                         { 8, 1, 0, 4 } },
        { "GeForce GTX TITAN X",                             { 16, 0, 0, 4 } },
        { "TITAN X (Pascal)",                                { 8, 0, 0, 4 } },
        { "Tesla K20m",                                      { 8, 0, 0, 4 } },
        { "Tesla K40m",                                      { 8, 1, 0, 4 } },
        { "default",                                         { 8, 1, 0, 4 } },
      }
    },
    { // QUALCOMM GPUs
      kDeviceTypeGPU, "QUALCOMM", {
        { "QUALCOMM Adreno(TM)",                             { 8, 1, 1, 4 } },
        { "default",                                         { 8, 1, 1, 4 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 8, 0, 1, 4 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry TransposeComplexSingle = {
  "Transpose", Precision::kComplexSingle, {"TRA_DIM", "TRA_PAD", "TRA_SHUFFLE", "TRA_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 4, 1, 1, 4 } },
        { "ATI Radeon HD 6750M",                             { 16, 1, 1, 1 } },
        { "Ellesmere",                                       { 4, 0, 1, 4 } },
        { "Fiji",                                            { 8, 1, 1, 2 } },
        { "Hawaii",                                          { 16, 0, 1, 1 } },
        { "Oland",                                           { 4, 0, 1, 2 } },
        { "Pitcairn",                                        { 8, 0, 1, 1 } },
        { "Tahiti",                                          { 16, 0, 1, 1 } },
        { "Tonga",                                           { 16, 0, 1, 1 } },
        { "Turks",                                           { 8, 1, 1, 4 } },
        { "default",                                         { 8, 0, 1, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 16, 0, 0, 2 } },
        { "default",                                         { 16, 0, 0, 2 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 4, 0, 1, 8 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 8, 0, 0, 2 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 4, 0, 0, 4 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 4, 1, 0, 8 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 4, 1, 0, 8 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 16, 0, 0, 4 } },
        { "default",                                         { 4, 1, 0, 8 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 16, 1, 1, 2 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 8, 0, 0, 2 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 8, 0, 0, 2 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 16, 1, 1, 2 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 16, 1, 0, 2 } },
        { "Iris",                                            { 8, 0, 0, 2 } },
        { "Iris Pro",                                        { 16, 1, 0, 2 } },
        { "default",                                         { 16, 1, 0, 2 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 16, 1, 1, 1 } },
        { "GeForce GTX 1070",                                { 16, 1, 1, 1 } },
        { "GeForce GTX 1080",                                { 16, 1, 0, 1 } },
        { "GeForce GTX 480",                                 { 16, 1, 0, 1 } },
        { "GeForce GTX 670",                                 { 16, 1, 1, 1 } },
        { "GeForce GTX 680",                                 { 16, 1, 1, 1 } },
        { "GeForce GTX 750",                                 { 16, 1, 0, 1 } },
        { "GeForce GTX 750 Ti",                              { 16, 1, 0, 1 } },
        { "GeForce GTX 980",                                 { 16, 1, 0, 1 } },
        { "GeForce GTX TITAN",                               { 16, 0, 0, 1 } },
        { "GeForce GTX TITAN Black",                         { 16, 1, 0, 1 } },
        { "GeForce GTX TITAN X",                             { 32, 1, 0, 1 } },
        { "TITAN X (Pascal)",                                { 8, 1, 0, 2 } },
        { "Tesla K20m",                                      { 16, 0, 0, 1 } },
        { "Tesla K40m",                                      { 16, 1, 0, 1 } },
        { "default",                                         { 16, 1, 0, 1 } },
      }
    },
    { // QUALCOMM GPUs
      kDeviceTypeGPU, "QUALCOMM", {
        { "QUALCOMM Adreno(TM)",                             { 16, 1, 0, 1 } },
        { "default",                                         { 16, 1, 0, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 8, 1, 1, 2 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry TransposeDouble = {
  "Transpose", Precision::kDouble, {"TRA_DIM", "TRA_PAD", "TRA_SHUFFLE", "TRA_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 4, 0, 1, 4 } },
        { "Ellesmere",                                       { 4, 0, 1, 4 } },
        { "Fiji",                                            { 8, 1, 1, 2 } },
        { "Hawaii",                                          { 16, 0, 1, 1 } },
        { "Oland",                                           { 8, 1, 1, 2 } },
        { "Pitcairn",                                        { 4, 0, 1, 2 } },
        { "Tahiti",                                          { 4, 1, 1, 4 } },
        { "Tonga",                                           { 4, 0, 1, 4 } },
        { "default",                                         { 4, 0, 1, 4 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 8, 0, 0, 1 } },
        { "default",                                         { 8, 0, 0, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 4, 1, 0, 8 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 4, 0, 0, 4 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 4, 1, 0, 4 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 4, 1, 0, 8 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 4, 0, 0, 16 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 4, 0, 0, 8 } },
        { "default",                                         { 4, 1, 0, 8 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 32, 1, 0, 1 } },
        { "default",                                         { 32, 1, 0, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 16, 1, 1, 2 } },
        { "GeForce GTX 1070",                                { 8, 0, 1, 2 } },
        { "GeForce GTX 1080",                                { 8, 0, 0, 2 } },
        { "GeForce GTX 480",                                 { 8, 1, 0, 2 } },
        { "GeForce GTX 670",                                 { 16, 1, 1, 2 } },
        { "GeForce GTX 680",                                 { 16, 1, 1, 2 } },
        { "GeForce GTX 750",                                 { 16, 1, 0, 1 } },
        { "GeForce GTX 750 Ti",                              { 32, 1, 0, 2 } },
        { "GeForce GTX 980",                                 { 16, 0, 0, 2 } },
        { "GeForce GTX TITAN",                               { 8, 0, 0, 2 } },
        { "GeForce GTX TITAN Black",                         { 16, 1, 0, 2 } },
        { "GeForce GTX TITAN X",                             { 32, 1, 0, 1 } },
        { "TITAN X (Pascal)",                                { 16, 1, 0, 2 } },
        { "Tesla K20m",                                      { 16, 1, 0, 2 } },
        { "Tesla K40m",                                      { 16, 1, 1, 2 } },
        { "default",                                         { 16, 1, 1, 2 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 16, 1, 1, 2 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry TransposeComplexDouble = {
  "Transpose", Precision::kComplexDouble, {"TRA_DIM", "TRA_PAD", "TRA_SHUFFLE", "TRA_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 4, 1, 1, 2 } },
        { "Ellesmere",                                       { 16, 0, 1, 1 } },
        { "Fiji",                                            { 16, 0, 1, 1 } },
        { "Hawaii",                                          { 4, 0, 1, 2 } },
        { "Oland",                                           { 16, 0, 1, 1 } },
        { "Pitcairn",                                        { 4, 0, 1, 1 } },
        { "Tahiti",                                          { 16, 0, 1, 1 } },
        { "Tonga",                                           { 8, 1, 1, 2 } },
        { "default",                                         { 16, 0, 1, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 8, 0, 0, 1 } },
        { "default",                                         { 8, 0, 0, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 4, 0, 1, 8 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 4, 0, 0, 4 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 4, 0, 0, 4 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 4, 1, 0, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 4, 0, 1, 8 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 16, 1, 0, 2 } },
        { "default",                                         { 4, 0, 0, 8 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 16, 1, 1, 1 } },
        { "GeForce GTX 1070",                                { 8, 0, 0, 1 } },
        { "GeForce GTX 1080",                                { 8, 0, 0, 1 } },
        { "GeForce GTX 480",                                 { 8, 1, 0, 1 } },
        { "GeForce GTX 670",                                 { 16, 1, 1, 1 } },
        { "GeForce GTX 680",                                 { 16, 1, 1, 1 } },
        { "GeForce GTX 750",                                 { 16, 1, 0, 1 } },
        { "GeForce GTX 750 Ti",                              { 16, 1, 0, 1 } },
        { "GeForce GTX 980",                                 { 32, 1, 0, 1 } },
        { "GeForce GTX TITAN",                               { 16, 1, 0, 1 } },
        { "GeForce GTX TITAN Black",                         { 16, 0, 0, 1 } },
        { "GeForce GTX TITAN X",                             { 32, 1, 0, 1 } },
        { "TITAN X (Pascal)",                                { 8, 0, 0, 1 } },
        { "Tesla K20m",                                      { 16, 1, 0, 1 } },
        { "Tesla K40m",                                      { 16, 1, 0, 1 } },
        { "default",                                         { 16, 1, 0, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 16, 1, 1, 1 } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
