
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xaxpy' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XaxpyHalf = {
  "Xaxpy", Precision::kHalf, {"VW" "WGS" "WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { 4, 128, 4 } },
        { "default",                                         { 4, 128, 4 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 1, 64, 1 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 8, 64, 1 } },
        { "default",                                         { 8, 64, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 8, 256, 4 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XaxpySingle = {
  "Xaxpy", Precision::kSingle, {"VW" "WGS" "WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 1, 128, 1 } },
        { "ATI Radeon HD 6750M",                             { 1, 256, 2 } },
        { "Ellesmere",                                       { 1, 64, 4 } },
        { "Fiji",                                            { 4, 64, 1 } },
        { "Hawaii",                                          { 2, 64, 2 } },
        { "Oland",                                           { 1, 128, 1 } },
        { "Pitcairn",                                        { 2, 128, 1 } },
        { "Tahiti",                                          { 2, 64, 1 } },
        { "Tonga",                                           { 1, 256, 8 } },
        { "Turks",                                           { 2, 256, 1 } },
        { "default",                                         { 2, 256, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 4, 256, 1 } },
        { "default",                                         { 4, 256, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 8, 512, 1 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 1, 512, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 1, 128, 2 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 4, 256, 1 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 2, 1024, 1 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 1, 128, 1 } },
        { "default",                                         { 8, 512, 1 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 1, 128, 1 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 1, 256, 1 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 1, 64, 1 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 1, 64, 1 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 8, 512, 1 } },
        { "Iris",                                            { 1, 64, 1 } },
        { "Iris Pro",                                        { 1, 128, 2 } },
        { "default",                                         { 4, 256, 1 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 2, 1024, 2 } },
        { "default",                                         { 2, 1024, 2 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 2, 64, 1 } },
        { "GeForce GT 650M",                                 { 2, 1024, 1 } },
        { "GeForce GTX 1070",                                { 1, 64, 4 } },
        { "GeForce GTX 1080",                                { 1, 256, 1 } },
        { "GeForce GTX 480",                                 { 2, 128, 1 } },
        { "GeForce GTX 670",                                 { 2, 64, 1 } },
        { "GeForce GTX 680",                                 { 1, 128, 1 } },
        { "GeForce GTX 750",                                 { 1, 64, 1 } },
        { "GeForce GTX 750 Ti",                              { 2, 64, 1 } },
        { "GeForce GTX 980",                                 { 1, 1024, 1 } },
        { "GeForce GTX TITAN",                               { 4, 256, 1 } },
        { "GeForce GTX TITAN Black",                         { 4, 128, 4 } },
        { "GeForce GTX TITAN X",                             { 1, 64, 1 } },
        { "TITAN X (Pascal)",                                { 4, 128, 1 } },
        { "Tesla K20m",                                      { 4, 128, 1 } },
        { "Tesla K40m",                                      { 4, 128, 1 } },
        { "default",                                         { 4, 1024, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 4, 256, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XaxpyComplexSingle = {
  "Xaxpy", Precision::kComplexSingle, {"VW" "WGS" "WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 2, 64, 8 } },
        { "ATI Radeon HD 6750M",                             { 1, 64, 1 } },
        { "Ellesmere",                                       { 2, 256, 1 } },
        { "Fiji",                                            { 1, 128, 2 } },
        { "Hawaii",                                          { 1, 128, 2 } },
        { "Oland",                                           { 1, 128, 1 } },
        { "Pitcairn",                                        { 1, 64, 1 } },
        { "Tahiti",                                          { 1, 64, 1 } },
        { "Tonga",                                           { 1, 256, 8 } },
        { "Turks",                                           { 2, 256, 1 } },
        { "default",                                         { 1, 128, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 1, 256, 1 } },
        { "default",                                         { 1, 256, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 4, 1024, 1 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 4, 256, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 4, 1024, 1 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 1, 1024, 2 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 4, 1024, 1 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 2, 1024, 1 } },
        { "default",                                         { 8, 1024, 1 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 4, 64, 2 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 1, 64, 1 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 1, 64, 1 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 1, 64, 1 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 4, 64, 1 } },
        { "Iris",                                            { 2, 128, 1 } },
        { "Iris Pro",                                        { 1, 256, 8 } },
        { "default",                                         { 4, 64, 1 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 1, 1024, 1 } },
        { "default",                                         { 1, 1024, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 1, 512, 1 } },
        { "GeForce GTX 1070",                                { 1, 64, 2 } },
        { "GeForce GTX 1080",                                { 2, 64, 1 } },
        { "GeForce GTX 480",                                 { 1, 256, 1 } },
        { "GeForce GTX 670",                                 { 1, 256, 1 } },
        { "GeForce GTX 680",                                 { 1, 256, 1 } },
        { "GeForce GTX 750",                                 { 1, 512, 1 } },
        { "GeForce GTX 750 Ti",                              { 1, 512, 1 } },
        { "GeForce GTX 980",                                 { 1, 64, 1 } },
        { "GeForce GTX TITAN",                               { 1, 256, 1 } },
        { "GeForce GTX TITAN Black",                         { 1, 128, 2 } },
        { "GeForce GTX TITAN X",                             { 1, 512, 1 } },
        { "TITAN X (Pascal)",                                { 2, 512, 1 } },
        { "Tesla K20m",                                      { 1, 128, 1 } },
        { "Tesla K40m",                                      { 1, 128, 1 } },
        { "default",                                         { 1, 256, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 1, 128, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XaxpyDouble = {
  "Xaxpy", Precision::kDouble, {"VW" "WGS" "WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 1, 256, 1 } },
        { "Ellesmere",                                       { 2, 64, 4 } },
        { "Fiji",                                            { 2, 64, 4 } },
        { "Hawaii",                                          { 1, 64, 2 } },
        { "Oland",                                           { 1, 64, 1 } },
        { "Pitcairn",                                        { 1, 128, 1 } },
        { "Tahiti",                                          { 1, 64, 1 } },
        { "Tonga",                                           { 1, 128, 4 } },
        { "default",                                         { 2, 64, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 2, 128, 2 } },
        { "default",                                         { 2, 128, 2 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 4, 64, 1 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 1, 1024, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 2, 1024, 1 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 8, 64, 1 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 8, 256, 1 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 8, 2048, 1 } },
        { "default",                                         { 8, 64, 1 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 2, 512, 1 } },
        { "default",                                         { 2, 512, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 1, 64, 1 } },
        { "GeForce GTX 1070",                                { 1, 64, 8 } },
        { "GeForce GTX 1080",                                { 1, 128, 1 } },
        { "GeForce GTX 480",                                 { 1, 128, 1 } },
        { "GeForce GTX 670",                                 { 1, 64, 1 } },
        { "GeForce GTX 680",                                 { 1, 64, 1 } },
        { "GeForce GTX 750",                                 { 1, 128, 1 } },
        { "GeForce GTX 750 Ti",                              { 1, 256, 2 } },
        { "GeForce GTX 980",                                 { 1, 256, 1 } },
        { "GeForce GTX TITAN",                               { 2, 1024, 1 } },
        { "GeForce GTX TITAN Black",                         { 2, 128, 1 } },
        { "GeForce GTX TITAN X",                             { 1, 512, 1 } },
        { "TITAN X (Pascal)",                                { 2, 512, 1 } },
        { "Tesla K20m",                                      { 2, 128, 1 } },
        { "Tesla K40m",                                      { 2, 128, 1 } },
        { "default",                                         { 1, 128, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 2, 256, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XaxpyComplexDouble = {
  "Xaxpy", Precision::kComplexDouble, {"VW" "WGS" "WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 1, 128, 1 } },
        { "Ellesmere",                                       { 1, 128, 1 } },
        { "Fiji",                                            { 1, 64, 1 } },
        { "Hawaii",                                          { 2, 64, 1 } },
        { "Oland",                                           { 1, 256, 1 } },
        { "Pitcairn",                                        { 1, 128, 1 } },
        { "Tahiti",                                          { 1, 128, 1 } },
        { "Tonga",                                           { 1, 64, 1 } },
        { "default",                                         { 1, 128, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 1, 64, 8 } },
        { "default",                                         { 1, 64, 8 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 4, 1024, 1 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 8, 128, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 8, 128, 1 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 8, 512, 1 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 8, 1024, 1 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 1, 256, 1 } },
        { "default",                                         { 8, 256, 1 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 1, 1024, 1 } },
        { "default",                                         { 1, 1024, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 1, 64, 1 } },
        { "GeForce GTX 1070",                                { 1, 64, 2 } },
        { "GeForce GTX 1080",                                { 1, 256, 1 } },
        { "GeForce GTX 480",                                 { 1, 128, 1 } },
        { "GeForce GTX 670",                                 { 1, 256, 1 } },
        { "GeForce GTX 680",                                 { 1, 64, 1 } },
        { "GeForce GTX 750",                                 { 1, 1024, 1 } },
        { "GeForce GTX 750 Ti",                              { 1, 64, 2 } },
        { "GeForce GTX 980",                                 { 1, 1024, 1 } },
        { "GeForce GTX TITAN",                               { 1, 64, 4 } },
        { "GeForce GTX TITAN Black",                         { 1, 128, 4 } },
        { "GeForce GTX TITAN X",                             { 1, 1024, 1 } },
        { "TITAN X (Pascal)",                                { 1, 256, 2 } },
        { "Tesla K20m",                                      { 1, 64, 1 } },
        { "Tesla K40m",                                      { 1, 64, 1 } },
        { "default",                                         { 1, 64, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 1, 256, 1 } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
