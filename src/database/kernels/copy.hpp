
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
namespace database {
// =================================================================================================

const Database::DatabaseEntry CopyHalf = {
  "Copy", Precision::kHalf, {"COPY_DIMX", "COPY_DIMY", "COPY_VW", "COPY_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { 16, 8, 4, 4 } },
        { "default",                                         { 16, 8, 4, 4 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 8, 16, 8, 4 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 8, 32, 4, 8 } },
        { "default",                                         { 8, 32, 4, 8 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 16, 8, 4, 4 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry CopySingle = {
  "Copy", Precision::kSingle, {"COPY_DIMX", "COPY_DIMY", "COPY_VW", "COPY_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 32, 8, 4, 1 } },
        { "ATI Radeon HD 6750M",                             { 16, 8, 2, 1 } },
        { "Ellesmere",                                       { 8, 8, 4, 8 } },
        { "Fiji",                                            { 16, 16, 1, 2 } },
        { "Hawaii",                                          { 32, 8, 2, 2 } },
        { "Oland",                                           { 32, 8, 4, 2 } },
        { "Pitcairn",                                        { 8, 16, 4, 1 } },
        { "Tahiti",                                          { 32, 8, 2, 2 } },
        { "Tonga",                                           { 32, 8, 4, 4 } },
        { "Turks",                                           { 8, 8, 4, 2 } },
        { "default",                                         { 8, 16, 4, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 32, 8, 2, 4 } },
        { "default",                                         { 32, 8, 2, 4 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 32, 16, 8, 1 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 32, 16, 8, 2 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 32, 8, 4, 4 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 32, 16, 8, 1 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 32, 16, 8, 2 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 32, 8, 8, 1 } },
        { "default",                                         { 32, 16, 8, 2 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 8, 8, 2, 1 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 32, 16, 4, 1 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 32, 16, 4, 1 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 16, 8, 2, 1 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 16, 8, 4, 8 } },
        { "Iris",                                            { 16, 8, 1, 2 } },
        { "Iris Pro",                                        { 32, 8, 4, 4 } },
        { "default",                                         { 8, 8, 2, 1 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 32, 8, 8, 1 } },
        { "default",                                         { 32, 8, 8, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 16, 8, 4, 1 } },
        { "GeForce GT 650M",                                 { 16, 16, 4, 2 } },
        { "GeForce GTX 1070",                                { 8, 16, 4, 1 } },
        { "GeForce GTX 1080",                                { 8, 32, 4, 1 } },
        { "GeForce GTX 480",                                 { 8, 8, 4, 1 } },
        { "GeForce GTX 670",                                 { 16, 32, 4, 1 } },
        { "GeForce GTX 680",                                 { 32, 16, 4, 1 } },
        { "GeForce GTX 750",                                 { 32, 8, 2, 2 } },
        { "GeForce GTX 750 Ti",                              { 16, 32, 2, 2 } },
        { "GeForce GTX 980",                                 { 32, 16, 1, 1 } },
        { "GeForce GTX TITAN",                               { 32, 8, 2, 4 } },
        { "GeForce GTX TITAN Black",                         { 8, 32, 4, 8 } },
        { "GeForce GTX TITAN X",                             { 32, 8, 1, 2 } },
        { "TITAN X (Pascal)",                                { 8, 32, 4, 1 } },
        { "Tesla K20m",                                      { 8, 8, 4, 4 } },
        { "Tesla K40m",                                      { 8, 8, 4, 2 } },
        { "default",                                         { 8, 32, 4, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32, 8, 4, 4 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry CopyComplexSingle = {
  "Copy", Precision::kComplexSingle, {"COPY_DIMX", "COPY_DIMY", "COPY_VW", "COPY_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 32, 8, 1, 1 } },
        { "ATI Radeon HD 6750M",                             { 8, 8, 1, 1 } },
        { "Ellesmere",                                       { 16, 16, 1, 4 } },
        { "Fiji",                                            { 16, 8, 1, 2 } },
        { "Hawaii",                                          { 32, 8, 1, 2 } },
        { "Oland",                                           { 8, 16, 1, 1 } },
        { "Pitcairn",                                        { 8, 8, 1, 2 } },
        { "Tahiti",                                          { 8, 8, 2, 2 } },
        { "Tonga",                                           { 8, 32, 1, 2 } },
        { "Turks",                                           { 32, 8, 4, 1 } },
        { "default",                                         { 16, 8, 1, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 32, 16, 4, 2 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 16, 16, 8, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 32, 8, 4, 2 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 32, 8, 2, 2 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 32, 32, 4, 1 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 16, 8, 8, 1 } },
        { "default",                                         { 32, 8, 8, 1 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 16, 8, 2, 1 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 16, 16, 2, 2 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 8, 8, 1, 1 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 8, 32, 2, 4 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 8, 8, 2, 1 } },
        { "Iris",                                            { 16, 8, 1, 2 } },
        { "Iris Pro",                                        { 32, 16, 1, 4 } },
        { "default",                                         { 16, 8, 1, 2 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 32, 8, 4, 1 } },
        { "default",                                         { 32, 8, 4, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 16, 8, 1, 1 } },
        { "GeForce GTX 1070",                                { 16, 8, 1, 1 } },
        { "GeForce GTX 1080",                                { 32, 8, 1, 2 } },
        { "GeForce GTX 480",                                 { 16, 16, 1, 1 } },
        { "GeForce GTX 670",                                 { 16, 8, 1, 1 } },
        { "GeForce GTX 750",                                 { 16, 8, 1, 2 } },
        { "GeForce GTX 750 Ti",                              { 16, 32, 1, 1 } },
        { "GeForce GTX 980",                                 { 8, 8, 1, 1 } },
        { "GeForce GTX TITAN Black",                         { 16, 8, 1, 1 } },
        { "GeForce GTX TITAN X",                             { 16, 8, 1, 1 } },
        { "TITAN X (Pascal)",                                { 8, 16, 2, 1 } },
        { "Tesla K20m",                                      { 8, 8, 1, 4 } },
        { "Tesla K40m",                                      { 16, 8, 1, 1 } },
        { "default",                                         { 32, 8, 1, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 16, 8, 1, 2 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry CopyDouble = {
  "Copy", Precision::kDouble, {"COPY_DIMX", "COPY_DIMY", "COPY_VW", "COPY_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 32, 8, 1, 1 } },
        { "Ellesmere",                                       { 32, 8, 1, 4 } },
        { "Fiji",                                            { 16, 8, 1, 2 } },
        { "Hawaii",                                          { 32, 8, 1, 2 } },
        { "Oland",                                           { 32, 8, 2, 8 } },
        { "Pitcairn",                                        { 32, 8, 1, 1 } },
        { "Tahiti",                                          { 8, 32, 2, 1 } },
        { "Tonga",                                           { 8, 32, 2, 4 } },
        { "default",                                         { 16, 8, 2, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 16, 8, 8, 2 } },
        { "default",                                         { 16, 8, 8, 2 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 16, 32, 8, 1 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 16, 8, 8, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 32, 16, 8, 1 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 16, 32, 2, 1 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 16, 32, 8, 1 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 16, 16, 8, 1 } },
        { "default",                                         { 16, 8, 8, 1 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 8, 8, 8, 1 } },
        { "default",                                         { 8, 8, 8, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 32, 16, 2, 1 } },
        { "GeForce GTX 1070",                                { 8, 8, 4, 1 } },
        { "GeForce GTX 1080",                                { 8, 8, 4, 1 } },
        { "GeForce GTX 480",                                 { 8, 8, 2, 1 } },
        { "GeForce GTX 670",                                 { 8, 8, 2, 1 } },
        { "GeForce GTX 680",                                 { 16, 32, 2, 1 } },
        { "GeForce GTX 750",                                 { 8, 16, 2, 1 } },
        { "GeForce GTX 750 Ti",                              { 16, 8, 2, 1 } },
        { "GeForce GTX 980",                                 { 32, 8, 2, 1 } },
        { "GeForce GTX TITAN",                               { 16, 32, 2, 2 } },
        { "GeForce GTX TITAN Black",                         { 16, 8, 2, 8 } },
        { "GeForce GTX TITAN X",                             { 32, 16, 1, 1 } },
        { "TITAN X (Pascal)",                                { 8, 8, 2, 2 } },
        { "Tesla K20m",                                      { 8, 8, 2, 1 } },
        { "Tesla K40m",                                      { 8, 8, 2, 2 } },
        { "default",                                         { 32, 32, 2, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 16, 8, 2, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry CopyComplexDouble = {
  "Copy", Precision::kComplexDouble, {"COPY_DIMX", "COPY_DIMY", "COPY_VW", "COPY_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 8, 16, 1, 1 } },
        { "Ellesmere",                                       { 8, 32, 1, 2 } },
        { "Fiji",                                            { 8, 16, 1, 1 } },
        { "Hawaii",                                          { 32, 8, 2, 8 } },
        { "Oland",                                           { 8, 16, 1, 1 } },
        { "Pitcairn",                                        { 16, 8, 1, 1 } },
        { "Tahiti",                                          { 8, 16, 1, 1 } },
        { "Tonga",                                           { 16, 8, 2, 1 } },
        { "default",                                         { 8, 16, 1, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 32, 8, 1, 2 } },
        { "default",                                         { 32, 8, 1, 2 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 8, 8, 8, 1 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 32, 8, 8, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 32, 16, 2, 1 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 32, 32, 8, 1 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 32, 16, 8, 4 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 8, 8, 8, 1 } },
        { "default",                                         { 16, 8, 8, 1 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 32, 8, 8, 1 } },
        { "default",                                         { 32, 8, 8, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 8, 8, 1, 1 } },
        { "GeForce GTX 1070",                                { 8, 32, 1, 4 } },
        { "GeForce GTX 1080",                                { 8, 8, 1, 1 } },
        { "GeForce GTX 480",                                 { 16, 8, 1, 1 } },
        { "GeForce GTX 670",                                 { 16, 8, 1, 1 } },
        { "GeForce GTX 680",                                 { 8, 8, 1, 1 } },
        { "GeForce GTX 750",                                 { 32, 8, 1, 1 } },
        { "GeForce GTX 750 Ti",                              { 16, 16, 1, 1 } },
        { "GeForce GTX 980",                                 { 8, 8, 1, 1 } },
        { "GeForce GTX TITAN",                               { 16, 16, 1, 1 } },
        { "GeForce GTX TITAN Black",                         { 8, 8, 1, 2 } },
        { "GeForce GTX TITAN X",                             { 16, 8, 1, 1 } },
        { "TITAN X (Pascal)",                                { 8, 8, 1, 2 } },
        { "Tesla K20m",                                      { 8, 8, 1, 2 } },
        { "Tesla K40m",                                      { 8, 8, 1, 1 } },
        { "default",                                         { 8, 8, 1, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 16, 8, 1, 1 } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
