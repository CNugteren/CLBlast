
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
  "Pad", Precision::kHalf, {"PAD_DIMX", "PAD_DIMY", "PAD_WPTX", "PAD_WPTY"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { 16, 8, 1, 2 } },
        { "default",                                         { 16, 8, 1, 2 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 8, 8, 4, 1 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 8, 32, 2, 2 } },
        { "default",                                         { 8, 8, 2, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 8, 8, 2, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadSingle = {
  "Pad", Precision::kSingle, {"PAD_DIMX", "PAD_DIMY", "PAD_WPTX", "PAD_WPTY"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 32, 8, 1, 1 } },
        { "ATI Radeon HD 6750M",                             { 8, 16, 2, 1 } },
        { "Ellesmere",                                       { 32, 8, 2, 2 } },
        { "Fiji",                                            { 16, 16, 1, 2 } },
        { "Hawaii",                                          { 32, 8, 1, 4 } },
        { "Oland",                                           { 8, 8, 1, 2 } },
        { "Pitcairn",                                        { 32, 8, 1, 2 } },
        { "Tahiti",                                          { 32, 8, 1, 2 } },
        { "Tonga",                                           { 16, 16, 2, 2 } },
        { "Turks",                                           { 32, 8, 2, 1 } },
        { "default",                                         { 8, 16, 1, 2 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 32, 8, 1, 4 } },
        { "default",                                         { 32, 8, 1, 4 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 32, 32, 4, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 32, 16, 4, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 32, 8, 2, 4 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 16, 32, 4, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 32, 16, 4, 4 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 32, 8, 4, 1 } },
        { "default",                                         { 32, 8, 4, 2 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 32, 8, 2, 4 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 32, 8, 2, 4 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 16, 8, 1, 2 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 16, 8, 4, 1 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 32, 8, 4, 2 } },
        { "Iris",                                            { 32, 16, 2, 1 } },
        { "Iris Pro",                                        { 16, 8, 2, 1 } },
        { "default",                                         { 32, 8, 4, 2 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 32, 16, 2, 1 } },
        { "default",                                         { 32, 16, 2, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 32, 8, 2, 1 } },
        { "GeForce GT 650M",                                 { 32, 16, 2, 2 } },
        { "GeForce GTX 1070",                                { 16, 8, 1, 1 } },
        { "GeForce GTX 1080",                                { 16, 8, 1, 1 } },
        { "GeForce GTX 480",                                 { 32, 8, 1, 4 } },
        { "GeForce GTX 670",                                 { 32, 8, 2, 2 } },
        { "GeForce GTX 680",                                 { 16, 8, 4, 1 } },
        { "GeForce GTX 750",                                 { 32, 16, 4, 2 } },
        { "GeForce GTX 750 Ti",                              { 16, 8, 4, 1 } },
        { "GeForce GTX 980",                                 { 16, 8, 1, 1 } },
        { "GeForce GTX TITAN",                               { 32, 8, 2, 1 } },
        { "GeForce GTX TITAN Black",                         { 32, 8, 1, 2 } },
        { "GeForce GTX TITAN X",                             { 16, 16, 1, 1 } },
        { "TITAN X (Pascal)",                                { 16, 8, 1, 2 } },
        { "Tesla K20m",                                      { 32, 8, 2, 1 } },
        { "Tesla K40m",                                      { 32, 8, 1, 1 } },
        { "default",                                         { 32, 8, 4, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32, 8, 2, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadComplexSingle = {
  "Pad", Precision::kComplexSingle, {"PAD_DIMX", "PAD_DIMY", "PAD_WPTX", "PAD_WPTY"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 32, 8, 1, 1 } },
        { "ATI Radeon HD 6750M",                             { 16, 8, 2, 1 } },
        { "Ellesmere",                                       { 16, 16, 2, 4 } },
        { "Fiji",                                            { 16, 8, 1, 2 } },
        { "Hawaii",                                          { 32, 8, 1, 2 } },
        { "Oland",                                           { 8, 32, 1, 1 } },
        { "Pitcairn",                                        { 8, 8, 1, 2 } },
        { "Tahiti",                                          { 16, 16, 1, 1 } },
        { "Tonga",                                           { 16, 8, 1, 2 } },
        { "Turks",                                           { 16, 8, 4, 4 } },
        { "default",                                         { 16, 8, 1, 2 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 32, 8, 1, 4 } },
        { "default",                                         { 32, 8, 1, 4 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 32, 8, 4, 2 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 32, 8, 2, 2 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 32, 8, 1, 2 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 32, 32, 4, 1 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 32, 8, 2, 4 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 32, 16, 4, 1 } },
        { "default",                                         { 32, 8, 4, 2 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 8, 8, 1, 2 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 8, 8, 1, 1 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 8, 8, 1, 1 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 32, 8, 1, 1 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 32, 8, 1, 1 } },
        { "Iris",                                            { 32, 16, 2, 4 } },
        { "Iris Pro",                                        { 32, 8, 2, 1 } },
        { "default",                                         { 32, 8, 1, 4 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 32, 8, 1, 1 } },
        { "default",                                         { 32, 8, 1, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 16, 16, 1, 1 } },
        { "GeForce GTX 1070",                                { 8, 32, 1, 1 } },
        { "GeForce GTX 1080",                                { 32, 8, 1, 1 } },
        { "GeForce GTX 480",                                 { 16, 8, 2, 1 } },
        { "GeForce GTX 670",                                 { 16, 8, 1, 2 } },
        { "GeForce GTX 680",                                 { 16, 32, 1, 2 } },
        { "GeForce GTX 750",                                 { 32, 8, 2, 1 } },
        { "GeForce GTX 750 Ti",                              { 16, 8, 1, 1 } },
        { "GeForce GTX 980",                                 { 16, 16, 1, 1 } },
        { "GeForce GTX TITAN",                               { 16, 8, 2, 1 } },
        { "GeForce GTX TITAN Black",                         { 16, 8, 1, 2 } },
        { "GeForce GTX TITAN X",                             { 16, 8, 1, 1 } },
        { "TITAN X (Pascal)",                                { 32, 32, 1, 2 } },
        { "Tesla K20m",                                      { 32, 8, 1, 2 } },
        { "Tesla K40m",                                      { 16, 8, 1, 1 } },
        { "default",                                         { 32, 8, 1, 2 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32, 8, 1, 2 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadDouble = {
  "Pad", Precision::kDouble, {"PAD_DIMX", "PAD_DIMY", "PAD_WPTX", "PAD_WPTY"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 32, 8, 1, 1 } },
        { "Ellesmere",                                       { 8, 32, 2, 1 } },
        { "Fiji",                                            { 8, 16, 1, 2 } },
        { "Hawaii",                                          { 32, 8, 1, 2 } },
        { "Oland",                                           { 8, 32, 1, 1 } },
        { "Pitcairn",                                        { 8, 8, 1, 2 } },
        { "Tahiti",                                          { 32, 8, 1, 1 } },
        { "Tonga",                                           { 32, 8, 4, 1 } },
        { "default",                                         { 16, 16, 1, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 32, 8, 4, 2 } },
        { "default",                                         { 32, 8, 4, 2 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 32, 8, 4, 2 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 32, 8, 4, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 32, 16, 2, 2 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 32, 32, 4, 1 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 32, 32, 4, 1 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 32, 8, 2, 1 } },
        { "default",                                         { 32, 16, 4, 1 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 32, 8, 1, 1 } },
        { "default",                                         { 32, 8, 1, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 32, 8, 1, 1 } },
        { "GeForce GTX 1070",                                { 8, 8, 1, 1 } },
        { "GeForce GTX 1080",                                { 32, 32, 2, 1 } },
        { "GeForce GTX 480",                                 { 16, 8, 1, 1 } },
        { "GeForce GTX 670",                                 { 16, 16, 2, 1 } },
        { "GeForce GTX 680",                                 { 32, 32, 1, 2 } },
        { "GeForce GTX 750",                                 { 32, 16, 1, 1 } },
        { "GeForce GTX 750 Ti",                              { 8, 16, 1, 1 } },
        { "GeForce GTX 980",                                 { 8, 16, 1, 1 } },
        { "GeForce GTX TITAN",                               { 32, 8, 1, 1 } },
        { "GeForce GTX TITAN Black",                         { 16, 8, 1, 1 } },
        { "GeForce GTX TITAN X",                             { 16, 8, 1, 1 } },
        { "TITAN X (Pascal)",                                { 8, 32, 4, 1 } },
        { "Tesla K20m",                                      { 32, 8, 1, 1 } },
        { "Tesla K40m",                                      { 16, 8, 1, 2 } },
        { "default",                                         { 32, 8, 1, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32, 8, 1, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadComplexDouble = {
  "Pad", Precision::kComplexDouble, {"PAD_DIMX", "PAD_DIMY", "PAD_WPTX", "PAD_WPTY"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 16, 8, 1, 1 } },
        { "Ellesmere",                                       { 8, 16, 1, 2 } },
        { "Fiji",                                            { 32, 8, 2, 1 } },
        { "Hawaii",                                          { 32, 8, 1, 1 } },
        { "Oland",                                           { 8, 16, 2, 1 } },
        { "Pitcairn",                                        { 16, 8, 1, 1 } },
        { "Tahiti",                                          { 8, 16, 1, 1 } },
        { "Tonga",                                           { 8, 16, 1, 1 } },
        { "default",                                         { 8, 16, 1, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 16, 8, 4, 1 } },
        { "default",                                         { 16, 8, 4, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 16, 16, 4, 1 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 32, 8, 2, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 32, 8, 2, 2 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 16, 32, 4, 1 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 32, 32, 2, 2 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 32, 8, 2, 1 } },
        { "default",                                         { 32, 8, 2, 2 } },
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
        { "GRID K520",                                       { 8, 8, 1, 1 } },
        { "GeForce GTX 1070",                                { 8, 8, 2, 2 } },
        { "GeForce GTX 1080",                                { 8, 8, 1, 1 } },
        { "GeForce GTX 480",                                 { 16, 8, 1, 1 } },
        { "GeForce GTX 670",                                 { 32, 8, 1, 1 } },
        { "GeForce GTX 680",                                 { 8, 8, 1, 1 } },
        { "GeForce GTX 750",                                 { 8, 8, 1, 1 } },
        { "GeForce GTX 750 Ti",                              { 16, 32, 1, 1 } },
        { "GeForce GTX 980",                                 { 16, 16, 1, 1 } },
        { "GeForce GTX TITAN",                               { 8, 32, 1, 2 } },
        { "GeForce GTX TITAN Black",                         { 16, 8, 1, 4 } },
        { "GeForce GTX TITAN X",                             { 16, 8, 1, 1 } },
        { "TITAN X (Pascal)",                                { 8, 16, 1, 1 } },
        { "Tesla K20m",                                      { 8, 8, 1, 2 } },
        { "Tesla K40m",                                      { 8, 8, 1, 1 } },
        { "default",                                         { 16, 8, 1, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32, 8, 1, 1 } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
