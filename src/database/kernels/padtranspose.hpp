
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
  "Padtranspose", Precision::kHalf, {"PADTRA_PAD" "PADTRA_TILE" "PADTRA_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { 0, 16, 4 } },
        { "default",                                         { 0, 16, 4 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 0, 8, 1 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 1, 8, 2 } },
        { "default",                                         { 0, 8, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 0, 8, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadtransposeSingle = {
  "Padtranspose", Precision::kSingle, {"PADTRA_PAD" "PADTRA_TILE" "PADTRA_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 0, 16, 4 } },
        { "ATI Radeon HD 6750M",                             { 1, 16, 1 } },
        { "Ellesmere",                                       { 1, 8, 4 } },
        { "Fiji",                                            { 0, 16, 2 } },
        { "Hawaii",                                          { 1, 16, 4 } },
        { "Oland",                                           { 0, 16, 4 } },
        { "Pitcairn",                                        { 0, 16, 4 } },
        { "Tahiti",                                          { 0, 16, 4 } },
        { "Tonga",                                           { 0, 16, 2 } },
        { "Turks",                                           { 1, 16, 1 } },
        { "default",                                         { 0, 16, 4 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 0, 8, 2 } },
        { "default",                                         { 0, 8, 2 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 0, 8, 8 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 0, 16, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 0, 32, 1 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 0, 8, 8 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 0, 8, 8 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 0, 32, 1 } },
        { "default",                                         { 0, 8, 8 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 1, 16, 2 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 0, 16, 4 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 1, 16, 2 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 0, 16, 4 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 1, 16, 2 } },
        { "Iris",                                            { 1, 16, 2 } },
        { "Iris Pro",                                        { 1, 16, 2 } },
        { "default",                                         { 1, 16, 2 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 0, 16, 2 } },
        { "default",                                         { 0, 16, 2 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 1, 32, 2 } },
        { "GeForce GTX 1070",                                { 0, 16, 1 } },
        { "GeForce GTX 1080",                                { 1, 16, 2 } },
        { "GeForce GTX 480",                                 { 1, 16, 2 } },
        { "GeForce GTX 670",                                 { 1, 32, 2 } },
        { "GeForce GTX 680",                                 { 1, 16, 2 } },
        { "GeForce GTX 750",                                 { 1, 32, 2 } },
        { "GeForce GTX 750 Ti",                              { 1, 32, 2 } },
        { "GeForce GTX 980",                                 { 0, 16, 1 } },
        { "GeForce GTX TITAN",                               { 1, 16, 2 } },
        { "GeForce GTX TITAN Black",                         { 1, 32, 2 } },
        { "GeForce GTX TITAN X",                             { 1, 32, 1 } },
        { "TITAN X (Pascal)",                                { 1, 16, 2 } },
        { "Tesla K20m",                                      { 1, 16, 2 } },
        { "Tesla K40m",                                      { 1, 32, 2 } },
        { "default",                                         { 1, 32, 2 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 1, 16, 2 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadtransposeComplexSingle = {
  "Padtranspose", Precision::kComplexSingle, {"PADTRA_PAD" "PADTRA_TILE" "PADTRA_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 0, 16, 4 } },
        { "ATI Radeon HD 6750M",                             { 1, 16, 1 } },
        { "Ellesmere",                                       { 0, 8, 4 } },
        { "Fiji",                                            { 1, 16, 2 } },
        { "Hawaii",                                          { 0, 16, 2 } },
        { "Oland",                                           { 0, 8, 4 } },
        { "Pitcairn",                                        { 0, 8, 4 } },
        { "Tahiti",                                          { 0, 16, 2 } },
        { "Tonga",                                           { 0, 16, 2 } },
        { "Turks",                                           { 0, 16, 4 } },
        { "default",                                         { 0, 8, 4 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 1, 16, 2 } },
        { "default",                                         { 1, 16, 2 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 0, 8, 8 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 1, 8, 4 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 0, 8, 4 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 0, 8, 8 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 0, 8, 8 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 0, 8, 4 } },
        { "default",                                         { 0, 8, 8 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 1, 16, 2 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 0, 16, 2 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 1, 16, 2 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 0, 16, 2 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 0, 16, 4 } },
        { "Iris",                                            { 0, 16, 2 } },
        { "Iris Pro",                                        { 1, 16, 2 } },
        { "default",                                         { 1, 16, 2 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 1, 16, 1 } },
        { "default",                                         { 1, 16, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 1, 16, 1 } },
        { "GeForce GTX 1070",                                { 1, 16, 1 } },
        { "GeForce GTX 1080",                                { 0, 8, 1 } },
        { "GeForce GTX 480",                                 { 1, 16, 1 } },
        { "GeForce GTX 670",                                 { 1, 16, 1 } },
        { "GeForce GTX 680",                                 { 1, 16, 1 } },
        { "GeForce GTX 750",                                 { 1, 16, 2 } },
        { "GeForce GTX 750 Ti",                              { 1, 16, 1 } },
        { "GeForce GTX 980",                                 { 0, 16, 1 } },
        { "GeForce GTX TITAN",                               { 1, 16, 1 } },
        { "GeForce GTX TITAN Black",                         { 0, 16, 1 } },
        { "GeForce GTX TITAN X",                             { 1, 32, 1 } },
        { "TITAN X (Pascal)",                                { 1, 8, 1 } },
        { "Tesla K20m",                                      { 0, 16, 1 } },
        { "Tesla K40m",                                      { 1, 16, 1 } },
        { "default",                                         { 1, 16, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 1, 16, 2 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadtransposeDouble = {
  "Padtranspose", Precision::kDouble, {"PADTRA_PAD" "PADTRA_TILE" "PADTRA_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 0, 16, 4 } },
        { "Ellesmere",                                       { 0, 16, 4 } },
        { "Fiji",                                            { 0, 16, 2 } },
        { "Hawaii",                                          { 0, 16, 2 } },
        { "Oland",                                           { 0, 16, 4 } },
        { "Pitcairn",                                        { 0, 8, 4 } },
        { "Tahiti",                                          { 1, 16, 2 } },
        { "Tonga",                                           { 0, 8, 2 } },
        { "default",                                         { 0, 16, 4 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 0, 16, 2 } },
        { "default",                                         { 0, 16, 2 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 0, 8, 8 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 1, 8, 4 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 0, 64, 1 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 0, 8, 8 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 0, 8, 8 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 1, 32, 1 } },
        { "default",                                         { 1, 8, 4 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 0, 16, 1 } },
        { "default",                                         { 0, 16, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 1, 16, 1 } },
        { "GeForce GTX 1070",                                { 1, 16, 1 } },
        { "GeForce GTX 1080",                                { 0, 8, 1 } },
        { "GeForce GTX 480",                                 { 1, 16, 1 } },
        { "GeForce GTX 670",                                 { 1, 16, 1 } },
        { "GeForce GTX 680",                                 { 1, 16, 1 } },
        { "GeForce GTX 750",                                 { 1, 16, 2 } },
        { "GeForce GTX 750 Ti",                              { 1, 32, 2 } },
        { "GeForce GTX 980",                                 { 1, 32, 1 } },
        { "GeForce GTX TITAN",                               { 0, 16, 1 } },
        { "GeForce GTX TITAN Black",                         { 0, 16, 1 } },
        { "GeForce GTX TITAN X",                             { 1, 32, 1 } },
        { "TITAN X (Pascal)",                                { 0, 8, 1 } },
        { "Tesla K20m",                                      { 0, 16, 1 } },
        { "Tesla K40m",                                      { 1, 16, 1 } },
        { "default",                                         { 1, 16, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 1, 16, 2 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry PadtransposeComplexDouble = {
  "Padtranspose", Precision::kComplexDouble, {"PADTRA_PAD" "PADTRA_TILE" "PADTRA_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 0, 8, 4 } },
        { "Ellesmere",                                       { 0, 8, 4 } },
        { "Fiji",                                            { 0, 8, 2 } },
        { "Hawaii",                                          { 0, 8, 4 } },
        { "Oland",                                           { 0, 8, 4 } },
        { "Pitcairn",                                        { 0, 8, 4 } },
        { "Tahiti",                                          { 0, 8, 2 } },
        { "Tonga",                                           { 0, 8, 2 } },
        { "default",                                         { 0, 8, 4 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 0, 8, 1 } },
        { "default",                                         { 0, 8, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 0, 8, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 1, 8, 2 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 1, 16, 2 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 1, 8, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 0, 8, 4 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 1, 8, 4 } },
        { "default",                                         { 0, 8, 4 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 0, 16, 1 } },
        { "default",                                         { 0, 16, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 1, 16, 1 } },
        { "GeForce GTX 1070",                                { 1, 16, 1 } },
        { "GeForce GTX 1080",                                { 1, 8, 1 } },
        { "GeForce GTX 480",                                 { 1, 16, 1 } },
        { "GeForce GTX 670",                                 { 1, 16, 1 } },
        { "GeForce GTX 680",                                 { 1, 32, 1 } },
        { "GeForce GTX 750",                                 { 1, 16, 1 } },
        { "GeForce GTX 750 Ti",                              { 1, 8, 2 } },
        { "GeForce GTX 980",                                 { 0, 16, 1 } },
        { "GeForce GTX TITAN",                               { 1, 16, 1 } },
        { "GeForce GTX TITAN Black",                         { 0, 16, 1 } },
        { "GeForce GTX TITAN X",                             { 1, 32, 1 } },
        { "TITAN X (Pascal)",                                { 1, 8, 1 } },
        { "Tesla K20m",                                      { 1, 16, 1 } },
        { "Tesla K40m",                                      { 1, 16, 1 } },
        { "default",                                         { 1, 16, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 0, 8, 2 } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
