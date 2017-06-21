
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xgemm' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XgemmHalf = {
  "Xgemm", Precision::kHalf, {"KWG", "KWI", "MDIMA", "MDIMC", "MWG", "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { 32, 2, 8, 8, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 } },
        { "default",                                         { 32, 2, 8, 8, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 32, 2, 8, 8, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 } },
        { "default",                                         { 32, 2, 8, 8, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32, 2, 8, 8, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmSingle = {
  "Xgemm", Precision::kSingle, {"KWG", "KWI", "MDIMA", "MDIMC", "MWG", "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 32, 2, 16, 16, 64, 8, 16, 128, 0, 0, 0, 0, 2, 8 } },
        { "ATI Radeon HD 6750M",                             { 32, 2, 8, 16, 128, 8, 8, 128, 0, 0, 1, 1, 8, 8 } },
        { "Ellesmere",                                       { 32, 2, 8, 8, 16, 16, 16, 64, 1, 1, 0, 0, 1, 2 } },
        { "Fiji",                                            { 32, 2, 16, 16, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 } },
        { "Hawaii",                                          { 16, 2, 16, 32, 128, 32, 8, 64, 1, 1, 1, 1, 4, 2 } },
        { "Oland",                                           { 16, 2, 32, 16, 64, 32, 16, 128, 1, 1, 1, 0, 2, 4 } },
        { "Pitcairn",                                        { 16, 2, 16, 8, 32, 16, 16, 128, 0, 0, 1, 0, 1, 1 } },
        { "Tahiti",                                          { 32, 2, 16, 32, 128, 16, 8, 64, 0, 0, 0, 0, 4, 1 } },
        { "Tonga",                                           { 16, 2, 16, 32, 64, 16, 8, 128, 1, 1, 0, 0, 2, 8 } },
        { "Turks",                                           { 32, 2, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 4 } },
        { "default",                                         { 32, 2, 8, 8, 32, 8, 8, 64, 0, 0, 0, 0, 4, 4 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 16, 2, 8, 8, 64, 8, 16, 16, 0, 0, 1, 1, 8, 1 } },
        { "default",                                         { 16, 2, 8, 8, 64, 8, 16, 16, 0, 0, 1, 1, 8, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 16, 2, 8, 8, 128, 16, 8, 128, 0, 1, 1, 1, 1, 8 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 32, 8, 32, 32, 64, 32, 16, 64, 1, 1, 1, 0, 2, 2 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 32, 2, 16, 8, 128, 16, 8, 64, 0, 0, 1, 0, 1, 2 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 32, 2, 32, 8, 128, 8, 8, 128, 1, 1, 1, 1, 2, 8 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 16, 2, 8, 8, 128, 8, 8, 128, 1, 1, 1, 0, 1, 8 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 32, 8, 16, 16, 64, 32, 32, 64, 0, 1, 1, 0, 1, 2 } },
        { "default",                                         { 32, 2, 8, 8, 32, 8, 8, 64, 1, 1, 0, 0, 4, 4 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 32, 2, 8, 8, 128, 32, 16, 64, 0, 0, 1, 0, 4, 2 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 32, 8, 8, 8, 64, 32, 16, 64, 1, 1, 1, 1, 4, 2 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 16, 2, 16, 8, 32, 8, 16, 128, 1, 1, 1, 1, 2, 4 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 32, 2, 16, 16, 64, 16, 8, 64, 1, 1, 1, 0, 2, 4 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4 } },
        { "Iris",                                            { 16, 8, 16, 8, 128, 32, 16, 64, 1, 1, 1, 1, 4, 1 } },
        { "Iris Pro",                                        { 16, 2, 16, 8, 64, 32, 32, 128, 1, 1, 1, 0, 4, 4 } },
        { "default",                                         { 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 32, 2, 32, 32, 32, 32, 8, 128, 0, 0, 1, 0, 1, 4 } },
        { "default",                                         { 32, 2, 32, 32, 32, 32, 8, 128, 0, 0, 1, 0, 1, 4 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 16, 2, 16, 8, 32, 8, 16, 64, 1, 1, 1, 1, 2, 4 } },
        { "GeForce GT 650M",                                 { 32, 2, 8, 8, 32, 32, 32, 64, 1, 1, 0, 0, 4, 2 } },
        { "GeForce GTX 1070",                                { 16, 2, 32, 16, 128, 32, 8, 128, 1, 1, 1, 0, 4, 1 } },
        { "GeForce GTX 1080",                                { 32, 2, 16, 8, 64, 8, 8, 64, 1, 1, 1, 1, 4, 8 } },
        { "GeForce GTX 480",                                 { 16, 2, 16, 8, 64, 32, 16, 64, 1, 1, 1, 1, 2, 2 } },
        { "GeForce GTX 670",                                 { 16, 2, 8, 8, 64, 16, 16, 64, 1, 1, 1, 0, 2, 4 } },
        { "GeForce GTX 680",                                 { 32, 8, 8, 16, 64, 32, 16, 128, 1, 1, 0, 0, 4, 2 } },
        { "GeForce GTX 750",                                 { 16, 2, 16, 16, 64, 32, 8, 128, 1, 1, 1, 1, 1, 2 } },
        { "GeForce GTX 750 Ti",                              { 16, 2, 16, 16, 128, 32, 8, 64, 1, 1, 0, 1, 8, 2 } },
        { "GeForce GTX 980",                                 { 16, 2, 16, 16, 64, 16, 8, 128, 1, 1, 1, 0, 4, 8 } },
        { "GeForce GTX TITAN",                               { 16, 8, 32, 16, 64, 8, 8, 64, 1, 1, 1, 0, 2, 2 } },
        { "GeForce GTX TITAN Black",                         { 16, 2, 16, 8, 64, 16, 16, 64, 1, 1, 1, 0, 4, 1 } },
        { "GeForce GTX TITAN X",                             { 16, 2, 8, 16, 128, 8, 8, 128, 1, 1, 1, 1, 4, 8 } },
        { "TITAN X (Pascal)",                                { 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 1 } },
        { "Tesla K20m",                                      { 16, 2, 32, 16, 64, 16, 8, 64, 1, 1, 1, 0, 2, 4 } },
        { "Tesla K40m",                                      { 16, 8, 16, 8, 64, 16, 16, 128, 1, 1, 1, 0, 2, 4 } },
        { "default",                                         { 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 2 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmComplexSingle = {
  "Xgemm", Precision::kComplexSingle, {"KWG", "KWI", "MDIMA", "MDIMC", "MWG", "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 32, 2, 32, 32, 64, 8, 8, 64, 0, 0, 1, 1, 2, 8 } },
        { "ATI Radeon HD 6750M",                             { 32, 2, 8, 8, 32, 16, 16, 64, 1, 1, 0, 0, 4, 4 } },
        { "Ellesmere",                                       { 32, 2, 16, 16, 32, 8, 8, 32, 1, 1, 0, 0, 1, 4 } },
        { "Fiji",                                            { 32, 2, 16, 16, 32, 16, 16, 32, 1, 1, 0, 0, 1, 2 } },
        { "Hawaii",                                          { 32, 2, 32, 8, 32, 8, 16, 32, 1, 0, 1, 0, 1, 1 } },
        { "Oland",                                           { 32, 2, 16, 8, 32, 32, 32, 128, 1, 0, 0, 1, 2, 4 } },
        { "Pitcairn",                                        { 16, 2, 8, 8, 32, 8, 8, 32, 0, 1, 1, 1, 4, 2 } },
        { "Tahiti",                                          { 16, 2, 8, 8, 32, 8, 16, 32, 1, 0, 0, 1, 2, 1 } },
        { "Tonga",                                           { 16, 2, 32, 8, 64, 16, 32, 64, 1, 1, 1, 0, 2, 1 } },
        { "Turks",                                           { 16, 2, 8, 8, 32, 32, 8, 32, 0, 1, 0, 0, 2, 1 } },
        { "default",                                         { 32, 2, 16, 16, 32, 16, 16, 32, 1, 1, 0, 0, 1, 2 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 16, 2, 8, 16, 128, 16, 8, 128, 0, 0, 0, 1, 8, 1 } },
        { "default",                                         { 16, 2, 8, 16, 128, 16, 8, 128, 0, 0, 0, 1, 8, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 16, 2, 32, 8, 128, 16, 16, 128, 1, 1, 0, 1, 1, 2 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 32, 2, 32, 16, 32, 16, 16, 64, 0, 1, 1, 0, 1, 2 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 32, 2, 16, 16, 64, 8, 16, 64, 0, 1, 0, 0, 4, 4 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 32, 2, 8, 8, 128, 16, 32, 128, 0, 0, 0, 0, 1, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 32, 2, 8, 8, 128, 32, 8, 128, 0, 0, 0, 0, 1, 4 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 32, 2, 8, 16, 16, 16, 16, 128, 0, 0, 1, 1, 1, 4 } },
        { "default",                                         { 32, 2, 16, 16, 64, 8, 8, 32, 0, 0, 0, 0, 4, 2 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 16, 8, 8, 16, 64, 32, 8, 32, 0, 0, 0, 0, 2, 1 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 16, 8, 8, 8, 32, 16, 16, 64, 1, 0, 0, 0, 4, 4 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 32, 8, 16, 16, 64, 16, 16, 64, 1, 1, 1, 1, 2, 1 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 32, 2, 16, 16, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 32, 2, 16, 16, 64, 16, 16, 64, 0, 0, 0, 0, 4, 2 } },
        { "Iris",                                            { 32, 8, 32, 16, 64, 8, 16, 64, 1, 0, 1, 0, 1, 1 } },
        { "Iris Pro",                                        { 16, 2, 8, 8, 32, 32, 8, 32, 1, 1, 1, 1, 1, 1 } },
        { "default",                                         { 32, 2, 8, 8, 32, 8, 8, 32, 1, 1, 0, 0, 4, 2 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 32, 2, 32, 32, 32, 32, 16, 128, 1, 0, 0, 0, 1, 4 } },
        { "default",                                         { 32, 2, 32, 32, 32, 32, 16, 128, 1, 0, 0, 0, 1, 4 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 16, 8, 32, 32, 64, 32, 16, 128, 1, 0, 1, 0, 1, 4 } },
        { "GeForce GTX 1070",                                { 16, 2, 16, 16, 128, 16, 16, 64, 1, 1, 1, 1, 2, 4 } },
        { "GeForce GTX 1080",                                { 16, 2, 32, 16, 64, 32, 8, 64, 1, 1, 0, 0, 1, 2 } },
        { "GeForce GTX 480",                                 { 16, 2, 16, 16, 32, 32, 16, 128, 0, 1, 1, 1, 2, 2 } },
        { "GeForce GTX 670",                                 { 16, 2, 32, 32, 64, 32, 8, 32, 1, 1, 1, 1, 1, 1 } },
        { "GeForce GTX 680",                                 { 16, 2, 32, 16, 64, 32, 32, 128, 1, 0, 0, 0, 2, 2 } },
        { "GeForce GTX 750",                                 { 16, 8, 16, 16, 64, 16, 16, 64, 1, 1, 1, 0, 2, 2 } },
        { "GeForce GTX 750 Ti",                              { 16, 2, 16, 8, 32, 32, 16, 64, 1, 1, 1, 0, 1, 2 } },
        { "GeForce GTX 980",                                 { 32, 8, 32, 32, 64, 16, 16, 64, 1, 1, 1, 0, 2, 1 } },
        { "GeForce GTX TITAN",                               { 16, 8, 16, 16, 64, 32, 16, 64, 1, 1, 1, 0, 1, 1 } },
        { "GeForce GTX TITAN Black",                         { 16, 2, 8, 16, 64, 8, 8, 32, 0, 1, 1, 0, 1, 2 } },
        { "GeForce GTX TITAN X",                             { 16, 2, 8, 8, 64, 8, 8, 32, 1, 0, 1, 1, 1, 4 } },
        { "TITAN X (Pascal)",                                { 32, 2, 32, 32, 64, 8, 8, 32, 1, 1, 0, 0, 2, 4 } },
        { "Tesla K20m",                                      { 32, 2, 8, 16, 64, 8, 16, 64, 1, 0, 0, 0, 1, 4 } },
        { "Tesla K40m",                                      { 16, 2, 32, 32, 32, 32, 8, 64, 0, 1, 0, 0, 1, 1 } },
        { "default",                                         { 32, 2, 8, 8, 16, 32, 32, 64, 1, 1, 0, 0, 1, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32, 2, 16, 16, 32, 8, 8, 32, 1, 1, 0, 0, 2, 1 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmDouble = {
  "Xgemm", Precision::kDouble, {"KWG", "KWI", "MDIMA", "MDIMC", "MWG", "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 32, 2, 16, 16, 64, 8, 8, 32, 0, 0, 0, 0, 4, 4 } },
        { "Ellesmere",                                       { 32, 2, 16, 16, 32, 16, 16, 64, 1, 1, 0, 0, 2, 2 } },
        { "Fiji",                                            { 32, 2, 16, 16, 32, 16, 16, 32, 1, 1, 0, 0, 2, 2 } },
        { "Hawaii",                                          { 16, 8, 32, 8, 128, 8, 8, 32, 0, 1, 0, 0, 1, 4 } },
        { "Oland",                                           { 16, 2, 8, 16, 64, 16, 8, 16, 0, 0, 1, 1, 1, 1 } },
        { "Pitcairn",                                        { 32, 2, 32, 16, 64, 8, 16, 32, 0, 0, 0, 0, 1, 2 } },
        { "Tahiti",                                          { 32, 2, 16, 8, 16, 8, 8, 32, 0, 0, 0, 1, 1, 4 } },
        { "Tonga",                                           { 32, 2, 16, 16, 32, 16, 16, 32, 1, 1, 0, 0, 2, 2 } },
        { "default",                                         { 32, 2, 16, 16, 32, 16, 16, 32, 1, 1, 0, 0, 2, 2 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 32, 2, 8, 8, 64, 8, 8, 16, 0, 1, 1, 0, 8, 2 } },
        { "default",                                         { 32, 2, 8, 8, 64, 8, 8, 16, 0, 1, 1, 0, 8, 2 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 16, 2, 32, 8, 128, 16, 16, 128, 1, 1, 1, 1, 2, 8 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 32, 2, 32, 16, 128, 16, 16, 64, 0, 1, 1, 0, 1, 2 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 32, 2, 32, 16, 128, 16, 16, 128, 0, 0, 1, 0, 1, 2 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 32, 2, 16, 8, 128, 8, 8, 64, 1, 0, 0, 1, 2, 8 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 32, 2, 16, 8, 128, 8, 8, 128, 1, 0, 0, 0, 2, 8 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 32, 2, 8, 16, 128, 16, 8, 128, 0, 0, 1, 1, 1, 8 } },
        { "default",                                         { 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 1, 4 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 32, 8, 8, 16, 16, 16, 16, 128, 0, 0, 1, 0, 1, 4 } },
        { "default",                                         { 32, 8, 8, 16, 16, 16, 16, 128, 0, 0, 1, 0, 1, 4 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 16, 2, 8, 8, 16, 8, 8, 32, 1, 0, 0, 1, 2, 2 } },
        { "GeForce GTX 1070",                                { 16, 2, 8, 16, 32, 8, 8, 64, 0, 0, 1, 1, 2, 8 } },
        { "GeForce GTX 1080",                                { 32, 2, 16, 16, 32, 16, 16, 64, 0, 0, 0, 0, 2, 4 } },
        { "GeForce GTX 480",                                 { 16, 2, 8, 16, 32, 32, 8, 64, 1, 1, 1, 0, 1, 2 } },
        { "GeForce GTX 670",                                 { 32, 8, 16, 32, 128, 16, 8, 32, 0, 1, 1, 0, 1, 1 } },
        { "GeForce GTX 680",                                 { 32, 8, 8, 8, 32, 16, 32, 128, 1, 0, 0, 1, 2, 4 } },
        { "GeForce GTX 750",                                 { 32, 8, 16, 32, 64, 16, 8, 128, 0, 0, 0, 1, 2, 1 } },
        { "GeForce GTX 750 Ti",                              { 32, 2, 8, 8, 32, 16, 16, 32, 0, 0, 0, 0, 4, 2 } },
        { "GeForce GTX 980",                                 { 32, 8, 16, 8, 64, 32, 32, 128, 0, 0, 1, 0, 2, 4 } },
        { "GeForce GTX TITAN",                               { 16, 8, 16, 8, 32, 16, 32, 128, 1, 1, 1, 1, 2, 2 } },
        { "GeForce GTX TITAN Black",                         { 16, 2, 16, 8, 16, 16, 8, 16, 1, 1, 1, 0, 1, 1 } },
        { "GeForce GTX TITAN X",                             { 16, 8, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 1, 1 } },
        { "TITAN X (Pascal)",                                { 32, 2, 32, 32, 32, 16, 16, 32, 0, 0, 0, 0, 1, 2 } },
        { "Tesla K20m",                                      { 16, 2, 32, 8, 32, 16, 16, 64, 1, 0, 0, 0, 1, 1 } },
        { "Tesla K40m",                                      { 32, 2, 16, 8, 64, 16, 32, 128, 1, 0, 1, 1, 2, 4 } },
        { "default",                                         { 32, 2, 16, 16, 32, 16, 16, 64, 0, 0, 0, 0, 2, 4 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32, 2, 8, 8, 32, 8, 8, 64, 0, 0, 0, 0, 4, 4 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmComplexDouble = {
  "Xgemm", Precision::kComplexDouble, {"KWG", "KWI", "MDIMA", "MDIMC", "MWG", "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 32, 8, 8, 16, 32, 16, 16, 32, 0, 0, 1, 1, 2, 2 } },
        { "Ellesmere",                                       { 32, 2, 16, 16, 16, 16, 16, 16, 1, 1, 0, 0, 1, 1 } },
        { "Fiji",                                            { 32, 2, 16, 16, 16, 16, 16, 16, 1, 1, 0, 0, 1, 1 } },
        { "Hawaii",                                          { 16, 2, 16, 16, 16, 16, 16, 32, 1, 0, 0, 0, 1, 2 } },
        { "Oland",                                           { 16, 2, 16, 8, 16, 16, 32, 128, 0, 0, 0, 0, 1, 4 } },
        { "Pitcairn",                                        { 32, 2, 16, 8, 32, 8, 32, 32, 0, 1, 1, 0, 1, 1 } },
        { "Tahiti",                                          { 16, 2, 16, 8, 16, 8, 8, 16, 0, 0, 1, 0, 1, 1 } },
        { "Tonga",                                           { 16, 2, 32, 16, 32, 16, 16, 16, 1, 1, 1, 1, 1, 1 } },
        { "default",                                         { 32, 2, 16, 16, 16, 16, 16, 16, 1, 1, 0, 0, 1, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 16, 2, 8, 8, 64, 32, 8, 64, 0, 0, 1, 0, 8, 1 } },
        { "default",                                         { 16, 2, 8, 8, 64, 32, 8, 64, 0, 0, 1, 0, 8, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 16, 2, 32, 8, 64, 16, 8, 128, 0, 1, 0, 1, 2, 1 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 32, 2, 16, 32, 128, 16, 16, 64, 0, 1, 0, 0, 2, 4 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 32, 2, 16, 32, 128, 16, 8, 32, 0, 1, 0, 0, 4, 1 } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { 32, 2, 8, 8, 128, 8, 16, 128, 0, 0, 0, 1, 1, 8 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 32, 2, 8, 8, 128, 32, 8, 128, 0, 0, 0, 0, 1, 4 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 32, 8, 8, 32, 32, 8, 8, 32, 0, 1, 0, 0, 1, 2 } },
        { "default",                                         { 32, 2, 8, 8, 16, 8, 8, 32, 1, 1, 0, 0, 1, 2 } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { 32, 2, 16, 16, 16, 16, 8, 32, 0, 0, 1, 0, 1, 1 } },
        { "default",                                         { 32, 2, 16, 16, 16, 16, 8, 32, 0, 0, 1, 0, 1, 1 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 32, 8, 16, 16, 16, 8, 16, 64, 1, 0, 1, 1, 1, 1 } },
        { "GeForce GTX 1070",                                { 32, 8, 32, 16, 32, 8, 8, 32, 0, 0, 0, 1, 1, 4 } },
        { "GeForce GTX 1080",                                { 32, 2, 16, 16, 16, 8, 8, 16, 0, 0, 0, 0, 1, 2 } },
        { "GeForce GTX 480",                                 { 16, 2, 32, 32, 32, 32, 8, 32, 0, 0, 1, 0, 1, 1 } },
        { "GeForce GTX 670",                                 { 32, 8, 16, 8, 16, 16, 32, 64, 1, 0, 0, 1, 1, 2 } },
        { "GeForce GTX 680",                                 { 16, 8, 16, 8, 64, 16, 32, 32, 0, 1, 1, 0, 1, 1 } },
        { "GeForce GTX 750",                                 { 32, 2, 8, 32, 32, 8, 8, 64, 0, 0, 1, 0, 1, 4 } },
        { "GeForce GTX 750 Ti",                              { 32, 2, 8, 8, 16, 8, 8, 32, 0, 0, 0, 0, 1, 1 } },
        { "GeForce GTX 980",                                 { 16, 2, 16, 8, 32, 8, 16, 128, 0, 0, 1, 1, 2, 2 } },
        { "GeForce GTX TITAN Black",                         { 16, 2, 16, 16, 32, 16, 8, 32, 0, 1, 1, 1, 1, 1 } },
        { "GeForce GTX TITAN X",                             { 32, 8, 16, 16, 128, 16, 16, 32, 0, 0, 1, 0, 1, 1 } },
        { "TITAN X (Pascal)",                                { 32, 2, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 1, 1 } },
        { "Tesla K20m",                                      { 32, 2, 32, 8, 32, 16, 16, 64, 0, 0, 1, 0, 1, 1 } },
        { "Tesla K40m",                                      { 16, 8, 8, 8, 32, 32, 16, 32, 0, 0, 1, 0, 1, 1 } },
        { "default",                                         { 32, 2, 16, 16, 32, 16, 16, 32, 0, 0, 0, 0, 1, 1 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32, 2, 32, 32, 32, 8, 8, 32, 1, 1, 0, 0, 1, 1 } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
