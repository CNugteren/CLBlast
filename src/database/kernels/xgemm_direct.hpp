
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xgemm_Direct' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XgemmDirectHalf = {
  "XgemmDirect", Precision::kHalf, {"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { 8, 32, 8, 8, 32, 1, 1, 1, 1, 32 } },
        { "default",                                         { 8, 32, 8, 8, 32, 1, 1, 1, 1, 32 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 2, 8, 8, 8, 8, 1, 1, 1, 1, 8 } },
        { "default",                                         { 2, 8, 8, 8, 8, 1, 1, 1, 1, 8 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 2, 16, 16, 16, 16, 1, 1, 1, 1, 16 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmDirectSingle = {
  "XgemmDirect", Precision::kSingle, {"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 2, 8, 8, 8, 8, 1, 1, 2, 2, 32 } },
        { "ATI Radeon HD 6750M",                             { 8, 8, 16, 8, 8, 1, 0, 2, 2, 32 } },
        { "Ellesmere",                                       { 2, 8, 8, 32, 32, 1, 1, 2, 1, 32 } },
        { "Fiji",                                            { 2, 16, 16, 8, 8, 1, 1, 1, 1, 16 } },
        { "Tonga",                                           { 16, 16, 16, 32, 8, 0, 1, 1, 1, 32 } },
        { "Turks",                                           { 2, 8, 8, 8, 8, 1, 1, 1, 1, 16 } },
        { "default",                                         { 2, 16, 16, 8, 8, 1, 1, 1, 1, 16 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 2, 8, 8, 8, 8, 0, 0, 1, 8, 64 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 16, 16, 8, 8, 8, 0, 0, 2, 4, 32 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 2, 8, 8, 8, 8, 0, 0, 2, 2, 64 } },
        { "default",                                         { 2, 8, 8, 8, 8, 1, 1, 4, 2, 32 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 2, 8, 8, 8, 8, 1, 1, 1, 1, 8 } },
        { "Iris Pro",                                        { 2, 16, 16, 8, 8, 1, 1, 2, 4, 32 } },
        { "default",                                         { 2, 8, 8, 8, 8, 1, 1, 1, 1, 8 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GT 650M",                                 { 16, 16, 16, 8, 16, 1, 0, 2, 2, 32 } },
        { "GeForce GTX 1080",                                { 16, 16, 8, 16, 8, 1, 1, 1, 1, 32 } },
        { "GeForce GTX 750 Ti",                              { 2, 8, 8, 8, 8, 1, 1, 4, 2, 32 } },
        { "GeForce GTX TITAN Black",                         { 2, 8, 8, 16, 16, 1, 1, 4, 2, 32 } },
        { "TITAN X (Pascal)",                                { 8, 32, 8, 8, 16, 1, 1, 1, 1, 32 } },
        { "default",                                         { 2, 8, 8, 16, 16, 1, 1, 4, 2, 32 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 2, 8, 8, 8, 8, 1, 1, 4, 2, 32 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmDirectComplexSingle = {
  "XgemmDirect", Precision::kComplexSingle, {"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 2, 16, 16, 16, 16, 1, 1, 1, 1, 16 } },
        { "ATI Radeon HD 6750M",                             { 2, 8, 8, 8, 8, 1, 1, 1, 1, 8 } },
        { "Fiji",                                            { 2, 16, 16, 16, 16, 1, 1, 1, 1, 16 } },
        { "Tonga",                                           { 2, 16, 16, 16, 16, 1, 1, 2, 2, 32 } },
        { "Turks",                                           { 2, 8, 8, 8, 8, 1, 1, 2, 2, 16 } },
        { "default",                                         { 2, 16, 16, 16, 16, 1, 1, 2, 2, 32 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 2, 8, 8, 8, 8, 0, 0, 4, 4, 32 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 2, 16, 16, 8, 8, 1, 1, 1, 4, 32 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 2, 8, 8, 16, 8, 1, 1, 2, 1, 32 } },
        { "default",                                         { 2, 8, 8, 8, 8, 1, 1, 4, 4, 32 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 2, 16, 16, 16, 16, 1, 1, 1, 1, 16 } },
        { "Iris Pro",                                        { 2, 16, 16, 8, 8, 1, 1, 2, 2, 32 } },
        { "default",                                         { 2, 16, 16, 16, 16, 1, 1, 1, 1, 16 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 1080",                                { 8, 8, 16, 16, 8, 1, 1, 2, 2, 32 } },
        { "GeForce GTX 750 Ti",                              { 16, 8, 8, 16, 8, 1, 1, 2, 1, 16 } },
        { "GeForce GTX TITAN Black",                         { 2, 8, 8, 16, 16, 1, 1, 1, 1, 16 } },
        { "TITAN X (Pascal)",                                { 2, 16, 16, 8, 8, 1, 1, 1, 1, 16 } },
        { "default",                                         { 2, 8, 8, 8, 8, 1, 1, 1, 2, 16 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 2, 32, 32, 8, 8, 1, 1, 1, 1, 32 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmDirectDouble = {
  "XgemmDirect", Precision::kDouble, {"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 2, 16, 16, 8, 8, 1, 1, 1, 1, 16 } },
        { "Ellesmere",                                       { 8, 16, 16, 8, 16, 1, 1, 2, 1, 32 } },
        { "Fiji",                                            { 16, 8, 8, 8, 16, 1, 1, 1, 1, 16 } },
        { "Tonga",                                           { 2, 16, 16, 16, 16, 1, 1, 1, 1, 32 } },
        { "default",                                         { 2, 16, 16, 16, 16, 1, 1, 1, 1, 16 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 2, 8, 8, 8, 8, 1, 1, 4, 4, 32 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 8, 8, 8, 8, 8, 0, 0, 1, 4, 32 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 2, 8, 8, 8, 8, 1, 1, 4, 4, 32 } },
        { "default",                                         { 2, 8, 8, 8, 8, 1, 1, 4, 2, 32 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 1080",                                { 2, 16, 16, 8, 8, 1, 1, 1, 2, 16 } },
        { "GeForce GTX 750 Ti",                              { 2, 8, 8, 8, 8, 1, 1, 2, 4, 32 } },
        { "GeForce GTX TITAN Black",                         { 8, 16, 16, 16, 8, 1, 0, 1, 1, 16 } },
        { "TITAN X (Pascal)",                                { 2, 8, 8, 8, 8, 1, 1, 1, 1, 16 } },
        { "default",                                         { 2, 8, 8, 8, 8, 1, 1, 1, 2, 16 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 2, 8, 8, 8, 8, 1, 1, 2, 2, 16 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmDirectComplexDouble = {
  "XgemmDirect", Precision::kComplexDouble, {"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 2, 16, 16, 16, 16, 1, 1, 1, 1, 16 } },
        { "Ellesmere",                                       { 16, 32, 32, 16, 8, 0, 0, 1, 1, 32 } },
        { "Fiji",                                            { 2, 16, 16, 16, 16, 1, 1, 1, 1, 16 } },
        { "Tonga",                                           { 2, 16, 16, 16, 16, 1, 1, 1, 1, 16 } },
        { "default",                                         { 2, 16, 16, 16, 16, 1, 1, 1, 1, 16 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 2, 8, 8, 32, 8, 0, 0, 1, 1, 32 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 8, 16, 16, 8, 8, 0, 0, 2, 1, 32 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 8, 16, 8, 8, 8, 0, 0, 2, 2, 32 } },
        { "default",                                         { 2, 8, 8, 8, 8, 1, 1, 2, 2, 16 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 1080",                                { 2, 16, 16, 8, 8, 1, 1, 1, 1, 16 } },
        { "GeForce GTX 750 Ti",                              { 2, 32, 32, 8, 8, 1, 1, 1, 1, 32 } },
        { "GeForce GTX TITAN Black",                         { 2, 8, 8, 8, 8, 1, 1, 1, 1, 8 } },
        { "TITAN X (Pascal)",                                { 2, 16, 16, 8, 8, 1, 1, 1, 2, 16 } },
        { "default",                                         { 2, 16, 16, 8, 8, 1, 1, 1, 1, 16 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 2, 8, 8, 8, 8, 1, 1, 1, 2, 16 } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
