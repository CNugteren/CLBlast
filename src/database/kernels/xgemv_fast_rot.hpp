
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xgemv_Fast_Rot' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XgemvFastRotHalf = {
  "XgemvFastRot", Precision::kHalf, {"VW3", "WGS3", "WPT3"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { 8, 32, 32 } },
        { "default",                                         { 8, 32, 32 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 8, 128, 32 } },
        { "default",                                         { 8, 128, 32 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 8, 128, 32 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastRotSingle = {
  "XgemvFastRot", Precision::kSingle, {"VW3", "WGS3", "WPT3"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 8, 64, 32 } },
        { "ATI Radeon HD 6750M",                             { 8, 128, 16 } },
        { "Ellesmere",                                       { 8, 32, 32 } },
        { "Fiji",                                            { 4, 32, 16 } },
        { "Tonga",                                           { 8, 128, 32 } },
        { "Turks",                                           { 8, 128, 16 } },
        { "default",                                         { 8, 32, 32 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 4, 32, 32 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 8, 128, 8 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 4, 32, 32 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 8, 16, 8 } },
        { "default",                                         { 8, 32, 32 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 8, 64, 32 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 4, 64, 16 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 2, 32, 16 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 4, 64, 16 } },
        { "Iris Pro",                                        { 4, 16, 16 } },
        { "default",                                         { 4, 64, 16 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GT 650M",                                 { 8, 32, 16 } },
        { "GeForce GTX 1080",                                { 8, 32, 32 } },
        { "GeForce GTX 750 Ti",                              { 8, 32, 32 } },
        { "GeForce GTX TITAN",                               { 1, 16, 16 } },
        { "GeForce GTX TITAN Black",                         { 4, 128, 16 } },
        { "TITAN X (Pascal)",                                { 8, 64, 32 } },
        { "default",                                         { 8, 32, 32 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 8, 32, 32 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastRotComplexSingle = {
  "XgemvFastRot", Precision::kComplexSingle, {"VW3", "WGS3", "WPT3"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 8, 16, 16 } },
        { "ATI Radeon HD 6750M",                             { 8, 32, 8 } },
        { "Ellesmere",                                       { 2, 32, 16 } },
        { "Fiji",                                            { 4, 32, 32 } },
        { "Tonga",                                           { 4, 32, 32 } },
        { "Turks",                                           { 4, 32, 8 } },
        { "default",                                         { 8, 16, 16 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 4, 32, 32 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 8, 32, 8 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 4, 32, 32 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 4, 16, 16 } },
        { "default",                                         { 4, 32, 32 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 2, 16, 16 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 4, 128, 8 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 4, 32, 8 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 4, 64, 16 } },
        { "Iris Pro",                                        { 4, 16, 16 } },
        { "default",                                         { 2, 32, 8 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 4, 16, 16 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastRotDouble = {
  "XgemvFastRot", Precision::kDouble, {"VW3", "WGS3", "WPT3"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 4, 16, 16 } },
        { "Ellesmere",                                       { 4, 16, 16 } },
        { "Fiji",                                            { 4, 32, 32 } },
        { "Tonga",                                           { 4, 16, 16 } },
        { "default",                                         { 4, 16, 16 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 4, 32, 32 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 8, 16, 8 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 4, 32, 32 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 8, 16, 8 } },
        { "default",                                         { 8, 32, 32 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 1080",                                { 8, 32, 32 } },
        { "GeForce GTX 750 Ti",                              { 4, 32, 16 } },
        { "GeForce GTX TITAN",                               { 1, 16, 16 } },
        { "GeForce GTX TITAN Black",                         { 1, 16, 16 } },
        { "TITAN X (Pascal)",                                { 8, 32, 32 } },
        { "default",                                         { 4, 32, 16 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 4, 16, 16 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastRotComplexDouble = {
  "XgemvFastRot", Precision::kComplexDouble, {"VW3", "WGS3", "WPT3"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 4, 32, 16 } },
        { "Ellesmere",                                       { 4, 16, 16 } },
        { "Fiji",                                            { 4, 32, 8 } },
        { "Tonga",                                           { 4, 16, 8 } },
        { "default",                                         { 8, 32, 16 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 2, 16, 16 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 4, 64, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 2, 16, 16 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 8, 16, 16 } },
        { "default",                                         { 8, 16, 16 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 4, 16, 16 } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
