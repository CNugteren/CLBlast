
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xger' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XgerHalf = {
  "Xger", Precision::kHalf, {"WGS1", "WGS2", "WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { 64, 1, 2 } },
        { "default",                                         { 64, 1, 2 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 256, 1, 2 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 64, 1, 4 } },
        { "default",                                         { 4, 8, 2 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 64, 1, 2 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgerSingle = {
  "Xger", Precision::kSingle, {"WGS1", "WGS2", "WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 256, 1, 1 } },
        { "ATI Radeon HD 6750M",                             { 16, 16, 4 } },
        { "Ellesmere",                                       { 64, 4, 2 } },
        { "Fiji",                                            { 256, 1, 1 } },
        { "Hawaii",                                          { 64, 2, 1 } },
        { "Oland",                                           { 32, 4, 2 } },
        { "Pitcairn",                                        { 64, 1, 1 } },
        { "Tahiti",                                          { 256, 1, 1 } },
        { "Tonga",                                           { 256, 1, 2 } },
        { "Turks",                                           { 64, 4, 2 } },
        { "default",                                         { 16, 16, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 64, 4, 4 } },
        { "default",                                         { 64, 4, 4 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 32, 4, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 128, 2, 4 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 256, 16, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 256, 4, 4 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 128, 1, 4 } },
        { "default",                                         { 128, 8, 4 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 32, 1, 2 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 256, 2, 2 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 128, 1, 2 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 64, 1, 4 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 32, 4, 4 } },
        { "Iris Pro",                                        { 64, 1, 4 } },
        { "default",                                         { 32, 4, 2 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 128, 1, 2 } },
        { "GeForce GT 650M",                                 { 32, 16, 4 } },
        { "GeForce GTX 1070",                                { 512, 1, 1 } },
        { "GeForce GTX 1080",                                { 16, 4, 1 } },
        { "GeForce GTX 480",                                 { 256, 1, 4 } },
        { "GeForce GTX 670",                                 { 32, 8, 2 } },
        { "GeForce GTX 680",                                 { 128, 1, 4 } },
        { "GeForce GTX 750",                                 { 64, 16, 4 } },
        { "GeForce GTX 750 Ti",                              { 64, 1, 2 } },
        { "GeForce GTX TITAN",                               { 32, 4, 2 } },
        { "GeForce GTX TITAN Black",                         { 32, 4, 2 } },
        { "TITAN X (Pascal)",                                { 512, 2, 1 } },
        { "default",                                         { 128, 1, 2 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32, 4, 2 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgerComplexSingle = {
  "Xger", Precision::kComplexSingle, {"WGS1", "WGS2", "WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 64, 4, 1 } },
        { "ATI Radeon HD 6750M",                             { 16, 16, 1 } },
        { "Ellesmere",                                       { 16, 8, 2 } },
        { "Fiji",                                            { 128, 2, 1 } },
        { "Hawaii",                                          { 64, 1, 2 } },
        { "Oland",                                           { 4, 8, 1 } },
        { "Pitcairn",                                        { 128, 2, 1 } },
        { "Tahiti",                                          { 64, 2, 1 } },
        { "Tonga",                                           { 64, 1, 1 } },
        { "Turks",                                           { 128, 2, 1 } },
        { "default",                                         { 128, 2, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 128, 1, 1 } },
        { "default",                                         { 128, 1, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 128, 2, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 256, 1, 4 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 256, 8, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 256, 2, 4 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 512, 4, 2 } },
        { "default",                                         { 256, 2, 4 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 32, 1, 2 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 128, 2, 1 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 512, 1, 1 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 256, 1, 2 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 16, 1, 1 } },
        { "Iris Pro",                                        { 16, 2, 4 } },
        { "default",                                         { 128, 2, 2 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 64, 4, 2 } },
        { "GeForce GTX 1070",                                { 16, 64, 2 } },
        { "GeForce GTX 1080",                                { 32, 2, 1 } },
        { "GeForce GTX 480",                                 { 128, 2, 2 } },
        { "GeForce GTX 670",                                 { 16, 32, 2 } },
        { "GeForce GTX 680",                                 { 32, 4, 2 } },
        { "GeForce GTX 750",                                 { 32, 16, 4 } },
        { "GeForce GTX 750 Ti",                              { 32, 8, 2 } },
        { "GeForce GTX TITAN",                               { 16, 16, 2 } },
        { "GeForce GTX TITAN Black",                         { 16, 16, 2 } },
        { "TITAN X (Pascal)",                                { 32, 2, 1 } },
        { "default",                                         { 128, 2, 2 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 64, 2, 2 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgerDouble = {
  "Xger", Precision::kDouble, {"WGS1", "WGS2", "WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 32, 4, 1 } },
        { "Ellesmere",                                       { 64, 1, 4 } },
        { "Fiji",                                            { 256, 1, 2 } },
        { "Hawaii",                                          { 32, 4, 2 } },
        { "Oland",                                           { 128, 1, 2 } },
        { "Pitcairn",                                        { 64, 1, 1 } },
        { "Tahiti",                                          { 64, 2, 1 } },
        { "Tonga",                                           { 8, 16, 2 } },
        { "default",                                         { 128, 2, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 64, 4, 1 } },
        { "default",                                         { 64, 4, 1 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 256, 1, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 512, 16, 1 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 256, 1, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 256, 4, 4 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 512, 8, 2 } },
        { "default",                                         { 256, 1, 4 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 128, 8, 2 } },
        { "GeForce GTX 1070",                                { 32, 8, 1 } },
        { "GeForce GTX 1080",                                { 32, 2, 1 } },
        { "GeForce GTX 480",                                 { 32, 4, 2 } },
        { "GeForce GTX 670",                                 { 32, 32, 2 } },
        { "GeForce GTX 680",                                 { 128, 4, 2 } },
        { "GeForce GTX 750",                                 { 256, 2, 2 } },
        { "GeForce GTX 750 Ti",                              { 32, 16, 1 } },
        { "GeForce GTX TITAN",                               { 16, 8, 2 } },
        { "GeForce GTX TITAN Black",                         { 32, 4, 2 } },
        { "TITAN X (Pascal)",                                { 32, 2, 1 } },
        { "default",                                         { 128, 1, 2 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 128, 1, 2 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgerComplexDouble = {
  "Xger", Precision::kComplexDouble, {"WGS1", "WGS2", "WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 64, 1, 1 } },
        { "Ellesmere",                                       { 8, 16, 1 } },
        { "Fiji",                                            { 64, 4, 2 } },
        { "Hawaii",                                          { 128, 1, 1 } },
        { "Oland",                                           { 16, 16, 2 } },
        { "Pitcairn",                                        { 64, 4, 1 } },
        { "Tahiti",                                          { 32, 4, 1 } },
        { "Tonga",                                           { 16, 4, 1 } },
        { "default",                                         { 32, 4, 1 } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { 64, 2, 4 } },
        { "default",                                         { 64, 2, 4 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 128, 4, 4 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 512, 4, 2 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 256, 8, 4 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 512, 2, 2 } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { 256, 1, 2 } },
        { "default",                                         { 256, 2, 2 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 16, 8, 2 } },
        { "GeForce GTX 1070",                                { 8, 128, 1 } },
        { "GeForce GTX 1080",                                { 8, 4, 1 } },
        { "GeForce GTX 480",                                 { 64, 2, 2 } },
        { "GeForce GTX 670",                                 { 8, 16, 2 } },
        { "GeForce GTX 680",                                 { 8, 16, 1 } },
        { "GeForce GTX 750",                                 { 8, 32, 4 } },
        { "GeForce GTX 750 Ti",                              { 32, 8, 2 } },
        { "GeForce GTX TITAN",                               { 32, 4, 2 } },
        { "GeForce GTX TITAN Black",                         { 16, 16, 2 } },
        { "TITAN X (Pascal)",                                { 4, 8, 1 } },
        { "default",                                         { 16, 8, 2 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 64, 2, 2 } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
