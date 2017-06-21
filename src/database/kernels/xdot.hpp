
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xdot' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XdotHalf = {
  "Xdot", Precision::kHalf, {"WGS1", "WGS2"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { 256, 64 } },
        { "default",                                         { 256, 64 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 256, 32 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 128, 32 } },
        { "default",                                         { 128, 32 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 128, 32 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XdotSingle = {
  "Xdot", Precision::kSingle, {"WGS1", "WGS2"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 128, 32 } },
        { "ATI Radeon HD 6750M",                             { 256, 32 } },
        { "Ellesmere",                                       { 128, 32 } },
        { "Fiji",                                            { 256, 32 } },
        { "Oland",                                           { 256, 32 } },
        { "Pitcairn",                                        { 128, 32 } },
        { "Tahiti",                                          { 128, 32 } },
        { "Tonga",                                           { 64, 32 } },
        { "Turks",                                           { 128, 64 } },
        { "default",                                         { 256, 32 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 32, 32 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 1024, 32 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 64, 128 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 64, 32 } },
        { "default",                                         { 64, 64 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 64, 32 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 256, 32 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 64, 32 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 512, 128 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 32, 32 } },
        { "Iris Pro",                                        { 512, 64 } },
        { "default",                                         { 64, 32 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 128, 32 } },
        { "GeForce GT 650M",                                 { 128, 64 } },
        { "GeForce GTX 1070",                                { 128, 1024 } },
        { "GeForce GTX 1080",                                { 512, 64 } },
        { "GeForce GTX 480",                                 { 512, 32 } },
        { "GeForce GTX 670",                                 { 512, 1024 } },
        { "GeForce GTX 680",                                 { 128, 128 } },
        { "GeForce GTX 750",                                 { 128, 32 } },
        { "GeForce GTX 750 Ti",                              { 64, 32 } },
        { "GeForce GTX 980",                                 { 256, 32 } },
        { "GeForce GTX TITAN Black",                         { 512, 64 } },
        { "GeForce GTX TITAN X",                             { 256, 32 } },
        { "TITAN X (Pascal)",                                { 1024, 32 } },
        { "Tesla K20m",                                      { 1024, 32 } },
        { "default",                                         { 256, 64 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 128, 32 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XdotComplexSingle = {
  "Xdot", Precision::kComplexSingle, {"WGS1", "WGS2"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 64, 32 } },
        { "ATI Radeon HD 6750M",                             { 256, 256 } },
        { "Ellesmere",                                       { 256, 32 } },
        { "Fiji",                                            { 256, 64 } },
        { "Oland",                                           { 128, 32 } },
        { "Pitcairn",                                        { 256, 32 } },
        { "Tahiti",                                          { 64, 32 } },
        { "Tonga",                                           { 256, 64 } },
        { "Turks",                                           { 128, 32 } },
        { "default",                                         { 256, 32 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 128, 64 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 1024, 32 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 64, 32 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 256, 32 } },
        { "default",                                         { 256, 32 } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { 256, 32 } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { 256, 32 } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { 32, 32 } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { 512, 32 } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { 32, 256 } },
        { "Iris Pro",                                        { 32, 32 } },
        { "default",                                         { 32, 32 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 64, 32 } },
        { "GeForce GTX 1070",                                { 128, 32 } },
        { "GeForce GTX 1080",                                { 128, 64 } },
        { "GeForce GTX 480",                                 { 512, 32 } },
        { "GeForce GTX 670",                                 { 256, 32 } },
        { "GeForce GTX 680",                                 { 128, 64 } },
        { "GeForce GTX 750",                                 { 64, 32 } },
        { "GeForce GTX 750 Ti",                              { 64, 32 } },
        { "GeForce GTX 980",                                 { 256, 64 } },
        { "GeForce GTX TITAN Black",                         { 128, 64 } },
        { "GeForce GTX TITAN X",                             { 256, 32 } },
        { "TITAN X (Pascal)",                                { 256, 32 } },
        { "Tesla K20m",                                      { 512, 32 } },
        { "default",                                         { 512, 64 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 256, 32 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XdotDouble = {
  "Xdot", Precision::kDouble, {"WGS1", "WGS2"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 64, 128 } },
        { "Ellesmere",                                       { 128, 64 } },
        { "Fiji",                                            { 256, 32 } },
        { "Oland",                                           { 256, 32 } },
        { "Pitcairn",                                        { 128, 32 } },
        { "Tahiti",                                          { 256, 32 } },
        { "Tonga",                                           { 128, 64 } },
        { "default",                                         { 128, 64 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 64, 128 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 512, 64 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 64, 64 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 256, 32 } },
        { "default",                                         { 256, 64 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 128, 32 } },
        { "GeForce GTX 1070",                                { 128, 512 } },
        { "GeForce GTX 1080",                                { 128, 128 } },
        { "GeForce GTX 480",                                 { 512, 32 } },
        { "GeForce GTX 670",                                 { 256, 32 } },
        { "GeForce GTX 680",                                 { 128, 64 } },
        { "GeForce GTX 750",                                 { 64, 256 } },
        { "GeForce GTX 750 Ti",                              { 128, 64 } },
        { "GeForce GTX 980",                                 { 128, 32 } },
        { "GeForce GTX TITAN Black",                         { 128, 64 } },
        { "GeForce GTX TITAN X",                             { 256, 32 } },
        { "TITAN X (Pascal)",                                { 128, 32 } },
        { "Tesla K20m",                                      { 512, 32 } },
        { "default",                                         { 128, 128 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 128, 64 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XdotComplexDouble = {
  "Xdot", Precision::kComplexDouble, {"WGS1", "WGS2"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { 64, 32 } },
        { "Ellesmere",                                       { 256, 32 } },
        { "Fiji",                                            { 256, 32 } },
        { "Oland",                                           { 256, 32 } },
        { "Pitcairn",                                        { 256, 32 } },
        { "Tahiti",                                          { 256, 32 } },
        { "Tonga",                                           { 128, 64 } },
        { "default",                                         { 256, 32 } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { 32, 128 } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { 1024, 32 } },
        { "Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz", { 1024, 32 } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { 32, 32 } },
        { "default",                                         { 128, 32 } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { 64, 32 } },
        { "GeForce GTX 1070",                                { 128, 64 } },
        { "GeForce GTX 1080",                                { 128, 32 } },
        { "GeForce GTX 480",                                 { 512, 32 } },
        { "GeForce GTX 670",                                 { 512, 128 } },
        { "GeForce GTX 680",                                 { 256, 64 } },
        { "GeForce GTX 750",                                 { 256, 32 } },
        { "GeForce GTX 750 Ti",                              { 64, 32 } },
        { "GeForce GTX 980",                                 { 64, 32 } },
        { "GeForce GTX TITAN Black",                         { 128, 32 } },
        { "GeForce GTX TITAN X",                             { 128, 32 } },
        { "TITAN X (Pascal)",                                { 128, 64 } },
        { "Tesla K20m",                                      { 128, 32 } },
        { "default",                                         { 128, 64 } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 256, 32 } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
