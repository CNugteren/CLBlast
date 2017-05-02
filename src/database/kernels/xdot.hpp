
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
  "Xdot", Precision::kHalf, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { {"WGS1",256}, {"WGS2",64} } },
        { "default",                                         { {"WGS1",256}, {"WGS2",64} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"WGS1",256}, {"WGS2",32} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"WGS1",128}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",32} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",128}, {"WGS2",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XdotSingle = {
  "Xdot", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",128}, {"WGS2",32} } },
        { "ATI Radeon HD 6750M",                             { {"WGS1",256}, {"WGS2",32} } },
        { "Ellesmere",                                       { {"WGS1",128}, {"WGS2",32} } },
        { "Oland",                                           { {"WGS1",256}, {"WGS2",32} } },
        { "Pitcairn",                                        { {"WGS1",128}, {"WGS2",32} } },
        { "Tahiti",                                          { {"WGS1",128}, {"WGS2",32} } },
        { "Tonga",                                           { {"WGS1",64}, {"WGS2",32} } },
        { "Turks",                                           { {"WGS1",128}, {"WGS2",64} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",32} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"WGS1",32}, {"WGS2",32} } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",1024}, {"WGS2",32} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"WGS1",64}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",32} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"WGS1",64}, {"WGS2",32} } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"WGS1",256}, {"WGS2",32} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"WGS1",64}, {"WGS2",32} } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { {"WGS1",512}, {"WGS2",128} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"WGS1",32}, {"WGS2",32} } },
        { "Iris Pro",                                        { {"WGS1",512}, {"WGS2",64} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",32} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"WGS1",128}, {"WGS2",32} } },
        { "GeForce GTX 1070",                                { {"WGS1",128}, {"WGS2",1024} } },
        { "GeForce GTX 1080",                                { {"WGS1",512}, {"WGS2",64} } },
        { "GeForce GTX 480",                                 { {"WGS1",512}, {"WGS2",32} } },
        { "GeForce GTX 670",                                 { {"WGS1",512}, {"WGS2",1024} } },
        { "GeForce GTX 680",                                 { {"WGS1",128}, {"WGS2",128} } },
        { "GeForce GTX 750",                                 { {"WGS1",128}, {"WGS2",32} } },
        { "GeForce GTX 750 Ti",                              { {"WGS1",64}, {"WGS2",32} } },
        { "GeForce GTX 980",                                 { {"WGS1",256}, {"WGS2",32} } },
        { "GeForce GTX TITAN Black",                         { {"WGS1",512}, {"WGS2",64} } },
        { "GeForce GTX TITAN X",                             { {"WGS1",256}, {"WGS2",32} } },
        { "TITAN X (Pascal)",                                { {"WGS1",1024}, {"WGS2",32} } },
        { "Tesla K20m",                                      { {"WGS1",1024}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",256}, {"WGS2",64} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",128}, {"WGS2",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XdotComplexSingle = {
  "Xdot", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WGS2",32} } },
        { "ATI Radeon HD 6750M",                             { {"WGS1",256}, {"WGS2",256} } },
        { "Ellesmere",                                       { {"WGS1",256}, {"WGS2",32} } },
        { "Oland",                                           { {"WGS1",128}, {"WGS2",32} } },
        { "Pitcairn",                                        { {"WGS1",256}, {"WGS2",32} } },
        { "Tahiti",                                          { {"WGS1",64}, {"WGS2",32} } },
        { "Tonga",                                           { {"WGS1",256}, {"WGS2",64} } },
        { "Turks",                                           { {"WGS1",128}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",256}, {"WGS2",32} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"WGS1",128}, {"WGS2",64} } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",1024}, {"WGS2",32} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"WGS1",256}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",256}, {"WGS2",32} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"WGS1",256}, {"WGS2",32} } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"WGS1",256}, {"WGS2",32} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"WGS1",32}, {"WGS2",32} } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { {"WGS1",512}, {"WGS2",32} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"WGS1",32}, {"WGS2",256} } },
        { "Iris Pro",                                        { {"WGS1",32}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",32}, {"WGS2",32} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"WGS1",64}, {"WGS2",32} } },
        { "GeForce GTX 1070",                                { {"WGS1",128}, {"WGS2",32} } },
        { "GeForce GTX 1080",                                { {"WGS1",128}, {"WGS2",64} } },
        { "GeForce GTX 480",                                 { {"WGS1",512}, {"WGS2",32} } },
        { "GeForce GTX 670",                                 { {"WGS1",256}, {"WGS2",32} } },
        { "GeForce GTX 680",                                 { {"WGS1",128}, {"WGS2",64} } },
        { "GeForce GTX 750",                                 { {"WGS1",64}, {"WGS2",32} } },
        { "GeForce GTX 750 Ti",                              { {"WGS1",64}, {"WGS2",32} } },
        { "GeForce GTX 980",                                 { {"WGS1",256}, {"WGS2",64} } },
        { "GeForce GTX TITAN Black",                         { {"WGS1",128}, {"WGS2",64} } },
        { "GeForce GTX TITAN X",                             { {"WGS1",256}, {"WGS2",32} } },
        { "TITAN X (Pascal)",                                { {"WGS1",256}, {"WGS2",32} } },
        { "Tesla K20m",                                      { {"WGS1",512}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",512}, {"WGS2",64} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",256}, {"WGS2",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XdotDouble = {
  "Xdot", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WGS2",128} } },
        { "Ellesmere",                                       { {"WGS1",128}, {"WGS2",64} } },
        { "Oland",                                           { {"WGS1",256}, {"WGS2",32} } },
        { "Pitcairn",                                        { {"WGS1",128}, {"WGS2",32} } },
        { "Tahiti",                                          { {"WGS1",256}, {"WGS2",32} } },
        { "Tonga",                                           { {"WGS1",128}, {"WGS2",64} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",64} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"WGS1",64}, {"WGS2",128} } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",512}, {"WGS2",64} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"WGS1",256}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",256}, {"WGS2",64} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"WGS1",128}, {"WGS2",32} } },
        { "GeForce GTX 1070",                                { {"WGS1",128}, {"WGS2",512} } },
        { "GeForce GTX 1080",                                { {"WGS1",128}, {"WGS2",128} } },
        { "GeForce GTX 480",                                 { {"WGS1",512}, {"WGS2",32} } },
        { "GeForce GTX 670",                                 { {"WGS1",256}, {"WGS2",32} } },
        { "GeForce GTX 680",                                 { {"WGS1",128}, {"WGS2",64} } },
        { "GeForce GTX 750",                                 { {"WGS1",64}, {"WGS2",256} } },
        { "GeForce GTX 750 Ti",                              { {"WGS1",128}, {"WGS2",64} } },
        { "GeForce GTX 980",                                 { {"WGS1",128}, {"WGS2",32} } },
        { "GeForce GTX TITAN Black",                         { {"WGS1",128}, {"WGS2",64} } },
        { "GeForce GTX TITAN X",                             { {"WGS1",256}, {"WGS2",32} } },
        { "TITAN X (Pascal)",                                { {"WGS1",128}, {"WGS2",32} } },
        { "Tesla K20m",                                      { {"WGS1",512}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",128} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",128}, {"WGS2",64} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XdotComplexDouble = {
  "Xdot", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WGS2",32} } },
        { "Ellesmere",                                       { {"WGS1",256}, {"WGS2",32} } },
        { "Oland",                                           { {"WGS1",256}, {"WGS2",32} } },
        { "Pitcairn",                                        { {"WGS1",256}, {"WGS2",32} } },
        { "Tahiti",                                          { {"WGS1",256}, {"WGS2",32} } },
        { "Tonga",                                           { {"WGS1",128}, {"WGS2",64} } },
        { "default",                                         { {"WGS1",256}, {"WGS2",32} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"WGS1",32}, {"WGS2",128} } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",1024}, {"WGS2",32} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"WGS1",32}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",32} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"WGS1",64}, {"WGS2",32} } },
        { "GeForce GTX 1070",                                { {"WGS1",128}, {"WGS2",64} } },
        { "GeForce GTX 1080",                                { {"WGS1",128}, {"WGS2",32} } },
        { "GeForce GTX 480",                                 { {"WGS1",512}, {"WGS2",32} } },
        { "GeForce GTX 670",                                 { {"WGS1",512}, {"WGS2",128} } },
        { "GeForce GTX 680",                                 { {"WGS1",256}, {"WGS2",64} } },
        { "GeForce GTX 750",                                 { {"WGS1",256}, {"WGS2",32} } },
        { "GeForce GTX 750 Ti",                              { {"WGS1",64}, {"WGS2",32} } },
        { "GeForce GTX 980",                                 { {"WGS1",64}, {"WGS2",32} } },
        { "GeForce GTX TITAN Black",                         { {"WGS1",128}, {"WGS2",32} } },
        { "GeForce GTX TITAN X",                             { {"WGS1",128}, {"WGS2",32} } },
        { "TITAN X (Pascal)",                                { {"WGS1",128}, {"WGS2",64} } },
        { "Tesla K20m",                                      { {"WGS1",128}, {"WGS2",32} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",64} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",128}, {"WGS2",32} } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
