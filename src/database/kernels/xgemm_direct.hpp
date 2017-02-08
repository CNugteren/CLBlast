
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
  "XgemmDirect", Precision::kHalf, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",8} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",8} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",8} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmDirectSingle = {
  "XgemmDirect", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",2}, {"WGD",32} } },
        { "Tonga",                                           { {"KWID",16}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",32}, {"NDIMCD",8}, {"PADA",0}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",32} } },
        { "Turks",                                           { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",0}, {"PADB",0}, {"VWMD",1}, {"VWND",8}, {"WGD",64} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",0}, {"PADB",0}, {"VWMD",2}, {"VWND",2}, {"WGD",64} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",4}, {"WGD",32} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",8} } },
        { "Iris Pro",                                        { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",4}, {"WGD",32} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",8} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 1080",                                { {"KWID",16}, {"MDIMAD",16}, {"MDIMCD",8}, {"NDIMBD",16}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",32} } },
        { "GeForce GTX 750 Ti",                              { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",4}, {"VWND",2}, {"WGD",32} } },
        { "GeForce GTX TITAN Black",                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",4}, {"VWND",2}, {"WGD",32} } },
        { "TITAN X (Pascal)",                                { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",32} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",4}, {"VWND",2}, {"WGD",32} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",4}, {"VWND",2}, {"WGD",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmDirectComplexSingle = {
  "XgemmDirect", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
        { "Tonga",                                           { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",2}, {"WGD",32} } },
        { "Turks",                                           { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",2}, {"WGD",16} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",2}, {"WGD",32} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",0}, {"PADB",0}, {"VWMD",4}, {"VWND",4}, {"WGD",32} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",16}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",1}, {"WGD",32} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",4}, {"VWND",4}, {"WGD",32} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
        { "Iris Pro",                                        { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",2}, {"WGD",32} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 1080",                                { {"KWID",8}, {"MDIMAD",8}, {"MDIMCD",16}, {"NDIMBD",16}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",2}, {"WGD",32} } },
        { "GeForce GTX 750 Ti",                              { {"KWID",16}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",16}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",1}, {"WGD",16} } },
        { "GeForce GTX TITAN Black",                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
        { "TITAN X (Pascal)",                                { {"KWID",16}, {"MDIMAD",16}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",2}, {"WGD",16} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",2}, {"WGD",16} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmDirectDouble = {
  "XgemmDirect", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
        { "Tonga",                                           { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",32} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",4}, {"VWND",4}, {"WGD",32} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",4}, {"VWND",4}, {"WGD",32} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",4}, {"VWND",4}, {"WGD",32} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 1080",                                { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",2}, {"WGD",16} } },
        { "GeForce GTX 750 Ti",                              { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",4}, {"WGD",32} } },
        { "GeForce GTX TITAN Black",                         { {"KWID",8}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",16}, {"NDIMCD",8}, {"PADA",1}, {"PADB",0}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
        { "TITAN X (Pascal)",                                { {"KWID",8}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",1}, {"WGD",16} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",2}, {"WGD",16} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",2}, {"WGD",16} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmDirectComplexDouble = {
  "XgemmDirect", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
        { "Tonga",                                           { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",16}, {"NDIMCD",16}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",32}, {"NDIMCD",8}, {"PADA",0}, {"PADB",0}, {"VWMD",1}, {"VWND",1}, {"WGD",32} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"KWID",8}, {"MDIMAD",16}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",0}, {"PADB",0}, {"VWMD",2}, {"VWND",2}, {"WGD",32} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",2}, {"VWND",2}, {"WGD",16} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 1080",                                { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
        { "GeForce GTX 750 Ti",                              { {"KWID",2}, {"MDIMAD",32}, {"MDIMCD",32}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",32} } },
        { "GeForce GTX TITAN Black",                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",8} } },
        { "TITAN X (Pascal)",                                { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",16}, {"NDIMCD",8}, {"PADA",1}, {"PADB",0}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
        { "default",                                         { {"KWID",2}, {"MDIMAD",16}, {"MDIMCD",16}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"PADA",1}, {"PADB",1}, {"VWMD",1}, {"VWND",1}, {"WGD",16} } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
