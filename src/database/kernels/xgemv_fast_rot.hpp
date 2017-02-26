
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
  "XgemvFastRot", Precision::kHalf, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere",                                       { {"VW3",8}, {"WGS3",32}, {"WPT3",32} } },
        { "default",                                         { {"VW3",8}, {"WGS3",32}, {"WPT3",32} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"VW3",8}, {"WGS3",128}, {"WPT3",32} } },
        { "default",                                         { {"VW3",8}, {"WGS3",128}, {"WPT3",32} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW3",8}, {"WGS3",128}, {"WPT3",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastRotSingle = {
  "XgemvFastRot", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW3",8}, {"WGS3",64}, {"WPT3",32} } },
        { "Ellesmere",                                       { {"VW3",8}, {"WGS3",32}, {"WPT3",32} } },
        { "Tonga",                                           { {"VW3",8}, {"WGS3",128}, {"WPT3",32} } },
        { "Turks",                                           { {"VW3",8}, {"WGS3",128}, {"WPT3",16} } },
        { "default",                                         { {"VW3",8}, {"WGS3",128}, {"WPT3",32} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"VW3",4}, {"WGS3",32}, {"WPT3",32} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"VW3",4}, {"WGS3",32}, {"WPT3",32} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"VW3",8}, {"WGS3",16}, {"WPT3",8} } },
        { "default",                                         { {"VW3",4}, {"WGS3",32}, {"WPT3",32} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"VW3",8}, {"WGS3",64}, {"WPT3",32} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"VW3",4}, {"WGS3",64}, {"WPT3",16} } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { {"VW3",2}, {"WGS3",32}, {"WPT3",16} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"VW3",4}, {"WGS3",64}, {"WPT3",16} } },
        { "Iris Pro",                                        { {"VW3",4}, {"WGS3",32}, {"WPT3",16} } },
        { "default",                                         { {"VW3",4}, {"WGS3",64}, {"WPT3",16} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 1080",                                { {"VW3",8}, {"WGS3",32}, {"WPT3",32} } },
        { "GeForce GTX 750 Ti",                              { {"VW3",8}, {"WGS3",32}, {"WPT3",32} } },
        { "GeForce GTX TITAN",                               { {"VW3",1}, {"WGS3",16}, {"WPT3",16} } },
        { "GeForce GTX TITAN Black",                         { {"VW3",4}, {"WGS3",128}, {"WPT3",16} } },
        { "TITAN X (Pascal)",                                { {"VW3",8}, {"WGS3",32}, {"WPT3",32} } },
        { "default",                                         { {"VW3",8}, {"WGS3",32}, {"WPT3",32} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW3",8}, {"WGS3",32}, {"WPT3",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastRotComplexSingle = {
  "XgemvFastRot", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW3",8}, {"WGS3",16}, {"WPT3",16} } },
        { "Ellesmere",                                       { {"VW3",2}, {"WGS3",32}, {"WPT3",16} } },
        { "Tonga",                                           { {"VW3",4}, {"WGS3",32}, {"WPT3",32} } },
        { "Turks",                                           { {"VW3",4}, {"WGS3",32}, {"WPT3",8} } },
        { "default",                                         { {"VW3",8}, {"WGS3",32}, {"WPT3",32} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"VW3",4}, {"WGS3",32}, {"WPT3",32} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"VW3",4}, {"WGS3",32}, {"WPT3",32} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"VW3",4}, {"WGS3",16}, {"WPT3",16} } },
        { "default",                                         { {"VW3",4}, {"WGS3",32}, {"WPT3",32} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"VW3",2}, {"WGS3",16}, {"WPT3",16} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"VW3",4}, {"WGS3",128}, {"WPT3",8} } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { {"VW3",4}, {"WGS3",32}, {"WPT3",8} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"VW3",4}, {"WGS3",64}, {"WPT3",16} } },
        { "Iris Pro",                                        { {"VW3",4}, {"WGS3",16}, {"WPT3",16} } },
        { "default",                                         { {"VW3",2}, {"WGS3",32}, {"WPT3",8} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW3",4}, {"WGS3",64}, {"WPT3",16} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastRotDouble = {
  "XgemvFastRot", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW3",4}, {"WGS3",16}, {"WPT3",16} } },
        { "Ellesmere",                                       { {"VW3",4}, {"WGS3",16}, {"WPT3",16} } },
        { "Tonga",                                           { {"VW3",4}, {"WGS3",16}, {"WPT3",16} } },
        { "default",                                         { {"VW3",4}, {"WGS3",16}, {"WPT3",16} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"VW3",4}, {"WGS3",32}, {"WPT3",32} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"VW3",4}, {"WGS3",32}, {"WPT3",32} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"VW3",8}, {"WGS3",16}, {"WPT3",8} } },
        { "default",                                         { {"VW3",8}, {"WGS3",32}, {"WPT3",32} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 1080",                                { {"VW3",8}, {"WGS3",32}, {"WPT3",32} } },
        { "GeForce GTX 750 Ti",                              { {"VW3",4}, {"WGS3",32}, {"WPT3",16} } },
        { "GeForce GTX TITAN",                               { {"VW3",1}, {"WGS3",16}, {"WPT3",16} } },
        { "GeForce GTX TITAN Black",                         { {"VW3",1}, {"WGS3",16}, {"WPT3",16} } },
        { "TITAN X (Pascal)",                                { {"VW3",8}, {"WGS3",32}, {"WPT3",16} } },
        { "default",                                         { {"VW3",4}, {"WGS3",32}, {"WPT3",16} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW3",4}, {"WGS3",16}, {"WPT3",16} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastRotComplexDouble = {
  "XgemvFastRot", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW3",4}, {"WGS3",32}, {"WPT3",16} } },
        { "Ellesmere",                                       { {"VW3",4}, {"WGS3",16}, {"WPT3",16} } },
        { "Tonga",                                           { {"VW3",4}, {"WGS3",16}, {"WPT3",8} } },
        { "default",                                         { {"VW3",8}, {"WGS3",32}, {"WPT3",16} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"VW3",2}, {"WGS3",16}, {"WPT3",16} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"VW3",2}, {"WGS3",16}, {"WPT3",16} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"VW3",8}, {"WGS3",16}, {"WPT3",16} } },
        { "default",                                         { {"VW3",2}, {"WGS3",16}, {"WPT3",16} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW3",4}, {"WGS3",16}, {"WPT3",16} } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
