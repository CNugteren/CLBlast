
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file populates the database with best-found tuning parameters for the Transpose kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::TraSingle = {
  "Transpose", Precision::kSingle, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"TRA_DIM",16}, {"TRA_WPT",2}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0} } },
        { "Tesla K20m",       { {"TRA_DIM",16}, {"TRA_WPT",2}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0} } },
        { "Tesla K40m",       { {"TRA_DIM",16}, {"TRA_WPT",2}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"TRA_DIM",16}, {"TRA_WPT",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
        { "Iris",             { {"TRA_DIM",8}, {"TRA_WPT",4}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0} } },
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"TRA_DIM",16}, {"TRA_WPT",1}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::TraDouble = {
  "Transpose", Precision::kDouble, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"TRA_DIM",8}, {"TRA_WPT",2}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0} } },
        { "Tesla K20m",       { {"TRA_DIM",16}, {"TRA_WPT",2}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0} } },
        { "Tesla K40m",       { {"TRA_DIM",16}, {"TRA_WPT",2}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"TRA_DIM",16}, {"TRA_WPT",2}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"TRA_DIM",16}, {"TRA_WPT",1}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::TraComplexSingle = {
  "Transpose", Precision::kComplexSingle, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"TRA_DIM",16}, {"TRA_WPT",1}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0} } },
        { "Tesla K20m",       { {"TRA_DIM",16}, {"TRA_WPT",1}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0} } },
        { "Tesla K40m",       { {"TRA_DIM",16}, {"TRA_WPT",1}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"TRA_DIM",16}, {"TRA_WPT",2}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
        { "Iris",             { {"TRA_DIM",16}, {"TRA_WPT",1}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0} } },
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"TRA_DIM",16}, {"TRA_WPT",1}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::TraComplexDouble = {
  "Transpose", Precision::kComplexDouble, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"TRA_DIM",8}, {"TRA_WPT",1}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0} } },
        { "Tesla K20m",       { {"TRA_DIM",16}, {"TRA_WPT",1}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0} } },
        { "Tesla K40m",       { {"TRA_DIM",16}, {"TRA_WPT",1}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"TRA_DIM",16}, {"TRA_WPT",1}, {"TRA_PAD",0}, {"TRA_SHUFFLE",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"TRA_DIM",16}, {"TRA_WPT",1}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
