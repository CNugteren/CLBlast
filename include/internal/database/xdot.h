
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file populates the database with best-found tuning parameters for the Xdot kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::XdotSingle = {
  "Xdot", Precision::kSingle, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
        { "Iris",             { {"WGS1",512}, {"WGS2",512} } },
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"WGS1",64}, {"WGS2",64} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XdotDouble = {
  "Xdot", Precision::kDouble, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"WGS1",64}, {"WGS2",64} } },
      }
    },
  }
};
// =================================================================================================

const Database::DatabaseEntry Database::XdotComplexSingle = {
  "Xdot", Precision::kComplexSingle, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
        { "Iris",             { {"WGS1",512}, {"WGS2",512} } },
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"WGS1",64}, {"WGS2",64} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XdotComplexDouble = {
  "Xdot", Precision::kComplexDouble, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"WGS1",64}, {"WGS2",64} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
