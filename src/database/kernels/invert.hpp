
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// Tuning parameters for the diagonal matrix inversion kernels
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry InvertHalf = {
  "Invert", Precision::kHalf, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"INTERNAL_BLOCK_SIZE",16} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry InvertSingle = {
  "Invert", Precision::kSingle, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"INTERNAL_BLOCK_SIZE",16} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry InvertComplexSingle = {
  "Invert", Precision::kComplexSingle, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"INTERNAL_BLOCK_SIZE",16} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry InvertDouble = {
  "Invert", Precision::kDouble, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"INTERNAL_BLOCK_SIZE",16} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry InvertComplexDouble = {
  "Invert", Precision::kComplexDouble, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"INTERNAL_BLOCK_SIZE",16} } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
