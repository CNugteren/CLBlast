
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file populates the database with best-found tuning parameters for the 'Xtrsv' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XtrsvHalf = {
  "Xtrsv", Precision::kHalf, {"TRSV_BLOCK_SIZE"}, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XtrsvSingle = {
  "Xtrsv", Precision::kSingle, {"TRSV_BLOCK_SIZE"}, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XtrsvComplexSingle = {
  "Xtrsv", Precision::kComplexSingle, {"TRSV_BLOCK_SIZE"}, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XtrsvDouble = {
  "Xtrsv", Precision::kDouble, {"TRSV_BLOCK_SIZE"}, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32 } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XtrsvComplexDouble = {
  "Xtrsv", Precision::kComplexDouble, {"TRSV_BLOCK_SIZE"}, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { 32 } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
