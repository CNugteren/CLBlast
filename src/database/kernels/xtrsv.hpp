
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
  "Xtrsv", Precision::kHalf, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"TRSV_BLOCK_SIZE",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XtrsvSingle = {
  "Xtrsv", Precision::kSingle, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"TRSV_BLOCK_SIZE",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XtrsvComplexSingle = {
  "Xtrsv", Precision::kComplexSingle, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"TRSV_BLOCK_SIZE",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XtrsvDouble = {
  "Xtrsv", Precision::kDouble, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"TRSV_BLOCK_SIZE",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XtrsvComplexDouble = {
  "Xtrsv", Precision::kComplexDouble, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"TRSV_BLOCK_SIZE",32} } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast
