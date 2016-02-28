
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
// =================================================================================================

const Database::DatabaseEntry Database::XgerSingle = {
  "Xger", Precision::kSingle, {
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",128}, {"WGS2",2}, {"WPT",4} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",2}, {"WPT",4} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",128}, {"WGS2",2}, {"WPT",4} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgerComplexSingle = {
  "Xger", Precision::kComplexSingle, {
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",512}, {"WGS2",8}, {"WPT",2} } },
        { "default",                                         { {"WGS1",512}, {"WGS2",8}, {"WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",512}, {"WGS2",8}, {"WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgerDouble = {
  "Xger", Precision::kDouble, {
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",512}, {"WGS2",16}, {"WPT",1} } },
        { "default",                                         { {"WGS1",512}, {"WGS2",16}, {"WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",512}, {"WGS2",16}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgerComplexDouble = {
  "Xger", Precision::kComplexDouble, {
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",512}, {"WGS2",1}, {"WPT",1} } },
        { "default",                                         { {"WGS1",512}, {"WGS2",1}, {"WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",512}, {"WGS2",1}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
