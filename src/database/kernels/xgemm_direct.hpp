
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xgemm' kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::XgemmDirectHalf = {
  "XgemmDirect", Precision::kHalf, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGD",32}, {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"VWMD",1}, {"VWND",1}, {"PADA",0}, {"PADB",0} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemmDirectSingle = {
  "XgemmDirect", Precision::kSingle, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGD",32}, {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"VWMD",1}, {"VWND",1}, {"PADA",0}, {"PADB",0} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemmDirectComplexSingle = {
  "XgemmDirect", Precision::kComplexSingle, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGD",32}, {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"VWMD",1}, {"VWND",1}, {"PADA",0}, {"PADB",0} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemmDirectDouble = {
  "XgemmDirect", Precision::kDouble, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGD",32}, {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"VWMD",1}, {"VWND",1}, {"PADA",0}, {"PADB",0} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemmDirectComplexDouble = {
  "XgemmDirect", Precision::kComplexDouble, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGD",32}, {"KWID",2}, {"MDIMAD",8}, {"MDIMCD",8}, {"NDIMBD",8}, {"NDIMCD",8}, {"VWMD",1}, {"VWND",1}, {"PADA",0}, {"PADB",0} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
