
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

const Database::DatabaseEntry Database::XgemmSingle = {
  "Xgemm", Precision::kSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"KWG",16}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",4}, {"VWN",1} } },
        { "default",                                       { {"KWG",16}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",4}, {"VWN",1} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"KWG",16}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",4} } },
        { "default",                                       { {"KWG",16}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",4} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"KWG",16}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemmComplexSingle = {
  "Xgemm", Precision::kComplexSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"KWG",32}, {"KWI",8}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "default",                                       { {"KWG",32}, {"KWI",8}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",64}, {"SA",0}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "default",                                       { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",64}, {"SA",0}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemmDouble = {
  "Xgemm", Precision::kDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",32}, {"NWG",128}, {"SA",1}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",4} } },
        { "default",                                       { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",32}, {"NWG",128}, {"SA",1}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",4} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",32}, {"NWG",128}, {"SA",1}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",4} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemmComplexDouble = {
  "Xgemm", Precision::kComplexDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"KWG",16}, {"KWI",8}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "default",                                       { {"KWG",16}, {"KWI",8}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"KWG",16}, {"KWI",8}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
