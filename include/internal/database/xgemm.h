
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file populates the database with best-found tuning parameters for the Xgemm kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::XgemmSingle = {
  "Xgemm", Precision::kSingle, {
    { // NVIDIA GPUs
      CL_DEVICE_TYPE_GPU, "NVIDIA Corporation", {
        { "GeForce GTX 480",  { {"MWG",128}, {"NWG",64}, {"KWG",32}, {"MDIMC",16}, {"NDIMC",16}, {"MDIMA",16}, {"NDIMB",16}, {"KWI",2}, {"VWM",2}, {"VWN",2}, {"STRM",1}, {"STRN",0}, {"SA",1}, {"SB",1} } },
        { "Tesla K20m",       { {"MWG",128}, {"NWG",64}, {"KWG",16}, {"MDIMC",16}, {"NDIMC",8}, {"MDIMA",16}, {"NDIMB",32}, {"KWI",2}, {"VWM",4}, {"VWN",1}, {"STRM",1}, {"STRN",0}, {"SA",1}, {"SB",1} } },
        { "Tesla K40m",       { {"MWG",128}, {"NWG",128}, {"KWG",16}, {"MDIMC",16}, {"NDIMC",16}, {"MDIMA",32}, {"NDIMB",16}, {"KWI",8}, {"VWM",2}, {"VWN",1}, {"STRM",1}, {"STRN",0}, {"SA",1}, {"SB",1} } },
        { kDefault,           { {"MWG",128}, {"NWG",64}, {"KWG",16}, {"MDIMC",16}, {"NDIMC",16}, {"MDIMA",16}, {"NDIMB",16}, {"KWI",2}, {"VWM",2}, {"VWN",1}, {"STRM",1}, {"STRN",0}, {"SA",1}, {"SB",1} } },
      }
    },
    { // AMD GPUs
      CL_DEVICE_TYPE_GPU, "AMD", {
        { "Tahiti",           { {"MWG",128}, {"NWG",128}, {"KWG",32}, {"MDIMC",16}, {"NDIMC",16}, {"MDIMA",32}, {"NDIMB",8}, {"KWI",2}, {"VWM",4}, {"VWN",4}, {"STRM",1}, {"STRN",1}, {"SA",1}, {"SB",1} } },
      }
    },
    { // Intel GPUs
      CL_DEVICE_TYPE_GPU, "Intel", {
        { "Iris",             { {"MWG",64}, {"NWG",64}, {"KWG",32}, {"MDIMC",8}, {"NDIMC",8}, {"MDIMA",16}, {"NDIMB",8}, {"KWI",8}, {"VWM",4}, {"VWN",4}, {"STRM",1}, {"STRN",0}, {"SA",1}, {"SB",0} } },
      }
    },
    { // Default
      CL_DEVICE_TYPE_ALL, kDefault, {
        { kDefault,           { {"MWG",32}, {"NWG",32}, {"KWG",16}, {"MDIMC",8}, {"NDIMC",8}, {"MDIMA",16}, {"NDIMB",16}, {"KWI",1}, {"VWM",1}, {"VWN",1}, {"STRM",1}, {"STRN",1}, {"SA",0}, {"SB",0} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemmDouble = {
  "Xgemm", Precision::kDouble, {
    { // NVIDIA GPUs
      CL_DEVICE_TYPE_GPU, "NVIDIA Corporation", {
        { "GeForce GTX 480",  { {"MWG",32}, {"NWG",64}, {"KWG",16}, {"MDIMC",8}, {"NDIMC",16}, {"MDIMA",16}, {"NDIMB",32}, {"KWI",2}, {"VWM",1}, {"VWN",2}, {"STRM",1}, {"STRN",0}, {"SA",1}, {"SB",1} } },
        { "Tesla K20m",       { {"MWG",64}, {"NWG",128}, {"KWG",16}, {"MDIMC",8}, {"NDIMC",32}, {"MDIMA",32}, {"NDIMB",32}, {"KWI",8}, {"VWM",2}, {"VWN",4}, {"STRM",1}, {"STRN",1}, {"SA",1}, {"SB",1} } },
        { "Tesla K40m",       { {"MWG",64}, {"NWG",64}, {"KWG",16}, {"MDIMC",16}, {"NDIMC",16}, {"MDIMA",16}, {"NDIMB",32}, {"KWI",2}, {"VWM",1}, {"VWN",1}, {"STRM",1}, {"STRN",0}, {"SA",1}, {"SB",1} } },
        { kDefault,           { {"MWG",32}, {"NWG",64}, {"KWG",16}, {"MDIMC",8}, {"NDIMC",16}, {"MDIMA",16}, {"NDIMB",32}, {"KWI",2}, {"VWM",1}, {"VWN",1}, {"STRM",1}, {"STRN",1}, {"SA",1}, {"SB",1} } },
      }
    },
    { // AMD GPUs
      CL_DEVICE_TYPE_GPU, "AMD", {
        { "Tahiti",           { {"MWG",128}, {"NWG",64}, {"KWG",16}, {"MDIMC",32}, {"NDIMC",8}, {"MDIMA",32}, {"NDIMB",16}, {"KWI",8}, {"VWM",1}, {"VWN",2}, {"STRM",1}, {"STRN",0}, {"SA",0}, {"SB",0} } },
      }
    },
    { // Intel GPUs
      CL_DEVICE_TYPE_GPU, "Intel", {
      }
    },
    { // Default
      CL_DEVICE_TYPE_ALL, kDefault, {
        { kDefault,           { {"MWG",32}, {"NWG",32}, {"KWG",16}, {"MDIMC",8}, {"NDIMC",8}, {"MDIMA",16}, {"NDIMB",16}, {"KWI",1}, {"VWM",1}, {"VWN",1}, {"STRM",1}, {"STRN",1}, {"SA",0}, {"SB",0} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemmComplexSingle = {
  "Xgemm", Precision::kComplexSingle, {
    { // NVIDIA GPUs
      CL_DEVICE_TYPE_GPU, "NVIDIA Corporation", {
        { "GeForce GTX 480",  { {"MWG",32}, {"NWG",64}, {"KWG",16}, {"MDIMC",16}, {"NDIMC",8}, {"MDIMA",16}, {"NDIMB",16}, {"KWI",2}, {"VWM",1}, {"VWN",1}, {"STRM",1}, {"STRN",1}, {"SA",1}, {"SB",1} } },
        { "Tesla K20m",       { {"MWG",32}, {"NWG",64}, {"KWG",16}, {"MDIMC",16}, {"NDIMC",8}, {"MDIMA",8}, {"NDIMB",8}, {"KWI",8}, {"VWM",2}, {"VWN",2}, {"STRM",1}, {"STRN",0}, {"SA",1}, {"SB",0} } },
        { "Tesla K40m",       { {"MWG",32}, {"NWG",64}, {"KWG",16}, {"MDIMC",8}, {"NDIMC",32}, {"MDIMA",32}, {"NDIMB",16}, {"KWI",8}, {"VWM",1}, {"VWN",1}, {"STRM",0}, {"STRN",1}, {"SA",1}, {"SB",1} } },
        { kDefault,           { {"MWG",32}, {"NWG",64}, {"KWG",16}, {"MDIMC",16}, {"NDIMC",8}, {"MDIMA",16}, {"NDIMB",16}, {"KWI",2}, {"VWM",1}, {"VWN",1}, {"STRM",1}, {"STRN",1}, {"SA",1}, {"SB",1} } },
      }
    },
    { // AMD GPUs
      CL_DEVICE_TYPE_GPU, "AMD", {
        { "Tahiti",           { {"MWG",16}, {"NWG",64}, {"KWG",32}, {"MDIMC",8}, {"NDIMC",8}, {"MDIMA",8}, {"NDIMB",16}, {"KWI",2}, {"VWM",1}, {"VWN",1}, {"STRM",1}, {"STRN",1}, {"SA",1}, {"SB",0} } },
      }
    },
    { // Intel GPUs
      CL_DEVICE_TYPE_GPU, "Intel", {
        { "Iris",             { {"MWG",32}, {"NWG",32}, {"KWG",16}, {"MDIMC",8}, {"NDIMC",8}, {"MDIMA",16}, {"NDIMB",16}, {"KWI",1}, {"VWM",1}, {"VWN",1}, {"STRM",1}, {"STRN",1}, {"SA",0}, {"SB",0} } },
      }
    },
    { // Default
      CL_DEVICE_TYPE_ALL, kDefault, {
        { kDefault,           { {"MWG",32}, {"NWG",32}, {"KWG",16}, {"MDIMC",8}, {"NDIMC",8}, {"MDIMA",16}, {"NDIMB",16}, {"KWI",1}, {"VWM",1}, {"VWN",1}, {"STRM",1}, {"STRN",1}, {"SA",0}, {"SB",0} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemmComplexDouble = {
  "Xgemm", Precision::kComplexDouble, {
    { // NVIDIA GPUs
      CL_DEVICE_TYPE_GPU, "NVIDIA Corporation", {
        { "GeForce GTX 480",  { {"MWG",16}, {"NWG",32}, {"KWG",16}, {"MDIMC",8}, {"NDIMC",8}, {"MDIMA",16}, {"NDIMB",8}, {"KWI",2}, {"VWM",1}, {"VWN",4}, {"STRM",1}, {"STRN",0}, {"SA",0}, {"SB",0} } },
        { "Tesla K20m",       { {"MWG",16}, {"NWG",128}, {"KWG",32}, {"MDIMC",8}, {"NDIMC",32}, {"MDIMA",8}, {"NDIMB",32}, {"KWI",2}, {"VWM",1}, {"VWN",4}, {"STRM",1}, {"STRN",1}, {"SA",1}, {"SB",0} } },
        { "Tesla K40m",       { {"MWG",32}, {"NWG",32}, {"KWG",16}, {"MDIMC",32}, {"NDIMC",8}, {"MDIMA",16}, {"NDIMB",32}, {"KWI",8}, {"VWM",1}, {"VWN",1}, {"STRM",1}, {"STRN",1}, {"SA",0}, {"SB",1} } },
        { kDefault,           { {"MWG",16}, {"NWG",32}, {"KWG",16}, {"MDIMC",8}, {"NDIMC",8}, {"MDIMA",16}, {"NDIMB",8}, {"KWI",2}, {"VWM",1}, {"VWN",4}, {"STRM",1}, {"STRN",0}, {"SA",0}, {"SB",0} } },
      }
    },
    { // AMD GPUs
      CL_DEVICE_TYPE_GPU, "AMD", {
        { "Tahiti",           { {"MWG",128}, {"NWG",32}, {"KWG",16}, {"MDIMC",32}, {"NDIMC",8}, {"MDIMA",32}, {"NDIMB",16}, {"KWI",8}, {"VWM",2}, {"VWN",1}, {"STRM",1}, {"STRN",1}, {"SA",0}, {"SB",0} } },
      }
    },
    { // Intel GPUs
      CL_DEVICE_TYPE_GPU, "Intel", {
      }
    },
    { // Default
      CL_DEVICE_TYPE_ALL, kDefault, {
        { kDefault,           { {"MWG",32}, {"NWG",32}, {"KWG",16}, {"MDIMC",8}, {"NDIMC",8}, {"MDIMA",16}, {"NDIMB",16}, {"KWI",1}, {"VWM",1}, {"VWN",1}, {"STRM",1}, {"STRN",1}, {"SA",0}, {"SB",0} } },
      }
    },
  }
};
// =================================================================================================
} // namespace clblast
