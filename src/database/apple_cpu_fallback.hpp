
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file provides overrides for Apple's OpenCL CPU implementation. It is a special case compared
// to all other implementations, as it only supports a 1-dimensional work-group size. In addition,
// that work-group size is limited to 1024 (in theory) or much lower (kernel resource dependent).
// Thus, instead of supporting this corner-case in the whole regular flow (starting from the tuner),
// we provide this file with some manual overrides.
//
// Note: These overrides are to make the Apple CPU work and not crash, they are not in any way
// optimized parameters. For decent speed don't use Apple's OpenCL CPU implementation.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XaxpyApple = {
  "Xaxpy", Precision::kAny, { { kDeviceTypeAll, "default", { { "default", { {"VW",8}, {"WGS",1}, {"WPT",4} } } } } }
};
const Database::DatabaseEntry XdotApple = {
  "Xdot", Precision::kAny, { {  kDeviceTypeAll, "default", { { "default", { {"WGS1",1}, {"WGS2",1} } } } } }
};
const Database::DatabaseEntry XgemvApple = {
  "Xgemv", Precision::kAny, { {  kDeviceTypeAll, "default", { { "default", { {"WGS1",1}, {"WPT1",4}, {"UNROLL1", 1} } } } } }
};
const Database::DatabaseEntry XgemvFastApple = {
  "XgemvFast", Precision::kAny, { {  kDeviceTypeAll, "default", { { "default", { {"VW2",1}, {"WGS2",1}, {"WPT2",1} } } } } }
};
const Database::DatabaseEntry XgemvFastRotApple = {
  "XgemvFastRot", Precision::kAny, { {  kDeviceTypeAll, "default", { { "default", { {"VW3",1}, {"WGS3",1}, {"WPT3",1} } } } } }
};
const Database::DatabaseEntry XgerApple = {
  "Xger", Precision::kAny, { {  kDeviceTypeAll, "default", { { "default", { {"WGS1",64}, {"WGS2",1}, {"WPT",2} } } } } }
};
const Database::DatabaseEntry XtrsvApple = {
  "Xtrsv", Precision::kAny, { {  kDeviceTypeAll, "default", { { "default", { {"TRSV_BLOCK_SIZE",32} } } } } }
};
const Database::DatabaseEntry XgemmApple = {
  "Xgemm", Precision::kAny, { {  kDeviceTypeAll, "default", { { "default", { {"KWG",1}, {"KWI",1}, {"MDIMA",1}, {"MDIMC",1}, {"MWG",1}, {"NDIMB",1}, {"NDIMC",1}, {"NWG",1}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } } } } }
};
const Database::DatabaseEntry XgemmDirectApple = {
  "XgemmDirect", Precision::kAny, { {  kDeviceTypeAll, "default", { { "default", { {"KWID",1}, {"MDIMAD",1}, {"MDIMCD",1}, {"NDIMBD",1}, {"NDIMCD",1}, {"PADA",0}, {"PADB",0}, {"VWMD",1}, {"VWND",1}, {"WGD",1} } } } } }
};
const Database::DatabaseEntry CopyApple = {
  "Copy", Precision::kAny, { {  kDeviceTypeAll, "default", { { "default", { {"COPY_DIMX",1}, {"COPY_DIMY",1}, {"COPY_VW",1}, {"COPY_WPT",1} } } } } }
};
const Database::DatabaseEntry PadApple = {
  "Pad", Precision::kAny, { {  kDeviceTypeAll, "default", { { "default", { {"PAD_DIMX",1}, {"PAD_DIMY",1}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } } } } }
};
const Database::DatabaseEntry TransposeApple = {
  "Transpose", Precision::kAny, { {  kDeviceTypeAll, "default", { { "default", { {"TRA_DIM",1}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } } } } }
};
const Database::DatabaseEntry PadtransposeApple = {
  "Padtranspose", Precision::kAny, { {  kDeviceTypeAll, "default", { { "default", { {"PADTRA_PAD",0}, {"PADTRA_TILE",1}, {"PADTRA_WPT",1} } } } } }
};
const Database::DatabaseEntry InvertApple = {
  "Invert", Precision::kAny, { {  kDeviceTypeAll, "default", { { "default", { {"INTERNAL_BLOCK_SIZE",16} } } } } }
};

// =================================================================================================
} // namespace database
} // namespace clblast
