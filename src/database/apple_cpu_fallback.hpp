
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
  "Xaxpy", Precision::kAny, {"VW", "WGS", "WPT"}, { { kDeviceTypeAll, "default", { { "default", { 8, 1, 4 } } } } }
};
const Database::DatabaseEntry XdotApple = {
  "Xdot", Precision::kAny, {"WGS1", "WGS2"}, { {  kDeviceTypeAll, "default", { { "default", { 1, 1 } } } } }
};
const Database::DatabaseEntry XgemvApple = {
  "Xgemv", Precision::kAny, {"WGS1", "WPT1", "UNROLL1"}, { {  kDeviceTypeAll, "default", { { "default", { 1, 4, 1 } } } } }
};
const Database::DatabaseEntry XgemvFastApple = {
  "XgemvFast", Precision::kAny, {"VW2", "WGS2", "WPT2"}, { {  kDeviceTypeAll, "default", { { "default", { 1, 1, 1 } } } } }
};
const Database::DatabaseEntry XgemvFastRotApple = {
  "XgemvFastRot", Precision::kAny, {"VW3", "WGS3", "WPT3"}, { {  kDeviceTypeAll, "default", { { "default", { 1, 1, 1 } } } } }
};
const Database::DatabaseEntry XgerApple = {
  "Xger", Precision::kAny, {"WGS1", "WGS2", "WPT"}, { {  kDeviceTypeAll, "default", { { "default", { 64, 1, 2 } } } } }
};
const Database::DatabaseEntry XtrsvApple = {
  "Xtrsv", Precision::kAny, {"TRSV_BLOCK_SIZE"}, { {  kDeviceTypeAll, "default", { { "default", { 32 } } } } }
};
const Database::DatabaseEntry XgemmApple = {
  "Xgemm", Precision::kAny, {"KWG", "KWI", "MDIMA", "MDIMC", "MWG", "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"}, { {  kDeviceTypeAll, "default", { { "default", { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1 } } } } }
};
const Database::DatabaseEntry XgemmDirectApple = {
  "XgemmDirect", Precision::kAny, {"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD"}, { {  kDeviceTypeAll, "default", { { "default", { 1, 1, 1, 1, 1, 0, 0, 1, 1, 1 } } } } }
};
const Database::DatabaseEntry CopyApple = {
  "Copy", Precision::kAny, {"COPY_DIMX", "COPY_DIMY", "COPY_VW", "COPY_WPT"}, { {  kDeviceTypeAll, "default", { { "default", { 1, 1, 1, 1 } } } } }
};
const Database::DatabaseEntry PadApple = {
  "Pad", Precision::kAny, {"PAD_DIMX", "PAD_DIMY", "PAD_WPTX", "PAD_WPTY"}, { {  kDeviceTypeAll, "default", { { "default", { 1, 1, 1, 1 } } } } }
};
const Database::DatabaseEntry TransposeApple = {
  "Transpose", Precision::kAny, {"TRA_DIM", "TRA_PAD", "TRA_SHUFFLE", "TRA_WPT"}, { {  kDeviceTypeAll, "default", { { "default", { 1, 0, 0, 1 } } } } }
};
const Database::DatabaseEntry PadtransposeApple = {
  "Padtranspose", Precision::kAny, {"PADTRA_PAD", "PADTRA_TILE", "PADTRA_WPT"}, { {  kDeviceTypeAll, "default", { { "default", { 0, 1, 1 } } } } }
};
const Database::DatabaseEntry InvertApple = {
  "Invert", Precision::kAny, {"INTERNAL_BLOCK_SIZE"}, { {  kDeviceTypeAll, "default", { { "default", { 16 } } } } }
};

// =================================================================================================
} // namespace database
} // namespace clblast
