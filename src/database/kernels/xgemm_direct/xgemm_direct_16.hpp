
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. It
// is auto-generated by the 'scripts/database/database.py' Python script.
//
// This file populates the database with best-found tuning parameters for the 'Xgemm_Direct16' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {

const DatabaseEntry XgemmDirectHalf = {
  "XgemmDirect", Precision::kHalf, {"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere", {
          { Name{"AMD Radeon RX 480                                 "}, Params{ 8, 32, 8, 8, 32, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"AMD Radeon RX 580 2048SP                          "}, Params{ 2, 8, 8, 8, 8, 1, 1, 2, 2, 16, 0, 0, 0, 0, 0, 0 } },
          { Name{"AMD Radeon RX590 GME                              "}, Params{ 2, 16, 16, 16, 16, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 16, 16, 16, 16, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "default", {
          { kDeviceNameDefault                                        , Params{ 2, 16, 16, 8, 8, 1, 1, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1010:xnack-", {
          { Name{"AMD Radeon RX 5700                                "}, Params{ 2, 8, 8, 16, 16, 1, 1, 4, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"AMD Radeon RX 5700 XT                             "}, Params{ 2, 8, 8, 16, 16, 1, 1, 4, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 16, 16, 1, 1, 4, 2, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1030", {
          { Name{"AMD Radeon RX 6800 XT                             "}, Params{ 2, 16, 16, 16, 16, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"AMD Radeon RX 6900 XT                             "}, Params{ 16, 8, 8, 16, 8, 1, 0, 2, 1, 16, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 8, 8, 1, 1, 1, 2, 16, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1031", {
          { Name{"AMD Radeon RX 6700 XT                             "}, Params{ 16, 8, 8, 16, 8, 0, 0, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 16, 8, 8, 16, 8, 0, 0, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1032", {
          { Name{"AMD Radeon RX 6600 XT                             "}, Params{ 2, 8, 8, 16, 16, 1, 1, 4, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 16, 16, 1, 1, 4, 2, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1034", {
          { Name{"AMD Radeon RX 6500 XT                             "}, Params{ 2, 16, 16, 16, 16, 1, 1, 2, 4, 64, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 16, 16, 16, 16, 1, 1, 2, 4, 64, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1035", {
          { Name{"AMD Radeon Graphics                               "}, Params{ 16, 16, 32, 16, 8, 1, 1, 2, 1, 64, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 16, 16, 32, 16, 8, 1, 1, 2, 1, 64, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1100", {
          { Name{"Radeon RX 7900 XTX                                "}, Params{ 2, 8, 8, 8, 8, 1, 1, 1, 2, 16, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 8, 8, 1, 1, 1, 2, 16, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1101", {
          { Name{"AMD Radeon RX 7800 XT                             "}, Params{ 2, 8, 8, 16, 16, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 16, 16, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1102", {
          { Name{"AMD Radeon RX 7600                                "}, Params{ 2, 8, 8, 8, 8, 1, 1, 4, 4, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 8, 8, 1, 1, 4, 4, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1103", {
          { Name{"AMD Radeon 780M Graphics                          "}, Params{ 2, 8, 8, 8, 8, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 8, 8, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1201", {
          { Name{"AMD Radeon RX 9070 XT                             "}, Params{ 2, 8, 8, 8, 8, 0, 0, 1, 1, 8, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 8, 8, 0, 0, 1, 1, 8, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx902", {
          { Name{"AMD Radeon(TM) Graphics                           "}, Params{ 2, 8, 8, 8, 8, 1, 1, 4, 4, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"AMD Radeon(TM) RX Vega 10 Graphics                "}, Params{ 2, 16, 8, 8, 8, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 8, 8, 1, 1, 4, 4, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx906:sramecc+:xnack-", {
          { Name{"AMD Instinct MI50/MI60                            "}, Params{ 2, 8, 8, 8, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0 } },
          { Name{"AMD Radeon VII                                    "}, Params{ 2, 16, 16, 16, 16, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 8, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx90c", {
          { Name{"AMD Radeon(TM) Graphics                           "}, Params{ 2, 8, 8, 8, 8, 1, 1, 4, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 8, 8, 1, 1, 4, 2, 32, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "default", {
          { Name{"Mali-G57 MC2 r0p1                                 "}, Params{ 16, 16, 32, 8, 8, 0, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"Mali-T628                                         "}, Params{ 2, 16, 16, 8, 8, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"Mali-T760                                         "}, Params{ 2, 16, 16, 8, 8, 1, 1, 1, 4, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 16, 16, 8, 8, 1, 1, 1, 4, 32, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // Imagination Technologies GPUs
      kDeviceTypeGPU, "Imagination Technologies", {
        { "default", {
          { Name{"PowerVR B-Series BXE-4-32                         "}, Params{ 2, 8, 8, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "default", {
          { Name{"Intel(R) Arc(TM) A750 Graphics                    "}, Params{ 8, 32, 32, 16, 8, 1, 0, 1, 4, 64, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) Arc(TM) A770 Graphics                    "}, Params{ 2, 32, 32, 16, 16, 0, 0, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) HD Graphics 620                          "}, Params{ 2, 16, 16, 16, 16, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) HD Graphics Skylake ULT GT2              "}, Params{ 2, 8, 8, 8, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) Iris(R) Xe Graphics                      "}, Params{ 16, 32, 16, 16, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) RaptorLake-S Mobile Graphics Controller  "}, Params{ 8, 16, 16, 8, 8, 0, 0, 4, 2, 64, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) UHD Graphics 620                         "}, Params{ 2, 8, 16, 8, 8, 1, 1, 4, 4, 64, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 16, 16, 16, 16, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // QUALCOMM GPUs
      kDeviceTypeGPU, "QUALCOMM", {
        { "default", {
          { Name{"QUALCOMM Adreno(TM)                               "}, Params{ 16, 16, 16, 8, 8, 1, 1, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 16, 16, 16, 8, 8, 1, 1, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "OpenCL C 2.0 Adreno(TM) 640", {
          { Name{"QUALCOMM Adreno(TM)                               "}, Params{ 2, 16, 16, 8, 8, 1, 1, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 16, 16, 8, 8, 1, 1, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "OpenCL C 2.0 Adreno(TM) 650", {
          { Name{"QUALCOMM Adreno(TM)                               "}, Params{ 2, 16, 16, 8, 8, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 16, 16, 8, 8, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "OpenCL C 3.0 Adreno(TM) 730", {
          { Name{"QUALCOMM Adreno(TM)                               "}, Params{ 2, 16, 16, 8, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 16, 16, 8, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "OpenCL C 3.0 Adreno(TM) 740", {
          { Name{"QUALCOMM Adreno(TM)                               "}, Params{ 2, 8, 16, 8, 8, 0, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 16, 8, 8, 0, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default", {
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 8, 8, 1, 1, 2, 2, 16, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
  }
};

} // namespace database
} // namespace clblast
