
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. It
// is auto-generated by the 'scripts/database/database.py' Python script.
//
// This file populates the database with best-found tuning parameters for the 'Xaxpy16' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {

const DatabaseEntry XaxpyHalf = {
  "Xaxpy", Precision::kHalf, {"VW", "WGS", "WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere", {
          { Name{"AMD Radeon RX 480                                 "}, Params{ 4, 128, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { Name{"AMD Radeon RX 580 2048SP                          "}, Params{ 2, 128, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { Name{"AMD Radeon RX590 GME                              "}, Params{ 1, 128, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 8, 128, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "Fiji", {
          { Name{"AMD Radeon 500 Series                             "}, Params{ 2, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "Vega", {
          { Name{"Radeon RX Vega                                    "}, Params{ 4, 64, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 4, 64, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "default", {
          { kDeviceNameDefault                                        , Params{ 4, 128, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1010:xnack-", {
          { Name{"AMD Radeon RX 5700                                "}, Params{ 2, 64, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { Name{"AMD Radeon RX 5700 XT                             "}, Params{ 4, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 64, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1030", {
          { Name{"AMD Radeon RX 6800 XT                             "}, Params{ 4, 128, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { Name{"AMD Radeon RX 6900 XT                             "}, Params{ 2, 256, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 4, 128, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1031", {
          { Name{"AMD Radeon RX 6700 XT                             "}, Params{ 8, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 8, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1032", {
          { Name{"AMD Radeon RX 6600 XT                             "}, Params{ 4, 64, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 4, 64, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1034", {
          { Name{"AMD Radeon RX 6500 XT                             "}, Params{ 4, 64, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 4, 64, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1035", {
          { Name{"AMD Radeon Graphics                               "}, Params{ 4, 128, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 4, 128, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1100", {
          { Name{"Radeon RX 7900 XTX                                "}, Params{ 4, 128, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 4, 128, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1101", {
          { Name{"AMD Radeon RX 7800 XT                             "}, Params{ 8, 128, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 8, 128, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1102", {
          { Name{"AMD Radeon RX 7600                                "}, Params{ 4, 256, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 4, 256, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx902", {
          { Name{"AMD Radeon(TM) Graphics                           "}, Params{ 2, 64, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { Name{"AMD Radeon(TM) RX Vega 10 Graphics                "}, Params{ 2, 128, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 64, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx906:sramecc+:xnack-", {
          { Name{"AMD Radeon VII                                    "}, Params{ 4, 256, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 4, 256, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx90c", {
          { Name{"AMD Radeon(TM) Graphics                           "}, Params{ 1, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 8, 128, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "default", {
          { Name{"Mali-T628                                         "}, Params{ 8, 128, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { Name{"Mali-T760                                         "}, Params{ 4, 256, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 4, 256, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // Imagination Technologies GPUs
      kDeviceTypeGPU, "Imagination Technologies", {
        { "default", {
          { Name{"PowerVR B-Series BXE-4-32                         "}, Params{ 8, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 8, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "default", {
          { Name{"Intel(R) Arc(TM) A770 Graphics                    "}, Params{ 4, 256, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) HD Graphics 5500 BroadWell U-Processor GT"}, Params{ 1, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) HD Graphics 620                          "}, Params{ 1, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) HD Graphics Skylake ULT GT2              "}, Params{ 8, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) Iris(R) Xe Graphics                      "}, Params{ 2, 64, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) RaptorLake-S Mobile Graphics Controller  "}, Params{ 2, 64, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) UHD Graphics 620                         "}, Params{ 2, 128, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) UHD Graphics 770                         "}, Params{ 2, 64, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 8, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // QUALCOMM GPUs
      kDeviceTypeGPU, "QUALCOMM", {
        { "default", {
          { Name{"QUALCOMM Adreno(TM)                               "}, Params{ 4, 64, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 8, 128, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "OpenCL C 2.0 Adreno(TM) 640", {
          { Name{"QUALCOMM Adreno(TM)                               "}, Params{ 4, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 4, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "OpenCL C 3.0 Adreno(TM) 730", {
          { Name{"QUALCOMM Adreno(TM)                               "}, Params{ 8, 256, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 8, 256, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
        { "OpenCL C 3.0 Adreno(TM) 740", {
          { Name{"QUALCOMM Adreno(TM)                               "}, Params{ 8, 256, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 8, 256, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default", {
          { kDeviceNameDefault                                        , Params{ 8, 128, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
  }
};

} // namespace database
} // namespace clblast
