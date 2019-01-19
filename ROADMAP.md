CLBlast feature road-map
================

This file gives an overview of the main features planned for addition to CLBlast. A first-order indication time-frame for development time is provided:

| Issue#                                                         | When        | Who       | Status | What          |
| ---------------------------------------------------------------|-------------|-----------|--------|---------------|
| -                                                              | Oct '17     | CNugteren | ✔      | CUDA API for CLBlast |
| [#169](https://github.com/CNugteren/CLBlast/issues/169) & #195 | Oct-Nov '17 | CNugteren | ✔      | Auto-tuning the kernel selection parameter |
| [#181](https://github.com/CNugteren/CLBlast/issues/181) & #201 | Nov '17     | CNugteren | ✔      | Compilation for Android and testing on a device |
| -                                                              | Nov '17     | CNugteren | ✔      | Integration of CLTune for easy testing on Android / fewer dependencies |
| [#128](https://github.com/CNugteren/CLBlast/issues/128) & #205 | Nov-Dec '17 | CNugteren | ✔      | Pre-processor for loop unrolling and array-to-register-promotion for e.g. ARM Mali |
| [#207](https://github.com/CNugteren/CLBlast/issues/207)        | Dec '17     | CNugteren | ✔      | Tuning of the TRSM/TRSV routines |
| [#195](https://github.com/CNugteren/CLBlast/issues/195)        | Jan '18     | CNugteren | ✔      | Extra GEMM API with pre-allocated temporary buffer |
| [#95](https://github.com/CNugteren/CLBlast/issues/95)   & #237 | Jan '18     | CNugteren | ✔      | Implement strided batch GEMM |
| [#224](https://github.com/CNugteren/CLBlast/issues/224)        | Jan-Feb '18 | CNugteren | ✔      | Implement Hadamard product (element-wise vector-vector product) |
| [#233](https://github.com/CNugteren/CLBlast/issues/233)        | Feb '18     | CNugteren | ✔      | Add CLBlast to common package managers |
| [#223](https://github.com/CNugteren/CLBlast/issues/223)        | Feb '18     | CNugteren | ✔      | Python OpenCL interface |
| [#237](https://github.com/CNugteren/CLBlast/issues/237)        | Mar '18     | CNugteren | ✔      | Making tuning possible from the CLBlast API |
| [#228](https://github.com/CNugteren/CLBlast/issues/228)        | Mar-Apr '18 | CNugteren | ✔      | Improving performance for Qualcomm Adreno GPUs |
| [#270](https://github.com/CNugteren/CLBlast/issues/270)        | Oct '18     | CNugteren | ✔      | Implement col2im |
| -                                                              | ??          | CNugteren |        | Add support for OpenCL image buffers |
| [#267](https://github.com/CNugteren/CLBlast/issues/267)        | Jan '19     | vbkaisetsu| ✔      | Merge im2col and GEMM into a direct kernel |
| [#136](https://github.com/CNugteren/CLBlast/issues/136)        | ??          | CNugteren |        | Implement xAXPBY and xSET |
| [#169](https://github.com/CNugteren/CLBlast/issues/169)        | ??          | dividiti  |        | Problem-specific tuning parameter selection |
