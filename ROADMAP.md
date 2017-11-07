CLBlast feature road-map
================

This file gives an overview of the main features planned for addition to CLBlast. A first-order indication time-frame for development time is provided:

| Issue#     | When        | Who       | Status | What          |
| -----------|-------------|-----------|--------|---------------|
| -          | Oct '17     | CNugteren | ✔      | CUDA API for CLBlast |
| [#169](https://github.com/CNugteren/CLBlast/issues/169), [#195](https://github.com/CNugteren/CLBlast/issues/195) | Oct-Nov '17 | CNugteren | ✔      | Auto-tuning the kernel selection parameter |
| [#181](https://github.com/CNugteren/CLBlast/issues/181), [#201](https://github.com/CNugteren/CLBlast/issues/201) | Nov '17     | CNugteren | ✔      | Compilation for Android and testing on a device |
| -          | Nov '17     | CNugteren |        | Integration of CLTune for easy testing on Android / fewer dependencies |
| [#128](https://github.com/CNugteren/CLBlast/issues/128), [#205](https://github.com/CNugteren/CLBlast/issues/205) | Nov-Dec '17 | CNugteren |        | Pre-processor for loop unrolling and array-to-register-promotion for e.g. ARM Mali |
| [#207](https://github.com/CNugteren/CLBlast/issues/207)       | Dec '17     | CNugteren |        | Tuning of the TRSM/TRSV routines |
| [#169](https://github.com/CNugteren/CLBlast/issues/169)       | '17         | dividiti  |        | Problem-specific tuning parameter selection |
