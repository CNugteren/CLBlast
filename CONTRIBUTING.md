
CLBlast: Contributing guidelines
================

For information about the CLBlast library, see the [README](README.md) file instead.

Tuning results
-------------

A [dedicated GitHub issue](https://github.com/CNugteren/CLBlast/issues/1) is available to post new tuning results. If you compiled with the tuners (see the [README](README.md) for instructions), ran one of the tuners on your device (or all perhaps?), and feel that these results should be included in the next release of CLBlast, please post them there. You can do this by attaching the JSON files to the issue (archived in a .ZIP file).


Code improvements and additions
-------------

Pull requests are welcome as long as they:

* Contain unit additions or modifications
* Follow the CLBlast coding style, which is loosely based on the [Google C++ style guide](https://google-styleguide.googlecode.com/svn/trunk/cppguide.html) and the Effective C++ books by Scott Meyers. We use a tab-size of 2 spaces and a max-width of 100 characters.
* Are made against the `master` branch.
