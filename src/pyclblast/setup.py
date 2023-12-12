
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0.
# This file follows the PEP8 Python style guide and uses a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import platform
import os
import subprocess
import sys


# Command line flags forwarded to CMake
cmake_cmd_args = []
for arg in sys.argv:
    if arg.startswith('-D'):
        cmake_cmd_args.append(arg)

for arg in cmake_cmd_args:
    sys.argv.remove(arg)


# For setup.py with Cmake extensions, see
# https://martinopilia.com/posts/2018/09/15/building-python-extension.html
class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='.', **kwargs):
        Extension.__init__(self, name, sources=[], **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuildExt(build_ext):
    def build_extensions(self):
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('Cannot find CMake executable')

        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

            cmake_args = [
                '-DCMAKE_BUILD_TYPE=Release',
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={}'.format(extdir),
                '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE={}'.format(self.build_temp)
            ]

            # We can handle some platform-specific settings at our discretion
            if platform.system() == 'Windows':
                plat = ('x64' if platform.architecture()[0] == '64bit' else 'Win32')
                cmake_args += [
                    # These options are likely to be needed under Windows
                    '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE',
                    '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={}'.format(extdir),
                ]
                # Assuming that Visual Studio and MinGW are supported compilers
                if self.compiler.compiler_type == 'msvc':
                    cmake_args += [
                        '-DCMAKE_GENERATOR_PLATFORM=%s' % plat,
                    ]
                else:
                    cmake_args += [
                        '-G', 'MinGW Makefiles',
                    ]

            cmake_args += cmake_cmd_args

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            # Config
            subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args,
                                  cwd=self.build_temp)

            # Build
            subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'],
                                  cwd=self.build_temp)


setup(
    name="pyclblast",
    scripts=[],
    ext_modules=[CMakeExtension("pyclblast")],
    cmdclass={"build_ext": CMakeBuildExt}
)
