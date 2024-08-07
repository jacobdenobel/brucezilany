import os
import subprocess
import platform
from glob import glob
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.0.1"

ext = Pybind11Extension(
    "bruce.brucecpp", 
    [x for x in glob("src/*cpp") if "main.cpp" not in x], 
    include_dirs=["include"],
    cxx_std=17
)

if platform.system() in ("Linux", "Darwin"):
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
    ext._add_cflags(["-O3", "-pthread"])
else:
    ext._add_cflags(["/O2"])


with open(os.path.join(os.path.dirname(__file__), "README.md")) as f:
    description = f.read()

setup(
    name="bruce",
    author="Jacob de Nobel",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    description="",
    long_description=description,
    long_description_content_type="text/markdown",
    packages=["bruce"],
    zip_safe=False,
    version=__version__,
    install_requires=[
        "librosa",
        "matplotlib",
        "numpy",
        "scipy",
    ],
)
