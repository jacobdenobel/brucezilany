import os
import subprocess
import platform
from glob import glob
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.0.1"

ext = Pybind11Extension(
    "hearing_model.brucecpp", glob("src/*cpp"), include_dirs=["include"], cxx_std=17
)

if platform.system() in ("Linux", "Darwin"):
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
    ext._add_cflags(["-O3", "-pthread"])
    try:
        if subprocess.check_output("ldconfig -p | grep tbb", shell=True):
            ext._add_ldflags(["-ltbb"])
            ext._add_cflags(["-DHASTBB"])
    except subprocess.CalledProcessError:
        pass
else:
    ext._add_cflags(["/O2", "/DHASTBB"])


with open(os.path.join(os.path.dirname(__file__), "README.md")) as f:
    description = f.read()

setup(
    name="hearing_model",
    author="Jacob de Nobel",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    description="",
    long_description=description,
    long_description_content_type="text/markdown",
    packages=["hearing_model"],
    zip_safe=False,
    version=__version__,
)