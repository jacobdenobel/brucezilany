import os
import platform
from glob import glob
from setuptools import setup, find_packages

from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.0.1"

ext = Pybind11Extension(
    "brucezilany.brucezilanycpp", 
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


with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="brucezilany",
    version=__version__,
    author="Jacob de Nobel",
    author_email="jacobdenobel@gmail.com", 
    description="Python/C++ interface to the Bruce-Zilany-Carney auditory nerve model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jacobdenobel/brucezilany", 
    project_urls={
        "Original Model": "https://www.ece.mcmaster.ca/~ibruce/zbcANmodel/zbcANmodel.htm",
        "Source": "https://github.com/jacobdenobel/brucezilany"
    },
    license="BSD-3-Clause OR Other (if applicable)",  
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    packages=find_packages(include=["brucezilany", "brucezilany.*"]),
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "librosa",
    ],
    include_package_data=True,
)