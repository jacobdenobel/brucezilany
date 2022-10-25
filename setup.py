from setuptools import setup, Extension
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from glob import glob

import platform 
print(platform.architecture())


ext = Pybind11Extension(
    'hearing_model',
    [x for x in glob("hearing_model/*cpp") if not x.endswith("main.cpp")],
    include_dirs=["hearing_model", pybind11.get_include()],
    cxx_std=17
)


setup(
    name='hearing_model',
    version='1.0',
    description='',
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)