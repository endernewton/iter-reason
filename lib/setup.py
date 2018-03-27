import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

extensions = [
    Extension(
        "utils.cython_bbox",
        ["utils/bbox.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
]

setup(
    name='iter-reason',
    ext_modules=cythonize(extensions)
)
