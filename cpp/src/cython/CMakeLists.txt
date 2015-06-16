find_package(Numpy REQUIRED)
message("numpy: ${PYTHON_NUMPY_INCLUDE_DIR}")
include_directories(
	${PYTHON_NUMPY_INCLUDE_DIR}
)

set( CYTHON_ANNOTATE ON)
cython_add_module( cycontrol cycontrol.pyx)