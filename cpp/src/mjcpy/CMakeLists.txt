include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/3rdparty")
include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/3rdparty/Boost.NumPy")
boost_python_module(mjcpy mjcpy.cpp )
target_link_libraries(mjcpy mjc boost_numpy)