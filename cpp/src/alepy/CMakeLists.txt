include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/3rdparty/ale_0_4/src")
include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/3rdparty/Boost.NumPy")

boost_python_module(alepy alepy.cpp )
target_link_libraries(alepy ale boost_numpy)

