cmake_minimum_required(VERSION 3.18...3.26)

project(balsa_python LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

add_subdirectory(extern/balsa EXCLUDE_FROM_ALL)

Python_add_library(balsa_ext MODULE WITH_SOABI src/balsamodule.cpp)
set_target_properties(balsa_ext PROPERTIES OUTPUT_NAME "_balsa")
target_link_libraries(balsa_ext PRIVATE balsa-static Python::NumPy)

install(TARGETS balsa_ext LIBRARY DESTINATION balsa)
