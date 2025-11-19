cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Define the CMake project
project(balsa_rfc LANGUAGES C CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)

# Locate Python and NumPy for the extension
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Add Balsa core as a subdirectory, but exclude it from the default build/install
# This prevents binaries, tests, examples, headers from being installed
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core balsa_core EXCLUDE_FROM_ALL)

# Build the Python C extension
Python_add_library(c_extension MODULE src/balsamodule.cpp WITH_SOABI)
set_target_properties(c_extension PROPERTIES OUTPUT_NAME "_balsa")

# Link only the static Balsa library and NumPy
target_link_libraries(c_extension PRIVATE balsa-static Python::NumPy)

# Install only the Python extension module into the package
install(TARGETS c_extension LIBRARY DESTINATION balsa_rfc)

