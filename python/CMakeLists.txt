# Set minimum required CMake version.
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Define the CMake project.
project(balsa_python LANGUAGES C CXX)

# Set the version of the C++ standard to use.
set(CMAKE_CXX_STANDARD 17)

# Fetch a copy of the balsa C++ library.
include(FetchContent)
set(BALSA_GIT_URL "git@bitbucket.org:sron_earth/balsa.git" CACHE STRING "Git URL of the balsa C++ library")
set(BALSA_GIT_TAG "origin/main" CACHE STRING "Git hash, tag name, or branch name of the balsa C++ library to use")
FetchContent_Declare(
	balsa
	GIT_REPOSITORY "${BALSA_GIT_URL}"
	GIT_TAG "${BALSA_GIT_TAG}"
)
FetchContent_MakeAvailable(balsa)

# Exclude the balsa C++ library targets from the default build. For the Python
# extension module, only the static library is needed. This will be explicitly
# selected using a target_link_libraries() call (see below).
FetchContent_GetProperties(balsa)
set_property(DIRECTORY ${balsa_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL ON)

# Locate Python components needed for building the C extension.
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Define the Python C extension.
Python_add_library(c_extension MODULE src/balsamodule.cpp WITH_SOABI)
set_target_properties(c_extension PROPERTIES OUTPUT_NAME "_balsa")
target_link_libraries(c_extension PRIVATE balsa-static Python::NumPy)

# Install the C extension with the balsa package Python code.
install(TARGETS c_extension LIBRARY DESTINATION balsa)
