# -------------------------------------------------------
# Minimum CMake version and project
# -------------------------------------------------------
cmake_minimum_required(VERSION 3.5.0)

project(balsa VERSION 0.9.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# -------------------------------------------------------
# CMake paths and options
# -------------------------------------------------------
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# -------------------------------------------------------
# Add subdirectories
# -------------------------------------------------------
add_subdirectory("Sources")
add_subdirectory("Examples")

# -------------------------------------------------------
# Optional: top-level install rules
# (all critical install rules are already in Sources/CMakeLists.txt)
# -------------------------------------------------------
# You can add additional files to install here if needed
# e.g., documentation or examples

