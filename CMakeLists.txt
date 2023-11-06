# Define minimum CMake version requirements.
cmake_minimum_required( VERSION 3.5.0 )

# Set the path for additional CMake modules.
set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

# Set project language details.
project( randomforest CXX )
set( CMAKE_CXX_STANDARD 17 )

# Set build type.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler flags.
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add subdirectories.
add_subdirectory( "Sources" )
