# Define minimum CMake version requirements.
cmake_minimum_required( VERSION 3.5.0 )

# Set the path for additional CMake modules.
set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

# Set project language details.
project( gollum CXX )
set( CMAKE_CXX_STANDARD 17 )
add_definitions( "-Wall" )

# Add subdirectories.
add_subdirectory( "Sources" )
