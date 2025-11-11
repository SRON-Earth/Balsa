# Use environment variable if set, otherwise default to system location
if(DEFINED ENV{BALSA_INSTALL_PATH})
    set(BALSA_INCLUDE_DIR "$ENV{BALSA_INSTALL_PATH}/include")
    set(BALSA_LIB_DIR     "$ENV{BALSA_INSTALL_PATH}/lib")
else()
    set(BALSA_INCLUDE_DIR "/usr/local/include")
    set(BALSA_LIB_DIR     "/usr/local/lib")
endif()

include_directories(${BALSA_INCLUDE_DIR})
link_directories(${BALSA_LIB_DIR})

add_executable(trainingexample trainingexample.cpp)
target_link_libraries(trainingexample balsa)

add_executable(classificationexample classificationexample.cpp)
target_link_libraries(classificationexample balsa)

add_executable(customcontainerexample customcontainerexample.cpp)
target_link_libraries(customcontainerexample balsa)

