add_executable( trainingexample trainingexample.cpp )
target_link_libraries( trainingexample balsa )

add_executable( classificationexample classificationexample.cpp )
target_link_libraries( classificationexample balsa )

add_executable( customcontainerexample customcontainerexample.cpp )
target_link_libraries( customcontainerexample balsa )
