add_library( balsa SHARED weightedcoin.cpp )
target_include_directories(balsa PUBLIC ${CMAKE_CURRENT_LIST_DIR})

add_executable( balsa_train balsa_train.cpp )
target_link_libraries( balsa_train balsa )

add_executable( balsa_classify balsa_classify.cpp )
target_link_libraries( balsa_classify balsa )

add_executable( balsa_dataproc balsa_dataproc.cpp )
target_link_libraries( balsa_dataproc balsa )
