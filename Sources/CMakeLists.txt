add_library( balsa SHARED weightedcoin.cpp )
target_include_directories(balsa PUBLIC ${CMAKE_CURRENT_LIST_DIR})

add_executable( balsa_train balsa_train.cpp )
target_link_libraries( balsa_train balsa )

add_executable( balsa_classify balsa_classify.cpp )
target_link_libraries( balsa_classify balsa )

add_executable( balsa_generate balsa_generate.cpp )
target_link_libraries( balsa_generate balsa )

add_executable( balsa_print balsa_print.cpp )
target_link_libraries( balsa_print balsa )

add_executable( balsa_metrics balsa_metrics.cpp )
target_link_libraries( balsa_metrics balsa )
