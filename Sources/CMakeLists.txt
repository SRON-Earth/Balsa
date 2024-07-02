configure_file( config.h.in config.h )

add_test( NAME testsuite COMMAND balsa_test )

add_library( balsa SHARED fileio.cpp modelevaluation.cpp serdes.cpp weightedcoin.cpp )
target_include_directories( balsa PUBLIC ${CMAKE_CURRENT_LIST_DIR} )

add_library( balsa-static STATIC EXCLUDE_FROM_ALL fileio.cpp modelevaluation.cpp serdes.cpp weightedcoin.cpp )
target_include_directories( balsa-static PUBLIC ${CMAKE_CURRENT_LIST_DIR} )

add_executable( balsa_train balsa_train.cpp )
target_link_libraries( balsa_train balsa )

add_executable( balsa_classify balsa_classify.cpp )
target_link_libraries( balsa_classify balsa )

add_executable( balsa_generate balsa_generate.cpp )
target_link_libraries( balsa_generate balsa )

add_executable( balsa_print balsa_print.cpp )
target_link_libraries( balsa_print balsa )

add_executable( balsa_measure balsa_measure.cpp )
target_link_libraries( balsa_measure balsa )

add_executable( balsa_featureimportance balsa_featureimportance.cpp )
target_link_libraries( balsa_featureimportance balsa )

add_executable( balsa_convert balsa_convert.cpp )
target_link_libraries( balsa_convert balsa )

add_executable( balsa_test balsa_test.cpp )
target_link_libraries( balsa_test balsa )
