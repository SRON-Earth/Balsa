add_library( balsa SHARED ingestion.cpp weightedcoin.cpp )
target_include_directories(balsa PUBLIC ${CMAKE_CURRENT_LIST_DIR})

add_executable( balsa_train balsa_train.cpp trainers.cpp )
target_link_libraries( balsa_train balsa )

add_executable( balsa_classify balsa_classify.cpp )
target_link_libraries( balsa_classify balsa )

#add_executable( rftest balsa_test.cpp )
#target_link_libraries( balsa_test balsa )
