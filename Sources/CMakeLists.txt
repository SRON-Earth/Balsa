add_library( rfcommon STATIC ingestion.cpp decisiontrees.cpp weightedcoin.cpp )

#add_executable( rftrain rftrain.cpp trainers.cpp trainermark2.cpp )
#target_link_libraries( rftrain rfcommon )

add_executable( rfclassify rfclassify.cpp )
target_link_libraries( rfclassify rfcommon )

add_executable( rftest rftest.cpp )
target_link_libraries( rftest rfcommon )
