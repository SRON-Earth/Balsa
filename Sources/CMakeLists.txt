add_executable( rftrain rftrain.cpp trainers.cpp trainermark2.cpp ingestion.cpp decisiontrees.cpp weightedcoin.cpp )

add_executable( rfclassify rfclassify.cpp ingestion.cpp decisiontrees.cpp )
