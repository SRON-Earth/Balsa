#include <iostream>

#include "datamodel.h"

#include <vector>
#include <random>
#include <chrono>

#include "ingestion.h"

int main( int, char ** )
{
    // Load training data set.
    // auto dataSet = loadTrainingDataSet("training_data_100_points_7_features.ds");
    std::cout << "Ingesting data..." << std::endl;
    auto start = std::chrono::system_clock::now();
    auto dataSet = loadTrainingDataSet("largeset.ds");
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>( end - start );
    std::cout << "Dataset loaded: " << dataSet->size() << " points. (" << elapsed.count() << " seconds)." << std::endl;

    // Train a random forest on the data.
    std::cout << "Building indices..." << std::endl;
    start = std::chrono::system_clock::now();
    unsigned int MAX_DEPTH = 50;
    BinaryRandomForestTrainer trainer( dataSet, MAX_DEPTH );
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::seconds>( end - start );
    std::cout <<"Done (" << elapsed.count() << " seconds)." << std::endl;
    std::cout << "Training..." << std::endl;
    start = std::chrono::system_clock::now();
    trainer.train();
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::seconds>( end - start );
    std::cout << "Done (" << elapsed.count() << " seconds)." << std::endl;

    // Finish.
    return EXIT_SUCCESS;
}
