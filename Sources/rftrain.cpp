#include <iostream>

#include "datamodel.h"

#include <vector>
#include <random>
#include <chrono>

#include "ingestion.h"

int main( int, char ** )
{
    // Load training data set.
    auto dataSet = loadTrainingDataSet("training_data_100_points_7_features.ds");

    // Train a random forest on the data.
    BinaryRandomForestTrainer trainer( dataSet );
    trainer.train();
    std::cout << "Done." << std::endl;

    // Finish.
    return EXIT_SUCCESS;
}
