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

    // TrainingDataSet::SharedPointer dataSet( new TrainingDataSet( 2  ) );
    // DataPoint
    // p = { 0.0, 0.0 };
    // dataSet->appendDataPoint( p, false );
    // p = { 0.0, 1.0 };
    // dataSet->appendDataPoint( p, false );
    // p = { 1.0, 0.0 };
    // dataSet->appendDataPoint( p, false );
    // p = { 1.0, 1.0 };
    // dataSet->appendDataPoint( p, true );


    // Train a random forest on the data.
    BinaryRandomForestTrainer trainer( dataSet );
    trainer.train();
    std::cout << "Done." << std::endl;

    // Finish.
    return EXIT_SUCCESS;
}
