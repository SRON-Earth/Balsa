#include <iostream>

#include "datamodel.h"

#include <vector>
#include <random>
#include <chrono>

int main( int, char ** )
{
    // Some constants for testing purposes.
    const unsigned int FEATURE_COUNT = 12;
    const unsigned int LABEL_COUNT   = 2;
    const unsigned int TEST_SET_SIZE = 20000000;
    const unsigned int SEED          = 123;

    // Create an empty training data set.
    TrainingDataSet dataset( FEATURE_COUNT );

    // Generate random datapoints.
    std::cout << "Generating random data..." << std::endl;
    std::mt19937 generator( SEED );
    std::uniform_real_distribution<> feature_distribution( 0.0, 100.0           );
    std::uniform_int_distribution <> label_distribution  ( 0  , LABEL_COUNT - 1 );
    for ( unsigned int i = 0; i < TEST_SET_SIZE; ++i )
    {
        // Create a datapoint with random data.
        auto point = DataPoint();
        point.reserve( FEATURE_COUNT );
        for ( unsigned int i = 0; i < FEATURE_COUNT; ++i )
        {
            point.push_back( feature_distribution( generator ) );
        }

        // Create a random label.
        DataPointLabel label = label_distribution( generator );

        // Add the point and its label to the training set.
        dataset.appendDataPoint( point, label );
    }
    std::cout << "Generated " << dataset.size() << " points with " << dataset.getFeatureCount() << " features." << std::endl;

    // Create a feature index for fast, ordered traversal.
    std::cout << "Building feature traversal index..." << std::endl;
    FeatureIndex index( dataset );
    std::cout << "Done." << std::endl;

    //dataset.dump();

    std::cout << "Passing all features..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    double t = 0;
    for ( unsigned int f = 0; f < FEATURE_COUNT; ++ f )
    {
        for ( unsigned int i = 0; i < TEST_SET_SIZE; ++i )
        {
            t += dataset.getFeatureValue( i, f );
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "Done. " << t << std::endl;
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish-start);
    std::cout << microseconds.count() << " microseconds." << std::endl;

    // Finish.
    return EXIT_SUCCESS;
}
