#include <iostream>

#include "datamodel.h"

#include <vector>
#include <random>
#include <chrono>

int main( int, char ** )
{
    // Some constants for testing purposes.
    const unsigned int FEATURE_COUNT = 12;
    const unsigned int TEST_SET_SIZE = 1000;
    const unsigned int SEED          = 123;

    // Create an empty training data set.
    TrainingDataSet dataset( FEATURE_COUNT );

    // Generate random datapoints.
    std::cout << "Generating random data..." << std::endl;
    std::mt19937 generator( SEED );
    std::uniform_real_distribution<> feature_distribution( 0.0, 100.0 );
    std::uniform_int_distribution <> label_distribution  ( 0  , 1     );
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
        bool label = label_distribution( generator );

        // Add the point and its label to the training set.
        dataset.appendDataPoint( point, label );
    }
    std::cout << "Generated " << dataset.size() << " points with " << dataset.getFeatureCount() << " features." << std::endl;

    // Create a feature index for fast, ordered traversal.
    std::cout << "Building feature traversal index..." << std::endl;
    FeatureIndex index( dataset );
    std::cout << "Done." << std::endl;

    // Finish.
    return EXIT_SUCCESS;
}
