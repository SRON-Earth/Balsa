#include <iostream>

#include <balsa.h>

/**
 * This example program demonstrates basic classification using the Balsa
 * library.
 */
int main( int, char ** )
{
    // Create a random forest classifier.
    RandomForestClassifier classifier1( "demomodel_3_features.mod", 3 );

    // Create some data points and an output location for the labels.
    bool   labels1[3];
    double points1[] = { 1.0, 1.1, 1.2,   // Point 0
                         2.0, 2.1, 2.2,   // Point 1
                         3.0, 3.1, 3.2 }; // Point 2

    // Run the classifier.
    classifier1.classify( points1, points1 + 9, labels1 );

    // N.B. Subsequent examples show how the data can be read from- and written
    // to other types of containers.

    return EXIT_SUCCESS;
}

