#include <balsa.h>
#include <iostream>

using namespace balsa;

int main( int, char ** )
{
    // Load data and labels.
    auto dataSet      = readTableAs<double>( "fruit-data.balsa" );
    auto labels       = readTableAs<Label>( "fruit-labels.balsa" );
    auto featureCount = dataSet.getColumnCount();

    // Create an output stream for writing decision tree models to an ensemble file.
    EnsembleFileOutputStream outputStream( "fruit-model.balsa" );

    // Create a trainer and train it on the data.
    RandomForestTrainer trainer( outputStream );
    trainer.train( dataSet.begin(), dataSet.end(), featureCount, labels.begin() );

    return 0;
}
