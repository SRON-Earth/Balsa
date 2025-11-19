#include <iostream>
#include <balsa.h>

using namespace balsa;

int main( int, char ** )
{
    // Read (and possibly convert) the data.
    auto dataSet = readTableAs<double>( "fruit-points.balsa" );

    // Classify the data.
    Table<Label>           labels( dataSet.getRowCount(), 1 );
    RandomForestClassifier classifier( "fruit-model.balsa" );
    classifier.classify( dataSet.begin(), dataSet.end(), labels.begin() );

    // Write the result to a binary Balsa output file.
    writeTable( labels, "fruit-classifier-labels.balsa" );

    // Print the results as text (or write to a text file).
    std::cout << labels << std::endl;

    return 0;
}
