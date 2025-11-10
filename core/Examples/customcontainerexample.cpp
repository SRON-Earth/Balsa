#include <balsa.h>

using namespace balsa;

int main( int, char ** )
{
    // Define some alternative data- and label-containers.
    typedef std::vector<float>  Points;
    typedef std::valarray<int>  Labels;

    // Create fake data.
    const std::size_t POINTCOUNT   = 100;
    const std::size_t FEATURECOUNT = 4;
    Points points( POINTCOUNT * FEATURECOUNT );
    Labels labels( POINTCOUNT );

    // Classify the data.
    RandomForestClassifier classifier( "fruit-model.balsa" );
    classifier.classify( points.begin(), points.end(), std::begin( labels ) );

    return 0;
}
