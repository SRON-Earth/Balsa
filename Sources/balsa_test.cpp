#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>

#include "classifierfilestream.h"
#include "datagenerator.h"
#include "datatypes.h"
#include "ensembleclassifier.h"
#include "randomforesttrainer.h"
#include "table.h"

using namespace balsa;

/**
 * Represents of a file with a specified name. The file will be deleted when
 * when the associated NamedTemporaryFile object is destroyed.
 */
class NamedTemporaryFile
{
public:

    NamedTemporaryFile( const std::filesystem::path & path ):
    m_path( path )
    {
    }

    ~NamedTemporaryFile()
    {
        std::filesystem::remove( m_path );
    }

    operator std::string() const
    {
        return m_path;
    }

private:

    std::filesystem::path m_path;
};

template <typename FeatureType>
bool testCross2x2()
{
    // Create a square where the data points on one diagonal belong to class A,
    // and the data points on the other diagonal belong to class B.
    FeatureType  points[] = { -1, 1, 1, 1, -1, -1, 1, -1 };
    uint8_t truth[]  = { 0, 1, 1, 0 };

    // Train a single decision tree.
    NamedTemporaryFile modelFile( "balsa_test_cross_2x2.tmp" );
    {
        EnsembleFileOutputStream outputStream( modelFile );
        RandomForestTrainer<FeatureType *, uint8_t *> trainer( outputStream, 2, std::numeric_limits<unsigned int>::max(), 1.0, 1, 1 );
        trainer.train( points, points + 8, 2, truth );
    }

    // Classify the training data.
    uint8_t labels[4];
    ClassifierFileInputStream inputStream( modelFile, 0 );
    EnsembleClassifier classifier( inputStream, 0 );
    classifier.classify( points, points + 8, labels );

    // Ensure the classification result matches the ground truth exactly.
    return std::equal( labels, labels + 4, truth );
}

template <typename FeatureType>
bool testCheckerboard()
{
    // Construct a multi-source model with a 2-D checkerboard.
    typename CheckerboardFeatureGenerator<FeatureType>::SharedPointer black( new CheckerboardFeatureGenerator<FeatureType>( CheckerboardFeatureGenerator<FeatureType>::Color::BLACK ) );
    black->addDimension( 16, 1.0 );
    black->addDimension( 32, 0.75 );
    typename CheckerboardFeatureGenerator<FeatureType>::SharedPointer white( new CheckerboardFeatureGenerator<FeatureType>( CheckerboardFeatureGenerator<FeatureType>::Color::WHITE ) );
    white->addDimension( 16, 1.0 );
    white->addDimension( 32, 0.75 );
    typename SingleSourceGenerator<FeatureType>::SharedPointer blackSource( new SingleSourceGenerator<FeatureType>() );
    blackSource->addFeatureGenerator( black );
    typename SingleSourceGenerator<FeatureType>::SharedPointer whiteSource( new SingleSourceGenerator<FeatureType>() );
    whiteSource->addFeatureGenerator( white );
    MultiSourceGenerator<FeatureType> generator( 0, 2 );
    generator.addSource( 1, blackSource );
    generator.addSource( 1, whiteSource );

    // Generate a data- and label set.
    Table<FeatureType> points( 2 );
    Table<Label>       truth( 1 );
    generator.generate( 10000, points, truth );

    // Train a single decision tree.
    NamedTemporaryFile modelFile( "balsa_test_checkerboard.tmp" );
    {
        EnsembleFileOutputStream outputStream( modelFile );
        RandomForestTrainer<typename Table<FeatureType>::ConstIterator> trainer( outputStream, generator.getFeatureCount(), std::numeric_limits<unsigned int>::max(), 1.0, 1, 1 );
        trainer.train( points.begin(), points.end(), points.getColumnCount(), truth.begin() );
    }

    // Classify the training data.
    Table<Label> labels( points.getRowCount(), 1 );
    ClassifierFileInputStream inputStream( modelFile, 0 );
    EnsembleClassifier classifier( inputStream, 0 );
    classifier.classify( points.begin(), points.end(), labels.begin() );

    // Ensure the classification result matches the ground truth exactly.
    return labels == truth;
}

template <typename FeatureType>
bool testConcentricRings()
{
    // Construct a multi-source model with three concentric rings.
    typename SingleSourceGenerator<FeatureType>::SharedPointer ring0( new SingleSourceGenerator<FeatureType>() );
    typename SingleSourceGenerator<FeatureType>::SharedPointer ring1( new SingleSourceGenerator<FeatureType>() );
    typename SingleSourceGenerator<FeatureType>::SharedPointer ring2( new SingleSourceGenerator<FeatureType>() );
    ring0->addFeatureGenerator( typename FeatureGenerator<FeatureType>::SharedPointer( new AnnulusFeatureGenerator<FeatureType>( 0.0, 2.0 ) ) );
    ring1->addFeatureGenerator( typename FeatureGenerator<FeatureType>::SharedPointer( new AnnulusFeatureGenerator<FeatureType>( 2.25, 3.25 ) ) );
    ring2->addFeatureGenerator( typename FeatureGenerator<FeatureType>::SharedPointer( new AnnulusFeatureGenerator<FeatureType>( 3.5, 7.0 ) ) );
    MultiSourceGenerator<FeatureType> generator( 0, 2 );
    generator.addSource( 1, ring0 );
    generator.addSource( 1, ring1 );
    generator.addSource( 1, ring2 );

    // Generate a data- and label set.
    Table<FeatureType> points( 2 );
    Table<Label>       truth( 1 );
    generator.generate( 10000, points, truth );

    // Train a single decision tree.
    NamedTemporaryFile modelFile( "balsa_test_concentric_rings.tmp" );
    {
        EnsembleFileOutputStream outputStream( modelFile );
        RandomForestTrainer<typename Table<FeatureType>::ConstIterator> trainer( outputStream, generator.getFeatureCount(), std::numeric_limits<unsigned int>::max(), 1.0, 1, 1 );
        trainer.train( points.begin(), points.end(), points.getColumnCount(), truth.begin() );
    }

    // Classify the training data.
    Table<Label> labels( points.getRowCount(), 1 );
    ClassifierFileInputStream inputStream( modelFile, 0 );
    EnsembleClassifier classifier( inputStream, 0 );
    classifier.classify( points.begin(), points.end(), labels.begin() );

    // Ensure the classification result matches the ground truth exactly.
    return labels == truth;
}

bool execute_test( const std::string & name, bool ( *test )( void ) )
{
    // Run a single test and return the test result.
    bool              result  = true;
    const std::size_t padding = ( name.size() > 60 ? 0 : 60 - name.size() );
    std::cout << name << std::string( padding, '.' ) << " ";
    try
    {
        result = test();
    }
    catch ( Exception & e )
    {
        std::cout << "FAIL" << std::endl;
        throw;
    }
    std::cout << ( result ? "PASS" : "FAIL" ) << std::endl;
    return result;
}

int main()
{
    // Define the combined test result.
    bool result = true;

    // Run all tests (even if one or more tests fail).
    try
    {
        result &= execute_test( "testCross2x2<float>", testCross2x2<float> );
        result &= execute_test( "testCross2x2<double>", testCross2x2<double> );
        result &= execute_test( "testCheckerboard<float>", testCheckerboard<float> );
        result &= execute_test( "testCheckerboard<double>", testCheckerboard<double> );
        result &= execute_test( "testConcentricRings<float>", testConcentricRings<float> );
        result &= execute_test( "testConcentricRings<double>", testConcentricRings<double> );
    }
    catch ( Exception & e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return ( result ? EXIT_SUCCESS : EXIT_FAILURE );
}
