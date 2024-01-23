#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "exceptions.h"
#include "ingestion.h"
#include "randomforestclassifier.h"
#include "serdes.h"
#include "timing.h"

namespace
{
class Options
{
public:

    Options()
    : threadCount( 1 )
    , maxPreload( 1 )
    {
    }

    static std::string getUsage()
    {
        std::stringstream ss;
        ss << "Usage:" << std::endl
           << std::endl
           << "   balsa_classify [options] <model file> <datapoint file> <output file>" << std::endl
           << std::endl
           << " Options:" << std::endl
           << std::endl
           << "   -t <thread count> :  Sets the number of threads (default is 1)." << std::endl
           << "   -p <preload count>: Sets the number of trees to preload (default is 1)." << std::endl;
        return ss.str();
    }

    static Options parseOptions( int argc, char ** argv )
    {
        // Put all arguments in a stringstream.
        std::stringstream args;
        for ( int i = 0; i < argc; ++i ) args << ' ' << argv[i];

        // Discard the executable name.
        std::string token;
        args >> token;

        // Parse all flags.
        Options options;
        while ( args >> token )
        {
            // Stop if the token is not a flag.
            assert( token.size() );
            if ( token[0] != '-' ) break;

            // Parse the '-t <threadcount>' option.
            if ( token == "-t" )
            {
                if ( !( args >> options.threadCount ) ) throw ParseError( "Missing parameter to -t option." );
            }
            else if ( token == "-p" )
            {
                if ( !( args >> options.maxPreload ) ) throw ParseError( "Missing parameter to -p option." );
            }
            else
            {
                throw ParseError( std::string( "Unknown option: " ) + token );
            }
        }

        // Parse the filenames.
        if ( token.size() == 0 ) throw ParseError( getUsage() );
        options.modelFile = token;
        if ( !( args >> options.dataFile ) ) throw ParseError( getUsage() );
        if ( !( args >> options.outputFile ) ) throw ParseError( getUsage() );

        // Return  results.
        return options;
    }

    std::string modelFile;
    std::string dataFile;
    std::string outputFile;
    unsigned int threadCount;
    unsigned int maxPreload;
};

void writeLabels( const std::vector<bool> & labels, const std::string & filename )
{
    // Open the output file stream.
    std::ofstream os( filename.c_str(), std::ofstream::binary );
    assert( os.good() );

    // Write the number of columns.
    serialize<std::uint32_t>( os, 1 );

    // Write the label values.
    for ( const bool & label : labels ) serialize<float>( os, label );
}
} // namespace

int main( int argc, char ** argv )
{
    try
    {
        // Parse the command-line arguments.
        Options options = Options::parseOptions( argc, argv );

        // Debug.
        std::cout << "Model File : " << options.modelFile << std::endl;
        std::cout << "Data File  : " << options.dataFile << std::endl;
        std::cout << "Output File: " << options.outputFile << std::endl;
        std::cout << "Threads    : " << options.threadCount << std::endl;
        std::cout << "Preload    : " << options.maxPreload << std::endl;
        std::cout << std::endl;
        assert( options.threadCount > 0 );

        // Load the data.
        StopWatch watch;
        std::cout << "Ingesting data..." << std::endl;
        watch.start();
        auto dataSet = loadDataSet( options.dataFile );
        std::cout << "Dataset loaded: " << dataSet->getFeatureCount() << " features x " << dataSet->size() << " points."
                  << std::endl;
        const auto dataLoadTime = watch.getElapsedTime();

        // Classify the data points.
        watch.start();
        std::vector<bool> labels( dataSet->size(), false );
        typedef typename std::decay_t<decltype( dataSet->getData() )>::const_iterator FeatureIteratorType;
        typedef typename std::vector<bool>::iterator OutputIteratorType;
        typedef RandomForestClassifier<FeatureIteratorType, OutputIteratorType, double, bool> ClassifierType;
        ClassifierType classifier( options.modelFile,
            dataSet->getFeatureCount(),
            options.threadCount - 1,
            options.maxPreload );
        classifier.classify( dataSet->getData().begin(), dataSet->getData().end(), labels.begin() );
        watch.stop();
        const auto classificationTime = watch.getElapsedTime();

        // Store the labels.
        watch.start();
        writeLabels( labels, options.outputFile );
        watch.stop();
        const auto labelStoreTime = watch.getElapsedTime();

        std::cout << "Timings:"
                  << std::endl
                  // << "Model Load Time: " << modelLoadTime << std::endl
                  << "Data Load Time: " << dataLoadTime << std::endl
                  << "Classification Time: " << classificationTime << std::endl
                  << "Label Store Time: " << labelStoreTime << std::endl;
    }
    catch ( Exception & e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
